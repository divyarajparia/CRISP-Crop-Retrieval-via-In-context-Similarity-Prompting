"""
GAICD calibration head — Phase 3 of plan_novelty_2.md.

Why this exists:
    The four prompt/retrieval/generation novelty ideas (visual grounding,
    rank-anchored refinement, multi-temperature, diverse ICL) all came back
    noise-level (±0.012 IoU) on the deterministic 30-sample slice. Phase 1
    re-weighted the existing CombinedScorer (vila=1.0, area=0.0) and also
    landed flat (-0.0043 IoU) — VILA and (VILA+Area) already mostly agree on
    Mantis's R=6 candidates, so re-mixing existing components adds no
    information.

    The remaining lever with realistic upside is *adding a new signal that
    has actually seen GAICD MOS labels*. GAICD train has 2,636 images x ~90
    annotated crops = ~237k labeled (crop, MOS) pairs that nothing in the
    pipeline learns from.

    This module trains a small head on those labels using:
        - features: CLIP ViT-L/14 image embedding (768-d) on the cropped
          image, concatenated with 6 normalized geometry features that
          encode where the crop sits in the original image
        - loss: pairwise ranking (RankNet for Ridge/MLP; lambdarank for
          LightGBM) — the inference task is "pick the best of R candidates"
          so train to the same objective
        - architecture ablation: Ridge / 2-layer MLP / LGBMRanker. Pick the
          winner on a 90/10 image-level holdout by val pair-ranking accuracy.

    The trained head is then exposed as `GaicdCalibrationScorer`, a new
    BaseScorer that plugs into CombinedScorer.scorers via the same
    weight-forwarding plumbing Phase 1 added (`weights={...}` argument to
    create_scorer).

Design choices:
    - CLIP ViT-L/14 (frozen) instead of VILA pooled features. VILA's
      _vila_score:360-411 only exposes a scalar `quality_scores`; extracting
      a pooled embedding would require hooking the JAX/Flax internals, which
      is brittle. open_clip ViT-L/14 is already a known-good loader path
      (LAION fallback at scorer.py:248-313 uses it) and gives 768-d embeddings
      out of the box.
    - Ranking loss, not regression. We never need calibrated MOS at inference
      time — we only need ordering across R Mantis candidates.
    - "Smart middle": Ridge + MLP + LightGBM ablation, not vanilla Ridge alone
      and not a cross-encoder transformer. The user explicitly chose this
      complexity tier in planning.

This file contains:
    1. CLIP feature extractor (used by both training and inference)
    2. Geometry feature builder (6-d, used by both training and inference)
    3. RankNet pair sampler + losses
    4. Three head trainers (Ridge / MLP / LGBMRanker)
    5. GaicdCalibrationScorer — BaseScorer subclass loaded at inference time

The training-side helpers are imported by build_calhead_features.py and
train_calhead.py (sibling top-level scripts). The GaicdCalibrationScorer
class is imported by models/scorer.py:create_scorer when "gaicd_cal" appears
in the scorer_config string.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .scorer import BaseScorer

logger = logging.getLogger(__name__)


# ============================================================================
# Feature extractor (CLIP ViT-L/14, frozen) — shared by training + inference
# ============================================================================


CLIP_EMBED_DIM = 768
GEOMETRY_DIM = 6
FEATURE_DIM = CLIP_EMBED_DIM + GEOMETRY_DIM  # 774


class ClipViTL14Encoder:
    """Frozen CLIP ViT-L/14 image encoder.

    Mirrors the loader pattern in scorer.py:_try_load_laion (lines 248-313).
    Wraps a single `encode(pil_image)` call that returns a 768-d float32
    numpy array (not normalized — downstream code handles normalization
    if needed).

    The encoder is loaded once per process. At training time we batch many
    crops through it; at inference time GaicdCalibrationScorer calls it on
    one crop at a time inside the refinement loop, so a small per-call
    overhead is fine.
    """

    def __init__(self, device: str = "cuda"):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info("ClipViTL14Encoder loaded on %s", device)

    @torch.no_grad()
    def encode(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL image to a 768-d numpy float32 vector."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(tensor)
        return feats.squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images. Returns (N, 768) float32."""
        if not images:
            return np.zeros((0, CLIP_EMBED_DIM), dtype=np.float32)
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        feats = self.model.encode_image(tensors)
        return feats.float().cpu().numpy()


# ============================================================================
# Geometry features
# ============================================================================


def geometry_features(
    crop_box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Compute 6-d normalized geometry features for a crop.

    Args:
        crop_box: (x1, y1, x2, y2) in pixel coordinates
        image_size: (W, H) of the original image (PIL .size order)

    Returns:
        6-d float32 array:
            [x1/W, y1/H, x2/W, y2/H,
             aspect_ratio = (x2-x1)/(y2-y1),
             area_ratio   = ((x2-x1)*(y2-y1)) / (W*H)]

    Pooled CLIP embeddings discard spatial position, so a head that only sees
    the cropped pixels can't tell whether a 600x400 crop was taken from the
    top-left or center of a 1200x800 image. Geometry features close that gap
    cheaply.
    """
    x1, y1, x2, y2 = crop_box
    W, H = image_size
    W = max(1, W)
    H = max(1, H)
    cw = max(1, x2 - x1)
    ch = max(1, y2 - y1)
    return np.array(
        [
            x1 / W,
            y1 / H,
            x2 / W,
            y2 / H,
            cw / ch,
            (cw * ch) / (W * H),
        ],
        dtype=np.float32,
    )


# ============================================================================
# Pair sampling for ranking loss
# ============================================================================


@dataclass
class PairBatch:
    """A batch of (positive, negative) feature pairs for RankNet training."""

    feat_pos: np.ndarray  # (B, FEATURE_DIM)
    feat_neg: np.ndarray  # (B, FEATURE_DIM)


def sample_pairs(
    features: np.ndarray,
    mos: np.ndarray,
    image_idx: np.ndarray,
    pairs_per_image: int = 20,
    mos_margin: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (a, b) pairs grouped by image where mos[a] > mos[b] + margin.

    Args:
        features: (N, FEATURE_DIM) — all crops, all images
        mos: (N,) — MOS labels in GAICD's native range
        image_idx: (N,) — image group id; pairs are only sampled within
            the same group so the head learns to rank crops from the same
            photo, which is exactly the inference task
        pairs_per_image: K from the plan
        mos_margin: skip near-ties (filters out the long flat tail at the
            top of GAICD's MOS distribution where annotator agreement is
            low)
        rng: numpy Generator for reproducibility

    Returns:
        (idx_pos, idx_neg): two int64 arrays of equal length giving the
        feature row indices on each side of the pair. Caller materializes
        the actual feature differences for Ridge or feeds rows directly
        into the MLP.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pos_list: List[int] = []
    neg_list: List[int] = []

    unique_imgs = np.unique(image_idx)
    for img in unique_imgs:
        mask = image_idx == img
        rows = np.where(mask)[0]
        if len(rows) < 2:
            continue
        img_mos = mos[rows]
        order = np.argsort(-img_mos)  # descending
        rows_sorted = rows[order]
        mos_sorted = img_mos[order]

        for _ in range(pairs_per_image):
            # Sample two distinct positions; require margin
            for _attempt in range(10):
                i, j = rng.integers(0, len(rows_sorted), size=2)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if mos_sorted[i] - mos_sorted[j] >= mos_margin:
                    pos_list.append(int(rows_sorted[i]))
                    neg_list.append(int(rows_sorted[j]))
                    break

    return np.array(pos_list, dtype=np.int64), np.array(neg_list, dtype=np.int64)


# ============================================================================
# Three head trainers
# ============================================================================


def train_ridge_ranking(
    features: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    alpha: float = 1.0,
):
    """RankSVM-style Ridge: fit linear w on (x_a - x_b) -> +1.

    Closed form via sklearn Ridge. Returns a fitted sklearn estimator.
    Inference: score(x) = w . x.
    """
    from sklearn.linear_model import Ridge

    diffs = features[pos_idx] - features[neg_idx]
    targets = np.ones(len(diffs), dtype=np.float32)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(diffs, targets)
    return model


class RankNetMLP(torch.nn.Module):
    """2-layer MLP scoring head, trained with RankNet loss."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_ranking(
    features: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    val_pos_idx: Optional[np.ndarray] = None,
    val_neg_idx: Optional[np.ndarray] = None,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> RankNetMLP:
    """Train RankNetMLP with pairwise log-loss.

    Loss: L = log(1 + exp(-(s_pos - s_neg)))
    """
    model = RankNetMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    feats_t = torch.from_numpy(features).float().to(device)

    n_pairs = len(pos_idx)
    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(n_pairs)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch_size):
            batch = order[start : start + batch_size]
            p = pos_idx[batch]
            n = neg_idx[batch]
            s_pos = model(feats_t[p])
            s_neg = model(feats_t[n])
            loss = torch.nn.functional.softplus(-(s_pos - s_neg)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        val_acc = -1.0
        if val_pos_idx is not None and val_neg_idx is not None:
            val_acc = pair_accuracy_torch(model, feats_t, val_pos_idx, val_neg_idx)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(
            "MLP epoch %d/%d: train_loss=%.4f val_pair_acc=%.4f",
            epoch + 1,
            epochs,
            avg_loss,
            val_acc,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_lightgbm_ranking(
    features: np.ndarray,
    mos: np.ndarray,
    image_idx: np.ndarray,
    train_mask: np.ndarray,
):
    """LightGBM lambdarank head.

    LightGBM's ranker takes per-row labels + a `group` array giving the
    sizes of consecutive groups. We sort the train rows by image_idx so
    rows from the same image are contiguous, then build the group sizes.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError(
            "LightGBM not installed; pip install lightgbm or skip this head"
        ) from e

    train_rows = np.where(train_mask)[0]
    order = np.argsort(image_idx[train_rows], kind="stable")
    train_rows = train_rows[order]

    X = features[train_rows]
    # LightGBM ranker expects integer relevance labels; map MOS into ints by
    # rounding to one decimal then scaling. The exact range doesn't matter
    # as long as ordering is preserved.
    y_float = mos[train_rows]
    y = np.round((y_float - y_float.min()) * 10).astype(int)

    grp_ids = image_idx[train_rows]
    _, group_sizes = np.unique(grp_ids, return_counts=True)
    # np.unique returns sorted unique values, but we sorted train_rows by
    # image_idx already so the order matches.

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=5,
        random_state=42,
    )
    model.fit(X, y, group=group_sizes)
    return model


# ============================================================================
# Pair accuracy helpers (used to pick the winning head)
# ============================================================================


def pair_accuracy_linear(w: np.ndarray, features: np.ndarray, pos_idx, neg_idx) -> float:
    s_pos = features[pos_idx] @ w
    s_neg = features[neg_idx] @ w
    return float((s_pos > s_neg).mean())


def pair_accuracy_torch(model: RankNetMLP, feats_t: torch.Tensor, pos_idx, neg_idx) -> float:
    model.eval()
    with torch.no_grad():
        s_pos = model(feats_t[pos_idx])
        s_neg = model(feats_t[neg_idx])
    return float((s_pos > s_neg).float().mean().item())


def pair_accuracy_lgbm(model, features: np.ndarray, pos_idx, neg_idx) -> float:
    s_pos = model.predict(features[pos_idx])
    s_neg = model.predict(features[neg_idx])
    return float((s_pos > s_neg).mean())


# ============================================================================
# Persistence wrapper — single .pkl that train_calhead.py writes and
# GaicdCalibrationScorer reads
# ============================================================================


@dataclass
class CalHeadCheckpoint:
    """Serialized form of the trained head.

    head_type:
        "ridge" -> coef is a (FEATURE_DIM,) numpy weight vector
        "mlp"   -> coef is the MLP state_dict (dict of CPU tensors)
        "lgbm"  -> coef is the lightgbm Booster (pickled directly)
    """

    head_type: str
    coef: object
    val_pair_acc: float
    feature_dim: int = FEATURE_DIM


def save_checkpoint(ckpt: CalHeadCheckpoint, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    logger.info("Saved CalHeadCheckpoint head_type=%s val_acc=%.4f -> %s",
                ckpt.head_type, ckpt.val_pair_acc, path)


def load_checkpoint(path: str) -> CalHeadCheckpoint:
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# Inference-time scorer — the BaseScorer that slots into CombinedScorer
# ============================================================================


class GaicdCalibrationScorer(BaseScorer):
    """BaseScorer wrapping the trained calibration head.

    Plugged into CombinedScorer.scorers via models/scorer.py:create_scorer
    when "gaicd_cal" appears in the scorer_config string.

    Unlike the existing scorers, this one needs both the cropped image
    (for CLIP encoding) AND the original image dimensions + the crop box
    in pixel space (for the geometry features). The original image is set
    via set_original() — the same hook CombinedScorer already calls — and
    the per-call crop_box is forwarded by the modified CombinedScorer.score
    / score_batch signatures.
    """

    def __init__(
        self,
        head_path: str,
        device: str = "cuda",
        encoder: Optional[ClipViTL14Encoder] = None,
    ):
        self.device = device
        self.head_path = head_path
        self.encoder = encoder if encoder is not None else ClipViTL14Encoder(device=device)
        self.ckpt = load_checkpoint(head_path)
        self.original_size: Optional[Tuple[int, int]] = None

        # Pre-materialize a torch model for the MLP path so we don't rebuild
        # it on every call.
        self._mlp_model: Optional[RankNetMLP] = None
        if self.ckpt.head_type == "mlp":
            self._mlp_model = RankNetMLP(in_dim=self.ckpt.feature_dim)
            self._mlp_model.load_state_dict(self.ckpt.coef)
            self._mlp_model.eval()

        logger.info(
            "GaicdCalibrationScorer ready (head_type=%s val_pair_acc=%.4f)",
            self.ckpt.head_type,
            self.ckpt.val_pair_acc,
        )

    def set_original(self, image: Image.Image):
        self.original_size = image.size

    def set_original_size(self, size: Tuple[int, int]):
        # Mirror AreaScorer's interface in case CombinedScorer happens to
        # call this one instead.
        self.original_size = size

    def score(
        self,
        crop_image: Image.Image,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """Score one crop. Returns a value in [0, 1] (sigmoid-squashed)."""
        emb = self.encoder.encode(crop_image)  # (768,)

        if crop_box is not None and self.original_size is not None:
            geom = geometry_features(crop_box, self.original_size)
        else:
            # Fallback if upstream forgot to pass the box: use the crop's
            # own dimensions, full-image bbox, aspect ratio = w/h, area = 1.0.
            cw, ch = crop_image.size
            geom = np.array(
                [0.0, 0.0, 1.0, 1.0, cw / max(1, ch), 1.0],
                dtype=np.float32,
            )

        feat = np.concatenate([emb, geom]).astype(np.float32)

        if self.ckpt.head_type == "ridge":
            raw = float(feat @ self.ckpt.coef)
        elif self.ckpt.head_type == "mlp":
            with torch.no_grad():
                raw = float(self._mlp_model(torch.from_numpy(feat).unsqueeze(0)).item())
        elif self.ckpt.head_type == "lgbm":
            raw = float(self.ckpt.coef.predict(feat.reshape(1, -1))[0])
        else:
            raise ValueError(f"Unknown head_type: {self.ckpt.head_type}")

        # Sigmoid squash so the unbounded ranking score lands in [0, 1] and
        # plays nicely with CombinedScorer's weighted average.
        return float(1.0 / (1.0 + np.exp(-raw)))

    def score_batch(
        self,
        crop_images: List[Image.Image],
        crop_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[float]:
        """Score multiple crops, batching the CLIP forward pass."""
        if not crop_images:
            return []
        embs = self.encoder.encode_batch(crop_images)  # (N, 768)

        if crop_boxes is not None and self.original_size is not None:
            geoms = np.stack(
                [geometry_features(b, self.original_size) for b in crop_boxes]
            )
        else:
            geoms = np.zeros((len(crop_images), GEOMETRY_DIM), dtype=np.float32)
            for i, img in enumerate(crop_images):
                cw, ch = img.size
                geoms[i] = [0.0, 0.0, 1.0, 1.0, cw / max(1, ch), 1.0]

        feats = np.concatenate([embs, geoms], axis=1).astype(np.float32)

        if self.ckpt.head_type == "ridge":
            raws = feats @ self.ckpt.coef
        elif self.ckpt.head_type == "mlp":
            with torch.no_grad():
                raws = self._mlp_model(torch.from_numpy(feats)).cpu().numpy()
        elif self.ckpt.head_type == "lgbm":
            raws = self.ckpt.coef.predict(feats)
        else:
            raise ValueError(f"Unknown head_type: {self.ckpt.head_type}")

        return [float(1.0 / (1.0 + np.exp(-r))) for r in raws]
