#!/usr/bin/env python
"""
Phase 3 — Step 2: train + ablate the three calibration head architectures.

What this does:
    1. Loads cache/gaicd_train_features.npz (output of build_calhead_features.py)
    2. Splits images 90/10 by image_id (image-level holdout, not crop-level)
    3. Samples K=20 pairs per image with mos margin >= 0.5
    4. Trains:
        - Ridge (RankSVM-style: fit on (x_a - x_b) -> +1)
        - 2-layer MLP with RankNet log-loss
        - LightGBM LGBMRanker (lambdarank objective)
    5. Evaluates each on the val pair set by pair-ranking accuracy
    6. Picks the winner (best val acc, tiebreak Ridge for stability) and
       saves to cache/gaicd_cal_head.pkl
    7. Prints a small ablation table for the writeup

This script is CPU-only — no GPU required, runs in ~5 minutes.

Usage (no tmux needed, but fine to run there):
    cd /data1/es22btech11013/divya/AFCIL/divya/cv-project/cropper
    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python train_calhead.py

Output:
    cropper/cache/gaicd_cal_head.pkl  (CalHeadCheckpoint pickle)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models.gaicd_calibration_head import (  # noqa: E402
    FEATURE_DIM,
    CalHeadCheckpoint,
    pair_accuracy_lgbm,
    pair_accuracy_linear,
    pair_accuracy_torch,
    sample_pairs,
    save_checkpoint,
    train_lightgbm_ranking,
    train_mlp_ranking,
    train_ridge_ranking,
)
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("train_calhead")


# ============================================================================
# Config
# ============================================================================

FEATURES_PATH = SCRIPT_DIR / "cache" / "gaicd_train_features.npz"
OUT_PATH      = SCRIPT_DIR / "cache" / "gaicd_cal_head.pkl"
SEED          = 42
VAL_FRAC      = 0.10
PAIRS_PER_IMG = 20
MOS_MARGIN    = 0.5
RIDGE_ALPHA   = 1.0
MLP_EPOCHS    = 5
MLP_LR        = 1e-3
MLP_BATCH     = 256


# ============================================================================
# Helpers
# ============================================================================


def split_by_image(image_idx: np.ndarray, val_frac: float, seed: int):
    """Image-level holdout: pick val_frac of unique image_idx values for val."""
    unique = np.unique(image_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = int(round(val_frac * len(unique)))
    val_imgs = set(unique[:n_val].tolist())

    val_mask = np.array([int(i) in val_imgs for i in image_idx])
    train_mask = ~val_mask
    return train_mask, val_mask


# ============================================================================
# Main
# ============================================================================


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run build_calhead_features.py first."
        )

    logger.info("Loading features from %s", FEATURES_PATH)
    data = np.load(FEATURES_PATH, allow_pickle=False)
    features  = data["features"].astype(np.float32)
    mos       = data["mos"].astype(np.float32)
    image_idx = data["image_idx"].astype(np.int64)

    n_total = features.shape[0]
    n_images = int(image_idx.max()) + 1
    logger.info("  features : %s", features.shape)
    logger.info("  mos      : range=[%.2f, %.2f] mean=%.2f",
                float(mos.min()), float(mos.max()), float(mos.mean()))
    logger.info("  images   : %d (mean %.1f crops/img)", n_images, n_total / n_images)
    assert features.shape[1] == FEATURE_DIM, (
        f"feature dim mismatch: cached {features.shape[1]} vs expected {FEATURE_DIM}"
    )

    # Image-level 90/10 split
    train_mask, val_mask = split_by_image(image_idx, VAL_FRAC, SEED)
    logger.info(
        "Split: train rows=%d (%.1f%%), val rows=%d (%.1f%%)",
        int(train_mask.sum()), 100 * train_mask.mean(),
        int(val_mask.sum()),   100 * val_mask.mean(),
    )

    # Pair sampling — use only train rows / val rows respectively
    rng = np.random.default_rng(SEED)
    train_rows = np.where(train_mask)[0]
    val_rows   = np.where(val_mask)[0]
    train_feats  = features[train_rows]
    train_mos    = mos[train_rows]
    train_imgidx = image_idx[train_rows]
    val_feats    = features[val_rows]
    val_mos      = mos[val_rows]
    val_imgidx   = image_idx[val_rows]

    logger.info("Sampling train pairs (K=%d, margin=%.2f)...", PAIRS_PER_IMG, MOS_MARGIN)
    t0 = time.time()
    tr_pos, tr_neg = sample_pairs(
        train_feats, train_mos, train_imgidx,
        pairs_per_image=PAIRS_PER_IMG, mos_margin=MOS_MARGIN, rng=rng,
    )
    logger.info("  %d train pairs in %.1fs", len(tr_pos), time.time() - t0)

    logger.info("Sampling val pairs...")
    t0 = time.time()
    va_pos, va_neg = sample_pairs(
        val_feats, val_mos, val_imgidx,
        pairs_per_image=PAIRS_PER_IMG, mos_margin=MOS_MARGIN,
        rng=np.random.default_rng(SEED + 1),
    )
    logger.info("  %d val pairs in %.1fs", len(va_pos), time.time() - t0)

    if len(tr_pos) == 0 or len(va_pos) == 0:
        raise RuntimeError(
            "Pair sampling produced no pairs — check MOS_MARGIN and dataset"
        )

    results = {}

    # ---- Head A: Ridge -----------------------------------------------------
    logger.info("== Training Ridge head ==")
    t0 = time.time()
    ridge = train_ridge_ranking(train_feats, tr_pos, tr_neg, alpha=RIDGE_ALPHA)
    ridge_w = ridge.coef_.astype(np.float32).reshape(-1)
    val_acc_ridge = pair_accuracy_linear(ridge_w, val_feats, va_pos, va_neg)
    train_acc_ridge = pair_accuracy_linear(ridge_w, train_feats, tr_pos, tr_neg)
    logger.info("  ridge: train_acc=%.4f val_acc=%.4f (%.1fs)",
                train_acc_ridge, val_acc_ridge, time.time() - t0)
    results["ridge"] = (val_acc_ridge, ridge_w)

    # ---- Head B: 2-layer MLP ----------------------------------------------
    logger.info("== Training MLP head ==")
    t0 = time.time()
    torch.manual_seed(SEED)
    mlp = train_mlp_ranking(
        train_feats, tr_pos, tr_neg,
        val_pos_idx=va_pos, val_neg_idx=va_neg,
        epochs=MLP_EPOCHS, batch_size=MLP_BATCH, lr=MLP_LR,
        device="cpu",
    )
    feats_t_train = torch.from_numpy(train_feats).float()
    feats_t_val   = torch.from_numpy(val_feats).float()
    train_acc_mlp = pair_accuracy_torch(mlp, feats_t_train, tr_pos, tr_neg)
    val_acc_mlp   = pair_accuracy_torch(mlp, feats_t_val,   va_pos, va_neg)
    logger.info("  mlp:   train_acc=%.4f val_acc=%.4f (%.1fs)",
                train_acc_mlp, val_acc_mlp, time.time() - t0)
    results["mlp"] = (val_acc_mlp, {k: v.cpu() for k, v in mlp.state_dict().items()})

    # ---- Head C: LightGBM --------------------------------------------------
    try:
        logger.info("== Training LightGBM head ==")
        t0 = time.time()
        # train_lightgbm_ranking expects the *original* feature/index/mask layout
        # since it builds its own group order from image_idx
        lgbm = train_lightgbm_ranking(features, mos, image_idx, train_mask)
        val_acc_lgbm = pair_accuracy_lgbm(lgbm, features, val_rows[va_pos], val_rows[va_neg])
        train_acc_lgbm = pair_accuracy_lgbm(lgbm, features, train_rows[tr_pos], train_rows[tr_neg])
        logger.info("  lgbm:  train_acc=%.4f val_acc=%.4f (%.1fs)",
                    train_acc_lgbm, val_acc_lgbm, time.time() - t0)
        results["lgbm"] = (val_acc_lgbm, lgbm)
    except Exception as e:
        logger.warning("LightGBM head failed: %s — skipping", e)

    # ---- Ablation table ----------------------------------------------------
    print()
    print("=" * 56)
    print("  Calibration head ablation (val pair-ranking accuracy)")
    print("=" * 56)
    print(f"  {'head':<10}  {'val_acc':>8}")
    print(f"  {'-'*10}  {'-'*8}")
    for name in ("ridge", "mlp", "lgbm"):
        if name in results:
            print(f"  {name:<10}  {results[name][0]:>8.4f}")
    print("=" * 56)

    # ---- Pick winner -------------------------------------------------------
    # Highest val acc; tiebreak Ridge > LGBM > MLP for stability/speed
    best_name = max(
        results.keys(),
        key=lambda n: (
            results[n][0],
            {"ridge": 2, "lgbm": 1, "mlp": 0}[n],
        ),
    )
    best_acc, best_coef = results[best_name]
    logger.info("Winning head: %s (val_acc=%.4f)", best_name, best_acc)

    if best_acc < 0.60:
        logger.warning(
            "Winning val accuracy %.4f is below 0.60 — features or pair sampling "
            "may be broken. Inspect feature stats and pair counts before relying "
            "on this head.",
            best_acc,
        )

    ckpt = CalHeadCheckpoint(
        head_type=best_name,
        coef=best_coef,
        val_pair_acc=float(best_acc),
        feature_dim=FEATURE_DIM,
    )
    save_checkpoint(ckpt, str(OUT_PATH))
    logger.info("Done.")


if __name__ == "__main__":
    main()
