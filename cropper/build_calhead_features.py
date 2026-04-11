#!/usr/bin/env python
"""
Phase 3 — Step 1: build the GAICD train feature cache for the calibration head.

What this does:
    For every image in GAICD train (2,636 images, ~90 crops each), crop the
    image at each annotated box, run CLIP ViT-L/14 on the cropped pixels to
    get a 768-d embedding, concatenate with 6 normalized geometry features,
    and dump everything to a single .npz file.

    The output cache is the *only* GPU-bound step in Phase 3 — once written,
    train_calhead.py runs entirely on CPU in a few minutes.

Why CLIP ViT-L/14:
    See cropper/models/gaicd_calibration_head.py docstring. TL;DR: VILA's
    pooled features are not exposed cleanly through the JAX checkpoint, but
    open_clip ViT-L/14 (the same loader pattern as the LAION fallback at
    cropper/models/scorer.py:248-313) gives 768-d embeddings out of the box.

Usage (run in tmux, GPU 1):
    cd /data1/es22btech11013/divya/AFCIL/divya/cv-project/cropper
    CUDA_VISIBLE_DEVICES=1 \\
      /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \\
      build_calhead_features.py

Output:
    cropper/cache/gaicd_train_features.npz with arrays:
        features    : (N, 774) float32
        mos         : (N,) float32       — original GAICD MOS values
        image_idx   : (N,) int64         — group id for ranking-loss pair sampling
        image_ids   : (n_images,) U32    — string image ids in image_idx order
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make sibling packages importable when running this script directly
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.datasets import GAICDDataset  # noqa: E402
from models.gaicd_calibration_head import (  # noqa: E402
    CLIP_EMBED_DIM,
    FEATURE_DIM,
    ClipViTL14Encoder,
    geometry_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("build_calhead_features")


# ============================================================================
# Config
# ============================================================================

DATA_ROOT      = SCRIPT_DIR / "data" / "GAICD"
SPLIT          = "train"
OUT_PATH       = SCRIPT_DIR / "cache" / "gaicd_train_features.npz"
DEVICE         = "cuda"
BATCH_CROPS    = 64        # encode this many crops in one CLIP forward
LOG_EVERY_IMG  = 100       # progress log frequency


# ============================================================================
# Main
# ============================================================================


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"GAICD root not found: {DATA_ROOT}")

    logger.info("Loading GAICD %s split from %s", SPLIT, DATA_ROOT)
    dataset = GAICDDataset(
        root_dir=str(DATA_ROOT),
        split=SPLIT,
        cache_embeddings=False,
    )
    n_images = len(dataset)
    logger.info("GAICD %s: %d images", SPLIT, n_images)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; CLIP encode will run on CPU and will be slow.")
        device = "cpu"
    else:
        device = DEVICE

    encoder = ClipViTL14Encoder(device=device)

    # Pre-allocate growing lists. Each GAICD train image has ~90 crops, so
    # ~237k total. We collect into Python lists then stack at the end.
    feats_chunks: list[np.ndarray] = []
    mos_all:  list[float] = []
    img_idx_all: list[int] = []
    image_ids_kept: list[str] = []

    t0 = time.time()
    n_crops_total = 0
    n_images_kept = 0

    for img_pos in range(n_images):
        sample = dataset[img_pos]
        img_id = sample["image_id"]
        image: Image.Image = sample["image"]
        crops = sample["crops"]  # list of (mos, x1, y1, x2, y2) in pixel space

        if not crops:
            continue

        W, H = image.size

        # Build cropped PIL images and parallel arrays
        cropped_imgs = []
        crop_geoms = []
        crop_mos = []
        for (mos, x1, y1, x2, y2) in crops:
            x1c = max(0, int(x1))
            y1c = max(0, int(y1))
            x2c = min(W, int(x2))
            y2c = min(H, int(y2))
            if x2c - x1c < 4 or y2c - y1c < 4:
                continue
            try:
                cropped = image.crop((x1c, y1c, x2c, y2c))
            except Exception as e:
                logger.debug("crop failed on %s: %s", img_id, e)
                continue
            cropped_imgs.append(cropped)
            crop_geoms.append(geometry_features((x1c, y1c, x2c, y2c), (W, H)))
            crop_mos.append(float(mos))

        if not cropped_imgs:
            continue

        # Batched encode
        emb_chunks = []
        for start in range(0, len(cropped_imgs), BATCH_CROPS):
            chunk = cropped_imgs[start : start + BATCH_CROPS]
            emb_chunks.append(encoder.encode_batch(chunk))
        embs = np.concatenate(emb_chunks, axis=0)  # (n_c, 768)
        geoms = np.stack(crop_geoms)               # (n_c, 6)
        feats = np.concatenate([embs, geoms], axis=1).astype(np.float32)
        assert feats.shape[1] == FEATURE_DIM, f"unexpected feature dim {feats.shape}"

        feats_chunks.append(feats)
        mos_all.extend(crop_mos)
        img_idx_all.extend([n_images_kept] * len(crop_mos))
        image_ids_kept.append(img_id)
        n_images_kept += 1
        n_crops_total += len(crop_mos)

        if (img_pos + 1) % LOG_EVERY_IMG == 0:
            elapsed = time.time() - t0
            rate = (img_pos + 1) / max(1e-6, elapsed)
            eta_min = (n_images - img_pos - 1) / max(1e-6, rate) / 60.0
            logger.info(
                "  [%5d/%5d] %.1f img/s | crops=%d | eta=%.1f min",
                img_pos + 1, n_images, rate, n_crops_total, eta_min,
            )

    elapsed = time.time() - t0
    logger.info(
        "Encoded %d crops across %d images in %.1f min (rate=%.1f img/s)",
        n_crops_total, n_images_kept, elapsed / 60, n_images_kept / max(1e-6, elapsed),
    )

    if not feats_chunks:
        raise RuntimeError("No features collected; check GAICD train annotations")

    features  = np.concatenate(feats_chunks, axis=0)
    mos_arr   = np.array(mos_all, dtype=np.float32)
    image_idx = np.array(img_idx_all, dtype=np.int64)
    image_ids_arr = np.array(image_ids_kept, dtype="U32")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        features=features,
        mos=mos_arr,
        image_idx=image_idx,
        image_ids=image_ids_arr,
    )

    # Sanity stats
    logger.info("Saved -> %s", OUT_PATH)
    logger.info("  features shape : %s", features.shape)
    logger.info("  features mean  : %.4f", float(features.mean()))
    logger.info("  features std   : %.4f", float(features.std()))
    logger.info("  features NaNs  : %d", int(np.isnan(features).sum()))
    logger.info("  mos range      : [%.3f, %.3f]", float(mos_arr.min()), float(mos_arr.max()))
    logger.info("  mos mean       : %.3f", float(mos_arr.mean()))
    logger.info("  unique images  : %d", int(image_ids_arr.shape[0]))
    logger.info("Total crops      : %d", int(features.shape[0]))


if __name__ == "__main__":
    main()
