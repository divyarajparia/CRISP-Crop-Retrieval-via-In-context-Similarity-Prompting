# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Replication of **Cropper** (CVPR 2025) — a training-free image cropping framework using Vision-Language Models (VLMs) with In-Context Learning (ICL). The implementation uses **Mantis-8B-Idefics2** (open-source) in place of Gemini 1.5 Pro. Target benchmark: GAICD free-form cropping (Table 9 of the paper).

**Target metrics (Mantis-8B-Idefics2):**

| IoU   | Acc@5 | Acc@10 | SRCC  | PCC   |
|-------|-------|--------|-------|-------|
| 0.672 | 80.2% | 88.6%  | 0.874 | 0.797 |

All runnable code lives under `cropper/`. Run every command from that directory.

## Commands

### Setup
```bash
cd cropper
pip install -r requirements.txt
```

### Free-form cropping (main benchmark — GAICD)
```bash
python scripts/run_freeform.py \
  --config configs/default.yaml \
  --data_dir data/GAICD \
  --output_dir results/freeform \
  --device cuda
```

Quick smoke-test (20 samples, resumable):
```bash
python scripts/run_freeform.py \
  --config configs/default.yaml \
  --data_dir data/GAICD \
  --output_dir results/freeform_smoke \
  --max_samples 20 \
  --device cuda
```

### Subject-aware cropping (SACD)
```bash
python scripts/run_subject_aware.py \
  --config configs/default.yaml \
  --data_dir data/SACD \
  --output_dir results/subject_aware \
  --device cuda
```

### Aspect-ratio-aware cropping (FCDB + GAICD retrieval)
```bash
python scripts/run_aspect_ratio.py \
  --config configs/default.yaml \
  --data_dir data \
  --output_dir results/aspect_ratio \
  --device cuda
```

Note: `--data_dir` must be `data/` (not `data/FCDB`) because the script needs both `data/FCDB` and `data/GAICD`.

### Ablations
```bash
python scripts/ablation.py --data_dir data --ablation S    # sweep S
python scripts/ablation.py --data_dir data --ablation R    # sweep R
python scripts/ablation.py --data_dir data --ablation L    # sweep L
python scripts/ablation.py --data_dir data --ablation scorer
python scripts/ablation.py --data_dir data --ablation retrieval
```

## Architecture

### Pipeline flow (`pipeline/cropper.py:Cropper.crop()`)

```
query image
  → CLIP retrieval (clip_retriever.py)        # top-S training images by cosine sim
  → task-specific crop selection (retrieval.py) # top-T crops by MOS / mask / aspect ratio
  → prompt assembly (prompt_builder.py)        # interleaved images + text
  → VLM generation (vlm.py)                   # outputs R crop tuples
  → iterative refinement ×L (refinement.py)   # score → feedback → new crops
  → best crop returned
```

### Key hyperparameters (`configs/default.yaml`)

| Param | Free-form | Subject-aware | Aspect-ratio | Meaning |
|-------|-----------|---------------|--------------|---------|
| S | 30 (paper) / 10 (Mantis default) | 30 | 10 | Retrieved training images |
| T | 5 | 1 | 1 | GT crops per ICL example |
| R | 6 (paper) / 5 (Mantis default) | 5 | 6 | VLM crop candidates |
| L | 2 | 10 | 2 | Refinement iterations |

### Coordinate systems — critical difference between tasks
- **Free-form**: `(s, x1, y1, x2, y2)` where coords are in `[1, 1000]`
- **Subject-aware**: `(x1, y1, x2, y2)` where coords are in `[0, 1]`
- **Aspect-ratio**: `(x1, y1, x2, y2)` in pixel space

All normalization/denormalization goes through `utils/coord_utils.py`.

### Scorer (`models/scorer.py`)
`create_scorer(task, device, require_exact_components)` returns a `CombinedScorer`:
- Free-form & subject-aware: VILA aesthetic scorer + Area scorer
- Aspect-ratio: CLIP content scorer only

VILA loads from TensorFlow Hub (`tfhub.dev/google/vila/image/1`). Falls back to LAION aesthetic predictor (from `weights/aesthetic_predictor_v2.pth`) if TFHub is unavailable. Set `require_exact_components=True` to raise instead of silently falling back.

### Metrics (`evaluation/metrics.py`)
`MetricsCalculator` accumulates per-image results; call `.compute()` for final aggregates.
- **IoU**: standard box IoU against best-matching GT crop
- **AccK/N**: whether top-K predicted crops (by score) overlap (IoU > 0.5) with top-N GT crops
- **SRCC/PCC**: per-image correlation between predicted scores and matched GT MOS; averaged across images

The `update()` call requires `pred_crops_all` (all crops across all refinement iterations, deduplicated by coordinates) and `pred_scores` to compute AccK/N and SRCC/PCC correctly.

### Resumability
`run_freeform.py` writes a `checkpoint.json` per sample to `--output_dir`. Rerunning the same command with the same `--output_dir` resumes from where it left off.

## Dataset Structure

```
cropper/data/
  GAICD/
    images/          # (or images/train, images/val, images/test)
    annotations/
    splits/train.txt, val.txt, test.txt
  FCDB/
    images/
    cropping_testing_set.json
  SACD/
    images/  masks/  annotations/
    splits/train.txt, val.txt, test.txt
```

## Weights

```
cropper/weights/
  aesthetic_predictor_v2.pth      # LAION aesthetic predictor (PyTorch)
  vila_rank_tuned/checkpoint_0/   # VILA-R JAX checkpoint (1.5 GB, downloaded separately)
```

Download VILA weights: `python scripts/download_vila.py`
