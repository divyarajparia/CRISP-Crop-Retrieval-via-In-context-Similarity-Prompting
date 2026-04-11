# Cropper: Project Guide

## 1. Overview

This project is an open-source replication of **Cropper** (CVPR 2025) — a training-free image cropping framework that uses Vision-Language Models (VLMs) with In-Context Learning (ICL) to produce aesthetically pleasing crops, without any task-specific training or fine-tuning.

**Paper**: *Cropper: Vision-Language Model for Image Cropping through In-Context Learning*, CVPR 2025.

**Key idea**: Given a query image, retrieve visually similar images whose good crops are already known, feed them as examples to a VLM, and ask it to propose crops. An aesthetic scorer then iteratively guides the VLM toward better crops.

**Our implementation** uses **Mantis-8B-Idefics2** (open-source, HuggingFace) in place of the paper's Gemini 1.5 Pro. Everything else follows the paper exactly.

### Tasks Covered

| Task | Dataset | Benchmark |
|------|---------|-----------|
| Free-form cropping | GAICD | Primary benchmark |
| Subject-aware cropping | SACD | Crop around a masked subject |
| Aspect-ratio-aware cropping | FCDB | Crop to a target width:h10.1109/QCE65121.2025.00096zeight ratio |

### Target Metrics (Table 9, Mantis-8B-Idefics2, GAICD free-form)

| IoU | Acc@5 | Acc@10 | SRCC | PCC |
|-----|-------|--------|------|-----|
| 0.672 | 80.2% | 88.6% | 0.874 | 0.797 |

---

## 2. Repository Layout

```
cv-project/
├── Cropper.pdf                  # Paper (CVPR 2025)
├── ClipCrop.pdf                 # Related baseline paper
├── PROJECT_GUIDE.md             # This file
└── cropper/                     # All code lives here
    ├── run_experiment.py        # MAIN LAUNCHER — edit this to run experiments
    ├── configs/
    │   └── default.yaml         # All hyperparameters
    ├── scripts/
    │   ├── run_freeform.py      # Free-form evaluation script
    │   ├── run_subject_aware.py # Subject-aware evaluation script
    │   ├── run_aspect_ratio.py  # Aspect-ratio evaluation script
    │   ├── ablation.py          # Ablation study runner
    │   └── download_vila.py     # Download VILA-R weights
    ├── pipeline/
    │   ├── cropper.py           # Top-level pipeline orchestration
    │   ├── retrieval.py         # CLIP-based ICL example retrieval
    │   ├── prompt_builder.py    # Builds VLM prompts per task
    │   └── refinement.py        # Iterative refinement loop
    ├── models/
    │   ├── vlm.py               # Mantis-8B-Idefics2 wrapper
    │   ├── clip_retriever.py    # CLIP ViT-B/32 retriever
    │   ├── scorer.py            # VILA-R / LAION aesthetic scorer
    │   └── vila/                # VILA-R model code (JAX/Flax)
    ├── data/
    │   ├── datasets.py          # Dataset loaders (GAICD, FCDB, SACD)
    │   ├── GAICD/               # Free-form cropping dataset
    │   ├── FCDB/                # Aspect-ratio dataset
    │   └── SACD/                # Subject-aware dataset
    ├── evaluation/
    │   ├── metrics.py           # IoU, Disp, SRCC, PCC, AccK/N
    │   └── evaluate.py          # Batch evaluation utilities
    ├── utils/
    │   ├── coord_utils.py       # Coordinate normalization / denormalization
    │   └── visualization.py     # Crop visualization
    ├── weights/
    │   ├── vila_rank_tuned/     # VILA-R checkpoint (~1.5 GB)
    │   └── aesthetic_predictor_v2.pth  # LAION fallback weights
    ├── results/                 # All experiment outputs land here
    └── requirements.txt
```

---

## 3. Local Setup

All commands are run from `cv-project/cropper/` unless noted otherwise.

### Step 1 — Clone the repository

```bash
git clone <repo_url>
cd cv-project/cropper
```

### Step 2 — Create the conda environment

The environment is named **`cv_project`**.

```bash
conda create -n cv_project python=3.9 -y
conda activate cv_project
pip install --upgrade pip
pip install -r requirements.txt
```

Then install the VILA-R dependencies (not in `requirements.txt` because they require specific versions):

```bash
# JAX stack — versions must match exactly
pip install jax==0.4.26 jaxlib==0.4.26 flax==0.8.2 orbax-checkpoint==0.5.9

# PAX/Lingvo stack for VILA-R checkpoint loading
pip install lingvo==0.12.7 paxml==1.4.0 praxis==1.4.0

# TensorFlow (used by lingvo internally)
pip install tensorflow==2.9.3 tensorflow-hub==0.16.1
```

> **Note**: `lingvo` requires Linux. Version pinning above is what is confirmed working on this server.

### Step 3 — Download datasets

Datasets go inside `data/`. The expected directory structure for each is shown below.

#### GAICD (free-form cropping)

- **Download**: https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch
- **Size**: ~329 MB
- **Split**: train=2,636 / val=200 / test=500 images
- **Annotations**: ~90 crop boxes per image, each with a MOS (Mean Opinion Score)

```
data/GAICD/
├── images/
│   ├── train/      # 2,636 images
│   ├── val/        # 200 images
│   └── test/       # 500 images
└── annotations/
    ├── train/      # .txt files, one per image
    ├── val/
    └── test/
```

Each annotation line: `x1 y1 x2 y2 MOS`

#### FCDB (aspect-ratio-aware cropping)

- **Download**: https://github.com/yiling-chen/flickr-cropping-dataset
- **Size**: ~330 MB
- **Split**: 348 test images only (no train split used)
- **Annotations**: single crop per image with aspect ratio

```
data/FCDB/
├── images/                      # test images
└── cropping_testing_set.json    # annotations
```

#### SACD (subject-aware cropping)

- **Download**: https://github.com/bcmi/Human-Centric-Image-Cropping
- **Size**: ~16 MB (annotations only; images referenced from GAICD/FCDB/FLMS)
- **Split**: train=2,326 / val=290 / test=290

```
data/SACD/
└── human_bboxes/
    ├── GAICD/      # crop boxes for GAICD images
    ├── FCDB/       # crop boxes for FCDB images
    ├── FLMS/       # crop boxes for FLMS images
    └── CPC/        # crop boxes for CPC images
```

### Step 4 — Download model weights

#### Mantis-8B-Idefics2 (VLM)

Downloaded automatically from HuggingFace on first run. Requires ~16 GB disk space and ~20 GB GPU VRAM. No manual action needed.

#### VILA-R (aesthetic scorer)

```bash
python scripts/download_vila.py
```

This downloads the VILA-R rank-tuned checkpoint (~1.5 GB) to `weights/vila_rank_tuned/`. Expected structure after download:

```
weights/vila_rank_tuned/
└── checkpoint_0/
    ├── metadata/
    └── state/
```

#### LAION aesthetic predictor (fallback)

Already present in the repo at `weights/aesthetic_predictor_v2.pth` (3.6 MB). No download needed.

### Step 5 — Verify setup

```bash
conda activate cv_project
cd cv-project/cropper

python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'models')
from models.scorer import create_scorer
from PIL import Image
import numpy as np
scorer = create_scorer(task='freeform', device='cuda')
print('Scorer backend:', scorer.scorers['vila'].scorer_type)
img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
print('Test score:', scorer.score(img))
"
```

Expected output: `Scorer backend: vila` and a float score. If it prints `laion` instead, the VILA-R weights are not loading correctly — check `weights/vila_rank_tuned/checkpoint_0/` exists.

---

## 4. Architecture

### Pipeline Overview

```
Query Image
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: CLIP Retrieval  (models/clip_retriever.py) │
│  ViT-B/32 cosine similarity → top-S training images │
│  + select top-T crops per image by MOS              │
└────────────────────────┬────────────────────────────┘
                         │  S=30 images, T=5 crops each
                         ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Prompt Assembly  (pipeline/prompt_builder) │
│  Interleave ICL example images + GT crop coords     │
│  into a single multi-image text prompt              │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: VLM Generation   (models/vlm.py)           │
│  Mantis-8B-Idefics2 generates R=6 crop proposals   │
│  as (MOS, x1, y1, x2, y2) tuples in [1, 1000]      │
└────────────────────────┬────────────────────────────┘
                         │  R=6 initial crop candidates
                         ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: Iterative Refinement (pipeline/refinement) │
│  Repeat L=2 times:                                  │
│    • Score each crop with VILA-R scorer             │
│    • Feed scores back to VLM as feedback            │
│    • VLM proposes improved crops                    │
│  Keep best crop seen across all iterations          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
                    Final Crop (x1, y1, x2, y2)
```

### Components

#### CLIP Retriever (`models/clip_retriever.py`)

Uses OpenCLIP `ViT-B/32` (OpenAI weights) to embed images into 512-d vectors. Retrieves the top-S training images most visually similar to the query via cosine similarity. CLIP embeddings for the entire training set are computed once and cached to `results/<run>/cache/train_embeddings.pkl` — subsequent runs reuse the cache.

#### Prompt Builder (`pipeline/prompt_builder.py`)

Assembles the multi-image VLM prompt. Coordinate format and prompt wording differ by task:

| Task | Coordinate range | Crop format |
|------|-----------------|-------------|
| Free-form | `[1, 1000]` | `(MOS, x1, y1, x2, y2)` |
| Subject-aware | `[0, 1]` | `(x1, y1, x2, y2)` + mask center |
| Aspect-ratio | pixel values | `(x1, y1, x2, y2)` + target ratio |

The refinement prompt appends each crop image with its aesthetic score and asks the VLM to propose a better crop.

#### VLM (`models/vlm.py`)

Wraps `TIGER-Lab/Mantis-8B-Idefics2` via HuggingFace `transformers`. Runs in float16. Uses `temperature=0.05` (near-deterministic) as determined by the paper's ablation (Fig. 5). Output is parsed by regex to extract crop coordinate tuples; malformed outputs fall back to a center crop.

#### Scorer (`models/scorer.py`)

`create_scorer(task, device)` returns a `CombinedScorer`:

- **Free-form / Subject-aware**: `VILA + Area` (equal weights)
- **Aspect-ratio**: `CLIP` content similarity only

**VILA-R** (primary): JAX/Flax checkpoint, trained on AVA dataset. Scores aesthetic quality in [0, 1]. Requires `lingvo`, `paxml`, `praxis` — only available in the `cv_project` conda environment.

**LAION** (fallback): CLIP ViT-L/14 + 5-layer MLP, `weights/aesthetic_predictor_v2.pth`. Loads if VILA-R unavailable.

**Area scorer**: `crop_area / original_area` — penalizes crops that are too small.

#### Refinement Loop (`pipeline/refinement.py`)

Runs `L` iterations. Each iteration: extract crop images → score with VILA-R → build feedback prompt → VLM generates new R crops. The best crop across **all** iterations (not just the final one) is returned.

#### Metrics (`evaluation/metrics.py`)

| Metric | Description |
|--------|-------------|
| **IoU** | Intersection over Union between predicted crop and best-matching GT crop |
| **Disp** | Average L1 distance between normalized coordinates of predicted vs GT |
| **SRCC** | Spearman rank-order correlation between predicted aesthetic scores and GT MOS, computed per-image then averaged |
| **PCC** | Pearson correlation, same computation as SRCC |
| **AccK/N** | Whether any of the top-K predicted crops overlaps (IoU > 0.5) with the top-N GT crops by MOS |

---

## 5. Hyperparameter Reference

All defaults live in `configs/default.yaml`. `run_experiment.py` overrides them at runtime.

| Parameter | Free-form | Subject-aware | Aspect-ratio | Description |
|-----------|-----------|---------------|--------------|-------------|
| **S** | 30 | 30 | 10 | Training images retrieved via CLIP |
| **T** | 5 | 1 | 1 | GT crops selected per retrieved image |
| **R** | 6 | 5 | 6 | Crop candidates generated per VLM call |
| **L** | 2 | 10 | 2 | Refinement iterations |
| **temperature** | 0.05 | 0.05 | 0.05 | VLM sampling temperature |
| **scorer** | `vila+area` | `vila+area` | `clip` | Scorer combination |
| **coord_range** | `[1, 1000]` | `[0, 1]` | pixel | Coordinate normalization |

These values match the paper's ablation-determined defaults (Figures 3–5).

---

## 6. Experiment History

All experiments are on the **GAICD free-form cropping** task. Paper target: **IoU=0.672, SRCC=0.874**.

| Run directory | n | Scorer | IoU | SRCC | PCC | Notes |
|---------------|---|--------|-----|------|-----|-------|
| `freeform` | 500 (full) | LAION | 0.474 | 0.123 | 0.065 | First full baseline run |
| `freeform_50samples` | 50 | LAION | 0.495 | −0.159 | −0.163 | 50-sample ablation |
| `freeform_50samples_paperish_v2` | 50 | LAION | 0.527 | 0.022 | −0.014 | Prompt tuned to match paper Table 1 exactly; best LAION run |
| `test_freeform2` | 5 | LAION | **0.628** | 0.258 | 0.474 | Small-sample sanity check; high variance |
| `freeform_S30_R6_L2__vila_area__n50` | — | VILA-R | *(in progress)* | — | — | First run with VILA-R working |

### Current Gap vs Paper Target

| Metric | Best achieved (50-sample) | Paper target | Gap |
|--------|--------------------------|--------------|-----|
| IoU | 0.527 | 0.672 | −0.145 |
| SRCC | 0.022 | 0.874 | −0.852 |
| PCC | −0.014 | 0.797 | −0.811 |

### Root Cause of Gap

The near-zero SRCC/PCC comes from using the **LAION aesthetic predictor** as the scorer. LAION scores internet image aesthetics; GAICD MOS scores rate crop composition — these two distributions do not naturally correlate.

**VILA-R** is trained on the AVA aesthetic dataset (closer in spirit to GAICD MOS) and is what the paper uses. VILA-R was previously failing to load due to two bugs:
1. Wrong conda environment — `lingvo`/`paxml`/`praxis` are only installed in `cv_project`.
2. Wrong `sys.path` — `coca_vila.py` imports `from vila import ...`, requiring `models/` (not `models/vila/`) on the path.

Both are now fixed. The next run with VILA-R will show how much SRCC/PCC improves.

---

## 7. Running Experiments

### Quick start

```bash
cd cv-project/cropper
conda activate cv_project
python run_experiment.py
```

### Configuring a run

Open `run_experiment.py` and edit only the `EXP` dictionary at the top:

```python
EXP = dict(
    task         = "freeform",   # "freeform" | "subject_aware" | "aspect_ratio"
    max_samples  = 50,           # None = full test set
    gpu          = 0,            # GPU index (sets CUDA_VISIBLE_DEVICES)

    data_dir     = "data/GAICD", # "data/SACD" for subject_aware, "data" for aspect_ratio

    S            = 30,           # retrieved training images
    T            = 5,            # GT crops per ICL example
    R            = 6,            # VLM crop candidates per iteration
    L            = 2,            # refinement iterations
    temperature  = 0.05,

    scorer       = "vila+area",  # "vila+area" | "clip" | "laion+area"

    output_dir   = None,         # None = auto-name from params + timestamp
    seed         = 42,
    resume       = None,         # path to checkpoint.json to resume a crashed run
)
```

When `output_dir = None`, the run name is generated automatically, e.g.:
```
results/freeform_S30_R6_L2__vila_area__n50__0409_1630/
```

### Reading results

```bash
# Print metrics from a run
python -c "
import json
d = json.load(open('results/<run_name>/freeform_results.json'))
for k, v in d['metrics'].items():
    print(f'{k}: {v:.4f}')
"
```

```bash
# Compare two runs side by side
python -c "
import json
a = json.load(open('results/freeform_50samples_paperish_v2/freeform_results.json'))['metrics']
b = json.load(open('results/<new_run>/freeform_results.json'))['metrics']
print(f'{'Metric':<10} {'Old':>10} {'New':>10} {'Delta':>10}')
for k in a:
    print(f'{k:<10} {a[k]:>10.4f} {b[k]:>10.4f} {b[k]-a[k]:>+10.4f}')
"
```

### Resuming a crashed run

If a run crashes partway, set `resume` to its checkpoint file:

```python
resume     = "results/freeform_S30_R6_L2__vila_area__n50__0409_1630/checkpoint.json",
output_dir = "results/freeform_S30_R6_L2__vila_area__n50__0409_1630",
```

---

## 8. Key Files Quick Reference

| File | Purpose |
|------|---------|
| `run_experiment.py` | **Start here.** Edit `EXP` dict and run. |
| `configs/default.yaml` | All default hyperparameters. |
| `pipeline/cropper.py` | Top-level `Cropper.crop()` — orchestrates the full pipeline. |
| `pipeline/retrieval.py` | CLIP-based ICL example retrieval (Equations 1 & 2 from paper). |
| `pipeline/prompt_builder.py` | Builds exact prompts from paper Tables 1, 13, 15. |
| `pipeline/refinement.py` | L-iteration refinement loop; returns best crop across all iterations. |
| `models/vlm.py` | Mantis-8B wrapper: generation + robust coordinate parsing. |
| `models/clip_retriever.py` | CLIP ViT-B/32 encoding and cosine similarity retrieval. |
| `models/scorer.py` | VILA-R scorer (JAX); LAION fallback; Area scorer; CombinedScorer factory. |
| `data/datasets.py` | `GAICDDataset`, `FCDBDataset`, `SACDDataset` loaders. |
| `evaluation/metrics.py` | IoU, Disp, SRCC, PCC, AccK/N implementations. |
| `utils/coord_utils.py` | Normalize/denormalize coordinates across task coordinate systems. |
| `scripts/download_vila.py` | Download VILA-R checkpoint to `weights/vila_rank_tuned/`. |
