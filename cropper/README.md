# Cropper

Training-free image cropping with vision-language models and in-context learning.

This project evaluates three tasks:
- Free-form cropping (GAICD)
- Subject-aware cropping (SACD)
- Aspect-ratio-aware cropping (FCDB + GAICD retrieval)

## 1) What This Repository Contains

- `scripts/`: runnable entry points (`run_freeform.py`, `run_subject_aware.py`, `run_aspect_ratio.py`, `ablation.py`)
- `pipeline/`: end-to-end cropping pipeline (retrieval, prompting, refinement)
- `models/`: VLM wrapper, CLIP retriever, scorers
- `data/`: dataset loaders and dataset download helper script
- `evaluation/`: metrics and reporting
- `configs/default.yaml`: default hyperparameters and model choices
- `results/`: saved outputs and metrics JSON files
- `weights/`: model weight folders (including VILA if used)

## 2) Environment Setup

From the `cropper/` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- GPU is recommended (`--device cuda`) for practical runtime.
- If `cuda` is unavailable, use `--device cpu`.

## 3) Dataset Setup

Use the helper script for guidance:

```bash
bash data/download.sh
```

Expected structure (important):

```text
data/
  GAICD/
    images/                # or images/train, images/val, images/test
    annotations/           # matching txt/json annotations
    splits/
      train.txt
      val.txt
      test.txt

  FCDB/
    images/
    cropping_testing_set.json

  SACD/
    images/
    masks/
    annotations/
    splits/
      train.txt
      val.txt
      test.txt
```

## 4) Core Run Commands

Run all commands from the `cropper/` directory.

### A) Free-form cropping (GAICD)

```bash
python scripts/run_freeform.py \
  --config configs/default.yaml \
  --data_dir data/GAICD \
  --output_dir results/freeform \
  --device cuda
```

Quick smoke run:

```bash
python scripts/run_freeform.py \
  --config configs/default.yaml \
  --data_dir data/GAICD \
  --output_dir results/freeform_smoke \
  --max_samples 20 \
  --device cuda
```

### B) Subject-aware cropping (SACD)

```bash
python scripts/run_subject_aware.py \
  --config configs/default.yaml \
  --data_dir data/SACD \
  --output_dir results/subject_aware \
  --device cuda
```

### C) Aspect-ratio-aware cropping (FCDB test + GAICD train retrieval)

For this script, pass `data/` (not just `data/FCDB`) because it expects both `data/FCDB` and `data/GAICD`.

```bash
python scripts/run_aspect_ratio.py \
  --config configs/default.yaml \
  --data_dir data \
  --output_dir results/aspect_ratio \
  --device cuda
```

### D) Ablation studies

```bash
python scripts/ablation.py \
  --config configs/default.yaml \
  --data_dir data \
  --output_dir results/ablation \
  --ablation all \
  --max_samples 50 \
  --device cuda
```

Run only one ablation:

```bash
python scripts/ablation.py --data_dir data --ablation S
python scripts/ablation.py --data_dir data --ablation R
python scripts/ablation.py --data_dir data --ablation L
python scripts/ablation.py --data_dir data --ablation scorer
python scripts/ablation.py --data_dir data --ablation retrieval
```

## 5) Workflow of the Code

High-level flow used by `Cropper.crop(...)`:

1. Load task config (`S`, `T`, `R`, `L`, scoring setup, coordinate format).
2. Retrieve top-`S` similar training images using CLIP embedding similarity.
3. Build a task-specific in-context prompt with retrieved examples.
4. Generate initial crop candidates with the VLM.
5. Score candidates with configured scorer(s) (VILA / CLIP / area depending on task).
6. Run iterative refinement for `L` rounds.
7. Return final crop and optional details.

Task differences:
- Free-form: multiple candidate crops with MOS-style tuple output.
- Subject-aware: conditioning on subject mask center.
- Aspect-ratio-aware: enforces a target width:height ratio.

## 6) Key Configuration Knobs

Edit `configs/default.yaml` to control behavior:

- `freeform.S`, `freeform.T`, `freeform.R`, `freeform.L`
- `subject_aware.S`, `subject_aware.R`, `subject_aware.L`
- `aspect_ratio.S`, `aspect_ratio.R`, `aspect_ratio.L`
- `vlm_model`, `clip_model`, `clip_pretrained`
- scorer mode per task (`vila+area`, `clip`, etc.)

## 7) Outputs and Where to Look

Each run writes to its `--output_dir`:

- `*_results.json`: final metrics + per-image predictions
- `checkpoint.json` (free-form): resumable progress
- `cache/train_embeddings.pkl` or similar: CLIP retrieval cache
- `config.yaml`: saved run config snapshot

Useful result folders:
- `results/freeform`
- `results/subject_aware`
- `results/aspect_ratio`
- `results/ablation`

## 8) Common Troubleshooting

- Out-of-memory on GPU:
  - reduce `--max_samples` for quick tests
  - switch to smaller models if you modify config
  - run with `--device cpu` for debugging
- Dataset not found / empty split:
  - verify paths and expected folder structure in Section 3
- Slow first run:
  - normal if CLIP embedding cache is being built
- Parsing or generation instability:
  - run with `--debug` to get full tracebacks

## 9) Reproducibility Tips

- Use fixed `--seed` (default is `42`).
- Keep a copy of `configs/default.yaml` used for each run.
- Start with `--max_samples` sanity checks before full benchmarks.

## 10) Minimal End-to-End Example

```bash
# 1) Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) (Optional) get dataset download instructions
bash data/download.sh

# 3) Quick free-form sanity run
python scripts/run_freeform.py \
  --config configs/default.yaml \
  --data_dir data/GAICD \
  --output_dir results/freeform_smoke \
  --max_samples 20 \
  --device cuda
```

If the smoke run is successful, remove `--max_samples` for full evaluation.
