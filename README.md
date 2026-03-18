# CV Project

This is the parent workspace for the image-cropping project implementation in [cropper](cropper).

## Project Layout

- [cropper](cropper): main codebase (models, pipeline, scripts, configs, evaluation)
- [ClipCrop.pdf](ClipCrop.pdf): project/reference document
- [Cropper.pdf](Cropper.pdf): paper/reference document

## Quick Start (from this folder)

```bash
cd cropper
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Commands (from this folder)

### Free-form cropping

```bash
cd cropper
python scripts/run_freeform.py \
  --config configs/default.yaml \
  --data_dir data/GAICD \
  --output_dir results/freeform \
  --device cuda
```

### Subject-aware cropping

```bash
cd cropper
python scripts/run_subject_aware.py \
  --config configs/default.yaml \
  --data_dir data/SACD \
  --output_dir results/subject_aware \
  --device cuda
```

### Aspect-ratio-aware cropping

```bash
cd cropper
python scripts/run_aspect_ratio.py \
  --config configs/default.yaml \
  --data_dir data \
  --output_dir results/aspect_ratio \
  --device cuda
```

## Workflow Summary

The pipeline used in [cropper/pipeline/cropper.py](cropper/pipeline/cropper.py):

1. Retrieve in-context examples using CLIP similarity.
2. Build a task-specific prompt with those examples.
3. Generate crop candidates with the VLM.
4. Score candidates (VILA/CLIP/area depending on config).
5. Iteratively refine and output final crop.

## Detailed Documentation

For full setup details, dataset structure, troubleshooting, and ablations, see:
- [cropper/README.md](cropper/README.md)
