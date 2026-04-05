# Text-Guided Cropping — CRISP Task 2

Part of the **CRISP** project (*Crop Retrieval via In-Context Similarity Prompting*),
a reimplementation of [Cropper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Cropper_Vision-Language_Model_for_Image_Cropping_through_In-Context_Learning_CVPR_2025_paper.pdf) using
open-source models.

---

## What is Text-Guided Cropping?

Standard image croppers produce a single "best aesthetic crop" with no way to
express **user intent**.  Text-guided cropping accepts a natural language instruction
alongside the image and returns a crop that honours that intent.

| Prompt | Effect |
|---|---|
| `"Focus on the person on the left"` | Subject selection |
| `"Wide cinematic crop"` | Style preference |
| `"Crop for Instagram square post"` | Platform-specific aspect ratio |
| `"Emphasize the mountain, not the lake"` | Compositional intent |
| `"Portrait orientation for mobile"` | Orientation / format |

---

## Architecture

```
Input Image + Text Prompt
        │
        ▼
┌───────────────────────┐
│   ICL Retriever       │  CLIP image similarity
│  (text-crop re-rank)  │+ text-crop CLIP alignment  ← Task-2 innovation
└───────────┬───────────┘
            │  S examples (image + crop box)
            ▼
┌───────────────────────┐
│   VLM Crop Generator  │  Mantis-8B-Idefics2
│  (interleaved prompt) │  with text intent in prompt
└───────────┬───────────┘
            │  R candidate boxes
            ▼
┌───────────────────────┐
│  Composite Scorer     │  aesthetic + CLIP-text + area
└───────────┬───────────┘
            │  scores
            ▼
┌───────────────────────┐
│  Iterative Refiner    │  L rounds of VLM feedback
└───────────┬───────────┘
            │  best crop box
            ▼
       Cropped Image
```

### Key components

| Class | File | Purpose |
|---|---|---|
| `TextGuidedCropper` | `text_guided_crop.py` | Main pipeline orchestrator |
| `ICLRetriever` | `text_guided_crop.py` | CLIP retrieval + text-crop re-ranking |
| `VLMCropGenerator` | `text_guided_crop.py` | Mantis-8B prompt builder + parser |
| `CompositeScorer` | `text_guided_crop.py` | Aesthetic + text-alignment + area scorer |
| `CropBox` | `text_guided_crop.py` | Bounding box dataclass (normalised & pixel) |
| `TextGuidedCropResult` | `text_guided_crop.py` | Result container |

---

## VRAM Requirements & 8 GB GPU Options

| Model | fp16 | 4-bit | Fits 8 GB? |
|---|---|---|---|
| Mantis-8B-Idefics2 | ~16 GB | ~5 GB ✓ | **4-bit only** |
| idefics2-8b | ~16 GB | ~5 GB ✓ | **4-bit only** |
| Qwen2-VL-7B-Instruct | ~14 GB | ~5 GB ✓ | **4-bit only** |
| **Qwen2-VL-2B-Instruct** | **~4 GB ✓** | ~2 GB ✓ | **YES (fp16)** |
| **SmolVLM-500M** | **~1 GB ✓** | ~1 GB ✓ | **YES (fp16)** |
| llava-1.5-7b-hf | ~14 GB | ~5 GB ✓ | **4-bit only** |

### Option A — 4-bit quantization (any 8B model, ~5 GB)

```bash
pip install bitsandbytes accelerate
```

```python
vlm_model, vlm_processor = load_vlm_model(
    model_name="TIGER-Lab/Mantis-8B-Idefics2",
    device="cuda",
    quantize=4,          # ← NF4, ~5 GB VRAM
)
```

```bash
python run_crop.py --image photo.jpg --prompt "Wide cinematic crop" \
    --vlm --quant 4 --output out.jpg
```

### Option B — Qwen2-VL-2B (fits in ~4 GB fp16, no quantization needed)

```python
vlm_model, vlm_processor = load_vlm_model(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    device="cuda",
)
```

```bash
python run_crop.py --image photo.jpg --prompt "Focus on the left person" \
    --vlm --vlm_model Qwen/Qwen2-VL-2B-Instruct --output out.jpg
```

### Option C — SmolVLM-500M (< 1 GB, fastest, lowest quality)

```bash
python run_crop.py --image photo.jpg --prompt "Portrait crop" \
    --vlm --vlm_model HuggingFaceTB/SmolVLM-500M --output out.jpg
```

### Show the VRAM guide from CLI

```bash
python run_crop.py --vram_guide
```

---

## Installation

```bash
pip install -r requirements.txt
```

For VLM support (requires ~20 GB VRAM):
```bash
pip install transformers accelerate
```

---

## Quick Start

### Heuristic mode (no GPU, no database)

```python
from PIL import Image
from text_guided_crop import TextGuidedCropper

cropper = TextGuidedCropper()   # no VLM, no CLIP
image   = Image.open("photo.jpg")

result = cropper.crop(image, "Wide cinematic crop")
result.cropped_image.save("output.jpg")
print(f"Box: {result.crop_box}  Score: {result.composite_score:.3f}")
```

### With CLIP retrieval

```python
from text_guided_crop import TextGuidedCropper, load_clip_model, build_database_from_gaicd

clip_model, clip_preprocess = load_clip_model()
cropper = TextGuidedCropper(clip_model=clip_model, clip_preprocess=clip_preprocess)

# Populate retrieval database
images, crops = build_database_from_gaicd("./data/GAICD")
cropper.add_examples(images, crops)

result = cropper.crop(Image.open("photo.jpg"), "Focus on the person on the left")
```

### With full VLM pipeline

```python
from text_guided_crop import TextGuidedCropper, load_clip_model, load_vlm_model

clip_model, clip_preprocess = load_clip_model()
vlm_model, vlm_processor   = load_vlm_model(device="cuda")

cropper = TextGuidedCropper(
    vlm_model=vlm_model,
    vlm_processor=vlm_processor,
    clip_model=clip_model,
    clip_preprocess=clip_preprocess,
    device="cuda",
    S=10, R=5, L=2,
    text_rerank_weight=0.5,   # weight for text-crop re-ranking
)
```

---

## Command-Line Interface

```bash
# Minimal (heuristic, no GPU)
python run_crop.py \
    --image photo.jpg \
    --prompt "Crop for Instagram square post" \
    --output out.jpg

# With CLIP + GAICD database
python run_crop.py \
    --image photo.jpg \
    --prompt "Wide cinematic crop" \
    --gaicd_root ./data/GAICD \
    --output out.jpg

# Full pipeline (VLM)
python run_crop.py \
    --image photo.jpg \
    --prompt "Focus on the person on the left" \
    --gaicd_root ./data/GAICD \
    --vlm --vlm_model TIGER-Lab/Mantis-8B-Idefics2 \
    --S 10 --R 5 --L 2 \
    --output out.jpg

# Fixed output size (backwards-compatible)
python run_crop.py \
    --image photo.jpg \
    --prompt "Portrait orientation for mobile" \
    --crop_size 720 1280 \
    --output out.jpg
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Input image path |
| `--prompt` | required | Text intent |
| `--output` | required | Output image path |
| `--crop_size W H` | None | Resize output (pixels) |
| `--clip_model` | `ViT-B-32` | OpenCLIP architecture |
| `--clip_pretrained` | `openai` | OpenCLIP weights |
| `--no_clip` | False | Disable CLIP entirely |
| `--vlm` | False | Enable Mantis-8B VLM |
| `--vlm_model` | Mantis-8B-Idefics2 | HuggingFace model ID |
| `--no_fp16` | False | Use float32 for VLM |
| `--gaicd_root` | None | GAICD dataset path |
| `--db_max_images` | 500 | Max DB images |
| `--S` | 10 | ICL examples |
| `--R` | 5 | Candidate crops |
| `--L` | 2 | Refinement iterations |
| `--text_rerank_weight` | 0.5 | Text re-rank weight |
| `--vila_weight` | 0.3 | Aesthetic score weight |
| `--text_weight` | 0.5 | CLIP-text score weight |
| `--area_weight` | 0.2 | Area score weight |
| `--device` | auto | `cuda` / `cpu` |
| `--verbose` | False | DEBUG logging |

---

## Design Notes

### Text-Crop Re-ranking (the Task-2 key contribution)

Standard Cropper retrieves ICL examples by **image similarity** only.
For text-guided cropping we introduce a second signal: the **cosine
similarity between the text prompt and the CLIP embedding of each
database crop region**.

```
combined_score_i = (1 - w) * image_sim(query, db_i)
                 +       w * text_sim(prompt, crop_region_i)
```

This retrieves examples whose *crops* (not just their full images)
visually match the user's intent — much better teachers for the VLM.

### Composite Scorer

The scorer weights text-alignment highly (default 0.5) so that among
visually similar crops the one matching the user's language intent is
preferred:

```
composite = 0.3 * aesthetic + 0.5 * clip_text_sim + 0.2 * area
```

### Fallback Mode

All three external dependencies (CLIP, VLM, GAICD) are optional.
When absent the system falls back to a keyword-driven heuristic
generator that still respects prompts like "cinematic", "square",
"left", "portrait", etc.

---

## Module API

```python
TextGuidedCropper(
    vlm_model, vlm_processor,  # optional
    clip_model, clip_preprocess,  # optional
    device="cpu",
    S=10, R=5, L=2,
    text_rerank_weight=0.5,
    coord_range=(1.0, 1000.0),
    vila_weight=0.3, text_weight=0.5, area_weight=0.2,
)

cropper.add_examples(images, crops, paths=None)  # populate DB
result = cropper.crop(image, text_prompt)         # main entrypoint
result = cropper.crop_with_size(image, text_prompt, (W, H))  # legacy API

# Result fields:
result.cropped_image    # PIL.Image
result.crop_box         # CropBox (pixel coords)
result.composite_score  # float in [0, 1]
result.all_candidates   # List[CropBox] if return_all_candidates=True
```

---

## References

- Lee et al., *Cropper: Vision-Language Model for Image Cropping through In-Context Learning*, CVPR 2025.
- Zhong et al., *ClipCrop: Conditioned Cropping Driven by Vision-Language Model*, ICCV workshop.
- Rajparia, Virani, Neeli, *CRISP: Crop Retrieval via In-Context Similarity Prompting*, 2025.
