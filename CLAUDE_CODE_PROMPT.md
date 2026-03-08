# Cropper Replication: Full Implementation Prompt for Claude Code

## IMPORTANT: Attach the Cropper PDF paper alongside this prompt.

---

## Goal

Reimplement the **Cropper** paper (CVPR 2025) — a training-free image cropping framework using vision-language models with in-context learning. We are replicating using **open-source VLMs** (starting with Mantis-8B-Idefics2) instead of Gemini 1.5 Pro.

This is a research reimplementation. Code quality, modularity, and reproducibility matter.

---

## Project Structure

Create the following structure:

```
cropper/
├── configs/
│   └── default.yaml              # All hyperparameters
├── data/
│   ├── download.sh               # Script to download GAICD, FCDB, SACD datasets
│   └── datasets.py               # Dataset loaders for GAICD, FCDB, SACD
├── models/
│   ├── vlm.py                    # VLM wrapper (Mantis-8B-Idefics2, with easy swap to other VLMs)
│   ├── clip_retriever.py         # CLIP-based prompt retrieval (baseline)
│   └── scorer.py                 # VILA aesthetic scorer + CLIP content scorer + area scorer
├── pipeline/
│   ├── prompt_builder.py         # Builds VLM prompts for all 3 cropping tasks
│   ├── retrieval.py              # Top-S retrieval logic (Eq. 1 and Eq. 2 from paper)
│   ├── refinement.py             # Iterative crop refinement loop
│   └── cropper.py                # Main Cropper pipeline combining everything
├── evaluation/
│   ├── metrics.py                # IoU, Disp, SRCC, PCC, AccK/N
│   └── evaluate.py               # Run evaluation on test sets
├── utils/
│   ├── coord_utils.py            # Coordinate normalization, denormalization, crop extraction
│   └── visualization.py          # Visualize crops, comparisons, iterative refinement
├── scripts/
│   ├── run_freeform.py           # Run free-form cropping evaluation
│   ├── run_subject_aware.py      # Run subject-aware cropping evaluation
│   ├── run_aspect_ratio.py       # Run aspect-ratio-aware cropping evaluation
│   └── ablation.py               # Run ablation studies
├── requirements.txt
└── README.md                     # Setup instructions, how to run, expected results
```

---

## Detailed Implementation Specifications

### 1. Datasets (`data/`)

**GAICD dataset** (free-form cropping):
- 3,336 images total: 2,636 train, 200 val, 500 test
- Each image has ~90 annotated crops, each with a MOS (mean opinion score)
- Crops are axis-aligned bounding boxes
- Download from: https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch
- The dataset loader should return: image, list of crops (each crop = (MOS, x1, y1, x2, y2))

**FCDB dataset** (free-form + aspect-ratio-aware cropping):
- 348 test images (we only use test set)
- Each image has a single user-annotated crop box
- For aspect-ratio-aware cropping: treat the aspect ratio of the annotated box as the target
- Download from: https://github.com/yiling-chen/flickr-cropping-dataset

**SACD dataset** (subject-aware cropping):
- 2,906 images: 2,326 train, 290 val, 290 test
- Each image has multiple subject masks and corresponding ground-truth crops
- Download from: https://github.com/bcmi/Human-Centric-Image-Cropping

For `download.sh`: write wget/curl commands to download these. If direct links aren't available, provide instructions for manual download with expected directory structure.

For `datasets.py`: implement PyTorch-style dataset classes for each. Each should support:
- Loading images
- Loading ground-truth crops with scores
- Loading masks (for SACD)
- Precomputing and caching CLIP embeddings for all images

---

### 2. VLM Wrapper (`models/vlm.py`)

Implement a wrapper for **Mantis-8B-Idefics2** from HuggingFace.

```python
# Model: TIGER-Lab/Mantis-8B-Idefics2
# This model supports multi-image input, which is critical for ICL
```

The wrapper should:
- Load model with `accelerate` for multi-GPU distribution
- Accept a list of PIL images + text prompt
- Return generated text (which contains crop coordinates)
- Support configurable temperature (default 0.05) and max_new_tokens
- Parse the output text to extract crop coordinates as tuples
- Handle batch inference if possible for efficiency

**Important**: The coordinate parsing is critical. The VLM outputs text like:
```
(0.8, 120, 50, 800, 700), (0.7, 200, 100, 900, 800)
```
Write robust regex-based parsing that handles:
- Variable whitespace
- Missing/extra parentheses
- Coordinates that are out of range (clamp to valid range)
- Cases where the VLM outputs garbage (return None/empty)

Also implement an **abstract base class** so we can easily swap in other VLMs later:

```python
class BaseVLM:
    def generate(self, images: List[PIL.Image], prompt: str, temperature: float, num_outputs: int) -> str:
        raise NotImplementedError
    
    def parse_crops(self, output_text: str, task: str) -> List[Tuple]:
        raise NotImplementedError
```

**NOTE on Mantis-8B-Idefics2 specifics**:
- The paper used S=30 ICL examples for Gemini, but Mantis has a smaller context window
- For Mantis, the paper used S=10, R=5 crops, L=2 iterations
- Use these reduced hyperparameters for Mantis

---

### 3. CLIP Retriever (`models/clip_retriever.py`)

Use OpenCLIP with ViT-B/32 (matching the paper):

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
```

Implement:
- `encode_images(images) -> embeddings`: Batch encode all images in the dataset
- `retrieve_top_s(query_embedding, database_embeddings, S) -> indices`: Cosine similarity, return top-S
- `select_ground_truth(query, retrieved_images, task) -> List[crops]`: Task-specific G metric
  - Free-form: Select top-T crops by MOS score
  - Subject-aware: Select crop with closest mask center point (negative L2 distance)
  - Aspect-ratio: Select crop with matching aspect ratio

Precompute and save all database embeddings to disk for efficiency. Use FAISS for fast retrieval if the dataset is large.

---

### 4. Scorer (`models/scorer.py`)

Implement three scorers:

**VILA-R aesthetic scorer**:
- Use the VILA-R model from: https://github.com/google-research/google-research/tree/master/vila
- If VILA-R is hard to set up, use a substitute aesthetic scorer like LAION aesthetic predictor or NIMA
- Input: cropped image -> Output: aesthetic score in [0, 1]

**CLIP content preservation scorer**:
- Cosine similarity between CLIP embeddings of cropped image and original image
- Already in [0, 1] range (cosine similarity)

**Area scorer**:
- A = (H_crop * W_crop) / (H * W), already in [0, 1]

**Combined scorer**:
- Normalize all scores to [0, 1]
- For free-form cropping: use VILA + Area (paper's choice from Table 5)
- For subject-aware cropping: use VILA + Area (paper's choice from Table 14)
- For aspect-ratio-aware cropping: use CLIP only (paper's choice from Table 16)
- Make the combination configurable

---

### 5. Prompt Builder (`pipeline/prompt_builder.py`)

Build the exact prompts from the paper. There are 3 different prompt templates:

**Free-form cropping (Table 1 in paper)**:

Initial prompt format:
```
"Localize the aesthetic part of the image. (s, x1, y1, x2, y2) represents the region. x1 and x2 are the left and right most positions, normalized into 1 to 1000, where 1 is the left and 1000 is the right. y1 and y2 are the top and bottom positions, normalized into 1 to 1000 where 1 is the top and 1000 is the bottom. s is MOS score. We provide several images here.
{image 1} (s1, x1, y1, x2, y2), (s2, x1, y1, x2, y2), ..., (sT, x1, y1, x2, y2),
{image 2} (s1, x1, y1, x2, y2), ...
...
{image S} ...
{Query image},"
```

Refinement prompt format:
```
[Initial prompt] +
"{Cropped image 1} (s1, x1, y1, x2, y2), Score is {score1}
{Cropped image 2} (s2, x1, y1, x2, y2), Score is {score2}
...
Propose similar crop that has high score. The region should be represented by (s, x1, y1, x2, y2)."
```

**Subject-aware cropping (Table 13 in paper)**:
- Coordinates normalized to 0-1 (NOT 1-1000 like free-form!)
- Includes mask center point (cx, cy) for each example
- No MOS score in the crop tuple, just (x1, y1, x2, y2)
- Refinement prompt says: "Propose different crop. The region should be represented by (x1, y1, x2, y2). Output:"

**Aspect-ratio-aware cropping (Table 15 in paper)**:
- Includes image size (w, h) and crop ratio in the prompt
- Coordinates are in pixel space (not normalized!)
- Initial prompt says: "Find visually appealing crop. Give the best crop in the form of a crop box and make sure the crop has certain width:height."
- Refinement asks to "Propose a different better crop with the given ratio."

**Critical implementation notes**:
- The prompt includes ACTUAL IMAGES interleaved with text. The VLM receives a multi-modal prompt.
- For Mantis-8B-Idefics2, format images using the model's expected chat template
- Coordinate normalization differs between tasks — be very careful here

---

### 6. Retrieval (`pipeline/retrieval.py`)

Implement Equations 1 and 2 from the paper:

```python
def retrieve_icl_examples(query_image, database, clip_retriever, task, S, T):
    """
    Args:
        query_image: PIL Image (and optionally mask for subject-aware, aspect_ratio for AR-aware)
        database: Dataset with images and ground-truth crops
        clip_retriever: CLIP-based retriever
        task: 'freeform' | 'subject_aware' | 'aspect_ratio'
        S: number of images to retrieve
        T: number of ground-truth crops per image (only for free-form)
    
    Returns:
        List of (image, crops) tuples for ICL
    """
    # Step 1: Retrieve top-S images using Q (CLIP similarity)
    # Step 2: For each retrieved image, select best crop(s) using G (task-specific)
```

Task-specific G metric:
- Free-form: G = MOS score -> select top-T crops by MOS
- Subject-aware: G = -L2(center(query_mask), center(retrieved_mask)) -> select crop for closest mask
- Aspect-ratio: G = similarity between crop's aspect ratio and target aspect ratio -> select matching crop

---

### 7. Iterative Refinement (`pipeline/refinement.py`)

```python
def iterative_refinement(vlm, scorer, query_image, initial_crops, prompt_builder, L=2, task='freeform'):
    """
    Args:
        vlm: VLM model wrapper
        scorer: Combined scorer
        query_image: PIL Image
        initial_crops: List of R crop coordinates from initial VLM output
        prompt_builder: Prompt builder for the task
        L: number of refinement iterations
        task: cropping task type
    
    Returns:
        Final best crop coordinates
    """
    current_crops = initial_crops
    
    for iteration in range(L):
        # 1. Crop the image according to each proposal
        # 2. Score each cropped image using the scorer
        # 3. Build refinement prompt with cropped images + scores
        # 4. Ask VLM to generate improved crop
        # 5. Parse output and update current_crops
    
    # For free-form and subject-aware: return crop from final iteration
    # For aspect-ratio: return highest-scoring crop across all iterations
    # (This is based on Table 8 in the paper)
```

---

### 8. Main Pipeline (`pipeline/cropper.py`)

```python
class Cropper:
    def __init__(self, vlm, clip_retriever, scorer, database, config):
        ...
    
    def crop(self, query_image, task='freeform', mask=None, aspect_ratio=None):
        # 1. Retrieve ICL examples
        # 2. Build initial prompt
        # 3. Generate initial crop candidates from VLM
        # 4. Run iterative refinement
        # 5. Return final crop
```

---

### 9. Metrics (`evaluation/metrics.py`)

Implement ALL metrics from the paper:

- **IoU** (Intersection over Union): Standard bounding box IoU
- **Disp** (Boundary displacement error): Average L1 distance between GT and predicted normalized coordinates
- **SRCC** (Spearman rank-order correlation): Between predicted MOS and GT MOS (use top-5 crops)
- **PCC** (Pearson correlation): Between predicted MOS and GT MOS (use top-5 crops)
- **AccK/N**: Whether top-K predictions fall within top-N GT crops by MOS
  - Compute: Acc1/5, Acc2/5, Acc3/5, Acc4/5, Acc1/10, Acc2/10, Acc3/10, Acc4/10

---

### 10. Evaluation Scripts (`scripts/`)

Each script should:
- Load the appropriate dataset
- Initialize all models (VLM, CLIP retriever, scorer)
- Run Cropper on all test images
- Compute and print all relevant metrics
- Save results to a JSON file
- Support resuming from checkpoints (save intermediate results) — this is important because VLM inference is slow

Include proper logging, progress bars (tqdm), and estimated time remaining.

---

### 11. Hyperparameters (`configs/default.yaml`)

```yaml
# For Mantis-8B-Idefics2 (reduced from Gemini settings)
freeform:
  S: 10              # number of ICL examples (30 for Gemini, 10 for Mantis)
  T: 5               # number of GT crops per example
  R: 5               # number of candidate crops (6 for Gemini, 5 for Mantis)
  L: 2               # refinement iterations
  temperature: 0.05
  scorer: "vila+area"
  coord_range: [1, 1000]   # coordinate normalization range

subject_aware:
  S: 30              # number of ICL examples
  T: 1               # one crop per example
  R: 5               # number of candidate crops
  L: 10              # refinement iterations (higher for subject-aware!)
  temperature: 0.05
  scorer: "vila+area"
  coord_range: [0.0, 1.0]  # normalized to 0-1

aspect_ratio:
  S: 10              # number of ICL examples
  T: 1               # one crop per example
  R: 6               # number of candidate crops
  L: 2               # refinement iterations
  temperature: 0.05
  scorer: "clip"     # CLIP only for aspect-ratio task
  coord_range: "pixel"     # pixel coordinates

# Model
vlm_model: "TIGER-Lab/Mantis-8B-Idefics2"
clip_model: "ViT-B-32"
clip_pretrained: "openai"
```

---

### 12. Key Implementation Gotchas

1. **Coordinate systems differ between tasks!** Free-form uses 1-1000, subject-aware uses 0-1, aspect-ratio uses pixel coordinates. Handle this carefully in prompt_builder.py and coord_utils.py.

2. **VLM output parsing will fail sometimes.** The VLM might output malformed coordinates, text instead of numbers, or completely irrelevant content. Build robust parsing with fallbacks (e.g., return center crop if parsing fails entirely).

3. **Memory management.** Loading 10-30 images into a single VLM prompt is memory-intensive. Use float16/bfloat16 for the VLM. Consider image resizing if needed.

4. **Caching.** Cache CLIP embeddings, VLM outputs, and scorer results. VLM inference is the bottleneck — if a run crashes, you don't want to redo everything.

5. **Mantis-8B context window.** Mantis has a much smaller context window than Gemini 1.5 Pro. If 10 images don't fit, reduce S further. Log warnings when the prompt is truncated.

6. **VILA-R availability.** If VILA-R is hard to set up, use NIMA or the LAION aesthetic predictor as a substitute. Document which scorer you used.

7. **Reproducibility.** Set all random seeds. Log all hyperparameters. Save the exact prompts sent to the VLM for debugging.

---

### 13. Expected Results (Targets to Verify Against)

From Table 9 of the paper, Mantis-8B-Idefics2 on GAICD free-form cropping:

| Metric | Paper's Number |
|--------|---------------|
| Acc5   | 80.2          |
| Acc10  | 88.6          |
| SRCC   | 0.874         |
| PCC    | 0.797         |
| IoU    | 0.672         |

Your replication should be close to these numbers. Small differences are expected due to implementation details, but if you're way off (e.g., IoU < 0.5), something is wrong with the prompt construction or coordinate parsing.

---

### 14. README.md

Write a clear README with:
- One-line description
- Setup instructions (conda env, pip install, model downloads)
- Dataset download and preparation steps
- How to run each evaluation (with exact commands)
- Expected results table
- How to swap in different VLMs
- Known limitations and differences from the original paper

---

## Summary of Priorities

1. **Get free-form cropping on GAICD working first** — this is the main benchmark
2. **Match the Mantis-8B numbers from Table 9** — this validates your implementation
3. **Then extend to subject-aware and aspect-ratio tasks**
4. **Save everything** — intermediate results, prompts, VLM outputs — for debugging and analysis

Start with free-form cropping on GAICD. Get that working end-to-end before touching the other tasks.
