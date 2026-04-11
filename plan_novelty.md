# Plan: Novelty Ideas Implementation for Cropper

## Context

This plan was created after completing `PROJECT_GUIDE.md` and `novelty.md`. The goal is to push Mantis-8B-Idefics2's performance ceiling higher on the GAICD free-form benchmark by making targeted changes to the ICL prompt format and search strategy. These ideas do NOT require training — they are all inference-time modifications.

**Baseline target** (paper, Mantis row, Table 9): IoU=0.672, Acc@5=80.2%, SRCC=0.874, PCC=0.797

Full idea descriptions live in `novelty.md` at the repo root.

---

## Implementation Order: 2 → 1 → 3 → 4

---

## Idea 2: Rank-Anchored Refinement Feedback *(Low effort, Medium-High impact)*

### Problem
Refinement prompt shows crops in arbitrary order with vague instruction "Propose similar crop that has high score." The VLM averages over all shown crops instead of exploiting the best one.

### Change
**File**: `cropper/pipeline/prompt_builder.py`  
**Method**: `PromptBuilder._build_freeform_refinement()`

Steps:
1. Sort `(crop_img, coords, score)` triples by score **descending**
2. First crop → label as "Best crop so far — Score X.XX:"
3. Remaining crops → label as "Other candidates:"
4. Replace instruction text:
   - Old: `"Propose similar crop that has high score. The region should be represented by (s, x1, y1, x2, y2). Only output the new crop tuples."`
   - New: `"Propose {R} variations of the best crop above. Adjust the boundaries to further improve aesthetic quality. Output (s, x1, y1, x2, y2). Only output crop tuples."`
5. Pass `R` from `build_refinement_prompt` → `_build_freeform_refinement` via `task_params.get("R", 6)`

Also update `FREEFORM_REFINEMENT_TEMPLATE`: since the new method builds the string directly, the template is no longer used for freeform — build the full prompt string inline.

### New prompt structure
```
{initial_prompt}

Best crop so far — Score 0.81:
{Cropped image 3} (0.81, 150, 90, 870, 710)

Other candidates:
{Cropped image 1} (0.72, 120, 80, 890, 720), Score 0.72
{Cropped image 2} (0.65, 100, 60, 850, 700), Score 0.65

Propose 6 variations of the best crop above. Adjust the boundaries to further improve aesthetic quality. Output (s, x1, y1, x2, y2). Only output crop tuples.
```

---

## Idea 1: Visual Crop Grounding in ICL Examples *(Medium effort, High impact)*

### Problem
ICL prompt shows full image + abstract coordinates. Mantis must mentally decode what `(120, 80, 890, 720)` looks like — lossy for a weaker model vs Gemini.

### Change
**File**: `cropper/pipeline/prompt_builder.py`  
**Method**: `PromptBuilder._build_freeform_initial()`

Steps:
1. For each ICL example, after appending the full image, extract and append the crop image
2. Crop coords are in [1,1000] range → use existing utility:
   ```python
   from utils.coord_utils import crop_from_normalized
   crop_img = crop_from_normalized(example["image"], (x1, y1, x2, y2), (1, 1000))
   ```
3. Use only the **top-1 crop** (first crop in `example["crops"]`) per example to avoid token explosion
4. Update `example_lines` format:
   - Old: `"{image 1}, (0.85, 120, 80, 890, 720)"`
   - New: `"{image 1}, {crop 1}, (0.85, 120, 80, 890, 720)"` 
   (where `{crop N}` is a new placeholder for the extracted crop image)
5. Update `format_prompt_for_mantis()` to also replace `{crop N}` tokens with `<image>`

**Image list order**: for example i, append `example["image"]` then immediately append `crop_image` so the <image> tokens in the text align with the image list.

**Placeholder naming convention**: `{crop 1}`, `{crop 2}`, ... for the N extracted crops.

---

## Idea 3: Multi-Temperature Diverse Candidates *(Low effort, Medium impact)*

### Problem
R=6 crops from temperature=0.05 cluster tightly. VILA-R picks best from a narrow band.

### Change
**Files**: `cropper/models/vlm.py`, `cropper/pipeline/cropper.py`, `cropper/configs/default.yaml`

**`vlm.py` — `MantisVLM.generate()`**:
- Add optional param `temperatures: List[float] = None`
- When provided, call generate once per temperature, parse crops from each, concatenate results
- Return combined crop string (or handle in caller by collecting parsed crops)

**`cropper.py` — `Cropper.crop()`**:
- Read `temperatures` from `task_config` (default: `None`)
- If set, pass to `vlm.generate()` along with per-temperature R = R // len(temperatures)

**`configs/default.yaml`** addition under `freeform:`:
```yaml
temperatures: [0.05, 0.8]   # half exploit, half explore
```

---

## Idea 4: Diversity-Enforced ICL Selection *(Low effort, Small-Medium impact)*

### Problem
CLIP retrieves top-S semantically similar images, which may all have same composition. VLM gets redundant ICL signal.

### Change
**File**: `cropper/pipeline/retrieval.py`  
**Function**: `retrieve_icl_examples()`

Steps (after `retrieve_top_s()` returns top-2S results):
1. Extract CLIP embeddings for retrieved candidates from `clip_retriever.database_embeddings`
2. Apply `sklearn.cluster.KMeans(n_clusters=k)` where `k = max(2, S // 5)` (e.g., k=6 for S=30)
3. For each cluster, select the candidate with highest CLIP similarity to query
4. Fallback: if sklearn unavailable or < k images, return original top-S

**Requires**: retrieve `2*S` candidates initially (change `S` arg to `retrieve_top_s` → pass `2*S`), then cluster down to S.

---

## Critical Files Summary

| File | Ideas |
|------|-------|
| `cropper/pipeline/prompt_builder.py` | 1, 2 |
| `cropper/pipeline/retrieval.py` | 4 |
| `cropper/models/vlm.py` | 3 |
| `cropper/pipeline/cropper.py` | 3 (config wiring) |
| `cropper/configs/default.yaml` | 3 (temperatures param) |
| `cropper/utils/coord_utils.py` | 1 (reuse `crop_from_normalized`) |

---

## Evaluation Protocol

After each change:
```bash
cd cropper
# Quick smoke (5 samples, verify no crash)
python scripts/run_freeform.py \
  --config configs/default.yaml \
  --data_dir data/GAICD \
  --output_dir results/novelty_idea2_test \
  --max_samples 5 \
  --device cuda

# Full eval (n=50, compare IoU/SRCC/PCC to baseline)
python run_experiment.py   # edit EXP dict output_dir per idea
```

Baseline numbers to beat: IoU=0.629, SRCC=0.258 (before VILA-R fix); post-fix baseline not yet measured.
