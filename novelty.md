# Novelty Ideas: Improving Cropper with Mantis-8B

These are proposed improvements to the baseline Cropper pipeline targeting genuine IoU/SRCC gains with Mantis-8B-Idefics2. The goal is not to match Gemini — it is to push Mantis's own ceiling higher.

---

## Idea 1: Visual Crop Grounding in ICL Examples

### Problem
The paper's ICL prompt shows the full image alongside coordinate tuples:
```
{image 1}, (0.85, 120, 80, 890, 720)
```
The VLM must mentally map abstract numbers to a spatial region. For a weaker model like Mantis (vs Gemini), this abstraction is lossy — the model may not reliably understand what region `(120, 80, 890, 720)` refers to within the image.

### Proposed Change
Show the **extracted crop image** alongside the full image and coordinates:
```
{full image 1}, {cropped region 1}, (0.85, 120, 80, 890, 720)
```
Now the VLM has direct visual evidence of what a high-quality crop of this scene looks like. It no longer needs to mentally decode coordinates — it can see the answer.

### Why This Helps
- Directly compensates for Mantis's weaker spatial coordinate reasoning vs Gemini
- Visual ICL is more grounded than numeric ICL for vision models
- The VLM can reason: "this part of a scene like this is aesthetically good" rather than "these numbers describe a region I should crop to"

### Implementation
- `pipeline/prompt_builder.py`: In `_build_freeform_initial()`, for each ICL example extract the crop using `utils/coord_utils.py:extract_crop()` and append the crop image to the image list alongside its coordinates
- `pipeline/prompt_builder.py`: Update `_build_freeform_refinement()` similarly — already shows crop images, so this is consistent

### Expected Impact: High

---

## Idea 2: Rank-Anchored Refinement Feedback

### Problem
The current refinement prompt treats all R crops equally:
```
{Cropped image 1} (0.72, ...), Score is 0.72
{Cropped image 2} (0.65, ...), Score is 0.65
{Cropped image 3} (0.81, ...), Score is 0.81
Propose similar crop that has high score.
```
Problems:
- Crops are shown in arbitrary order — the VLM has no clear anchor
- "Propose similar crop that has high score" is vague — similar to which crop?
- The VLM may average over all shown crops rather than converging toward the best one

### Proposed Change
Sort crops by score descending, explicitly label the best, and anchor the instruction to it:
```
Best crop so far — Score 0.81:
{Cropped image 3} (0.81, 150, 90, 870, 710)

Other candidates:
{Cropped image 1} (0.72, ...), Score 0.72
{Cropped image 2} (0.65, ...), Score 0.65

Propose 6 variations of the best crop above. Adjust the boundaries to further improve aesthetic quality. Output (s, x1, y1, x2, y2).
```

### Why This Helps
- Gives the VLM a clear optimization target (best crop = anchor)
- "Variations of X" is a more actionable instruction than "similar to all of these"
- Sorting by score descending means the VLM's attention is weighted toward the good examples
- Directly addresses Mantis's tendency to average/blend across examples rather than exploit the best one

### Implementation
- `pipeline/prompt_builder.py`: In `_build_freeform_refinement()`, sort `(crop_img, coords, score)` tuples by score descending, reformat feedback string to separate "best" from "others", update instruction text

### Expected Impact: Medium-High

---

## Idea 3: Diverse Initial Candidate Generation (Multi-Temperature Sampling)

### Problem
R=6 crops from a single VLM call at temperature=0.05 (near-deterministic) cluster tightly — they are small variations of the same crop. The VILA-R scorer then picks the best from a narrow band of candidates. The pipeline never explores the full crop space.

### Proposed Change
Split the R candidates into two groups generated at different temperatures:
- R/2 crops at temperature=0.05 — confident, deterministic (exploit current best guess)
- R/2 crops at temperature=0.8 — more diverse, exploratory (explore alternative crops)

Score all R candidates with VILA-R, keep the best. Wider search coverage increases the probability of landing near the true optimal crop.

### Why This Helps
- More diverse candidates → higher ceiling for VILA-R to select from
- Low-temperature crops anchor around the VLM's confident prediction
- High-temperature crops explore regions the VLM considers plausible but less certain
- No additional model calls if both groups are generated in sequence and scored together

### Implementation
- `models/vlm.py`: In `generate()`, accept a `temperatures` list and make one call per temperature, concatenating parsed crops
- `pipeline/cropper.py`: Pass `temperatures=[0.05, 0.8]` and `R` per temperature to VLM; collect all crops for scoring

### Expected Impact: Medium

---

## Idea 4: Diversity-Enforced ICL Example Selection

### Problem
CLIP retrieves the 30 most semantically similar training images. If the query is a bird on a branch, all 30 retrieved images may be birds — all with centered compositions. The VLM receives a redundant ICL signal that only covers one compositional mode. It gets no guidance for handling off-center subjects, wide landscapes, or background-heavy scenes.

### Proposed Change
After CLIP retrieval of top-2S candidates, apply k-means clustering (k = S/5 = 6) on their CLIP embeddings. Select the top-ranked example from each cluster to form the final S ICL examples.

This enforces compositional diversity without any training — just a reranking step on top of existing CLIP embeddings.

### Why This Helps
- VLM sees diverse compositional examples → more robust crop proposals
- Reduces retrieval collapse (all similar images suggesting the same crop)
- Particularly helps on queries that don't fit the dominant mode of similar images

### Implementation
- `pipeline/retrieval.py`: After `retrieve_top_s()`, apply `sklearn.cluster.KMeans` on retrieved embeddings and select top-1 per cluster
- Fallback: if fewer than k clusters, return original retrieval

### Expected Impact: Small-Medium

---

## Implementation Priority

| # | Idea | Files | Effort | Expected Gain |
|---|------|-------|--------|---------------|
| 1 | Visual crop grounding in ICL | `pipeline/prompt_builder.py` | Medium | High |
| 2 | Rank-anchored refinement feedback | `pipeline/prompt_builder.py` | Low | Medium-High |
| 3 | Multi-temperature diverse candidates | `models/vlm.py`, `pipeline/cropper.py` | Low | Medium |
| 4 | Diversity-enforced ICL selection | `pipeline/retrieval.py` | Low | Small-Medium |

Implement in order 2 → 1 → 3 → 4 (easiest high-impact first).

---

## Evaluation Protocol

For each change, run:
```bash
# Baseline (current)
python run_experiment.py   # with scorer=vila+area, n=50

# With change applied
python run_experiment.py   # same settings, different output_dir
```

Compare IoU, SRCC, PCC side by side. All changes are ablatable independently.
