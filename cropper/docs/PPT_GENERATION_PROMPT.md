# Comprehensive Prompt for Cropper Presentation Generation

## Instructions for AI Tool
Create a professional 6-slide PowerPoint presentation about the Cropper paper replication project. Use a clean, academic visual style with diagrams, flowcharts, and the provided example images. Each slide should be visually engaging with minimal text and maximum clarity.

---

## SLIDE 1: Problem Statement & Motivation

### Title: "Training-Free Image Cropping with Vision-Language Models"

### Key Points to Visualize:
1. **The Problem**: Traditional image cropping methods require:
   - Large annotated datasets
   - Task-specific training
   - Limited generalization to new domains

2. **The Gap**: Existing methods fail to leverage the rich visual understanding already present in Vision-Language Models (VLMs)

3. **Our Approach**: Use VLMs with In-Context Learning (ICL) to perform aesthetic cropping WITHOUT any training

### Visual Elements:
- Show a comparison: "Traditional Pipeline (Training Required)" vs "Cropper Pipeline (Training-Free)"
- Use icons: Dataset icon → Training icon → Model icon (Traditional)
- vs: Example images → VLM → Crop output (Cropper)

### Key Insight:
> "VLMs like Gemini and Mantis already understand composition, aesthetics, and visual balance. We just need to show them examples and ask them to crop."

---

## SLIDE 2: Cropper System Architecture & Components

### Title: "System Architecture: End-to-End Pipeline"

### Main Components (Create a Flow Diagram):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CROPPER PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Input   │───▶│ CLIP Retrieval │───▶│  VLM Prompt  │───▶│  Initial     │ │
│  │  Image   │    │  (ViT-B/32)   │    │  Generation  │    │  Crop Gen    │ │
│  └──────────┘    └───────────────┘    └──────────────┘    └──────────────┘ │
│                          │                    │                   │         │
│                          ▼                    ▼                   ▼         │
│                  ┌───────────────┐    ┌──────────────┐    ┌──────────────┐ │
│                  │  S=10 Similar │    │  T=5 Best    │    │  R=5 Crop    │ │
│                  │  Images Found │    │  ICL Examples│    │  Candidates  │ │
│                  └───────────────┘    └──────────────┘    └──────────────┘ │
│                                                                   │         │
│                                                                   ▼         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    ITERATIVE REFINEMENT LOOP (L=2)                    │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │  │
│  │  │ VILA Scorer │───▶│ Score Crops │───▶│ Refinement  │──┐            │  │
│  │  │ (Aesthetic) │    │ [0.0-1.0]   │    │   Prompt    │  │            │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘  │            │  │
│  │         ▲                                    │          │            │  │
│  │         └────────────────────────────────────┘          │            │  │
│  │                    VLM generates new crops ◀────────────┘            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                   │         │
│                                                                   ▼         │
│                                                          ┌──────────────┐  │
│                                                          │ FINAL CROP   │  │
│                                                          │ (Best Score) │  │
│                                                          └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Details Table:

| Component | Model/Method | Purpose |
|-----------|--------------|---------|
| Image Encoder | CLIP ViT-B/32 | Extract visual features for similarity search |
| VLM | Mantis-8B-Idefics2 | Generate crop coordinates via ICL |
| Scorer | VILA (TFHub) | Aesthetic quality assessment [0-1] |
| Area Scorer | Geometric | Penalize tiny crops |

### Parameters:
- **S = 10**: Number of similar images retrieved initially
- **T = 5**: Number of ICL examples used in prompt
- **R = 5**: Number of crop candidates generated per iteration
- **L = 2**: Number of refinement iterations

---

## SLIDE 3: ICL Example Retrieval

### Title: "Step 1: In-Context Learning Example Selection"

### Process Flow:
```
Query Image → CLIP Encoder → Feature Vector (512-d)
                                    │
                                    ▼
              ┌─────────────────────────────────────┐
              │    GAICD Training Database          │
              │    (~1000 images with GT crops)     │
              │                                     │
              │  Each image has CLIP embedding      │
              │  + Multiple annotated crop boxes    │
              │  + Human MOS scores (1-5 scale)     │
              └─────────────────────────────────────┘
                                    │
                    Cosine Similarity Search
                                    │
                                    ▼
              ┌─────────────────────────────────────┐
              │    Top-S (10) Most Similar Images   │
              │                                     │
              │    Filter by:                       │
              │    - Crop quality (MOS > threshold) │
              │    - Diversity (avoid duplicates)   │
              │                                     │
              │    Select Top-T (5) as ICL Examples │
              └─────────────────────────────────────┘
```

### Visual Example:
Show: Query image (e.g., bird on branch) → 5 retrieved similar images with their GT crops highlighted

### Key Insight:
> "Similar images have similar good crops. If we know how to crop a similar bird photo well, we can use that as a guide."

---

## SLIDE 4: VLM Cropping - Initial Generation

### Title: "Step 2: VLM-Based Crop Generation"

### The Exact Prompt (Free-form Task):

```
Localize the aesthetic part of the image. (s, x1, y1, x2, y2) represents
the region. x1 and x2 are the left and right most positions, normalized
into 1 to 1000, where 1 is the left and 1000 is the right. y1 and y2 are
the top and bottom positions, normalized into 1 to 1000 where 1 is the
top and 1000 is the bottom. s is MOS score. We provide several images here.

{image 1}, (0.85, 120, 80, 890, 720)
{image 2}, (0.78, 50, 150, 950, 800)
{image 3}, (0.82, 200, 100, 850, 650)
{image 4}, (0.90, 100, 200, 900, 850)
{image 5}, (0.75, 150, 50, 800, 600)

{Query image}
Output 5 crops represented by (s, x1, y1, x2, y2). Only output crop tuples.
```

### Three Cropping Tasks:

| Task | Coordinate Range | Input | Output |
|------|------------------|-------|--------|
| **Free-form** | [1, 1000] normalized | Image only | R crops with MOS |
| **Subject-Aware** | [0, 1] normalized | Image + mask center (cx, cy) | Single crop |
| **Aspect-Ratio** | Pixel coordinates | Image + target ratio | R crops with ratio |

### VLM Output Example:
```
(0.82, 150, 100, 850, 700)
(0.78, 200, 150, 900, 750)
(0.75, 100, 80, 800, 650)
(0.71, 180, 120, 880, 720)
(0.68, 120, 90, 870, 680)
```

### Visual:
Show the query image with 5 different colored bounding boxes representing the R=5 crop candidates

---

## SLIDE 5: Iterative Refinement Loop

### Title: "Step 3: Score-Guided Iterative Refinement"

### The Refinement Loop (L=2 iterations):

```
┌────────────────────────────────────────────────────────────────┐
│                    ITERATION 1                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Initial R=5 Crops                                              │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ VILA Scorer: Score each crop [0.0 - 1.0]                │   │
│  │                                                          │   │
│  │ Crop 1: 0.72    Crop 2: 0.65    Crop 3: 0.81            │   │
│  │ Crop 4: 0.58    Crop 5: 0.69                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Refinement Prompt:                                             │
│  "Previous crops and scores:                                    │
│   {Crop 1} (0.72, 150, 100, 850, 700), Score: 0.72             │
│   {Crop 2} (0.65, 200, 150, 900, 750), Score: 0.65             │
│   ...                                                           │
│   Propose similar crop that has high score."                    │
│       │                                                         │
│       ▼                                                         │
│  VLM generates NEW R=5 crops (guided by feedback)              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    ITERATION 2                                  │
│                    (Same process)                               │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│              SELECT BEST CROP ACROSS ALL ITERATIONS            │
│                                                                 │
│   Compare all crops from iterations 0, 1, 2                    │
│   Select crop with HIGHEST VILA score                          │
│                                                                 │
│   Final Output: (x1, y1, x2, y2) in pixel coordinates          │
└────────────────────────────────────────────────────────────────┘
```

### VILA Scorer Details:

**VILA (Vision-Language Aesthetics):**
- Trained on AVA dataset with user comments
- Available via TensorFlow Hub: `tfhub.dev/google/vila/image/1`
- Input: JPEG image bytes
- Output: Quality score [0.0 - 1.0]

**Combined Scoring:**
```
Final Score = w1 * VILA_score + w2 * Area_score
```
Where Area_score penalizes crops that are too small.

---

## SLIDE 6: Future Directions

### Title: "Proposed Extensions & Future Work"

### Extension 1: Crop-Quality-Aware Prompt Retrieval

**Problem**: Current retrieval uses only visual similarity (CLIP). Not all similar images have good crops.

**Solution**: Add crop quality scoring to ICL example selection:
```
Final_Score = α * CLIP_similarity + β * Crop_quality (VILA score of GT crop)
```

**Impact**: Learn from "better teachers" - images with both visual similarity AND high-quality crops.

### Extension 2: Preference-Conditioned Multi-Crop Generation

**Problem**: Single "best" crop doesn't account for:
- Multiple subjects in image
- Different user preferences (portrait vs landscape, tight vs wide)
- Platform-specific requirements (Instagram square, YouTube thumbnail)

**Solution**:
- Accept text preference prompts: "Focus on the person on the left", "Wide cinematic crop"
- Generate diverse crops optimized for different styles
- Reframe as SET generation, not single-answer

### Extension 3: Video Cropping with Temporal Consistency

**Problem**: Frame-by-frame cropping causes jitter and instability.

**Solution**:
- Treat crop as trajectory over time
- Use optical flow / feature correspondences
- Add smoothness constraints:
```
Loss = Σ ||crop_t - crop_{t-1}||² (smoothness)
     + Σ VILA_score(crop_t)      (quality)
```

**Result**: Stable, aesthetically-pleasing video crops without training.

---

## Example Images to Include

Use images from: `results/freeform_50samples_paperish_v2/crops/`

### Recommended Images:
1. `210333.jpg` - Shows prediction vs GT crop comparison
2. `211117.jpg` - Another comparison example
3. `213811.jpg` - High-quality crop result
4. `215046.jpg` - Example with clear subject

### Image Annotations:
- Red box = Predicted crop
- Green box = Ground truth crop
- Show IoU score on visualization

---

## Visual Style Guidelines

1. **Color Scheme**: Use professional blues, grays, and accent orange
2. **Fonts**: Sans-serif (Arial, Helvetica, or similar)
3. **Diagrams**: Use flowchart style with rounded boxes and arrows
4. **Code Blocks**: Use monospace font with syntax highlighting
5. **Images**: Add subtle shadows/borders for visual separation

---

## Metrics to Highlight

### Target Performance (Mantis-8B-Idefics2 on GAICD):
| Metric | Target | Description |
|--------|--------|-------------|
| IoU | 0.672 | Intersection over Union with GT |
| Acc@5 | 80.2% | Top-5 pred overlaps with top-5 GT |
| Acc@10 | 88.6% | Top-10 overlap |
| SRCC | 0.874 | Spearman rank correlation |
| PCC | 0.797 | Pearson correlation |

---

## Summary Prompt for Document Generation

If you need to regenerate this presentation or create a detailed document, use this condensed prompt:

---

**CONDENSED PROMPT:**

Create a technical presentation about "Cropper: Training-Free Image Cropping via VLM In-Context Learning" with these slides:

1. **Problem & Motivation**: Traditional cropping needs training; we use VLMs with ICL instead
2. **Architecture**: CLIP retrieval (S=10→T=5) → VLM (Mantis-8B) → Iterative refinement (L=2, R=5) → VILA scoring
3. **ICL Retrieval**: CLIP similarity search on GAICD database, select diverse high-quality examples
4. **VLM Cropping**: Exact prompt format with (MOS, x1, y1, x2, y2) in [1,1000] range, three task types
5. **Refinement Loop**: VILA scores crops → feedback to VLM → new crops → select best across iterations
6. **Future Work**: (a) Quality-aware retrieval, (b) Multi-crop with preferences, (c) Temporal video cropping

Include flow diagrams, the exact prompts, parameter tables (S=10, T=5, R=5, L=2), and example crop visualizations showing red=predicted, green=GT boxes.

---
