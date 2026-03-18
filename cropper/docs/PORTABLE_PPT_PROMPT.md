# AI Presentation Generator Prompt

## Quick Instructions
Copy the entire content below and paste it into any AI presentation tool (Gamma, Tome, Canva AI, Beautiful.ai, etc.) to generate a professional slide deck.

---

# PROMPT START

Create a 6-slide professional academic presentation titled "Cropper: Training-Free Image Cropping using Vision-Language Models with In-Context Learning"

## Slide 1: Problem Statement & Motivation

**Title:** Training-Free Image Cropping with Vision-Language Models

**Content:**
- Traditional image cropping methods require large annotated datasets and task-specific training
- This limits generalization to new domains and requires significant compute resources
- Key Insight: Vision-Language Models (VLMs) already understand composition, aesthetics, and visual balance
- Our Approach: Use VLMs with In-Context Learning (ICL) to perform aesthetic cropping WITHOUT any training
- Simply show the VLM a few examples of good crops, and it learns to crop new images

**Visual:** Side-by-side comparison diagram showing:
- Left: "Traditional" - Dataset → Training → Model (complex, expensive)
- Right: "Cropper" - Example images → VLM → Output crop (simple, no training)

---

## Slide 2: System Architecture

**Title:** Cropper Pipeline: End-to-End Architecture

**Content - Create a horizontal flow diagram:**

```
Input Image → CLIP Retriever → ICL Examples → VLM Prompt → Initial Crops → Refinement Loop → Final Crop
                 |                |               |              |               |
            ViT-B/32        S=10 → T=5      Mantis-8B        R=5 crops     VILA Scorer
```

**Key Components Table:**

| Component | Model | Purpose |
|-----------|-------|---------|
| Image Encoder | CLIP ViT-B/32 | Visual similarity search |
| VLM | Mantis-8B-Idefics2 | Crop generation via ICL |
| Aesthetic Scorer | VILA | Quality assessment [0-1] |

**Parameters:**
- S=10: Initial similar images retrieved
- T=5: ICL examples in prompt
- R=5: Crop candidates per iteration
- L=2: Refinement iterations

---

## Slide 3: In-Context Learning Example Retrieval

**Title:** Step 1: Retrieving High-Quality Cropping Examples

**Visual - Show a flow:**

Query Image → CLIP Encoding → Cosine Similarity Search → GAICD Database (1000+ images with expert crops) → Top-10 Similar → Filter by Quality → Top-5 Best Examples

**Key Points:**
- Use CLIP to find visually similar images from a database
- Each database image has human-annotated "best crops" with quality scores (MOS 1-5)
- Select diverse, high-quality examples as "teachers" for the VLM
- The VLM learns: "If similar images were cropped this way, I should crop similarly"

**Example visualization:** Show query image of a bird, with 5 similar bird images and their highlighted crop boxes

---

## Slide 4: VLM Cropping - Prompt & Generation

**Title:** Step 2: VLM-Based Crop Coordinate Generation

**The Exact Prompt Format:**
```
Localize the aesthetic part of the image. (s, x1, y1, x2, y2) represents
the region. x1, x2 are left/right positions [1-1000]. y1, y2 are top/bottom
positions [1-1000]. s is quality score.

{Example 1}, (0.85, 120, 80, 890, 720)
{Example 2}, (0.78, 50, 150, 950, 800)
{Example 3}, (0.82, 200, 100, 850, 650)
{Example 4}, (0.90, 100, 200, 900, 850)
{Example 5}, (0.75, 150, 50, 800, 600)

{Query image}
Output 5 crops represented by (s, x1, y1, x2, y2).
```

**Three Task Types:**

| Task | Coordinates | Special Input |
|------|-------------|---------------|
| Free-form | [1, 1000] | None |
| Subject-Aware | [0, 1] | Mask center (cx, cy) |
| Aspect-Ratio | Pixels | Target ratio (e.g., 16:9) |

**VLM Output:** 5 crop candidates with predicted quality scores

---

## Slide 5: Iterative Refinement with VILA Scoring

**Title:** Step 3: Score-Guided Iterative Refinement

**The Loop (repeat L=2 times):**

```
Initial 5 Crops
      ↓
VILA Scorer evaluates each crop → Scores: [0.72, 0.65, 0.81, 0.58, 0.69]
      ↓
Feedback to VLM: "Crop 3 scored 0.81, Crop 4 scored 0.58. Propose better crops."
      ↓
VLM generates NEW 5 crops (guided by score feedback)
      ↓
Repeat...
      ↓
Select BEST crop across ALL iterations
```

**VILA Scorer:**
- Vision-Language Image Aesthetics model (Google)
- Trained on AVA dataset with user comments
- Input: Image → Output: Quality score [0.0-1.0]
- Available via TensorFlow Hub

**Final Selection:** Choose crop with highest VILA score across all iterations

---

## Slide 6: Future Research Directions

**Title:** Proposed Extensions & Future Work

**1. Crop-Quality-Aware Retrieval**
- Current: Retrieve by visual similarity only
- Problem: Similar images may have poor crops
- Solution: Score = CLIP_similarity + Crop_quality
- Impact: Learn from better "teachers"

**2. Preference-Conditioned Multi-Crop Generation**
- Current: Single "best" crop assumption
- Problem: Multiple subjects, different user preferences
- Solution: Accept text prompts ("Focus on person on left", "Wide cinematic crop")
- Generate diverse crops for different styles/platforms

**3. Video Cropping with Temporal Consistency**
- Current: Image-only, frame-by-frame jitters
- Problem: Crop jumps around in video
- Solution: Smooth crop trajectory using:
  - Optical flow guidance
  - Temporal smoothness constraints
- Result: Stable, aesthetically-pleasing video crops

---

## Visual Style Requirements

- Clean, professional academic style
- Color scheme: Blues, grays, accent orange
- Use flow diagrams with arrows for pipelines
- Include code blocks for prompts (monospace font)
- Add example images showing:
  - Red bounding box = Model prediction
  - Green bounding box = Ground truth
  - IoU score displayed

## Target Metrics to Display

| Metric | Target |
|--------|--------|
| IoU | 0.672 |
| Acc@5 | 80.2% |
| Acc@10 | 88.6% |
| SRCC | 0.874 |

# PROMPT END

---

## Notes for Using This Prompt

1. **For Gamma.app**: Paste the entire prompt, it will auto-generate slides with diagrams
2. **For Tome**: Use the "Create presentation" feature with this as input
3. **For Canva AI**: Use "Magic Design" and paste sections individually
4. **For ChatGPT/Claude with artifacts**: Ask it to create slide-by-slide content
5. **For Google Slides + Gemini**: Use Gemini to generate content, then paste into slides

## Image Files to Upload

From `results/freeform_50samples_paperish_v2/crops/`:

**Best Images by IoU (High Quality Results):**
| Image | IoU | Use For |
|-------|-----|---------|
| 215046.jpg | 0.760 | Best result - show in architecture slide |
| 212000.jpg | 0.745 | Good for refinement example |
| 211117.jpg | 0.726 | ICL retrieval visualization |
| 219424.jpg | 0.709 | Multi-crop comparison |
| 220035.jpg | 0.704 | Future directions slide |

**Current Achieved Metrics:**
- IoU: 0.527 (target: 0.672)
- SRCC: 0.022 (target: 0.874)

These images show red (predicted) vs green (GT) crop boxes with IoU scores overlaid.
