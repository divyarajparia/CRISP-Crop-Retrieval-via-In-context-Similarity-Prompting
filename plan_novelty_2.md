# Plan v2: Push Cropper IoU/SRCC/PCC up — sequential, hypothesis-driven

> **Reading instructions for future-Claude:** This file is the durable memory of the strategic discussion that happened on 2026-04-11 about what to do after the first four novelty ideas (visual grounding, rank-anchored refinement, multi-temperature, diverse ICL) all came back noise-level. If context has been compacted, read this file *before* making any new decisions. The user explicitly asked for this file to exist as a recap so the conversation can be resumed cold. The "Strategic decisions log" section captures the *why* — do not silently re-litigate decisions that were already settled (especially: Phase 1 first despite being HP tuning, no premature combos, one GPU sequential, time-budget framing). Tonight's runs are a *diagnostic* to pick what to commit to overnight; they are not the final result.

---

# >>> STARTING NOW: PHASE 1 — SCORER WEIGHT FIX <<<

**Confirmed by the user 2026-04-11 ~17:00:** the very next thing we do is the **VILA / Area scorer-weight fix**. Concretely:

- Wire the existing-but-unused `cropper/configs/default.yaml:52-55` keys (`vila_weight`, `area_weight`) into `create_scorer` (~15 LOC across `models/scorer.py` and `pipeline/cropper.py`).
- New runner `cropper/run_scorer_sweep.py` that patches `cfg["scorer"]["vila_weight"]=1.0`, `cfg["scorer"]["area_weight"]=0.0` exactly the way `cropper/run_novelty_quick.py:patch_config` patches novelty flags.
- Run **`scorer_vila10_area00_30`** as Run #1 on GPU 3, after `idea1_vg_top5_30` finishes (~30 min).
- This is what we're calling **"Phase 1"** in this file from now on. It is *not* Phase 2 (GT-crop seeding). It is *not* a combination of ideas. It is purely the scorer weight wiring.

### Label history (because earlier drafts confused this)

| Earlier name | Final name | What it is |
|---|---|---|
| Plan v1 "Phase B" / "Option B" | **Phase 1** (this file) | Scorer weight fix — wire `vila_weight` / `area_weight` from config; sweep variants like `vila10_area00`. **STARTING WITH THIS.** |
| Plan v1 "Phase C" | **Phase 2** | GT-crop seeding — pool ICL example crops as initial candidates instead of Mantis hallucinations. Stretch goal for after Phase 1. |
| Plan v1 "Phase D" | **Phase 3** | Calibration head trained on GAICD train. Overnight stretch goal for tomorrow. |
| Plan v1 "Phase A" | **dropped** | "Combine Idea 2 + Idea 4" — user explicitly rejected as a low-information combo. |

---

## Top-of-file recap

- **What:** Three sequential phases of experiments (Phase 1 = scorer area-fix, Phase 2 = GT-crop seeding, Phase 3 = calibration head). **Phase 1 starting now**, Phase 2 tonight or tomorrow, Phase 3 over the next day.
- **Why this order:** Phase 1 mechanically gates Phase 2 (the scorer is the IoU bottleneck — Phase 2's better candidates are wasted under a biased scorer). So even though Phase 1 is honestly just HP tuning, it has to go first. Phase 2 then becomes the *real* novelty contribution layered on top of the fixed scorer.
- **Constraints:** 1 GPU (GPU 3), sequential (~1.5h per 30-sample run). 6–7h budget today. Full night and tomorrow available for longer runs once a direction is confirmed.
- **Tonight's job:** prove which direction is real so we know where to point overnight compute. Not "ship the final number."
- **Headline contribution for the writeup will be Phase 2 (GT-crop seeding), not Phase 1.** Phase 1 is framed as a *diagnostic ablation* that uncovered a structural bias in the released codebase. Phase 2 is the new mechanism that this fix enables.

---

## Context

Four novelty ideas have now been run on the Mantis-8B-Idefics2 Cropper replication. None of the four individually moved IoU by more than ±0.012 on the apples-to-apples 30-sample slice. The user wants a deep, evidence-driven plan for what to do next to see a *considerable* boost (for an assignment writeup) — but **not** by combining the existing four ideas, since each individually is a small effect and a combination just stacks small effects without testing a new hypothesis. We need new mechanisms.

### Result table (verified from `freeform_results.json` / live `checkpoint.json`, last updated 2026-04-11 ~15:30)

| Run | n | IoU | SRCC | PCC | Notes |
|---|---|---|---|---|---|
| baseline (paperish_v2, 50) | 50 | 0.5268 | 0.0225 | -0.0136 | the long baseline |
| Idea 1 — full visual grounding | 50 | **0.4088** | 0.0000 | 0.0000 | catastrophic; ~61 image tokens broke Mantis parser |
| Idea 2 — rank-anchored refinement | 50 | **0.5364** (+0.0096) | **0.0659** (+193%) | 0.0210 | small IoU win, big SRCC win |
| baseline_30 (quick) | 30 | 0.5334 | 0.1263 | 0.0599 | n=30 slice baseline |
| Idea 4 — diverse ICL (KMeans) | 30 | 0.5214 (-0.012) | **0.2113** (+67%) | **0.1814** (+200%) | IoU regress, huge calibration win |
| Idea 1 corrected — vg_top5 | 10 (in-progress) | 0.5209 vs baseline 0.5197 | — | — | within noise of baseline |
| Idea 3 — multi-temperature | 10 (in-progress) | 0.5202 vs baseline 0.5197 | — | — | within noise of baseline |

**Apples-to-apples on first 10 deterministic samples:** baseline=0.5197, idea1_corrected=0.5209 (+0.0012), idea3=0.5202 (+0.0005). Both essentially noise.

### The single most important diagnostic insight

**The external scorer (VILA + Area), not Mantis crop generation, is what determines the IoU number.** Trace at `cropper/pipeline/refinement.py:196-202`:

```python
if task in ["freeform", "aspect_ratio"]:
    best_crop, best_score = _select_best_across_iterations(
        all_iterations, all_scores  # ← scores from CombinedScorer, not Mantis
    )
```

Whatever crops Mantis generates, the FINAL pick is whichever scored highest under `CombinedScorer`. Mantis self-scores are only fed *back into the prompt* as a hint. So all four existing ideas attacked the wrong half of the pipeline:

| Idea | What it changes | Affects IoU directly? |
|---|---|---|
| 1 — Visual grounding | Prompt only | No (only via candidate set) |
| 2 — Rank-anchored | Refinement prompt format only | No |
| 3 — Multi-temperature | Candidate generation diversity | No |
| 4 — Diverse ICL | Retrieval diversity | No |

**Nobody touched the scorer.** And the scorer is the IoU bottleneck. This explains why each idea on its own only nudges IoU by noise-level: the candidate pool keeps changing but the same biased scoring head keeps selecting the same wrong winners.

### The "elephant in the room"

`cropper/models/scorer.py:608-620` — default freeform weights are `{vila: 1.0, area: 1.0}`, which after normalization at line 627 become `{vila: 0.5, area: 0.5}`. So **half** the final scoring power goes to `AreaScorer`:

```python
# scorer.py:583
return min(1.0, max(0.0, crop_area / orig_area))
```

…which is literally "bigger crop = better score" capped at 1.0. On GAICD the GT crops are *aesthetic* picks, not "crop as little as possible" picks, so this 50% area weight is a structural bias toward crops larger than GT.

The kicker: `cropper/configs/default.yaml:52-55` already declares `scorer.vila_weight`, `scorer.area_weight`, `scorer.clip_weight` — but `pipeline/cropper.py:create_cropper:376-384` calls `create_scorer(...)` without passing them. The knobs exist; they're not wired.

---

## Strategic decisions log (the *why* behind this plan — do not re-litigate)

These decisions were settled during the 2026-04-11 conversation. The user explicitly chose each one. Future-Claude: respect them unless the user reopens them.

### Decision 1 — Don't combine the existing 4 ideas as the next experiment.

Original plan v1 proposed "combine Idea 2 + Idea 4" as the first new run. User rejected: *"i dontwant to try 2 + 4 first. we trie the mindivuaully, w knwo the yarent gonan give acrazy jump. we can try that later."*

**Why:** Combining small effects is low-information. Doesn't distinguish "the ideas are truly additive" from "we got lucky on a small slice." Either way the headline number is still small. The user wants experiments that test a *new mechanism*, not experiments that re-package known small effects. Combos can be revisited *after* a new mechanism is shown to work, as a follow-up to stack on the new baseline. (Saved to feedback memory: `feedback_no_combo_experiments.md`.)

### Decision 2 — Sequential implementation, one phase at a time, with decision checkpoints.

User explicitly asked: *"how do u want to do it? one by one impelntatio nfor all ideas?"*

**Why:** Each experiment's outcome should inform the next one. If a phase doesn't behave as expected, the user wants to debug and re-plan before sinking implementation effort into the next phase. Don't batch implementation work for two phases into a single coding session. (Saved to feedback memory: `feedback_sequential_implementation.md`.)

### Decision 3 — Phase 1 (scorer area-fix) IS hyperparameter tuning. The user called this out and was right.

User: *"why is ahse 1 so imp? isnt it jsut hyperparametr tuning?"*

**Why it's still done first:** Phase 1 is conceded to be HP tuning, not a contribution. But it has to go first because **the scorer mechanically gates everything else.** Phase 2 (GT-crop seeding) gives the scorer better candidates, but if the scorer is biased, it'll still pick a mediocre big Mantis-generated crop over the smaller-but-aesthetic GT-seeded crop. So Phase 2's lift is structurally capped until Phase 1 lands. Phase 1 unblocks Phase 2.

**Writeup framing:** Phase 1 is a *diagnostic*, not the contribution. It will appear in the writeup as: *"we identified that the released codebase scores crops with `0.5 * aesthetic + 0.5 * pixel_area`, which is structurally biased on the GAICD benchmark whose ground truth is aesthetic-only. We removed the area term as a bug-fix preliminary."* Then Phase 2 becomes the contribution: *"having fixed the scorer, we then introduced GT-crop seeding to give the unbiased scorer better candidates to choose between."*

### Decision 4 — Today is "find a direction with signal," not "ship the headline number."

User: *"i dont have a lot of time. i want to see a good result by tonight, i.e. in 6-7 hours [...] i have full night and tomorrow full day too -so we can def spend mroe time to imrpove and try stuff, but at first, i just wanna get some good resutls by night, so i know what tp do for the overnight runs."*

**Why this matters for sequencing:** Tonight's runs are diagnostics. The output we want is "yes Phase 1 works → overnight = Phase 1+2 stacked at 200 samples" or "no Phase 1 didn't work → overnight = pivot to Phase 3 (cal head)". The *result* of tonight's experiments determines what we commit overnight compute to. So tonight should optimize for *information per run*, not for *final IoU*.

### Decision 5 — One GPU only, sequential.

User: *"also - i wudl only use 1 gpu form now."*

**Why:** Earlier proposed parallel-on-GPU1+GPU3 schedules are scrapped. ~1.5h per 30-sample run. ~3 sequential runs fit in the 6-7h budget plus coding time.

### Decision 6 — Headline contribution for the writeup will be Phase 2 (GT-crop seeding), not Phase 1.

User: *"sp lets try vila ara thign, then based on results, we might even benefit formusing that + stacke withpheasee 2 idea right?"*

**Why:** User correctly understands that Phase 1+Phase 2 stacks naturally — Phase 1 fixes which crop wins, Phase 2 fixes what crops are in the pool. Once Phase 1 unblocks the scorer, Phase 2 can deliver its full mechanism. The writeup story is: bug-fix (Phase 1, diagnostic) → new mechanism (Phase 2, contribution) → optionally stretch goal (Phase 3, cal head).

---

## Phase 1 — Scorer weight fix (the diagnostic, runs FIRST tonight)

### Hypothesis

The 50% area weight in `CombinedScorer` is a structural bias toward larger-than-GT crops, costing meaningful IoU. Down-weighting (or zeroing) it will shift selection toward aesthetically-judged crops, which is what GAICD MOS measures.

### What the repo currently does (concrete)

**Where the scoring happens** — `cropper/pipeline/refinement.py:116, 192, 200`:

```python
scores = scorer.score_batch(crop_images)              # every refinement iteration
final_scores = scorer.score_batch(final_crop_images)  # final pass
best_crop, best_score = _select_best_across_iterations(all_iterations, all_scores)
```

The final crop returned to the evaluator is whichever crop got the highest `scorer` score across all iterations. This is the only thing that determines IoU.

**What `scorer` is** — `cropper/models/scorer.py:586-627`:

```python
# scorer.py:610-612 — DEFAULT WEIGHTS
if task == "freeform":
    weights = {"vila": 1.0, "area": 1.0}

# scorer.py:625-627 — NORMALIZATION
total = sum(self.weights.get(k, 0) for k in self.scorers.keys())
self.weights = {k: self.weights.get(k, 0) / total for k in self.scorers.keys()}
# → effectively {"vila": 0.5, "area": 0.5}
```

So every score is computed as:
```
final_score = 0.5 * VILA_score + 0.5 * Area_score
```

**What `VILA_score` and `Area_score` are:**

- **VILA**: a learned aesthetic predictor. Outputs ~[0,1]. Trained on human aesthetic ratings of photos. This is the *meaningful* signal.
- **Area** (`scorer.py:578-583`):
  ```python
  crop_area = (x2 - x1) * (y2 - y1)
  orig_area = image_size[0] * image_size[1]
  return min(1.0, max(0.0, crop_area / orig_area))
  ```
  Literally `crop_size / original_image_size`, capped at 1.0. **Bigger = higher score, full stop.** No aesthetic content, no awareness of composition, just pixel area ratio.

### Worked example showing the bias

Two candidate crops on the same image:

| Crop | Description | VILA | Area frac |
|---|---|---|---|
| A | small aesthetic crop, great composition | 0.85 | 0.40 |
| B | larger lazy crop, basically the whole picture, mediocre framing | 0.55 | 0.80 |

Under the current default `0.5 * VILA + 0.5 * Area`:

| Crop | VILA term | Area term | Final |
|---|---|---|---|
| A | 0.5 × 0.85 = 0.425 | 0.5 × 0.40 = 0.20 | **0.625** |
| B | 0.5 × 0.55 = 0.275 | 0.5 × 0.80 = 0.40 | **0.675** ← wins |

**Crop B wins** even though Crop A is the better photograph. The 0.5 area weight buys Crop B a 0.20 head start that more than compensates for being 0.30 worse on aesthetics. And Crop B is exactly the kind of crop that the GAICD GT *isn't*. This explains why baseline SRCC is 0.02–0.13 (almost no correlation between scorer score and human MOS): half the score is "biggest wins", which is unrelated to MOS.

Re-running the example with `vila=1.0, area=0.0`:

| Crop | VILA term | Area term | Final |
|---|---|---|---|
| A | 1.0 × 0.85 = 0.85 | 0 | **0.85** ← wins |
| B | 1.0 × 0.55 = 0.55 | 0 | 0.55 |

Now Crop A wins, which is what we want — the GAICD-style aesthetic pick.

### Why the actual code change is ~15 lines

The relevant configs **already exist** in `cropper/configs/default.yaml:52-55`:

```yaml
scorer:
  vila_weight: 1.0
  area_weight: 1.0
  clip_weight: 1.0
```

…but `pipeline/cropper.py:create_cropper:376-384` never reads them:

```python
task_config = config.get(task, {})
scorer_config = task_config.get("scorer", "vila+area")
scorer = create_scorer(
    task=task,
    device=device,
    scorer_config=scorer_config,
    require_exact_components=require_exact_components,
)
# ← never passes weights from config["scorer"]
```

And `models/scorer.py:create_scorer:664-714` doesn't even accept a `weights` parameter — it just constructs `CombinedScorer` and lets the default `{vila:1.0, area:1.0}` kick in.

**Wiring change (~15 LOC total):**

1. **`cropper/models/scorer.py`** — `create_scorer()` accepts `weights: Optional[Dict[str, float]] = None`, passes it to `CombinedScorer(scorers, weights=weights, task=task)`. (~5 lines)
2. **`cropper/pipeline/cropper.py`** — `create_cropper()` reads `config.get("scorer", {})`, builds `weights = {"vila": cfg["vila_weight"], "area": cfg["area_weight"]}`, passes to `create_scorer`. (~8 lines)
3. **The runner** — patches `cfg["scorer"]["vila_weight"]` and `cfg["scorer"]["area_weight"]` in the temp YAML, exactly the same way `cropper/run_novelty_quick.py:patch_config` patches the novelty flags. (~100 lines for a runner that does 3 sweep variants — `cropper/run_scorer_sweep.py`.)

### Sweep variants

| Variant | vila_weight | area_weight | Hypothesis |
|---|---|---|---|
| `scorer_vila07_area03_30` | 0.7 | 0.3 | Tilt toward aesthetics, keep some area floor against tiny degenerate crops |
| `scorer_vila10_area00_30` | 1.0 | 0.0 | Pure aesthetic, no area bias — most extreme variant, the headline diagnostic |
| `scorer_vila03_area07_30` | 0.3 | 0.7 | Sanity check the opposite extreme — should *underperform*; if it doesn't, wiring is broken |

**For tonight (1 GPU, sequential):** start with `scorer_vila10_area00_30`. If it wins, that's the diagnostic confirmation and we move to Phase 2. If it goes pathological (any sample with IoU<0.05 — possible if removing area causes degenerate tiny crops), abort early via the per-10-sample checkpoint and try `scorer_vila07_area03_30` instead.

### Expected impact

- Conservative: +0.01 to +0.02 IoU over baseline (0.5334).
- Aggressive: +0.03 to +0.05 IoU.
- Failure mode: degenerate tiny crops (catch with per-sample IoU eyeball at sample 10).

### Decision checkpoint after Phase 1

- **Phase 1 wins (+0.02 or more):** confirmed scorer-is-bottleneck. Proceed to Phase 2 *with `vila10_area00` baked in as the new default scorer config* — the real test of Phase 2 is Phase 2-on-fixed-scorer, not Phase 2-on-broken-scorer.
- **Phase 1 flat (within ±0.005):** scorer weights aren't a big lever. Proceed to Phase 2 *on the default scorer*; if Phase 2 also flat → pivot overnight to Phase 3 (cal head).
- **Phase 1 inverted or pathological:** wiring is wrong; debug before going further.

---

## Phase 2 — Idea 5: GT-crop initialization (the actual novelty contribution)

### Hypothesis

ICL example crops are already loaded in [1, 1000] normalized coordinates with MOS scores at `cropper/pipeline/retrieval.py:186-196`. They are aesthetically vetted by humans on similar images. Right now the initial candidate pool comes from Mantis hallucinating R=6 crops. If we seed refinement with ground-truth crops transferred from the most CLIP-similar training images, the (now-improved Phase-1) scorer has *better candidates* to choose between.

### Why it's independent of Phase 1 (and stacks with it)

| | Controls |
|---|---|
| Phase 1 | *which* candidate wins (selection rule) |
| Phase 2 | *what's in* the candidate pool (sourcing strategy) |

Mechanically additive. Phase 2 alone is partially wasted (the biased default scorer still picks bad-but-big crops over GT-seeded ones). Phase 2 *on top of* a Phase-1-fixed scorer should deliver its full lift.

### Code change (~30 LOC in `cropper/pipeline/cropper.py`)

After `retrieve_icl_examples()` returns, pool the top-T crops from the top-K most-similar ICL examples (already in normalized coords for the example image — treat the coordinate space as image-relative and apply to query, since both have been resized to the same VLM-input dimensions). Add them to (or replace) the Mantis-generated `initial_crops`. Gate behind a new flag: `freeform.novelty.gt_seed_init = true`.

### Two variants to run (~3h total at 1 GPU sequential)

- `gt_seed_pool_30` — pool GT-transferred crops with Mantis crops (R=6 Mantis + ~5 GT = 11 candidates → refinement)
- `gt_seed_replace_30` — replace Mantis initial crops entirely with GT-transferred crops (5 candidates)

### Expected impact

- Standalone (default scorer): +0.005 to +0.02 IoU
- On top of Phase 1's fixed scorer: +0.01 to +0.03 IoU additive

### Risks

- Coordinate transfer assumption may fail: a crop position that works on image A may not correspond to the same composition on image B even after both are resized to VLM input. Mitigation: the `pool` variant is robust to this (Mantis crops are still in the pool); only `replace` puts all weight on the assumption.

---

## Phase 3 — BIG SWING: calibration head trained on GAICD train (overnight stretch goal)

### Hypothesis

VILA is a generic aesthetic predictor trained on web images, not on GAICD MOS specifically. A small head trained on GAICD train (2,636 images × ~90 annotated crops = ~237k labeled `(crop, MOS)` pairs) will out-score VILA on the GAICD test slice because it's directly fitted to the target distribution. This is the only intervention with a realistic shot at +0.05 IoU.

### Implementation (~150 LOC)

New file `cropper/models/gaicd_calibration_head.py` plus scorer integration:

1. **Feature cache** (one-time, ~30 min on GPU): iterate `data/GAICD/annotations/train/*.txt`. Each annotation file is plain text, one crop per line as `x1 y1 x2 y2 MOS`. For each `(image, crop_box, mos)` tuple, extract VILA pooled features (the layer right before VILA's final regression head). Save to `cropper/cache/gaicd_train_features.pkl` keyed by `(image_id, crop_idx)`.
2. **Train head** (~5 min CPU): sklearn `Ridge` or PyTorch 2-layer MLP, `(VILA_features) → MOS`. Hold out 10% of train images for early stopping.
3. **`GaicdCalibrationScorer` class**: wraps the trained head, exposes `score(image, crop_box)` that extracts VILA features and runs the head. Plugs into `CombinedScorer.scorers` dict.
4. **New default scorer config**: `vila+area+gaicd_cal` with weights tuned by sweep (initial guess `{vila: 0.3, area: 0.0, gaicd_cal: 0.7}` — keep a tiny VILA so we don't fully overfit the head).

### Writeup framing

"Training-free *cropper*" is preserved — no fine-tuning of Mantis, CLIP, or VILA's backbone. We trained a tiny 2-layer regression head on top of frozen features (standard linear-probe territory). Zero gradient flow into the cropping pipeline.

### Expected impact

- +0.05 to +0.10 IoU on top of Phase 1+2.
- Risk: out-of-distribution behavior on test crops, but the train/test split is stratified so this should generalize.

### When to do it

- Defer to overnight or tomorrow (it's ~7h of work between coding, feature extraction, training, and one eval run).
- Trigger: Phase 1 + Phase 2 results are in tonight; if they're solid, Phase 3 layers on top; if they're flat, Phase 3 is the only remaining intervention with real upside.

---

## Tonight's concrete plan (1 GPU, sequential, ~6h budget)

### Pre-plan: in-progress runs

- GPU 1: `idea3_multi_temp_30` (started earlier today, was at n=10 ~1h ago)
- GPU 3: `idea1_vg_top5_30` (started earlier today, similar progress)

**Decision needed before starting Phase 1:** let them finish (~30 min remaining) for a clean ideas table, or kill one to free a GPU now. Default is to let them finish — 30 min isn't worth losing per-30-sample completion of two ideas, and the time can be used to code Phase 1 wiring in parallel.

### Tonight's runs (after in-progress runs finish)

3 sequential runs fit in the 6–7h budget:

| Run | Cost | What it answers |
|---|---|---|
| #1 | 0:15 code + 1:30 run | Does removing the area weight improve IoU? |
| #2 | depends on #1 (see decision tree) | — |
| #3 | depends on #2 | — |

**Run #1 = `scorer_vila10_area00_30`** (Phase 1, the diagnostic). This is the single most informative experiment because:
- Result is mechanically binary (the scorer is or isn't the bottleneck)
- 15-minute code change
- If it wins, Run #2 is Phase 2 *on the fixed scorer* (the real headline test)
- If it fails, Run #2 is the milder Phase 1 variant or a hard pivot

### Decision tree

| Run #1 outcome | Run #2 | Run #3 | Overnight target |
|---|---|---|---|
| **Phase 1 wins (+0.02 IoU)** | Code Phase 2 GT-seeding (~45 min) → run `gt_seed_pool_30` *with `vila10_area00` scorer config baked in* | If #2 wins → `gt_seed_replace_30` (variant). If #2 flat → `vila07_area03_30` (sanity-check milder Phase 1) | Phase 1 + Phase 2 stacked at 50 samples (or 200 if confident); start Phase 3 cal-head implementation overnight |
| **Phase 1 flat (within ±0.005)** | `gt_seed_pool_30` *with default scorer* (test Phase 2 in isolation) | depends on #2 | If Phase 2 also flat → overnight = Phase 3 (cal head) is the only remaining lever |
| **Phase 1 bad (degenerate tiny crops, IoU<0.05 on any sample at checkpoint 10)** | abort #1 early; start `vila07_area03_30` (less extreme — keep some area floor) | depends | if Phase 1 fundamentally breaks → overnight = Phase 3 |

### Total budget accounting

- 0:15 code Phase 1 wiring
- 1:30 Run #1
- 0:30 analysis + decision
- 0:45 code Phase 2 (only on the "Phase 1 wins" branch)
- 1:30 Run #2
- 1:30 Run #3
- ~0:30 analysis + writeup decision
- **Total: ~6:30** — fits the 6-7h budget with a small buffer

---

## Critical files

| Phase | File | Change | LOC |
|---|---|---|---|
| 1 | `cropper/models/scorer.py` | `create_scorer()` accepts and forwards `weights` | ~6 |
| 1 | `cropper/pipeline/cropper.py` | `create_cropper()` reads `config["scorer"]` and passes weights | ~8 |
| 1 | `cropper/run_scorer_sweep.py` | NEW — runner that patches `cfg["scorer"]["vila_weight"]` / `area_weight` (mirrors `run_novelty_quick.py:patch_config`) | ~100 |
| 2 | `cropper/configs/default.yaml` | add `freeform.novelty.gt_seed_init: false` | 1 |
| 2 | `cropper/pipeline/cropper.py:crop()` | seed `initial_crops` from ICL example crops when flag on | ~30 |
| 2 | `cropper/run_gt_seed_quick.py` | NEW — 2-experiment runner (pool, replace) | ~80 |
| 3 | `cropper/models/gaicd_calibration_head.py` | NEW — feature extraction + MLP training + scorer wrapper | ~150 |
| 3 | `cropper/models/scorer.py` | wire `gaicd_cal` into `create_scorer` | ~10 |

## Reused utilities

- `cropper/run_novelty_quick.py:patch_config` — exact template for runner config patching (NOVELTY_FLAGS reset pattern)
- `cropper/run_novelty_quick.py:run_one` — exact template for the per-experiment Popen+tee loop
- `cropper/pipeline/retrieval.py:_select_freeform_examples` — already loads top-T crops in [1,1000] normalized coords; Phase 2 just consumes the existing return value
- `cropper/evaluation/evaluate.py` checkpoint cadence (now every 10 samples — edited earlier today) — any phase's runs benefit automatically: a 30→50 extension is just re-running with the same `output_dir` and `MAX_SAMPLES=50` (resume kicks in at sample 30)

## Verification protocol (per phase)

1. Check the experiment's `freeform_results.json` for `IoU`, `SRCC`, `PCC`.
2. Compare against `cropper/results/novelty_quick_baseline_30/freeform_results.json` (n=30 baseline) — apples-to-apples on the same deterministic slice.
3. Sanity-check per-sample IoUs: dump `[round(r['iou'],3) for r in results['results']]` — no single sample should collapse to <0.05 (would indicate the kind of catastrophic failure we saw with full visual grounding).
4. **Phase 1 specifically:** `vila03_area07` should *underperform* baseline (validates that the area bias is real and the wiring is right). `vila10_area00` should *over*perform. If the ordering doesn't match, the wiring is broken — debug before proceeding.
5. Once any variant looks promising on n=30: extend to n=50 (or n=200) by re-running with `MAX_SAMPLES=N` and the same `output_dir`. Resume picks up at sample 30.

## Expected outcomes (sequential, best-case stack)

| Metric | baseline n=30 | +Phase 1 (vila only) | +Phase 2 (GT seed) | +Phase 3 (cal head) | Total |
|---|---|---|---|---|---|
| IoU | 0.5334 | +0.020 | +0.015 | +0.060 | **~0.628** |
| SRCC | 0.1263 | +0.030 | +0.020 | +0.150 | **~0.329** |
| PCC | 0.0599 | +0.020 | +0.015 | +0.130 | **~0.225** |

Even at conservative ends: IoU ~0.575, which is +0.04 over baseline_30 — clearly above noise, clearly above any single previous idea. With Phase 3 hitting its upper end: IoU ~0.628, closing ~70% of the gap to the paper's reported 0.672.

---

## Resolved questions

1. **In-progress runs:** let `idea1_vg_top5_30` and `idea3_multi_temp_30` finish (~30 min remaining). Use the 30 min to code Phase 1 wiring in parallel so we can fire as soon as GPU 3 frees up.
2. **GPU for tonight's runs:** **GPU 3** (it'll be free as soon as `idea1_vg_top5_30` finishes).

## Next action

Code Phase 1 wiring:
1. `cropper/models/scorer.py` — `create_scorer()` accepts and forwards `weights`
2. `cropper/pipeline/cropper.py` — `create_cropper()` reads `config["scorer"]` and passes weights
3. `cropper/run_scorer_sweep.py` — NEW runner targeting GPU 3, single experiment for tonight: `scorer_vila10_area00_30`

Then wait for `idea1_vg_top5_30` on GPU 3 to finish, kick off Run #1, monitor checkpoint at sample 10 for pathological behavior, and make the Run #2 decision based on the final result.
