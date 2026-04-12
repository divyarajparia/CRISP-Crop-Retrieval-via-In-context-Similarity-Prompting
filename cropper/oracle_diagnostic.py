#!/usr/bin/env python
"""
Oracle diagnostic for the novelty_quick_baseline_30 run (CPU-only, ~1 min).

WHAT THIS ANSWERS
-----------------
The baseline_30 run logged IoU = 0.533 against the *single* top-MOS GT
crop per image. But GAICD annotates ~90 GT crops per image, each with
its own MOS. Two different questions:

    Q1 (what baseline measured)   : "how close is our pred to THE top-MOS GT?"
    Q2 (what oracle-over-GT asks) : "how close is our pred to ANY GT crop?"

The gap between Q1 and Q2 tells us what calhead (and Phase 3 in general)
can realistically do with nothing but a better scorer:

    gap small  (Q2 - Q1 < 0.05) : scorer already picks near its ceiling
                                  in the GT set. No room for re-ranking.
                                  Tomorrow: don't tune scoring, tune
                                  candidates (multi-temp, bigger R).

    gap medium (0.05 - 0.15)    : scorer lands in the GT cluster but
                                  not the top-MOS one. Calhead has
                                  moderate headroom by nudging toward
                                  high-MOS regions.

    gap large  (> 0.15)         : scorer picks crops that match LOW-MOS
                                  GTs. Calhead has huge headroom — a
                                  ranking head trained on MOS should
                                  directly fix this.

NOTES / LIMITATIONS
-------------------
This is NOT the full oracle over Mantis candidates. That would require
`pred_crops_all` (all R*L candidates across refinement iterations) to
be stored in the results JSON, which `evaluate.py` currently does NOT
dump. If tonight's 3 calhead runs come back flat, we'll add that dump
tomorrow and re-run baseline_30 to get the true candidate ceiling.

The diagnostic here is the best free signal we can extract from the
existing run in ~1 minute of CPU time.

USAGE
-----
    cd /data1/es22btech11013/divya/AFCIL/divya/cv-project/cropper
    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python oracle_diagnostic.py

OUTPUT
------
Prints a per-sample table and an aggregate block. Writes the same data
to results/novelty_quick_baseline_30/oracle_diagnostic.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median, stdev

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.datasets import GAICDDataset  # noqa: E402
from evaluation.metrics import compute_iou  # noqa: E402


RESULTS_PATH = SCRIPT_DIR / "results" / "novelty_quick_baseline_30" / "freeform_results.json"
OUT_PATH     = SCRIPT_DIR / "results" / "novelty_quick_baseline_30" / "oracle_diagnostic.json"
DATA_ROOT    = SCRIPT_DIR / "data" / "GAICD"


def _box(crop):
    """Extract (x1, y1, x2, y2) from either 4-tuple or (mos, x1, y1, x2, y2)."""
    if len(crop) == 5:
        return tuple(crop[1:])
    return tuple(crop)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Expected {RESULTS_PATH}")

    print(f"Loading {RESULTS_PATH}")
    with open(RESULTS_PATH) as f:
        payload = json.load(f)
    results = payload["results"]
    print(f"  {len(results)} samples")

    print(f"Loading GAICD test split from {DATA_ROOT}")
    dataset = GAICDDataset(root_dir=str(DATA_ROOT), split="test", cache_embeddings=False)
    print(f"  {len(dataset)} images in test split")

    # Index: image_id -> list of (mos, x1, y1, x2, y2)
    print("Building image_id -> GT crops index...")
    index: dict[str, list] = {}
    index_sizes: dict[str, tuple[int, int]] = {}
    for i in range(len(dataset)):
        s = dataset[i]
        index[str(s["image_id"])] = s["crops"]
        index_sizes[str(s["image_id"])] = s["image"].size  # (W, H)

    print(f"  {len(index)} images indexed")

    rows = []
    missing = 0
    for sample in results:
        img_id = str(sample["image_id"])
        if img_id not in index:
            missing += 1
            continue
        pred_box = _box(sample["pred_crop"])
        stored_iou = float(sample["iou"])
        gt_crops = index[img_id]

        # Per-GT IoU distribution
        ious = [compute_iou(pred_box, _box(c)) for c in gt_crops]
        gt_mos = [float(c[0]) for c in gt_crops]
        best_idx = max(range(len(ious)), key=lambda i: ious[i])
        oracle_iou = ious[best_idx]
        matched_mos = gt_mos[best_idx]

        # Top-MOS GT reference
        top_mos = max(gt_mos)

        rows.append({
            "image_id": img_id,
            "stored_iou": stored_iou,
            "oracle_iou_vs_any_gt": oracle_iou,
            "gap": oracle_iou - stored_iou,
            "matched_gt_mos": matched_mos,
            "top_mos_in_image": top_mos,
            "matched_mos_rank": sum(1 for m in gt_mos if m > matched_mos) + 1,  # 1 = best
            "n_gt_crops": len(gt_crops),
        })

    if missing:
        print(f"  WARNING: {missing} samples missing from test split index")

    # ----------------- Per-sample table -----------------
    print()
    print("=" * 96)
    print("  PER-SAMPLE ORACLE DIAGNOSTIC")
    print("=" * 96)
    print(f"  {'image_id':>10}  {'stored':>7}  {'oracle':>7}  {'gap':>6}  {'match_mos':>9}  {'top_mos':>7}  {'rank':>4}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*4}")
    for r in sorted(rows, key=lambda r: -r["gap"]):
        print(
            f"  {r['image_id']:>10}  {r['stored_iou']:>7.3f}  {r['oracle_iou_vs_any_gt']:>7.3f}  "
            f"{r['gap']:>6.3f}  {r['matched_gt_mos']:>9.3f}  {r['top_mos_in_image']:>7.3f}  "
            f"{r['matched_mos_rank']:>4}"
        )

    # ----------------- Aggregate -----------------
    def agg(key):
        vals = [r[key] for r in rows]
        return mean(vals), median(vals), (stdev(vals) if len(vals) > 1 else 0.0), min(vals), max(vals)

    m_stored, med_stored, sd_stored, min_stored, max_stored = agg("stored_iou")
    m_oracle, med_oracle, sd_oracle, min_oracle, max_oracle = agg("oracle_iou_vs_any_gt")
    m_gap,    med_gap,    sd_gap,    min_gap,    max_gap    = agg("gap")
    m_mos,    _,          _,         _,          _          = agg("matched_gt_mos")
    m_rank,   _,          _,         _,          _          = agg("matched_mos_rank")
    m_topmos, _,          _,         _,          _          = agg("top_mos_in_image")

    print()
    print("=" * 96)
    print("  AGGREGATE")
    print("=" * 96)
    print(f"  n samples                       : {len(rows)}")
    print(f"  mean stored IoU    (vs top-MOS) : {m_stored:.4f}   [median {med_stored:.4f}, std {sd_stored:.4f}, range {min_stored:.3f}..{max_stored:.3f}]")
    print(f"  mean oracle IoU    (vs any GT)  : {m_oracle:.4f}   [median {med_oracle:.4f}, std {sd_oracle:.4f}, range {min_oracle:.3f}..{max_oracle:.3f}]")
    print(f"  mean gap           (oracle-stored): {m_gap:+.4f}   [median {med_gap:+.4f}, std {sd_gap:.4f}, range {min_gap:+.3f}..{max_gap:+.3f}]")
    print(f"  mean matched-GT MOS             : {m_mos:.3f}   (higher is better; top_mos avg {m_topmos:.3f})")
    print(f"  mean matched-GT MOS rank        : {m_rank:.2f}   (1 = top-MOS GT; lower is better)")

    # ----------------- Interpretation -----------------
    print()
    print("=" * 96)
    print("  INTERPRETATION")
    print("=" * 96)
    if m_gap < 0.05:
        verdict = (
            "SMALL GAP -- the scorer already lands near its ceiling in the GT set.\n"
            "  Calhead has little to work with here. Flat Phase 3 results would NOT be\n"
            "  surprising. If tonight's runs come back flat, tomorrow's pivot should be\n"
            "  candidate generation, not scoring."
        )
    elif m_gap < 0.15:
        verdict = (
            "MEDIUM GAP -- scorer lands inside the GT cluster but not on the top-MOS\n"
            "  crop. Calhead has moderate headroom: a ranking head should nudge picks\n"
            "  toward higher-MOS crops. Expected Phase 3 delta: +0.01 to +0.04 IoU if\n"
            "  calhead learns meaningfully."
        )
    else:
        verdict = (
            "LARGE GAP -- scorer picks crops that match LOW-MOS GTs. Calhead has big\n"
            "  headroom; a ranking head trained directly on GAICD MOS should fix this.\n"
            "  Expected Phase 3 delta: +0.05+ IoU if the head's val pair accuracy > 0.65."
        )
    print(f"  {verdict}")

    # ----------------- Save -----------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "n_samples": len(rows),
                "mean_stored_iou": m_stored,
                "mean_oracle_iou": m_oracle,
                "mean_gap": m_gap,
                "mean_matched_mos": m_mos,
                "mean_matched_mos_rank": m_rank,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print()
    print(f"Saved per-sample details -> {OUT_PATH}")


if __name__ == "__main__":
    main()
