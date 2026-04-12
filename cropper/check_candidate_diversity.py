#!/usr/bin/env python
"""
Parse a run.log to check candidate diversity per sample.

WHAT IT CHECKS
--------------
For every "Scores: ['0.xxx', '0.xxx', ...]" line in the log, count
how many UNIQUE score values appear. If the count is 1, all candidates
got the same score — meaning either (a) all crops are duplicates, or
(b) the scorer is numerically collapsing to the same value.

Either way, if unique_count == 1 for most samples, there is nothing
for any re-ranking improvement (calhead included) to pick from. The
bottleneck would be candidate generation, not scoring.

INTERPRETATION
--------------
| mean unique scores per "Scores:" line | meaning                                         |
| 1 - 1.5                               | CATASTROPHIC — calhead cannot help at all       |
| 1.5 - 3                               | MILD — calhead has weak signal to re-rank       |
| 5 - 10                                | GOOD — real diversity, calhead can move things  |

USAGE
-----
    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        check_candidate_diversity.py results/novelty_quick_baseline_30/run.log
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

SCORES_RE = re.compile(r"Scores:\s*\[([^\]]+)\]")


def parse_scores_line(content_inside_brackets: str) -> list[float]:
    """Parse "'0.461', '0.461', ..." into a list of floats."""
    parts = [p.strip().strip("'\"") for p in content_inside_brackets.split(",")]
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def main():
    if len(sys.argv) < 2:
        log_path = Path("results/novelty_quick_baseline_30/run.log")
    else:
        log_path = Path(sys.argv[1])
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    print(f"Parsing {log_path}")
    text = log_path.read_text()

    rows = []
    for m in SCORES_RE.finditer(text):
        scores = parse_scores_line(m.group(1))
        if not scores:
            continue
        n = len(scores)
        uniq = len(set(round(s, 4) for s in scores))  # 4-decimal rounding
        rows.append((n, uniq, scores))

    if not rows:
        print("No 'Scores:' lines found in log — nothing to analyze.")
        return

    print(f"Found {len(rows)} 'Scores:' lines")
    print()
    print("=" * 88)
    print(f"  {'#':>4}  {'n_cands':>7}  {'n_uniq':>6}  {'min':>6}  {'max':>6}  {'spread':>6}  scores (head)")
    print(f"  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*30}")
    for i, (n, uniq, scores) in enumerate(rows, 1):
        smin, smax = min(scores), max(scores)
        head = ", ".join(f"{s:.3f}" for s in scores[:5])
        if len(scores) > 5:
            head += ", ..."
        print(f"  {i:>4}  {n:>7}  {uniq:>6}  {smin:>6.3f}  {smax:>6.3f}  {smax-smin:>6.3f}  {head}")

    uniq_counts = [r[1] for r in rows]
    cand_counts = [r[0] for r in rows]
    spreads    = [max(r[2]) - min(r[2]) for r in rows]

    print()
    print("=" * 88)
    print("  AGGREGATE")
    print("=" * 88)
    print(f"  total 'Scores:' lines     : {len(rows)}")
    print(f"  mean candidates per line  : {mean(cand_counts):.2f}")
    print(f"  mean unique per line      : {mean(uniq_counts):.2f}  (median {median(uniq_counts):.1f})")
    print(f"  mean max-min spread       : {mean(spreads):.4f}  (median {median(spreads):.4f})")
    print()
    hist = Counter(uniq_counts)
    print("  distribution of unique counts:")
    for k in sorted(hist.keys()):
        bar = "#" * hist[k]
        print(f"    {k:>3} unique : {hist[k]:>4}  {bar}")

    n1 = hist.get(1, 0)
    frac1 = n1 / len(rows)
    print()
    print("=" * 88)
    print("  VERDICT")
    print("=" * 88)
    print(f"  fraction of lines with ONLY 1 unique score : {frac1:.1%}  ({n1}/{len(rows)})")
    if frac1 > 0.8:
        verdict = (
            "CATASTROPHIC. >80% of 'Scores:' lines have all-identical scores.\n"
            "  No re-ranking can help here. Calhead is ~useless until candidate\n"
            "  diversity is fixed. Tomorrow's pivot: candidate generation\n"
            "  (multi-temp with bigger K, bigger R, prompt engineering)."
        )
    elif frac1 > 0.4:
        verdict = (
            "MIXED. Roughly half the samples have zero diversity. Calhead will\n"
            "  only move the half with real spread. Expected Phase 3 effect is\n"
            "  diluted proportionally."
        )
    else:
        verdict = (
            "HEALTHY. Most samples have meaningful score spread. Calhead can\n"
            "  re-rank candidates as intended. Phase 3 is worth running."
        )
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
