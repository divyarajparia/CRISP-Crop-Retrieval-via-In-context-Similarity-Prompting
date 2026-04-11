#!/usr/bin/env python
"""
Phase 1 — Scorer weight sweep on the same deterministic 30-sample slice.

Why this exists (see plan_novelty_2.md "Phase 1"):

The released CombinedScorer for free-form cropping mixes a learned aesthetic
predictor (VILA) with a literal pixel-area term ({"vila": 1.0, "area": 1.0}
that normalizes to 0.5/0.5). The area term is just `crop_area / orig_area`,
so half the final score is "bigger crop wins" — structurally biased on GAICD
where ground-truth crops are aesthetic picks, not pixel-area picks. The
config keys `scorer.vila_weight` / `scorer.area_weight` already exist in
configs/default.yaml:52-55 but were never wired through to create_scorer.

This runner consumes the wiring added today (see
cropper/models/scorer.py:create_scorer and cropper/pipeline/cropper.py:
create_cropper) and patches `cfg["scorer"]["vila_weight"]` and
`cfg["scorer"]["area_weight"]` directly in a temp YAML the same way
run_novelty_quick.py:patch_config patches the novelty flags.

For tonight (1-GPU sequential budget) only the headline diagnostic variant
runs: vila=1.0 / area=0.0 (pure aesthetic). Milder variants are listed in
the EXPERIMENTS table commented out so they can be re-enabled after the
decision checkpoint.

Usage:
    python run_scorer_sweep.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ============================================================================
# CONFIG
# ============================================================================

MAX_SAMPLES = 30
GPU         = 1          # GPU 1 is free; GPU 3 has another job (results/tp)
DATA_DIR    = "data"
SCORER      = "vila+area"
S, T, R, L  = 30, 5, 6, 2
TEMPERATURE = 0.05

# Pin the cv_project conda env's python explicitly. The VILA-R loader needs
# `lingvo` (and friends), which only live in cv_project. Using sys.executable
# would inherit whatever python launched this script, and a base-anaconda
# launch silently falls back to "VILA dependencies not available" → the run
# crashes at startup (already burned us once today).
CV_PROJECT_PYTHON = "/data1/es22btech11013/anaconda3/envs/cv_project/bin/python"

# Phase 1 sweep variants. Only the headline diagnostic runs tonight; the
# milder + sanity-check variants are listed for documentation and can be
# re-enabled after Run #1's decision checkpoint.
EXPERIMENTS = [
    dict(
        name           = "scorer_vila10_area00_30",
        output_dir     = "results/scorer_vila10_area00_30",
        scorer_weights = dict(vila_weight=1.0, area_weight=0.0),
    ),
    # dict(
    #     name           = "scorer_vila07_area03_30",
    #     output_dir     = "results/scorer_vila07_area03_30",
    #     scorer_weights = dict(vila_weight=0.7, area_weight=0.3),
    # ),
    # dict(
    #     name           = "scorer_vila03_area07_30",
    #     output_dir     = "results/scorer_vila03_area07_30",
    #     scorer_weights = dict(vila_weight=0.3, area_weight=0.7),
    # ),
]

NOVELTY_FLAGS = (
    "visual_crop_grounding",
    "rank_anchored_refinement",
    "multi_temperature",
    "diverse_icl",
)

# ============================================================================
# RUNNER
# ============================================================================


def patch_config(scorer_weights: dict, base_yaml: str = "configs/default.yaml") -> str:
    """Write a temp YAML that hardwires S/T/R/L/scorer, force-resets every
    novelty flag to False (so this is the pure-baseline candidate pool, only
    the scoring rule changes), and overwrites cfg["scorer"]["vila_weight"]
    and cfg["scorer"]["area_weight"] with the values for this experiment."""
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f) or {}

    ff = cfg.setdefault("freeform", {})
    ff["S"], ff["T"], ff["R"], ff["L"] = S, T, R, L
    ff["temperature"] = TEMPERATURE
    ff["scorer"]      = SCORER

    novelty = ff.setdefault("novelty", {})
    for k in NOVELTY_FLAGS:
        novelty[k] = False

    sc = cfg.setdefault("scorer", {})
    sc.update(scorer_weights)

    tmp = f"/tmp/scorer_sweep_{os.getpid()}_{int(time.time() * 1000)}.yaml"
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)
    return tmp


def run_one(exp: dict):
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT     : {exp['name']}")
    print(f"  Scorer weights : {exp['scorer_weights']}")
    print(f"  Output         : {exp['output_dir']}")
    print(f"  Samples        : {MAX_SAMPLES}")
    print(f"  GPU            : {GPU}")
    print("=" * 70 + "\n")

    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    tmp_cfg = patch_config(exp["scorer_weights"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]            = str(GPU)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"]   = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"]  = "0.15"
    env["TF_FORCE_GPU_ALLOW_GROWTH"]       = "true"
    env["PYTHONUNBUFFERED"]                = "1"

    cmd = [
        CV_PROJECT_PYTHON, "-u", "evaluation/evaluate.py",
        "--config",      tmp_cfg,
        "--data_dir",    DATA_DIR,
        "--output_dir",  exp["output_dir"],
        "--device",      "cuda",
        "--task",        "freeform",
        "--max_samples", str(MAX_SAMPLES),
    ]

    proc = None
    try:
        with open(log_path, "w", buffering=1) as log_f:
            header = (
                f"# {exp['name']}\n"
                f"# scorer_weights: {exp['scorer_weights']}\n"
                f"# samples:        {MAX_SAMPLES}\n"
                f"# gpu:            {GPU}\n"
                f"# cmd:            {' '.join(cmd)}\n"
                f"# started:        {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + "=" * 70 + "\n"
            )
            log_f.write(header)
            sys.stdout.write(header)
            sys.stdout.flush()

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
            proc.wait()

            if proc.returncode != 0:
                msg = (
                    f"\n!! {exp['name']} FAILED with exit code {proc.returncode} "
                    f"— continuing to next experiment\n"
                )
                sys.stdout.write(msg)
                log_f.write(msg)
    except KeyboardInterrupt:
        print(f"!! {exp['name']} interrupted — re-raising to abort sweep")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
        raise
    finally:
        Path(tmp_cfg).unlink(missing_ok=True)


def main():
    t0 = time.time()
    for i, exp in enumerate(EXPERIMENTS, 1):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n>>> [{i}/{len(EXPERIMENTS)}] starting {exp['name']} at {ts}")
        run_one(exp)
    elapsed_h = (time.time() - t0) / 3600
    print(f"\nAll experiments complete in {elapsed_h:.2f} hours")


if __name__ == "__main__":
    main()
