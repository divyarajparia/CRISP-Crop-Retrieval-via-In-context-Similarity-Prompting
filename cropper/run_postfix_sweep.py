#!/usr/bin/env python
"""
Post-fix IoU maximization sweep — up to 7 runs across 3 GPUs.

Context:
    Two coordinate-space bugs in the Cropper replication pipeline were fixed
    earlier today (data/datasets.py row-major transpose + prompt_builder
    pixel-vs-1000 normalization). Every prior GAICD result is invalid until
    `post_fix_baseline_30` (slot 1) completes and gives us a new ground
    truth to compare against.

    See plan: /data1/es22btech11013/.claude/plans/parallel-toasting-raccoon.md

Tiers:
    Tier 1 (slot 1): baseline. Must finish before anything else is trusted.
    Tier 2 (slots 2+3, first two runs each): high-confidence individual
            levers (multi_temperature, visual_crop_grounding, diverse_icl,
            bigger_R). All 4 must fit in budget.
    Tier 3 (slots 2+3, third run each): budget-permitting additional knobs
            (rank_anchored_refinement, bigger_L). May overrun the 9h window;
            kill manually if GPU must be freed.

Usage (in three tmux windows, one per GPU):
    cd /data1/es22btech11013/divya/AFCIL/divya/cv-project/cropper

    # Slot 1 (baseline) — expected 3h
    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        run_postfix_sweep.py --slot 1 --cuda-device 0

    # Slot 2 (multitemp -> grounding -> rank_anchored) — expected 6-9h
    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        run_postfix_sweep.py --slot 2 --cuda-device 1

    # Slot 3 (diverse_icl -> bigR -> bigL) — expected 6-9h
    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        run_postfix_sweep.py --slot 3 --cuda-device 2

The --slot arg selects which experiment list to run. The --cuda-device arg
sets CUDA_VISIBLE_DEVICES in the subprocess env so Mantis pins to the right
physical GPU regardless of whether the parent shell had the env var set.

Sanity gate:
    After slot 1 has processed ~3 samples (t ≈ 0:20), check
    results/post_fix_baseline_30/run.log. Look for image-dependent pred
    crops and multi-value Scores: lines. If broken → kill everything
    before slots 2/3 burn GPU time on a still-broken pipeline.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

# ============================================================================
# CONFIG
# ============================================================================

MAX_SAMPLES = 30
DATA_DIR    = "data"
BASE_CONFIG = "configs/default.yaml"

# Pin to cv_project env so VILA-R loads cleanly (lingvo import). Base
# anaconda python silently falls back to the LAION predictor and changes
# the scorer behavior between runs — don't rely on it.
CV_PROJECT_PYTHON = "/data1/es22btech11013/anaconda3/envs/cv_project/bin/python"

# All four freeform.novelty.* flags are force-reset to False before each
# run so every experiment starts from the paper-default baseline
# (S=30, T=5, R=6, L=2, scorer=vila+area) and only the explicit patches
# in the experiment dict differ.
NOVELTY_FLAGS = (
    "visual_crop_grounding",
    "rank_anchored_refinement",
    "multi_temperature",
    "diverse_icl",
)

EXPERIMENTS: Dict[int, List[Dict[str, Any]]] = {
    # ---- Slot 1: Tier 1 baseline (GPU 1, 3h window) ----
    1: [
        dict(
            name="post_fix_baseline_30",
            output_dir="results/post_fix_baseline_30",
            patches={},  # no overrides — pure paper-default
        ),
    ],
    # ---- Slot 2: Tier 2 × 2 + Tier 3 × 1 (GPU 2, 9h window) ----
    2: [
        dict(
            name="post_fix_multitemp_30",
            output_dir="results/post_fix_multitemp_30",
            patches={"freeform.novelty.multi_temperature": True},
        ),
        dict(
            name="post_fix_grounding_30",
            output_dir="results/post_fix_grounding_30",
            patches={"freeform.novelty.visual_crop_grounding": True},
        ),
        # Tier 3 — aspirational; kill manually if slot is behind schedule.
        dict(
            name="post_fix_rank_anchored_30",
            output_dir="results/post_fix_rank_anchored_30",
            patches={"freeform.novelty.rank_anchored_refinement": True},
        ),
    ],
    # ---- Slot 3: Tier 2 × 2 + Tier 3 × 1 (GPU 3, 9h window) ----
    3: [
        dict(
            name="post_fix_diverse_icl_30",
            output_dir="results/post_fix_diverse_icl_30",
            patches={"freeform.novelty.diverse_icl": True},
        ),
        dict(
            name="post_fix_bigR_30",
            output_dir="results/post_fix_bigR_30",
            patches={"freeform.R": 10},
        ),
        # Tier 3 — L=4 ~doubles refinement iterations, expected ~3h30m per
        # run. Will very likely overrun the 9h window. Kill if needed.
        dict(
            name="post_fix_bigL_30",
            output_dir="results/post_fix_bigL_30",
            patches={"freeform.L": 4},
        ),
    ],
}

# ============================================================================
# PATCHING
# ============================================================================


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set cfg[a][b][c] = value from a dotted key 'a.b.c'."""
    parts = dotted_key.split(".")
    cur = cfg
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value


def patch_config(patches: Dict[str, Any]) -> str:
    """Load default.yaml, reset novelty flags, apply patches, write temp yaml.

    Returns the temp yaml path (caller unlinks it after the run).
    """
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}

    ff = cfg.setdefault("freeform", {})
    novelty = ff.setdefault("novelty", {})
    for k in NOVELTY_FLAGS:
        novelty[k] = False

    for dotted_key, value in patches.items():
        _set_dotted(cfg, dotted_key, value)

    tmp = f"/tmp/postfix_sweep_{os.getpid()}_{int(time.time() * 1000)}.yaml"
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)
    return tmp


# ============================================================================
# RUNNER
# ============================================================================


def run_one(exp: Dict[str, Any], cuda_device: int, slot: int) -> int:
    print("\n" + "=" * 70)
    print(f"  SLOT           : {slot}")
    print(f"  EXPERIMENT     : {exp['name']}")
    print(f"  Patches        : {exp['patches']}")
    print(f"  Output         : {exp['output_dir']}")
    print(f"  Samples        : {MAX_SAMPLES}")
    print(f"  CUDA device    : {cuda_device}")
    print("=" * 70 + "\n")

    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    tmp_cfg = patch_config(exp["patches"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]           = str(cuda_device)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
    env["TF_FORCE_GPU_ALLOW_GROWTH"]      = "true"
    env["PYTHONUNBUFFERED"]               = "1"

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
    rc = -1
    try:
        with open(log_path, "w", buffering=1) as log_f:
            header = (
                f"# {exp['name']}\n"
                f"# slot:     {slot}\n"
                f"# patches:  {exp['patches']}\n"
                f"# samples:  {MAX_SAMPLES}\n"
                f"# cuda:     {cuda_device}\n"
                f"# cmd:      {' '.join(cmd)}\n"
                f"# started:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + "=" * 70 + "\n"
            )
            log_f.write(header)
            sys.stdout.write(header)
            sys.stdout.flush()

            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
            proc.wait()
            rc = proc.returncode

            if rc != 0:
                msg = (
                    f"\n!! {exp['name']} FAILED with exit code {rc} "
                    f"— continuing to next experiment in slot\n"
                )
                sys.stdout.write(msg)
                log_f.write(msg)
    except KeyboardInterrupt:
        print(f"!! {exp['name']} interrupted — re-raising to abort slot")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    finally:
        Path(tmp_cfg).unlink(missing_ok=True)
    return rc


def main():
    parser = argparse.ArgumentParser(
        description="Post-fix IoU sweep runner (per-slot experiment list).",
    )
    parser.add_argument(
        "--slot", type=int, required=True, choices=[1, 2, 3],
        help="Experiment slot: 1=baseline, 2=slot2 list, 3=slot3 list",
    )
    parser.add_argument(
        "--cuda-device", type=int, required=True,
        help="Physical CUDA device index passed to subprocesses via "
             "CUDA_VISIBLE_DEVICES",
    )
    args = parser.parse_args()

    experiments = EXPERIMENTS[args.slot]
    print(
        f"\n>>> Slot {args.slot}: {len(experiments)} experiment(s) queued "
        f"on CUDA device {args.cuda_device}"
    )
    for i, exp in enumerate(experiments, 1):
        print(f"    [{i}] {exp['name']}  patches={exp['patches']}")
    print()

    t0 = time.time()
    for i, exp in enumerate(experiments, 1):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"\n>>> [slot {args.slot}, run {i}/{len(experiments)}] "
            f"starting {exp['name']} at {ts}"
        )
        run_one(exp, cuda_device=args.cuda_device, slot=args.slot)

    elapsed_h = (time.time() - t0) / 3600
    print(
        f"\n>>> Slot {args.slot} complete — {len(experiments)} runs in "
        f"{elapsed_h:.2f} hours"
    )


if __name__ == "__main__":
    main()
