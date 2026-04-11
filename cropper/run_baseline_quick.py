#!/usr/bin/env python
"""
Quick 30-sample GPU-1 runner for [idea3_multi_temp_30, baseline_30].

Companion to run_novelty_quick.py (which runs Idea 4 + corrected Idea 1 on
GPU 3). This script runs Idea 3 (multi-temperature) and the fresh baseline on
GPU 1, sequentially, on the same deterministic 30 GAICD samples so all four
IoU numbers (baseline, idea1-corrected, idea3, idea4) are directly comparable.

Idea 3 runs first so that if anything goes sideways late, we still have the
baseline number from the second slot.

Usage:
    python run_baseline_quick.py
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
GPU         = 1          # GPU 1, after the user kills the existing main sweep
DATA_DIR    = "data"
SCORER      = "vila+area"
S, T, R, L  = 30, 5, 6, 2
TEMPERATURE = 0.05

EXPERIMENTS = [
    dict(
        name        = "baseline_30",
        output_dir  = "results/novelty_quick_baseline_30",
        novelty     = dict(),   # all flags False = baseline
    ),
    dict(
        name        = "idea3_multi_temp_30",
        output_dir  = "results/novelty_quick_idea3_multi_temp_30",
        novelty     = dict(multi_temperature=True),
    ),
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

def patch_config(novelty_overrides: dict, base_yaml: str = "configs/default.yaml") -> str:
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f) or {}
    ff = cfg.setdefault("freeform", {})
    ff["S"], ff["T"], ff["R"], ff["L"] = S, T, R, L
    ff["temperature"] = TEMPERATURE
    ff["scorer"]      = SCORER
    novelty = ff.setdefault("novelty", {})
    for k in NOVELTY_FLAGS:
        novelty[k] = False
    novelty.update(novelty_overrides)
    tmp = f"/tmp/baseline_quick_{os.getpid()}_{int(time.time() * 1000)}.yaml"
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)
    return tmp


def run_one(exp: dict):
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {exp['name']}")
    print(f"  Novelty   : {exp['novelty']}")
    print(f"  Output    : {exp['output_dir']}")
    print(f"  Samples   : {MAX_SAMPLES}")
    print(f"  GPU       : {GPU}")
    print("=" * 70 + "\n")

    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    tmp_cfg = patch_config(exp["novelty"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]            = str(GPU)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"]   = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"]  = "0.15"
    env["TF_FORCE_GPU_ALLOW_GROWTH"]       = "true"
    env["PYTHONUNBUFFERED"]                = "1"

    cmd = [
        sys.executable, "-u", "evaluation/evaluate.py",
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
                f"# novelty: {exp['novelty']}\n"
                f"# samples: {MAX_SAMPLES}\n"
                f"# gpu:     {GPU}\n"
                f"# cmd:     {' '.join(cmd)}\n"
                f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
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
