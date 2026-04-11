#!/usr/bin/env python
"""
Quick 30-sample GPU-1 validation sweep.

Runs two experiments on the same deterministic 30 GAICD samples:
  1. idea4_diverse_icl_30  — Idea 4: KMeans-diverse ICL downselection
  2. idea1_vg_top5_30      — corrected Idea 1: visual grounding for top-5 ICL examples only

Baseline for these 30 samples runs separately on GPU 0 via run_baseline_quick.py
after Idea 3 finishes on the main sweep.

Each experiment writes to its own results subdir with a tee'd run.log.
Failures are isolated (try/except CalledProcessError).

Usage:
    python run_novelty_quick.py
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
GPU         = 3
DATA_DIR    = "data"
SCORER      = "vila+area"
S, T, R, L  = 30, 5, 6, 2
TEMPERATURE = 0.05

EXPERIMENTS = [
    dict(
        name        = "idea4_diverse_icl_30",
        output_dir  = "results/novelty_quick_idea4_diverse_icl_30",
        novelty     = dict(diverse_icl=True),
    ),
    dict(
        name        = "idea1_vg_top5_30",
        output_dir  = "results/novelty_quick_idea1_vg_top5_30",
        novelty     = dict(visual_crop_grounding=True),   # top_k=5 via default.yaml
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
    """Write a temp YAML that hardwires S/T/R/L/scorer and flips exactly the
    novelty flags this experiment needs. All other flags are force-reset to
    False so a leftover setting from a previous patch can never leak through.
    visual_grounding_top_k is left at the default.yaml value (5)."""
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
    tmp = f"/tmp/novelty_quick_{os.getpid()}_{int(time.time() * 1000)}.yaml"
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

    # Create the output dir up-front so we can stream a per-experiment run.log
    # into it while the subprocess is still running.
    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    tmp_cfg = patch_config(exp["novelty"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]            = str(GPU)
    # JAX VILA-R shares the GPU with Mantis — cap preallocation so they coexist.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"]   = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"]  = "0.15"
    env["TF_FORCE_GPU_ALLOW_GROWTH"]       = "true"
    # Force unbuffered Python output so tee picks up lines in real time.
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
