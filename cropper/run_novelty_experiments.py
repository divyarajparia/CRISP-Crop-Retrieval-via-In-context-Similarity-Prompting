#!/usr/bin/env python
"""
Sequential runner for the 4 novelty ideas on the GAICD free-form benchmark.

Each experiment toggles exactly one flag in `freeform.novelty` and writes its
results to its own `results/` subdir. Failures are isolated: if one experiment
crashes (e.g. OOM), the runner logs it and moves on to the next one.

Usage:
    python run_novelty_experiments.py        # 30 samples, GPU 1 (parallel with baseline on GPU 0)
Edit MAX_SAMPLES / GPU below to change the sweep budget or GPU.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ============================================================================
# CONFIG — edit these
# ============================================================================

MAX_SAMPLES = 50          # 30 ≈ 12h sweep, 50 ≈ 20h sweep
GPU         = 1           # baseline is on GPU 0; sweep runs in parallel on GPU 1
DATA_DIR    = "data"
SCORER      = "vila+area"
S, T, R, L  = 30, 5, 6, 2
TEMPERATURE = 0.05

EXPERIMENTS = [
    # Order: easiest-first (Idea 2 → 1 → 3 → 4) per plan_novelty.md
    dict(
        name        = "idea2_rank_anchored",
        output_dir  = "results/novelty_idea2_rank_anchored",
        novelty     = dict(rank_anchored_refinement=True),
    ),
    dict(
        name        = "idea1_visual_grounding",
        output_dir  = "results/novelty_idea1_visual_grounding",
        novelty     = dict(visual_crop_grounding=True),
    ),
    dict(
        name        = "idea3_multi_temperature",
        output_dir  = "results/novelty_idea3_multi_temperature",
        novelty     = dict(multi_temperature=True),
    ),
    dict(
        name        = "idea4_diverse_icl",
        output_dir  = "results/novelty_idea4_diverse_icl",
        novelty     = dict(diverse_icl=True),
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
    False so a leftover setting from a previous patch can never leak through."""
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
    tmp = f"/tmp/novelty_exp_{os.getpid()}_{int(time.time() * 1000)}.yaml"
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
            # Write a small header into run.log so it's self-describing.
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
