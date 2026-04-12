#!/usr/bin/env python
"""
Final experiment sweep — two runs on two GPUs.

Slot 1 (GPU 2): Calhead rebuild + eval
    Step A: build_calhead_features.py (~30 min)
    Step B: train_calhead.py (~2 min, CPU)
    Step C: evaluate with scorer="vila+area+gaicd_cal" (~3h)

Slot 2 (GPU 3): Stacked positive ideas
    VILA-only scorer + R=10 + diverse_icl + multi_temperature

Usage:
    cd /data1/es22btech11013/divya/AFCIL/divya/cv-project/cropper

    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        run_final_sweep.py --slot 1 --cuda-device 2

    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        run_final_sweep.py --slot 2 --cuda-device 3
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

CV_PROJECT_PYTHON = "/data1/es22btech11013/anaconda3/envs/cv_project/bin/python"

NOVELTY_FLAGS = (
    "visual_crop_grounding",
    "rank_anchored_refinement",
    "multi_temperature",
    "diverse_icl",
    "anti_bias_prompt",
    "final_iter_selection",
)


# ============================================================================
# PATCHING
# ============================================================================


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value


def patch_config(patches: Dict[str, Any]) -> str:
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    ff = cfg.setdefault("freeform", {})
    novelty = ff.setdefault("novelty", {})
    for k in NOVELTY_FLAGS:
        novelty[k] = False
    for dotted_key, value in patches.items():
        _set_dotted(cfg, dotted_key, value)
    tmp = f"/tmp/final_sweep_{os.getpid()}_{int(time.time() * 1000)}.yaml"
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)
    return tmp


# ============================================================================
# RUNNER HELPERS
# ============================================================================


def run_subprocess(cmd, env, log_path, label):
    """Run a subprocess, tee stdout to log file, return exit code."""
    print(f"\n>>> [{label}] Running: {' '.join(cmd)}")
    proc = None
    rc = -1
    try:
        with open(log_path, "a", buffering=1) as log_f:
            header = (
                f"\n# {label}\n"
                f"# cmd: {' '.join(cmd)}\n"
                f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
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
                msg = f"\n!! {label} FAILED with exit code {rc}\n"
                sys.stdout.write(msg)
                log_f.write(msg)
    except KeyboardInterrupt:
        print(f"!! {label} interrupted")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    return rc


def run_eval(exp_name, output_dir, patches, cuda_device):
    """Run evaluation/evaluate.py with patched config."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    tmp_cfg = patch_config(patches)

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
        "--output_dir",  output_dir,
        "--device",      "cuda",
        "--task",        "freeform",
        "--max_samples", str(MAX_SAMPLES),
    ]

    try:
        rc = run_subprocess(cmd, env, str(log_path), exp_name)
    finally:
        Path(tmp_cfg).unlink(missing_ok=True)
    return rc


# ============================================================================
# SLOT 1: Calhead pipeline
# ============================================================================


def run_calhead_pipeline(cuda_device):
    """Build features -> train head -> evaluate."""
    out_dir = Path("results/final_calhead_30")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / "run.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]           = str(cuda_device)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
    env["TF_FORCE_GPU_ALLOW_GROWTH"]      = "true"
    env["PYTHONUNBUFFERED"]               = "1"

    # Step A: Build features (~30 min)
    print("\n" + "=" * 70)
    print("  STEP A: Rebuilding GAICD calhead features (correct coordinates)")
    print("=" * 70)
    rc = run_subprocess(
        [CV_PROJECT_PYTHON, "-u", "build_calhead_features.py"],
        env, log_path, "build_features",
    )
    if rc != 0:
        print("!! Feature build failed — aborting calhead pipeline")
        return rc

    # Step B: Train head (~2 min, CPU-only)
    print("\n" + "=" * 70)
    print("  STEP B: Training calibration head")
    print("=" * 70)
    rc = run_subprocess(
        [CV_PROJECT_PYTHON, "-u", "train_calhead.py"],
        env, log_path, "train_calhead",
    )
    if rc != 0:
        print("!! Calhead training failed — aborting")
        return rc

    # Step C: Evaluate with calhead scorer (~3h)
    print("\n" + "=" * 70)
    print("  STEP C: Evaluating with vila+area+gaicd_cal scorer")
    print("=" * 70)
    rc = run_eval(
        exp_name="final_calhead_30",
        output_dir="results/final_calhead_30",
        patches={
            "freeform.scorer": "vila+area+gaicd_cal",
            "scorer.gaicd_cal_weight": 1.0,
            "scorer.gaicd_cal_head_path": "cache/gaicd_cal_head.pkl",
        },
        cuda_device=cuda_device,
    )
    return rc


# ============================================================================
# SLOT 2: Stacked positive ideas
# ============================================================================


def run_stacked(cuda_device):
    """VILA-only + R=10 + diverse_icl + multi_temperature."""
    print("\n" + "=" * 70)
    print("  STACKED: VILA-only + R=10 + diverse_icl + multi_temp")
    print("=" * 70)
    return run_eval(
        exp_name="final_stacked_30",
        output_dir="results/final_stacked_30",
        patches={
            "freeform.scorer": "vila",
            "freeform.R": 10,
            "freeform.novelty.diverse_icl": True,
            "freeform.novelty.multi_temperature": True,
            "freeform.temperatures": [0.05, 0.8],
        },
        cuda_device=cuda_device,
    )


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Final experiment sweep.",
    )
    parser.add_argument(
        "--slot", type=int, required=True, choices=[1, 2],
        help="1=calhead pipeline, 2=stacked positive ideas",
    )
    parser.add_argument(
        "--cuda-device", type=int, required=True,
        help="Physical CUDA device index",
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.slot == 1:
        run_calhead_pipeline(args.cuda_device)
    else:
        run_stacked(args.cuda_device)

    elapsed_h = (time.time() - t0) / 3600
    print(f"\n>>> Slot {args.slot} complete in {elapsed_h:.2f} hours")


if __name__ == "__main__":
    main()
