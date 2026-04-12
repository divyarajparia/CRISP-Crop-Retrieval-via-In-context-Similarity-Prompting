#!/usr/bin/env python
"""
Phase 2 IoU-maximization sweep — scorer & selection fixes.

Context:
    Post-fix sweep (run_postfix_sweep.py) revealed that the VLM
    produces near-full-image crops 90% of the time, barely beating
    a trivial constant prediction (+0.007 IoU). Root cause: the Area
    scorer rewards bigger crops, creating a self-reinforcing loop
    during refinement.

Experiments:
    Slot 1: vila_only_scorer  — drop Area scorer entirely
    Slot 2: final_iter_select — use final-iteration selection (paper default)
    Slot 3: anti_bias_prompt  — prompt instruction discouraging full-image crops

Usage (one per GPU, or sequential on one GPU):
    cd /data1/es22btech11013/divya/AFCIL/divya/cv-project/cropper

    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        run_phase2_sweep.py --slot 1 --cuda-device 0

    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        run_phase2_sweep.py --slot 2 --cuda-device 1

    /data1/es22btech11013/anaconda3/envs/cv_project/bin/python \
        run_phase2_sweep.py --slot 3 --cuda-device 2
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

# All novelty flags are force-reset to False before each run.
NOVELTY_FLAGS = (
    "visual_crop_grounding",
    "rank_anchored_refinement",
    "multi_temperature",
    "diverse_icl",
    "anti_bias_prompt",
    "final_iter_selection",
)

EXPERIMENTS: Dict[int, List[Dict[str, Any]]] = {
    # Slot 1: Drop Area scorer — VILA-only
    1: [
        dict(
            name="phase2_vila_only_30",
            output_dir="results/phase2_vila_only_30",
            patches={"freeform.scorer": "vila"},
        ),
    ],
    # Slot 2: Final-iteration selection (paper's recommended strategy)
    2: [
        dict(
            name="phase2_final_iter_30",
            output_dir="results/phase2_final_iter_30",
            patches={"freeform.novelty.final_iter_selection": True},
        ),
    ],
    # Slot 3: Anti-bias prompt instruction
    3: [
        dict(
            name="phase2_anti_bias_30",
            output_dir="results/phase2_anti_bias_30",
            patches={"freeform.novelty.anti_bias_prompt": True},
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
    """Load default.yaml, reset novelty flags, apply patches, write temp yaml."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}

    ff = cfg.setdefault("freeform", {})
    novelty = ff.setdefault("novelty", {})
    for k in NOVELTY_FLAGS:
        novelty[k] = False

    for dotted_key, value in patches.items():
        _set_dotted(cfg, dotted_key, value)

    tmp = f"/tmp/phase2_sweep_{os.getpid()}_{int(time.time() * 1000)}.yaml"
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
                    f"-- continuing to next experiment in slot\n"
                )
                sys.stdout.write(msg)
                log_f.write(msg)
    except KeyboardInterrupt:
        print(f"!! {exp['name']} interrupted")
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
        description="Phase 2 IoU sweep (scorer & selection fixes).",
    )
    parser.add_argument(
        "--slot", type=int, required=True, choices=[1, 2, 3],
        help="Experiment slot: 1=vila_only, 2=final_iter, 3=anti_bias",
    )
    parser.add_argument(
        "--cuda-device", type=int, required=True,
        help="Physical CUDA device index",
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
        f"\n>>> Slot {args.slot} complete -- {len(experiments)} runs in "
        f"{elapsed_h:.2f} hours"
    )


if __name__ == "__main__":
    main()
