#!/usr/bin/env python
"""
Experiment launcher for Cropper.
Edit the CONFIG block below and run:  python run_experiment.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ==============================================================================
# EXPERIMENT CONFIG — edit this block to change what runs
# ==============================================================================

EXP = dict(
    # --- What to run ---
    task         = "freeform",          # "freeform" | "subject_aware" | "aspect_ratio"
    max_samples  = 50,                  # None = full test set; int = quick run
    gpu          = 3,                   # GPU index to use (0, 1, 2, ...)

    # --- Data ---
    data_dir     = "data",              # paperish_v2 used "data" (not "data/GAICD")

    # --- ICL / generation hyperparams (override config yaml) ---
    S            = 30,                  # retrieved training images
    T            = 5,                   # GT crops per ICL example
    R            = 6,                   # VLM crop candidates per iteration
    L            = 2,                   # refinement iterations
    temperature  = 0.05,

    # --- Scorer ---
    scorer       = "vila+area",           # "vila+area" | "clip" | "laion+area"

    # --- Output ---
    # Baseline replication of paperish_v2 (target: IoU=0.527)
    output_dir   = "results/tp",

    # --- Misc ---
    seed         = 42,
    resume       = None,                # path to checkpoint.json to resume from
    debug        = False,
)

# ==============================================================================
# END OF CONFIG — nothing below needs editing for normal use
# ==============================================================================

SCRIPT_MAP = {
    # Use evaluation/evaluate.py for freeform so this matches the paperish_v2
    # baseline run exactly (require_exact_components=True, --data_dir data, VILA-R).
    "freeform":       "evaluation/evaluate.py",
    "subject_aware":  "scripts/run_subject_aware.py",
    "aspect_ratio":   "scripts/run_aspect_ratio.py",
}

def make_output_dir(cfg: dict) -> str:
    if cfg["output_dir"]:
        return cfg["output_dir"]
    ts = datetime.now().strftime("%m%d_%H%M")
    scorer_tag = cfg["scorer"].replace("+", "_")
    n = cfg["max_samples"] if cfg["max_samples"] else "full"
    name = f"{cfg['task']}_S{cfg['S']}_R{cfg['R']}_L{cfg['L']}__{scorer_tag}__n{n}__{ts}"
    return f"results/{name}"


def patch_config_yaml(cfg: dict, config_path: str) -> str:
    """Write a temp yaml with overridden S/T/R/L/scorer so the script picks it up."""
    import yaml
    with open(config_path) as f:
        base = yaml.safe_load(f) or {}

    task_key = cfg["task"]
    if task_key not in base:
        base[task_key] = {}

    base[task_key]["S"]           = cfg["S"]
    base[task_key]["T"]           = cfg["T"]
    base[task_key]["R"]           = cfg["R"]
    base[task_key]["L"]           = cfg["L"]
    base[task_key]["temperature"] = cfg["temperature"]
    base[task_key]["scorer"]      = cfg["scorer"]

    tmp_path = f"/tmp/cropper_exp_config_{os.getpid()}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(base, f)
    return tmp_path


def main():
    cfg = EXP
    script = SCRIPT_MAP.get(cfg["task"])
    if script is None:
        sys.exit(f"Unknown task: {cfg['task']}")

    output_dir = make_output_dir(cfg)
    print(f"\n{'='*60}")
    print(f"  Task      : {cfg['task']}")
    print(f"  GPU       : {cfg['gpu']}")
    print(f"  S={cfg['S']}  T={cfg['T']}  R={cfg['R']}  L={cfg['L']}  temp={cfg['temperature']}")
    print(f"  Scorer    : {cfg['scorer']}")
    print(f"  Samples   : {cfg['max_samples'] or 'full test set'}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*60}\n")

    # Patch a temp config yaml with overrides
    base_config = "configs/default.yaml"
    tmp_config = patch_config_yaml(cfg, base_config)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    # JAX VILA-R shares the GPU with Mantis-8B. By default JAX preallocates
    # ~75% of VRAM at startup (~29 GB on our 48 GB card), which leaves
    # Mantis without enough headroom for its multi-image forward pass.
    # Disable preallocation AND cap JAX at 15% of VRAM (~7 GB) — VILA-R's
    # actual working set is only a few GB. Both are required: PREALLOCATE
    # alone still lets JAX grow into Mantis's territory and trigger OOM.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    cmd = [
        sys.executable, script,
        "--config",    tmp_config,
        "--data_dir",  cfg["data_dir"],
        "--output_dir", output_dir,
        "--device",    "cuda",
    ]

    # evaluation/evaluate.py takes --task but no --seed/--debug; the other
    # scripts (run_subject_aware / run_aspect_ratio) take --seed/--debug and
    # infer the task from the script name.
    if script.endswith("evaluate.py"):
        cmd += ["--task", cfg["task"]]
    else:
        cmd += ["--seed", str(cfg["seed"])]
        if cfg["debug"]:
            cmd += ["--debug"]

    if cfg["max_samples"]:
        cmd += ["--max_samples", str(cfg["max_samples"])]
    if cfg["resume"]:
        cmd += ["--resume", cfg["resume"]]

    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        Path(tmp_config).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
