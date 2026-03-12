#!/usr/bin/env python
"""
Run ablation studies for Cropper.

Studies include:
1. Effect of number of ICL examples (S)
2. Effect of number of crops (R)
3. Effect of refinement iterations (L)
4. Effect of different scorers
5. Effect of retrieval strategy (random vs CLIP)
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import GAICDDataset
from models.vlm import create_vlm
from models.clip_retriever import CLIPRetriever
from models.scorer import create_scorer
from pipeline.cropper import Cropper
from evaluation.metrics import compute_iou

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_ablation(
    cropper: Cropper,
    test_dataset,
    config_override: Dict,
    max_samples: int = 50,
) -> Dict[str, float]:
    """Run evaluation with specific config override."""
    iou_scores = []

    # Update cropper config temporarily
    original_config = cropper.config.copy()
    cropper.config.update(config_override)

    for idx in tqdm(range(min(max_samples, len(test_dataset))), desc="Ablation", leave=False):
        try:
            item = test_dataset[idx]
            image = item["image"]
            gt_crops = item["crops"]

            if not gt_crops:
                continue

            best_gt = gt_crops[0]
            gt_box = best_gt[1:5] if len(best_gt) == 5 else best_gt

            result = cropper.crop(
                query_image=image,
                task="freeform",
                return_details=True,
            )

            pred_crop = result["final_crop"]
            pred_box = pred_crop[1:5] if len(pred_crop) == 5 else pred_crop

            iou = compute_iou(pred_box, gt_box)
            iou_scores.append(iou)

        except Exception as e:
            logger.debug(f"Error: {e}")
            continue

    # Restore config
    cropper.config = original_config

    return {
        "IoU": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "IoU_std": float(np.std(iou_scores)) if iou_scores else 0.0,
        "n_samples": len(iou_scores),
    }


def ablation_icl_examples(
    cropper: Cropper,
    val_dataset,
    output_dir: Path,
    max_samples: int = 50,
):
    """Ablation: Number of ICL examples (S)."""
    logger.info("Running ablation: Number of ICL examples (S)")

    S_values = [1, 5, 10, 20, 30, 40, 50]
    results = []

    for S in S_values:
        logger.info(f"  Testing S={S}")
        config = {"freeform": {"S": S, "T": 5, "R": 5, "L": 2}}
        metrics = run_ablation(cropper, val_dataset, config, max_samples)
        metrics["S"] = S
        results.append(metrics)
        logger.info(f"    IoU: {metrics['IoU']:.4f}")

    # Save results
    with open(output_dir / "ablation_S.json", "w") as f:
        json.dump(results, f, indent=2)

    # Find best S
    best = max(results, key=lambda x: x["IoU"])
    logger.info(f"  Best S={best['S']} with IoU={best['IoU']:.4f}")

    return results


def ablation_num_crops(
    cropper: Cropper,
    val_dataset,
    output_dir: Path,
    max_samples: int = 50,
):
    """Ablation: Number of crops (R)."""
    logger.info("Running ablation: Number of crops (R)")

    R_values = [1, 3, 5, 7, 9, 11, 15, 21]
    results = []

    for R in R_values:
        logger.info(f"  Testing R={R}")
        config = {"freeform": {"S": 10, "T": 5, "R": R, "L": 2}}
        metrics = run_ablation(cropper, val_dataset, config, max_samples)
        metrics["R"] = R
        results.append(metrics)
        logger.info(f"    IoU: {metrics['IoU']:.4f}")

    with open(output_dir / "ablation_R.json", "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x["IoU"])
    logger.info(f"  Best R={best['R']} with IoU={best['IoU']:.4f}")

    return results


def ablation_iterations(
    cropper: Cropper,
    val_dataset,
    output_dir: Path,
    max_samples: int = 50,
):
    """Ablation: Number of refinement iterations (L)."""
    logger.info("Running ablation: Number of iterations (L)")

    L_values = [0, 1, 2, 3, 4, 5, 8, 10, 12]
    results = []

    for L in L_values:
        logger.info(f"  Testing L={L}")
        config = {"freeform": {"S": 10, "T": 5, "R": 5, "L": L}}
        metrics = run_ablation(cropper, val_dataset, config, max_samples)
        metrics["L"] = L
        results.append(metrics)
        logger.info(f"    IoU: {metrics['IoU']:.4f}")

    with open(output_dir / "ablation_L.json", "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x["IoU"])
    logger.info(f"  Best L={best['L']} with IoU={best['IoU']:.4f}")

    return results


def ablation_scorer(
    cropper: Cropper,
    val_dataset,
    output_dir: Path,
    device: str,
    max_samples: int = 50,
):
    """Ablation: Different scorer combinations."""
    logger.info("Running ablation: Scorer combinations")

    scorer_configs = [
        "vila",
        "area",
        "clip",
        "vila+area",
        "vila+clip",
        "area+clip",
        "vila+area+clip",
    ]
    results = []

    for scorer_config in scorer_configs:
        logger.info(f"  Testing scorer={scorer_config}")

        # Create new scorer
        scorer = create_scorer(
            task="freeform",
            device=device,
            scorer_config=scorer_config,
        )
        cropper.scorer = scorer

        config = {"freeform": {"S": 10, "T": 5, "R": 5, "L": 2, "scorer": scorer_config}}
        metrics = run_ablation(cropper, val_dataset, config, max_samples)
        metrics["scorer"] = scorer_config
        results.append(metrics)
        logger.info(f"    IoU: {metrics['IoU']:.4f}")

    with open(output_dir / "ablation_scorer.json", "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x["IoU"])
    logger.info(f"  Best scorer={best['scorer']} with IoU={best['IoU']:.4f}")

    return results


def ablation_retrieval(
    vlm,
    clip_retriever,
    val_dataset,
    train_dataset,
    output_dir: Path,
    device: str,
    max_samples: int = 50,
):
    """Ablation: Random vs CLIP retrieval."""
    logger.info("Running ablation: Retrieval strategy")

    results = []

    # CLIP-based retrieval (standard)
    logger.info("  Testing CLIP-based retrieval")
    scorer = create_scorer(task="freeform", device=device)
    cropper = Cropper(
        vlm=vlm,
        clip_retriever=clip_retriever,
        scorer=scorer,
        database=train_dataset,
        config={"freeform": {"S": 10, "T": 5, "R": 5, "L": 2}},
    )
    metrics = run_ablation(cropper, val_dataset, {}, max_samples)
    metrics["retrieval"] = "clip"
    results.append(metrics)
    logger.info(f"    IoU: {metrics['IoU']:.4f}")

    # Random retrieval (for comparison)
    logger.info("  Testing random retrieval")
    # This would require modifying the retrieval to be random
    # For now, we'll note this as a TODO
    metrics_random = {"retrieval": "random", "IoU": 0.0, "note": "TODO: Implement random retrieval"}
    results.append(metrics_random)

    with open(output_dir / "ablation_retrieval.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/ablation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["S", "R", "L", "scorer", "retrieval", "all"],
        default="all",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    data_dir = Path(args.data_dir)
    logger.info("Loading GAICD datasets...")
    train_dataset = GAICDDataset(data_dir / "GAICD", split="train")
    val_dataset = GAICDDataset(data_dir / "GAICD", split="val")
    logger.info(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Initialize models
    logger.info("Initializing models...")
    vlm = create_vlm(
        model_type="mantis",
        model_name=config.get("vlm_model", "TIGER-Lab/Mantis-8B-Idefics2"),
        device=args.device,
    )

    clip_retriever = CLIPRetriever(
        model_name=config.get("clip_model", "ViT-B-32"),
        pretrained=config.get("clip_pretrained", "openai"),
        device=args.device,
    )
    cache_path = output_dir / "cache" / "train_embeddings.pkl"
    clip_retriever.build_database(train_dataset, cache_path=cache_path)

    scorer = create_scorer(task="freeform", device=args.device)

    cropper = Cropper(
        vlm=vlm,
        clip_retriever=clip_retriever,
        scorer=scorer,
        database=train_dataset,
        config=config,
    )

    # Run ablations
    all_results = {}

    if args.ablation in ["S", "all"]:
        all_results["S"] = ablation_icl_examples(
            cropper, val_dataset, output_dir, args.max_samples
        )

    if args.ablation in ["R", "all"]:
        all_results["R"] = ablation_num_crops(
            cropper, val_dataset, output_dir, args.max_samples
        )

    if args.ablation in ["L", "all"]:
        all_results["L"] = ablation_iterations(
            cropper, val_dataset, output_dir, args.max_samples
        )

    if args.ablation in ["scorer", "all"]:
        all_results["scorer"] = ablation_scorer(
            cropper, val_dataset, output_dir, args.device, args.max_samples
        )

    if args.ablation in ["retrieval", "all"]:
        all_results["retrieval"] = ablation_retrieval(
            vlm, clip_retriever, val_dataset, train_dataset,
            output_dir, args.device, args.max_samples
        )

    # Save all results
    with open(output_dir / "all_ablations.json", "w") as f:
        json.dump({
            "results": all_results,
            "config": config,
            "max_samples": args.max_samples,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"\nAll ablation results saved to {output_dir}")


if __name__ == "__main__":
    main()
