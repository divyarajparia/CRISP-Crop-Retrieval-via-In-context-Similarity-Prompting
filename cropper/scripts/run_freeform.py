#!/usr/bin/env python
"""
Run free-form cropping evaluation on GAICD dataset.

This is the main benchmark for Cropper replication.
Target metrics from Table 9 (Mantis-8B-Idefics2):
    - Acc5: 80.2
    - Acc10: 88.6
    - SRCC: 0.874
    - PCC: 0.797
    - IoU: 0.672
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import GAICDDataset
from models.vlm import create_vlm
from models.clip_retriever import CLIPRetriever
from models.scorer import create_scorer
from pipeline.cropper import Cropper
from evaluation.metrics import MetricsCalculator, format_results_table, compute_iou

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Run free-form cropping evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to GAICD dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/freeform",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set seed
    set_seed(args.seed)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    # Get task-specific config
    task_config = config.get("freeform", {})
    S = task_config.get("S", 10)
    T = task_config.get("T", 5)
    R = task_config.get("R", 5)
    L = task_config.get("L", 2)

    logger.info(f"Free-form cropping config: S={S}, T={T}, R={R}, L={L}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load datasets
    data_dir = Path(args.data_dir)
    logger.info("Loading GAICD training set for ICL examples...")
    train_dataset = GAICDDataset(data_dir, split="train")
    logger.info(f"  Loaded {len(train_dataset)} training images")

    logger.info("Loading GAICD test set for evaluation...")
    test_dataset = GAICDDataset(data_dir, split="test")
    logger.info(f"  Loaded {len(test_dataset)} test images")

    # Initialize models
    logger.info("Initializing VLM...")
    vlm = create_vlm(
        model_type="mantis",
        model_name=config.get("vlm_model", "TIGER-Lab/Mantis-8B-Idefics2"),
        device=args.device,
    )

    logger.info("Initializing CLIP retriever...")
    clip_retriever = CLIPRetriever(
        model_name=config.get("clip_model", "ViT-B-32"),
        pretrained=config.get("clip_pretrained", "openai"),
        device=args.device,
    )

    # Build embedding database
    cache_path = output_dir / "cache" / "train_embeddings.pkl"
    clip_retriever.build_database(train_dataset, cache_path=cache_path)

    logger.info("Initializing scorer...")
    scorer = create_scorer(
        task="freeform",
        device=args.device,
        scorer_config=task_config.get("scorer", "vila+area"),
    )

    # Create Cropper
    cropper = Cropper(
        vlm=vlm,
        clip_retriever=clip_retriever,
        scorer=scorer,
        database=train_dataset,
        config=config,
    )

    # Evaluate
    logger.info("Starting evaluation...")
    metrics = MetricsCalculator()
    results = []

    # Resume from checkpoint
    start_idx = 0
    if args.resume and os.path.exists(args.resume):
        with open(args.resume, "r") as f:
            checkpoint = json.load(f)
            results = checkpoint.get("results", [])
            start_idx = len(results)
            logger.info(f"Resuming from index {start_idx}")

    # Determine samples to evaluate
    n_samples = len(test_dataset) if args.max_samples is None else min(args.max_samples, len(test_dataset))

    pbar = tqdm(range(start_idx, n_samples), desc="Evaluating")

    for idx in pbar:
        try:
            item = test_dataset[idx]
            image = item["image"]
            image_id = item["image_id"]
            gt_crops = item["crops"]

            if not gt_crops:
                logger.warning(f"No GT crops for {image_id}")
                continue

            # Best GT crop
            best_gt = gt_crops[0]
            gt_box = best_gt[1:5] if len(best_gt) == 5 else best_gt

            # Run cropper
            result = cropper.crop(
                query_image=image,
                task="freeform",
                return_details=True,
            )

            pred_crop = result["final_crop"]
            pred_box = pred_crop[1:5] if len(pred_crop) == 5 else pred_crop

            # Get ALL predicted crops from ALL iterations for AccK/N and SRCC/PCC
            # The result contains iterations and scores from refinement
            all_pred_crops = []
            all_pred_scores = []

            if "iterations" in result and "scores" in result:
                # Collect crops from ALL iterations (not just final)
                # This gives us enough crops for proper metric computation
                for iter_crops, iter_scores in zip(result["iterations"], result["scores"]):
                    for crop, score in zip(iter_crops, iter_scores):
                        if len(crop) == 5:
                            mos, x1, y1, x2, y2 = crop
                            all_pred_crops.append((score, x1, y1, x2, y2))  # Use scorer output as MOS
                        else:
                            x1, y1, x2, y2 = crop
                            all_pred_crops.append((score, x1, y1, x2, y2))
                        all_pred_scores.append(score)

                # Sort by score descending and remove duplicates
                # Use a dict keyed by coordinates to deduplicate
                seen_crops = {}
                for crop, score in zip(all_pred_crops, all_pred_scores):
                    box_key = (crop[1], crop[2], crop[3], crop[4])  # x1, y1, x2, y2
                    if box_key not in seen_crops or score > seen_crops[box_key][0]:
                        seen_crops[box_key] = (score, crop)

                # Sort by score descending
                sorted_items = sorted(seen_crops.values(), key=lambda x: x[0], reverse=True)
                all_pred_crops = [item[1] for item in sorted_items]
                all_pred_scores = [item[0] for item in sorted_items]
            else:
                # Fallback: use the final crop
                if len(pred_crop) == 5:
                    all_pred_crops = [pred_crop]
                else:
                    all_pred_crops = [(result.get("final_score", 0.5), *pred_crop)]
                all_pred_scores = [result.get("final_score", 0.5)]

            # Compute IoU between best prediction and best GT
            iou = compute_iou(pred_box, gt_box)

            # Update metrics with ALL crops for proper AccK/N computation
            metrics.update(
                pred_crop=pred_crop,
                gt_crop=best_gt,
                image_size=image.size,
                pred_crops_all=all_pred_crops,
                gt_crops_all=gt_crops,  # All GT crops sorted by MOS
                pred_scores=all_pred_scores,
            )

            results.append({
                "image_id": image_id,
                "pred_crop": [float(x) for x in pred_crop],
                "gt_crop": [float(x) for x in best_gt],
                "iou": iou,
                "score": result.get("final_score", 0),
                "n_pred_crops": len(all_pred_crops),
            })

            # Update progress bar
            avg_iou = sum(r["iou"] for r in results) / len(results)
            pbar.set_postfix({"IoU": f"{iou:.3f}", "Avg": f"{avg_iou:.3f}"})

            # Checkpoint
            if (idx + 1) % 50 == 0:
                checkpoint_path = output_dir / "checkpoint.json"
                with open(checkpoint_path, "w") as f:
                    json.dump({"results": results}, f)

        except Exception as e:
            logger.error(f"Error on image {idx}: {e}")
            if args.debug:
                raise
            continue

    # Compute final metrics
    final_metrics = metrics.compute()

    # Print results
    logger.info("\n" + format_results_table(final_metrics))

    # Compare with paper targets
    logger.info("\nComparison with paper (Table 9, Mantis-8B-Idefics2):")
    targets = {
        "Acc5": 80.2,
        "Acc10": 88.6,
        "SRCC": 0.874,
        "PCC": 0.797,
        "IoU": 0.672,
    }
    for metric, target in targets.items():
        achieved = final_metrics.get(metric, 0)
        diff = achieved - target
        status = "✓" if abs(diff) < 0.05 else "✗"
        logger.info(f"  {metric}: {achieved:.3f} (target: {target}, diff: {diff:+.3f}) {status}")

    # Save final results
    output_file = output_dir / "freeform_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metrics": final_metrics,
            "targets": targets,
            "results": results,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(results),
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
