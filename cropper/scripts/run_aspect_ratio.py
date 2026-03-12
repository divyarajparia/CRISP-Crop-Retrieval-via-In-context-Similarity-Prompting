#!/usr/bin/env python
"""
Run aspect-ratio-aware cropping evaluation on FCDB dataset.

Target metrics from Table 4:
    - IoU: 0.756
    - Disp: 0.053
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import FCDBDataset, GAICDDataset
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Run aspect-ratio cropping evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/aspect_ratio")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    set_seed(args.seed)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    task_config = config.get("aspect_ratio", {})
    S = task_config.get("S", 10)
    R = task_config.get("R", 6)
    L = task_config.get("L", 2)

    logger.info(f"Aspect-ratio cropping config: S={S}, R={R}, L={L}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    data_dir = Path(args.data_dir)

    # Use GAICD training set for ICL retrieval (as per paper)
    logger.info("Loading GAICD training set for ICL examples...")
    gaicd_train = GAICDDataset(data_dir / "GAICD", split="train")
    logger.info(f"  Loaded {len(gaicd_train)} training images")

    logger.info("Loading FCDB test set...")
    fcdb_test = FCDBDataset(data_dir / "FCDB")
    logger.info(f"  Loaded {len(fcdb_test)} test images")

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

    cache_path = output_dir / "cache" / "gaicd_train_embeddings.pkl"
    clip_retriever.build_database(gaicd_train, cache_path=cache_path)

    logger.info("Initializing scorer...")
    scorer = create_scorer(
        task="aspect_ratio",
        device=args.device,
        scorer_config=task_config.get("scorer", "clip"),  # CLIP only for aspect-ratio
    )

    # Create Cropper
    cropper = Cropper(
        vlm=vlm,
        clip_retriever=clip_retriever,
        scorer=scorer,
        database=gaicd_train,
        config=config,
    )

    # Evaluate
    logger.info("Starting evaluation...")
    metrics = MetricsCalculator()
    results = []

    n_samples = len(fcdb_test) if args.max_samples is None else min(args.max_samples, len(fcdb_test))

    for idx in tqdm(range(n_samples), desc="Evaluating"):
        try:
            item = fcdb_test[idx]
            image = item["image"]
            image_id = item["image_id"]
            gt_crop = item["crop"]
            aspect_ratio = item["aspect_ratio"]

            # Run cropper
            result = cropper.crop(
                query_image=image,
                task="aspect_ratio",
                aspect_ratio=aspect_ratio,
                return_details=True,
            )

            pred_crop = result["final_crop"]
            iou = compute_iou(pred_crop, gt_crop)

            metrics.update(
                pred_crop=pred_crop,
                gt_crop=gt_crop,
                image_size=image.size,
            )

            results.append({
                "image_id": image_id,
                "pred_crop": [float(x) for x in pred_crop],
                "gt_crop": [float(x) for x in gt_crop],
                "aspect_ratio": aspect_ratio,
                "iou": iou,
            })

        except Exception as e:
            logger.error(f"Error on image {idx}: {e}")
            if args.debug:
                raise
            continue

    # Compute metrics
    final_metrics = metrics.compute()
    logger.info("\n" + format_results_table(final_metrics))

    # Compare with targets
    logger.info("\nComparison with paper (Table 4):")
    targets = {"IoU": 0.756, "Disp": 0.053}
    for metric, target in targets.items():
        achieved = final_metrics.get(metric, 0)
        diff = achieved - target
        logger.info(f"  {metric}: {achieved:.4f} (target: {target}, diff: {diff:+.4f})")

    # Save results
    output_file = output_dir / "aspect_ratio_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metrics": final_metrics,
            "targets": targets,
            "results": results,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
