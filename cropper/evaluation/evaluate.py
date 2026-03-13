"""
Evaluation script for Cropper.
Runs evaluation on test sets and computes all metrics.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import GAICDDataset, FCDBDataset, SACDDataset
from pipeline.cropper import create_cropper, Cropper
from evaluation.metrics import MetricsCalculator, format_results_table, compute_iou
from utils.visualization import draw_crop_box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _log_runtime_environment(args):
    """Log enough runtime state to diagnose environment-dependent runs."""
    logger.info("Runtime cwd: %s", os.getcwd())
    logger.info("Config path: %s", args.config)
    logger.info("Data dir: %s", args.data_dir)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    try:
        import torch

        logger.info("torch version: %s", torch.__version__)
        logger.info("torch.cuda.is_available=%s", torch.cuda.is_available())
        logger.info("torch.cuda.device_count=%d", torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            logger.info("torch current device index=%d", torch.cuda.current_device())
            logger.info("torch current device name=%s", torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception as exc:
        logger.warning("Could not log torch CUDA environment: %s", exc)


def evaluate_freeform(
    cropper: Cropper,
    dataset: GAICDDataset,
    output_dir: Path,
    max_samples: Optional[int] = None,
    resume_from: Optional[str] = None,
    save_crops: bool = False,
) -> Dict[str, float]:
    """
    Evaluate free-form cropping on GAICD dataset.

    Args:
        cropper: Cropper instance
        dataset: GAICD test dataset
        output_dir: Directory to save results
        max_samples: Maximum number of samples to evaluate
        resume_from: Path to resume from (checkpoint file)

    Returns:
        Dict of evaluation metrics
    """
    logger.info(f"Evaluating free-form cropping on {len(dataset)} images")

    metrics = MetricsCalculator()
    results = []

    # Load checkpoint if resuming
    start_idx = 0
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, "r") as f:
            checkpoint = json.load(f)
            results = checkpoint.get("results", [])
            start_idx = len(results)
            logger.info(f"Resuming from checkpoint at index {start_idx}")

    # Determine number of samples
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    # Create progress bar
    pbar = tqdm(range(start_idx, n_samples), desc="Evaluating")

    for idx in pbar:
        try:
            item = dataset[idx]
            image = item["image"]
            image_id = item["image_id"]
            gt_crops = item["crops"]

            # Skip if no ground truth
            if not gt_crops:
                logger.warning(f"No GT crops for {image_id}, skipping")
                continue

            # Get best GT crop (highest MOS)
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

            # Compute IoU
            iou = compute_iou(pred_box, gt_box)

            # Update metrics
            metrics.update(
                pred_crop=pred_crop,
                gt_crop=best_gt,
                image_size=image.size,
            )

            # Store result
            results.append({
                "image_id": image_id,
                "pred_crop": list(pred_crop),
                "gt_crop": list(best_gt),
                "iou": iou,
                "score": result.get("final_score", 0),
            })

            # Save crop visualization
            if save_crops:
                crops_dir = output_dir / "crops"
                crops_dir.mkdir(exist_ok=True)
                # Draw GT (green) and pred (red) on original
                vis = draw_crop_box(image, gt_box, color="green", width=3, label=f"GT IoU={iou:.2f}")
                vis = draw_crop_box(vis, pred_box, color="red", width=3, label="Pred")
                # Tight crop
                tight = image.crop(pred_box)
                # Side-by-side: annotated original | tight crop
                aw, ah = vis.size
                tw, th = tight.size
                combined_h = max(ah, th)
                combined = Image.new("RGB", (aw + 10 + tw, combined_h), "white")
                combined.paste(vis, (0, (combined_h - ah) // 2))
                combined.paste(tight, (aw + 10, (combined_h - th) // 2))
                combined.save(crops_dir / f"{image_id}.jpg", quality=90)

            pbar.set_postfix({"IoU": f"{iou:.3f}", "Avg_IoU": f"{sum(r['iou'] for r in results)/len(results):.3f}"})

            # Save checkpoint periodically
            if (idx + 1) % 50 == 0:
                _save_checkpoint(results, output_dir / "checkpoint.json")

        except Exception as e:
            logger.error(f"Error processing {idx}: {e}")
            continue

    # Compute final metrics
    final_metrics = metrics.compute()

    # Save results
    output_file = output_dir / "freeform_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metrics": final_metrics,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(format_results_table(final_metrics))

    return final_metrics


def evaluate_subject_aware(
    cropper: Cropper,
    dataset: SACDDataset,
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate subject-aware cropping on SACD dataset.

    Args:
        cropper: Cropper instance
        dataset: SACD test dataset
        output_dir: Directory to save results
        max_samples: Maximum samples to evaluate

    Returns:
        Dict of evaluation metrics
    """
    logger.info(f"Evaluating subject-aware cropping on {len(dataset)} images")

    metrics = MetricsCalculator()
    results = []

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for idx in tqdm(range(n_samples), desc="Evaluating"):
        try:
            item = dataset[idx]
            image = item["image"]
            image_id = item["image_id"]
            mask = item["mask"]
            mask_center = item["mask_center"]
            gt_crop = item["crop"]

            # Run cropper
            result = cropper.crop(
                query_image=image,
                task="subject_aware",
                mask=mask,
                mask_center=mask_center,
                return_details=True,
            )

            pred_crop = result["final_crop"]

            # Compute IoU
            iou = compute_iou(pred_crop, gt_crop)

            # Update metrics
            metrics.update(
                pred_crop=pred_crop,
                gt_crop=gt_crop,
                image_size=image.size,
            )

            results.append({
                "image_id": image_id,
                "pred_crop": list(pred_crop),
                "gt_crop": list(gt_crop),
                "iou": iou,
            })

        except Exception as e:
            logger.error(f"Error processing {idx}: {e}")
            continue

    final_metrics = metrics.compute()

    output_file = output_dir / "subject_aware_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metrics": final_metrics,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(format_results_table(final_metrics))

    return final_metrics


def evaluate_aspect_ratio(
    cropper: Cropper,
    dataset: FCDBDataset,
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate aspect-ratio-aware cropping on FCDB dataset.

    Args:
        cropper: Cropper instance
        dataset: FCDB test dataset
        output_dir: Directory to save results
        max_samples: Maximum samples to evaluate

    Returns:
        Dict of evaluation metrics
    """
    logger.info(f"Evaluating aspect-ratio cropping on {len(dataset)} images")

    metrics = MetricsCalculator()
    results = []

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for idx in tqdm(range(n_samples), desc="Evaluating"):
        try:
            item = dataset[idx]
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

            # Compute IoU
            iou = compute_iou(pred_crop, gt_crop)

            # Update metrics
            metrics.update(
                pred_crop=pred_crop,
                gt_crop=gt_crop,
                image_size=image.size,
            )

            results.append({
                "image_id": image_id,
                "pred_crop": list(pred_crop),
                "gt_crop": list(gt_crop),
                "aspect_ratio": aspect_ratio,
                "iou": iou,
            })

        except Exception as e:
            logger.error(f"Error processing {idx}: {e}")
            continue

    final_metrics = metrics.compute()

    output_file = output_dir / "aspect_ratio_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metrics": final_metrics,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(format_results_table(final_metrics))

    return final_metrics


def _save_checkpoint(results: List[Dict], checkpoint_path: Path):
    """Save intermediate results to checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump({"results": results}, f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cropper")
    parser.add_argument(
        "--task",
        type=str,
        choices=["freeform", "subject_aware", "aspect_ratio", "all"],
        default="freeform",
        help="Cropping task to evaluate",
    )
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
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
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
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save_crops",
        action="store_true",
        default=False,
        help="Save crop visualizations (original with GT/pred boxes + tight crop) to output_dir/crops/",
    )

    args = parser.parse_args()
    _log_runtime_environment(args)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets based on task
    data_dir = Path(args.data_dir)

    if args.task in ["freeform", "all"]:
        logger.info("Loading GAICD dataset...")
        gaicd_dataset = GAICDDataset(data_dir / "GAICD", split="test")

        # Create cropper for freeform
        cropper = create_cropper(
            config_path=args.config,
            device=args.device,
            task="freeform",
            database=GAICDDataset(data_dir / "GAICD", split="train"),
            require_exact_components=True,
        )

        evaluate_freeform(
            cropper=cropper,
            dataset=gaicd_dataset,
            output_dir=output_dir,
            max_samples=args.max_samples,
            resume_from=args.resume,
            save_crops=args.save_crops,
        )

    if args.task in ["subject_aware", "all"]:
        logger.info("Loading SACD dataset...")
        sacd_dataset = SACDDataset(data_dir / "SACD", split="test")

        cropper = create_cropper(
            config_path=args.config,
            device=args.device,
            task="subject_aware",
            database=SACDDataset(data_dir / "SACD", split="train"),
            require_exact_components=True,
        )

        evaluate_subject_aware(
            cropper=cropper,
            dataset=sacd_dataset,
            output_dir=output_dir,
            max_samples=args.max_samples,
        )

    if args.task in ["aspect_ratio", "all"]:
        logger.info("Loading FCDB dataset...")
        fcdb_dataset = FCDBDataset(data_dir / "FCDB")

        cropper = create_cropper(
            config_path=args.config,
            device=args.device,
            task="aspect_ratio",
            database=GAICDDataset(data_dir / "GAICD", split="train"),  # Use GAICD for retrieval
        )

        evaluate_aspect_ratio(
            cropper=cropper,
            dataset=fcdb_dataset,
            output_dir=output_dir,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
