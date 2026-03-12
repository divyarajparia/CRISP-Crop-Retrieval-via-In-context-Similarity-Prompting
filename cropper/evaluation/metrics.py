"""
Evaluation metrics for Cropper.
Implements IoU, Disp, SRCC, PCC, AccK/N from the paper.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import spearmanr, pearsonr


def compute_iou(
    pred: Tuple[int, int, int, int],
    gt: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.

    Args:
        pred: Predicted (x1, y1, x2, y2) coordinates
        gt: Ground-truth (x1, y1, x2, y2) coordinates

    Returns:
        IoU score in [0, 1]
    """
    x1_p, y1_p, x2_p, y2_p = pred
    x1_g, y1_g, x2_g, y2_g = gt

    # Compute intersection
    x1_i = max(x1_p, x1_g)
    y1_i = max(y1_p, y1_g)
    x2_i = min(x2_p, x2_g)
    y2_i = min(y2_p, y2_g)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area_pred = (x2_p - x1_p) * (y2_p - y1_p)
    area_gt = (x2_g - x1_g) * (y2_g - y1_g)
    union = area_pred + area_gt - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def compute_displacement(
    pred: Tuple[float, float, float, float],
    gt: Tuple[float, float, float, float],
    image_size: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Compute boundary displacement error (Disp).

    Disp = average L1 distance between GT and predicted normalized coordinates.

    Args:
        pred: Predicted (x1, y1, x2, y2) coordinates
        gt: Ground-truth (x1, y1, x2, y2) coordinates
        image_size: (width, height) for normalization (if coordinates are in pixels)

    Returns:
        Average L1 displacement
    """
    x1_p, y1_p, x2_p, y2_p = pred
    x1_g, y1_g, x2_g, y2_g = gt

    if image_size:
        w, h = image_size
        # Normalize to [0, 1]
        disp = (
            abs(x1_p / w - x1_g / w) +
            abs(y1_p / h - y1_g / h) +
            abs(x2_p / w - x2_g / w) +
            abs(y2_p / h - y2_g / h)
        ) / 4
    else:
        # Assume already normalized
        disp = (
            abs(x1_p - x1_g) +
            abs(y1_p - y1_g) +
            abs(x2_p - x2_g) +
            abs(y2_p - y2_g)
        ) / 4

    return float(disp)


def compute_srcc(
    pred_scores: List[float],
    gt_scores: List[float],
) -> float:
    """
    Compute Spearman's rank-order correlation coefficient (SRCC).

    Args:
        pred_scores: Predicted MOS scores
        gt_scores: Ground-truth MOS scores

    Returns:
        SRCC value in [-1, 1]
    """
    if len(pred_scores) < 2 or len(gt_scores) < 2:
        return 0.0

    # Align lengths
    n = min(len(pred_scores), len(gt_scores))
    pred = pred_scores[:n]
    gt = gt_scores[:n]

    try:
        corr, _ = spearmanr(pred, gt)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def compute_pcc(
    pred_scores: List[float],
    gt_scores: List[float],
) -> float:
    """
    Compute Pearson correlation coefficient (PCC).

    Args:
        pred_scores: Predicted MOS scores
        gt_scores: Ground-truth MOS scores

    Returns:
        PCC value in [-1, 1]
    """
    if len(pred_scores) < 2 or len(gt_scores) < 2:
        return 0.0

    # Align lengths
    n = min(len(pred_scores), len(gt_scores))
    pred = pred_scores[:n]
    gt = gt_scores[:n]

    try:
        corr, _ = pearsonr(pred, gt)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def compute_acc_k_n(
    pred_crops: List[Tuple],
    gt_crops: List[Tuple],
    K: int,
    N: int,
) -> float:
    """
    Compute AccK/N metric.

    AccK/N measures whether any of the top-K predictions fall within
    the top-N ground-truth crops (by MOS).

    Args:
        pred_crops: Predicted crops with scores, sorted by score descending
        gt_crops: Ground-truth crops with MOS, sorted by MOS descending
        K: Number of top predictions to consider
        N: Number of top ground-truth crops to consider

    Returns:
        1.0 if any top-K prediction overlaps with top-N GT, 0.0 otherwise
    """
    if len(pred_crops) < K or len(gt_crops) < N:
        return 0.0

    # Get top-K predictions
    top_k_pred = pred_crops[:K]

    # Get top-N ground-truth (by MOS)
    top_n_gt = gt_crops[:N]

    # Check if any prediction overlaps with GT
    # We use IoU > 0.5 as the overlap criterion
    for pred in top_k_pred:
        pred_box = _extract_box(pred)

        for gt in top_n_gt:
            gt_box = _extract_box(gt)
            iou = compute_iou(pred_box, gt_box)

            if iou > 0.5:
                return 1.0

    return 0.0


def _extract_box(crop: Tuple) -> Tuple[int, int, int, int]:
    """Extract (x1, y1, x2, y2) from a crop tuple."""
    if len(crop) == 5:  # (mos, x1, y1, x2, y2)
        return crop[1:5]
    elif len(crop) == 4:  # (x1, y1, x2, y2)
        return crop
    else:
        raise ValueError(f"Invalid crop format: {crop}")


def compute_all_acc_metrics(
    pred_crops: List[Tuple],
    gt_crops: List[Tuple],
) -> Dict[str, float]:
    """
    Compute all AccK/N metrics.

    Computes: Acc1/5, Acc2/5, Acc3/5, Acc4/5, Acc1/10, Acc2/10, Acc3/10, Acc4/10

    Args:
        pred_crops: Predicted crops with scores
        gt_crops: Ground-truth crops with MOS

    Returns:
        Dict with all Acc metrics
    """
    results = {}

    for N in [5, 10]:
        for K in range(1, 5):
            key = f"Acc{K}/{N}"
            results[key] = compute_acc_k_n(pred_crops, gt_crops, K, N)

    # Also compute Acc5 and Acc10 (average of AccK/N for each N)
    results["Acc5"] = np.mean([results[f"Acc{K}/5"] for K in range(1, 5)]) * 100
    results["Acc10"] = np.mean([results[f"Acc{K}/10"] for K in range(1, 5)]) * 100

    return results


class MetricsCalculator:
    """
    Calculator for all evaluation metrics.

    Implements metrics from the Cropper paper:
    - IoU: Intersection over Union with best GT crop
    - Disp: Boundary displacement error
    - SRCC/PCC: Correlation between predicted scores and GT MOS for top-5 crops
    - AccK/N: Whether top-K predictions overlap with top-N GT crops
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.iou_scores = []
        self.disp_scores = []
        # For SRCC/PCC: store per-image correlation scores
        self.srcc_per_image = []
        self.pcc_per_image = []
        # For backward compatibility
        self.pred_mos_all = []
        self.gt_mos_all = []
        self.acc_results = []

    def update(
        self,
        pred_crop: Tuple,
        gt_crop: Tuple,
        image_size: Optional[Tuple[int, int]] = None,
        pred_crops_all: Optional[List[Tuple]] = None,
        gt_crops_all: Optional[List[Tuple]] = None,
        pred_scores: Optional[List[float]] = None,
    ):
        """
        Update metrics with a new prediction.

        Args:
            pred_crop: Predicted crop (x1, y1, x2, y2) or (mos, x1, y1, x2, y2)
            gt_crop: Ground-truth crop
            image_size: Image size for displacement calculation
            pred_crops_all: All predicted crops for AccK/N (sorted by score desc)
            gt_crops_all: All GT crops for AccK/N (sorted by MOS desc)
            pred_scores: Predicted scores for SRCC/PCC computation
        """
        # Extract boxes
        pred_box = _extract_box(pred_crop)
        gt_box = _extract_box(gt_crop)

        # IoU
        iou = compute_iou(pred_box, gt_box)
        self.iou_scores.append(iou)

        # Displacement
        disp = compute_displacement(pred_box, gt_box, image_size)
        self.disp_scores.append(disp)

        # MOS scores for correlation (backward compat)
        if len(pred_crop) == 5:
            self.pred_mos_all.append(pred_crop[0])
        if len(gt_crop) == 5:
            self.gt_mos_all.append(gt_crop[0])

        # AccK/N
        if pred_crops_all and gt_crops_all:
            acc = compute_all_acc_metrics(pred_crops_all, gt_crops_all)
            self.acc_results.append(acc)

            # SRCC/PCC per image (paper method):
            # Compare predicted scores of top-5 crops with their matched GT MOS
            if pred_scores:
                srcc, pcc = self._compute_per_image_correlation(
                    pred_crops_all[:5],  # Top-5 predictions
                    pred_scores[:5],
                    gt_crops_all,
                )
                self.srcc_per_image.append(srcc)
                self.pcc_per_image.append(pcc)

    def _compute_per_image_correlation(
        self,
        pred_crops: List[Tuple],
        pred_scores: List[float],
        gt_crops: List[Tuple],
    ) -> Tuple[float, float]:
        """
        Compute SRCC/PCC for a single image.

        For each predicted crop, find the best matching GT crop (by IoU)
        and use its GT MOS. Then compute correlation between predicted
        scores and matched GT MOS values.

        Args:
            pred_crops: Top-K predicted crops with scores
            pred_scores: Predicted scores for these crops
            gt_crops: All GT crops sorted by MOS descending

        Returns:
            (SRCC, PCC) for this image
        """
        if len(pred_crops) < 2 or len(gt_crops) < 1:
            return 0.0, 0.0

        # For each predicted crop, find best matching GT crop and get its MOS
        matched_gt_mos = []
        for pred_crop in pred_crops:
            pred_box = _extract_box(pred_crop)

            best_iou = 0.0
            best_gt_mos = 0.0

            for gt_crop in gt_crops:
                gt_box = _extract_box(gt_crop)
                iou = compute_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    # Get GT MOS (first element if 5-tuple)
                    if len(gt_crop) == 5:
                        best_gt_mos = gt_crop[0]
                    else:
                        # If no MOS, use IoU as proxy
                        best_gt_mos = iou

            matched_gt_mos.append(best_gt_mos)

        # Compute correlations
        if len(pred_scores) >= 2 and len(set(pred_scores)) > 1 and len(set(matched_gt_mos)) > 1:
            srcc = compute_srcc(pred_scores, matched_gt_mos)
            pcc = compute_pcc(pred_scores, matched_gt_mos)
        else:
            srcc = 0.0
            pcc = 0.0

        return srcc, pcc

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dict with all computed metrics
        """
        results = {}

        # Mean IoU
        if self.iou_scores:
            results["IoU"] = float(np.mean(self.iou_scores))
            results["IoU_std"] = float(np.std(self.iou_scores))

        # Mean Displacement
        if self.disp_scores:
            results["Disp"] = float(np.mean(self.disp_scores))
            results["Disp_std"] = float(np.std(self.disp_scores))

        # SRCC and PCC - prefer per-image computation (paper method)
        if self.srcc_per_image:
            # Average SRCC/PCC across images (paper method)
            # Filter out zeros for mean computation
            valid_srcc = [s for s in self.srcc_per_image if s != 0.0]
            valid_pcc = [p for p in self.pcc_per_image if p != 0.0]

            if valid_srcc:
                results["SRCC"] = float(np.mean(valid_srcc))
            else:
                results["SRCC"] = 0.0

            if valid_pcc:
                results["PCC"] = float(np.mean(valid_pcc))
            else:
                results["PCC"] = 0.0
        elif self.pred_mos_all and self.gt_mos_all:
            # Fallback: global correlation
            results["SRCC"] = compute_srcc(self.pred_mos_all, self.gt_mos_all)
            results["PCC"] = compute_pcc(self.pred_mos_all, self.gt_mos_all)

        # Average AccK/N
        if self.acc_results:
            for key in self.acc_results[0].keys():
                values = [r[key] for r in self.acc_results]
                results[key] = float(np.mean(values))

        return results

    def __str__(self) -> str:
        """Format metrics as string."""
        results = self.compute()
        lines = []
        for key, value in results.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


def format_results_table(results: Dict[str, float]) -> str:
    """
    Format results as a table string.

    Args:
        results: Dict of metric name to value

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Evaluation Results")
    lines.append("=" * 50)

    # Group metrics
    primary_metrics = ["IoU", "Disp", "SRCC", "PCC"]
    acc_metrics = [k for k in results.keys() if k.startswith("Acc")]

    for metric in primary_metrics:
        if metric in results:
            lines.append(f"{metric:>10}: {results[metric]:.4f}")

    if acc_metrics:
        lines.append("-" * 50)
        for metric in sorted(acc_metrics):
            lines.append(f"{metric:>10}: {results[metric]:.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)
