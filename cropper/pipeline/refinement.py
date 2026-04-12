"""
Iterative crop refinement module for Cropper.
Implements the iterative refinement loop from the paper.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

from models.vlm import BaseVLM
from models.scorer import CombinedScorer
from utils.coord_utils import (
    denormalize_coords,
    extract_crop,
    validate_crop,
    get_center_crop,
)
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


def iterative_refinement(
    vlm: BaseVLM,
    scorer: CombinedScorer,
    query_image: Image.Image,
    initial_crops: List[Tuple],
    prompt_builder: PromptBuilder,
    initial_prompt: str,
    initial_images: List[Image.Image],
    L: int = 2,
    task: str = "freeform",
    coord_range: Union[Tuple[int, int], str] = (1, 1000),
    temperature: float = 0.05,
    task_params: Optional[Dict] = None,
    return_all_iterations: bool = False,
) -> Union[Tuple, Dict]:
    """
    Perform iterative crop refinement.

    Args:
        vlm: VLM model wrapper
        scorer: Combined scorer for evaluating crops
        query_image: Original query image
        initial_crops: List of R crop coordinates from initial VLM output
        prompt_builder: Prompt builder for the task
        initial_prompt: The initial prompt used
        initial_images: Images from the initial prompt
        L: Number of refinement iterations
        task: Cropping task type
        coord_range: Coordinate range for normalization
        temperature: VLM sampling temperature
        task_params: Task-specific parameters
        return_all_iterations: Whether to return all iteration results

    Returns:
        If return_all_iterations=False:
            Final best crop coordinates
        If return_all_iterations=True:
            Dict with 'final_crop', 'iterations', 'scores'
    """
    task_params = task_params or {}
    image_size = query_image.size

    # Set original image for scorer
    scorer.set_original(query_image)

    # Convert initial crops to pixel coordinates
    current_crops = []
    for crop in initial_crops:
        if len(crop) == 5:  # (mos, x1, y1, x2, y2)
            mos, x1, y1, x2, y2 = crop
            pixel_crop = denormalize_coords((x1, y1, x2, y2), image_size, coord_range)
            current_crops.append((mos, *pixel_crop))
        else:  # (x1, y1, x2, y2)
            pixel_crop = denormalize_coords(crop, image_size, coord_range)
            current_crops.append(pixel_crop)

    # Validate crops
    current_crops = [
        _validate_crop_tuple(crop, image_size) for crop in current_crops
    ]

    # Filter out invalid crops
    current_crops = [c for c in current_crops if c is not None]

    if not current_crops:
        logger.warning("No valid initial crops. Using center crop as fallback.")
        center_crop = get_center_crop(image_size)
        current_crops = [center_crop]

    # Track all iterations
    all_iterations = []
    all_scores = []

    # Current prompt for refinement
    current_prompt = initial_prompt
    current_images = initial_images

    for iteration in range(L):
        logger.info(f"Refinement iteration {iteration + 1}/{L}")

        # 1. Extract crop images
        crop_images = []
        crop_boxes_pixel: List[Tuple[int, int, int, int]] = []
        for crop in current_crops:
            if len(crop) == 5:
                _, x1, y1, x2, y2 = crop
            else:
                x1, y1, x2, y2 = crop

            crop_img = extract_crop(query_image, (x1, y1, x2, y2))
            crop_images.append(crop_img)
            crop_boxes_pixel.append((int(x1), int(y1), int(x2), int(y2)))

        # 2. Score each crop. Pass parallel pixel-space boxes so the
        #    GaicdCalibrationScorer can compute geometry features. Other
        #    scorers ignore the kwarg.
        scores = scorer.score_batch(crop_images, crop_boxes=crop_boxes_pixel)

        # Store iteration results
        all_iterations.append(list(current_crops))
        all_scores.append(list(scores))

        logger.info(f"  Scores: {[f'{s:.3f}' for s in scores]}")

        # 3. Build refinement prompt
        # Convert crops back to normalized coordinates for the prompt
        normalized_crops = []
        for crop in current_crops:
            if len(crop) == 5:
                mos, x1, y1, x2, y2 = crop
                from utils.coord_utils import normalize_coords
                norm_coords = normalize_coords((x1, y1, x2, y2), image_size, coord_range)
                normalized_crops.append((mos, *norm_coords))
            else:
                x1, y1, x2, y2 = crop
                from utils.coord_utils import normalize_coords
                norm_coords = normalize_coords((x1, y1, x2, y2), image_size, coord_range)
                normalized_crops.append(norm_coords)

        refinement_prompt, refinement_images = prompt_builder.build_refinement_prompt(
            initial_prompt=initial_prompt,
            initial_images=initial_images,
            crop_images=crop_images,
            crop_coords=normalized_crops,
            scores=scores,
            query_image=query_image,
            task_params=task_params,
        )

        # 4. Generate new crop candidates
        vlm_output = vlm.generate(
            images=refinement_images,
            prompt=refinement_prompt,
            temperature=temperature,
        )

        logger.debug(f"  VLM output: {vlm_output[:200]}...")

        # 5. Parse new crops
        new_crops = vlm.parse_crops(vlm_output, task, image_size)

        if not new_crops:
            logger.warning(f"  No crops parsed at iteration {iteration + 1}. Keeping current crops.")
            continue

        # Convert to pixel coordinates and validate
        new_crops_pixel = []
        for crop in new_crops:
            if len(crop) == 5:  # (mos, x1, y1, x2, y2)
                mos, x1, y1, x2, y2 = crop
                pixel_crop = denormalize_coords((x1, y1, x2, y2), image_size, coord_range)
                validated = validate_crop(pixel_crop, image_size)
                new_crops_pixel.append((mos, *validated))
            else:  # (x1, y1, x2, y2)
                pixel_crop = denormalize_coords(crop, image_size, coord_range)
                validated = validate_crop(pixel_crop, image_size)
                new_crops_pixel.append(validated)

        if new_crops_pixel:
            current_crops = new_crops_pixel
            logger.info(f"  Updated to {len(current_crops)} new crops")

    # Final scoring
    final_crop_images = []
    final_crop_boxes: List[Tuple[int, int, int, int]] = []
    for crop in current_crops:
        if len(crop) == 5:
            _, x1, y1, x2, y2 = crop
        else:
            x1, y1, x2, y2 = crop
        crop_img = extract_crop(query_image, (x1, y1, x2, y2))
        final_crop_images.append(crop_img)
        final_crop_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    final_scores = scorer.score_batch(final_crop_images, crop_boxes=final_crop_boxes)
    all_iterations.append(list(current_crops))
    all_scores.append(list(final_scores))

    # Select final crop based on task
    final_iter_sel = task_params.get("final_iter_selection", False)
    if task in ["freeform", "aspect_ratio"] and not final_iter_sel:
        # Default: keep the best crop seen across all iterations.
        best_crop, best_score = _select_best_across_iterations(
            all_iterations, all_scores
        )
    else:
        # For subject-aware: return from final iteration
        best_idx = final_scores.index(max(final_scores))
        best_crop = current_crops[best_idx]
        best_score = final_scores[best_idx]

    logger.info(f"Final best crop score: {best_score:.3f}")

    if return_all_iterations:
        return {
            "final_crop": best_crop,
            "final_score": best_score,
            "iterations": all_iterations,
            "scores": all_scores,
        }
    else:
        return best_crop


def _validate_crop_tuple(
    crop: Tuple,
    image_size: Tuple[int, int],
) -> Optional[Tuple]:
    """Validate a crop tuple and return valid version or None."""
    try:
        if len(crop) == 5:
            mos, x1, y1, x2, y2 = crop
            validated = validate_crop((x1, y1, x2, y2), image_size)
            return (mos, *validated)
        elif len(crop) == 4:
            return validate_crop(crop, image_size)
        else:
            return None
    except Exception:
        return None


def _select_best_across_iterations(
    iterations: List[List[Tuple]],
    scores: List[List[float]],
) -> Tuple[Tuple, float]:
    """Select the best crop across all iterations by score."""
    best_crop = None
    best_score = -float("inf")

    for iter_crops, iter_scores in zip(iterations, scores):
        for crop, score in zip(iter_crops, iter_scores):
            if score > best_score:
                best_score = score
                best_crop = crop

    return best_crop, best_score


class IterativeRefiner:
    """
    High-level iterative refinement wrapper.
    """

    def __init__(
        self,
        vlm: BaseVLM,
        scorer: CombinedScorer,
        prompt_builder: PromptBuilder,
        task: str = "freeform",
        L: int = 2,
        temperature: float = 0.05,
        coord_range: Union[Tuple[int, int], str] = (1, 1000),
    ):
        """
        Initialize iterative refiner.

        Args:
            vlm: VLM model wrapper
            scorer: Combined scorer
            prompt_builder: Prompt builder
            task: Cropping task type
            L: Number of refinement iterations
            temperature: VLM temperature
            coord_range: Coordinate normalization range
        """
        self.vlm = vlm
        self.scorer = scorer
        self.prompt_builder = prompt_builder
        self.task = task
        self.L = L
        self.temperature = temperature
        self.coord_range = coord_range

    def refine(
        self,
        query_image: Image.Image,
        initial_crops: List[Tuple],
        initial_prompt: str,
        initial_images: List[Image.Image],
        task_params: Optional[Dict] = None,
        return_all: bool = False,
    ) -> Union[Tuple, Dict]:
        """
        Perform iterative refinement.

        Args:
            query_image: Query image
            initial_crops: Initial crop candidates
            initial_prompt: Initial prompt text
            initial_images: Initial prompt images
            task_params: Task-specific parameters
            return_all: Whether to return all iteration results

        Returns:
            Final crop or dict with full results
        """
        return iterative_refinement(
            vlm=self.vlm,
            scorer=self.scorer,
            query_image=query_image,
            initial_crops=initial_crops,
            prompt_builder=self.prompt_builder,
            initial_prompt=initial_prompt,
            initial_images=initial_images,
            L=self.L,
            task=self.task,
            coord_range=self.coord_range,
            temperature=self.temperature,
            task_params=task_params,
            return_all_iterations=return_all,
        )
