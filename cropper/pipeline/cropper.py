"""
Main Cropper pipeline combining all components.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from PIL import Image

from models.vlm import BaseVLM, create_vlm
from models.clip_retriever import CLIPRetriever
from models.scorer import CombinedScorer, create_scorer
from utils.coord_utils import (
    denormalize_coords,
    normalize_coords,
    extract_crop,
    get_center_crop,
)
from .prompt_builder import PromptBuilder
from .retrieval import retrieve_icl_examples
from .refinement import iterative_refinement

logger = logging.getLogger(__name__)


class Cropper:
    """
    Main Cropper pipeline for image cropping through in-context learning.

    Combines:
    - CLIP-based ICL example retrieval
    - VLM-based crop generation
    - Iterative refinement with scoring
    """

    def __init__(
        self,
        vlm: BaseVLM,
        clip_retriever: CLIPRetriever,
        scorer: CombinedScorer,
        database,
        config: Optional[Dict] = None,
    ):
        """
        Initialize Cropper pipeline.

        Args:
            vlm: Vision-Language Model wrapper
            clip_retriever: CLIP-based retriever for ICL examples
            scorer: Combined scorer for crop evaluation
            database: Dataset for ICL examples
            config: Configuration dictionary
        """
        self.vlm = vlm
        self.clip_retriever = clip_retriever
        self.scorer = scorer
        self.database = database
        self.config = config or {}

        # Prompt builders for each task
        self.prompt_builders = {
            "freeform": PromptBuilder(task="freeform"),
            "subject_aware": PromptBuilder(task="subject_aware"),
            "aspect_ratio": PromptBuilder(task="aspect_ratio"),
        }

        # Cache for saving intermediate results
        self.cache_dir = Path(self.config.get("cache_dir", "./cache"))
        self.save_prompts = self.config.get("save_prompts", True)

    def crop(
        self,
        query_image: Image.Image,
        task: str = "freeform",
        mask: Optional[Image.Image] = None,
        mask_center: Optional[Tuple[float, float]] = None,
        aspect_ratio: Optional[float] = None,
        return_details: bool = False,
    ) -> Union[Tuple[int, int, int, int], Dict]:
        """
        Generate a crop for the query image.

        Args:
            query_image: Input image to crop
            task: Cropping task type ('freeform', 'subject_aware', 'aspect_ratio')
            mask: Binary mask for subject-aware cropping
            mask_center: (cx, cy) center of subject mask (normalized 0-1)
            aspect_ratio: Target aspect ratio for aspect-ratio-aware cropping
            return_details: Whether to return detailed results

        Returns:
            If return_details=False:
                Final crop coordinates (x1, y1, x2, y2) in pixels
            If return_details=True:
                Dict with 'crop', 'score', 'iterations', etc.
        """
        # Get task-specific config
        task_config = self.config.get(task, {})
        S = task_config.get("S", 10)
        T = task_config.get("T", 5)
        R = task_config.get("R", 5)
        L = task_config.get("L", 2)
        temperature = task_config.get("temperature", 0.05)
        coord_range = task_config.get("coord_range", (1, 1000))

        if coord_range == "pixel":
            coord_range = "pixel"
        elif isinstance(coord_range, list):
            coord_range = tuple(coord_range)

        logger.info(f"Running {task} cropping with S={S}, T={T}, R={R}, L={L}")

        # Prepare task parameters
        task_params = {}
        if task == "subject_aware":
            if mask_center is None and mask is not None:
                mask_center = self._compute_mask_center(mask)
            task_params["mask_center"] = mask_center or (0.5, 0.5)

        elif task == "aspect_ratio":
            task_params["aspect_ratio"] = aspect_ratio or 1.0
            task_params["R"] = R

        # Step 1: Retrieve ICL examples
        logger.info("Step 1: Retrieving ICL examples...")
        icl_examples = retrieve_icl_examples(
            query_image=query_image,
            database=self.database,
            clip_retriever=self.clip_retriever,
            task=task,
            S=S,
            T=T,
            mask=mask,
            mask_center=mask_center,
            aspect_ratio=aspect_ratio,
        )

        if not icl_examples:
            logger.warning("No ICL examples retrieved. Using fallback.")
            fallback_crop = get_center_crop(query_image.size, aspect_ratio)
            if return_details:
                return {"crop": fallback_crop, "score": 0.5, "error": "no_icl_examples"}
            return fallback_crop

        logger.info(f"  Retrieved {len(icl_examples)} ICL examples")

        # Step 2: Build initial prompt
        logger.info("Step 2: Building initial prompt...")
        prompt_builder = self.prompt_builders[task]
        initial_prompt, initial_images = prompt_builder.build_initial_prompt(
            icl_examples=icl_examples,
            query_image=query_image,
            task_params=task_params,
        )

        if self.save_prompts:
            self._save_prompt(initial_prompt, "initial", task)

        # Step 3: Generate initial crop candidates
        logger.info("Step 3: Generating initial crop candidates...")
        vlm_output = self.vlm.generate(
            images=initial_images,
            prompt=initial_prompt,
            temperature=temperature,
        )

        logger.debug(f"  VLM output: {vlm_output[:200]}...")

        initial_crops = self.vlm.parse_crops(vlm_output, task, query_image.size)

        if not initial_crops:
            logger.warning("Failed to parse initial crops. Using fallback.")
            # Generate some default crops as fallback
            initial_crops = self._generate_fallback_crops(
                query_image.size, task, aspect_ratio, R
            )

        logger.info(f"  Generated {len(initial_crops)} initial crop candidates")

        # Step 4: Run iterative refinement
        logger.info("Step 4: Running iterative refinement...")
        result = iterative_refinement(
            vlm=self.vlm,
            scorer=self.scorer,
            query_image=query_image,
            initial_crops=initial_crops,
            prompt_builder=prompt_builder,
            initial_prompt=initial_prompt,
            initial_images=initial_images,
            L=L,
            task=task,
            coord_range=coord_range,
            temperature=temperature,
            task_params=task_params,
            return_all_iterations=return_details,
        )

        if return_details:
            return result
        else:
            # Extract pixel coordinates from result
            if isinstance(result, dict):
                final_crop = result["final_crop"]
            else:
                final_crop = result

            # Ensure we return (x1, y1, x2, y2)
            if len(final_crop) == 5:
                _, x1, y1, x2, y2 = final_crop
                return (x1, y1, x2, y2)
            else:
                return final_crop

    def _compute_mask_center(self, mask: Image.Image) -> Tuple[float, float]:
        """Compute normalized center of a binary mask."""
        import numpy as np
        mask_array = np.array(mask)

        if len(mask_array.shape) == 3:
            mask_array = mask_array.mean(axis=2)

        y_coords, x_coords = np.where(mask_array > 127)

        if len(x_coords) == 0:
            return 0.5, 0.5

        cx = x_coords.mean() / mask_array.shape[1]
        cy = y_coords.mean() / mask_array.shape[0]

        return float(cx), float(cy)

    def _generate_fallback_crops(
        self,
        image_size: Tuple[int, int],
        task: str,
        aspect_ratio: Optional[float],
        R: int,
    ) -> List[Tuple]:
        """Generate fallback crops when VLM fails."""
        w, h = image_size
        crops = []

        # Generate R different crops by varying margins
        for i in range(R):
            margin = 0.1 + i * 0.05
            crop = get_center_crop(image_size, aspect_ratio, margin)

            if task == "freeform":
                # Normalize to [1, 1000]
                x1, y1, x2, y2 = crop
                norm_crop = normalize_coords(crop, image_size, (1, 1000))
                mos = 0.7 - i * 0.05  # Decreasing MOS for variety
                crops.append((mos, *norm_crop))

            elif task == "subject_aware":
                # Normalize to [0, 1]
                norm_crop = normalize_coords(crop, image_size, (0.0, 1.0))
                crops.append(norm_crop)

            elif task == "aspect_ratio":
                # Keep as pixel coordinates
                crops.append(crop)

        return crops

    def _save_prompt(self, prompt: str, prompt_type: str, task: str):
        """Save prompt for debugging."""
        save_dir = self.cache_dir / "prompts" / task
        save_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = save_dir / f"{prompt_type}_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)


def create_cropper(
    config_path: Optional[str] = None,
    device: str = "cuda",
    task: str = "freeform",
    database=None,
) -> Cropper:
    """
    Factory function to create a Cropper instance.

    Args:
        config_path: Path to configuration YAML file
        device: Device to run models on
        task: Default cropping task
        database: Dataset for ICL examples

    Returns:
        Configured Cropper instance
    """
    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create VLM
    vlm_model = config.get("vlm_model", "TIGER-Lab/Mantis-8B-Idefics2")
    vlm = create_vlm(
        model_type="mantis" if "mantis" in vlm_model.lower() else "idefics2",
        model_name=vlm_model,
        device=device,
    )

    # Create CLIP retriever
    clip_model = config.get("clip_model", "ViT-B-32")
    clip_pretrained = config.get("clip_pretrained", "openai")
    clip_retriever = CLIPRetriever(
        model_name=clip_model,
        pretrained=clip_pretrained,
        device=device,
        cache_dir=config.get("cache", {}).get("cache_dir", "./cache"),
    )

    # Build database if provided
    if database is not None:
        cache_path = Path(config.get("cache", {}).get("cache_dir", "./cache"))
        clip_retriever.build_database(
            database,
            cache_path=cache_path / "clip_embeddings.pkl",
        )

    # Create scorer
    task_config = config.get(task, {})
    scorer_config = task_config.get("scorer", "vila+area")
    scorer = create_scorer(task=task, device=device, scorer_config=scorer_config)

    # Create Cropper
    return Cropper(
        vlm=vlm,
        clip_retriever=clip_retriever,
        scorer=scorer,
        database=database,
        config=config,
    )
