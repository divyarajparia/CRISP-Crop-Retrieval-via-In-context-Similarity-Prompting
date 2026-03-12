"""
Retrieval module for Cropper.
Implements Equations 1 and 2 from the paper for ICL example retrieval.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from models.clip_retriever import CLIPRetriever

logger = logging.getLogger(__name__)


def retrieve_icl_examples(
    query_image: Image.Image,
    database,
    clip_retriever: CLIPRetriever,
    task: str,
    S: int,
    T: int = 5,
    mask: Optional[Image.Image] = None,
    mask_center: Optional[Tuple[float, float]] = None,
    aspect_ratio: Optional[float] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Retrieve in-context learning examples.

    Implements Eq. 1 and Eq. 2 from the paper:
    - Eq. 1: Z = argmax_{z_i ∈ D} Q(z_q, z_i), |Z| = S
    - Eq. 2: H = argmax_{c_j ∈ C_j} G(z_q, c_j), z_j ∈ Z, |H| = T

    Args:
        query_image: PIL Image (query)
        database: Dataset with images and ground-truth crops
        clip_retriever: CLIP-based retriever
        task: 'freeform' | 'subject_aware' | 'aspect_ratio'
        S: Number of images to retrieve
        T: Number of ground-truth crops per image (only for free-form)
        mask: Binary mask for subject-aware cropping
        mask_center: (cx, cy) center of query mask (normalized 0-1)
        aspect_ratio: Target aspect ratio for aspect-ratio-aware cropping
        exclude_ids: Image IDs to exclude (e.g., query image if in database)

    Returns:
        List of ICL examples, each containing:
        - 'image': PIL Image
        - 'image_id': str
        - 'crops': List of crop tuples (format depends on task)
        - Additional task-specific fields
    """
    # Step 1: Retrieve top-S images using Q (CLIP similarity)
    query_embedding = clip_retriever.encode_image(query_image)
    top_s_results = clip_retriever.retrieve_top_s(
        query_embedding, S, exclude_ids=exclude_ids
    )

    retrieved_ids = [img_id for img_id, score in top_s_results]
    logger.info(f"Retrieved {len(retrieved_ids)} images for ICL")

    # Step 2: Select best crop(s) using G (task-specific)
    if task == "freeform":
        return _select_freeform_examples(database, retrieved_ids, T)

    elif task == "subject_aware":
        if mask_center is None and mask is not None:
            mask_center = _compute_mask_center(mask)
        elif mask_center is None:
            mask_center = (0.5, 0.5)  # Default to center
        return _select_subject_aware_examples(database, retrieved_ids, mask_center)

    elif task == "aspect_ratio":
        if aspect_ratio is None:
            aspect_ratio = 1.0  # Default to square
        return _select_aspect_ratio_examples(database, retrieved_ids, aspect_ratio)

    else:
        raise ValueError(f"Unknown task: {task}")


def _select_freeform_examples(
    database,
    retrieved_ids: List[str],
    T: int,
) -> List[Dict]:
    """
    Select top-T crops by MOS score for free-form cropping.

    G = MOS score
    """
    examples = []

    for img_id in retrieved_ids:
        # Get image from database
        item = _get_item_by_id(database, img_id)
        if item is None:
            continue

        # Get top-T crops by MOS
        crops = database.get_top_crops(img_id, T)

        if not crops:
            # Fallback: use any available crops
            crops = item.get("crops", [])[:T]

        examples.append({
            "image": item["image"],
            "image_id": img_id,
            "crops": crops,  # List of (MOS, x1, y1, x2, y2)
        })

    return examples


def _select_subject_aware_examples(
    database,
    retrieved_ids: List[str],
    query_mask_center: Tuple[float, float],
) -> List[Dict]:
    """
    Select crop with closest mask center for subject-aware cropping.

    G = -L2(center(query_mask), center(retrieved_mask))
    """
    examples = []

    for img_id in retrieved_ids:
        # Get all subjects for this image
        subjects = database.get_subjects_for_image(img_id)

        if not subjects:
            continue

        # Find subject with closest mask center (minimize L2 distance)
        best_subject = None
        best_distance = float("inf")

        for subject in subjects:
            # Get item to retrieve image and mask center
            item = _get_item_by_subject(database, subject)
            if item is None:
                continue

            center = item.get("mask_center", (0.5, 0.5))
            distance = np.sqrt(
                (center[0] - query_mask_center[0]) ** 2 +
                (center[1] - query_mask_center[1]) ** 2
            )

            if distance < best_distance:
                best_distance = distance
                best_subject = item

        if best_subject:
            examples.append({
                "image": best_subject["image"],
                "image_id": img_id,
                "mask_center": best_subject.get("mask_center", (0.5, 0.5)),
                "crop": best_subject["crop"],  # (x1, y1, x2, y2) normalized
            })

    return examples


def _select_aspect_ratio_examples(
    database,
    retrieved_ids: List[str],
    target_aspect_ratio: float,
) -> List[Dict]:
    """
    Select crop with matching aspect ratio.

    G = -|AR(crop) - target_AR|
    """
    examples = []

    for img_id in retrieved_ids:
        item = _get_item_by_id(database, img_id)
        if item is None:
            continue

        # Get crop and its aspect ratio
        crop = item.get("crop")
        if crop is None:
            continue

        x1, y1, x2, y2 = crop
        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_ar = crop_w / max(crop_h, 1)

        examples.append({
            "image": item["image"],
            "image_id": img_id,
            "crop": crop,  # (x1, y1, x2, y2) in pixels
            "aspect_ratio": crop_ar,
        })

    # Sort by aspect ratio similarity (closest first)
    examples.sort(key=lambda x: abs(x["aspect_ratio"] - target_aspect_ratio))

    return examples


def _get_item_by_id(database, img_id: str) -> Optional[Dict]:
    """Get dataset item by image ID."""
    for i in range(len(database)):
        item = database[i]
        if item["image_id"] == img_id:
            return item
    return None


def _get_item_by_subject(database, subject: Dict) -> Optional[Dict]:
    """Get dataset item by subject info."""
    img_id = subject.get("image_id")
    subj_idx = subject.get("subject_idx", 0)

    for i in range(len(database)):
        item = database[i]
        if (item["image_id"] == img_id and
            item.get("subject_idx", 0) == subj_idx):
            return item

    return None


def _compute_mask_center(mask: Image.Image) -> Tuple[float, float]:
    """Compute normalized center of a binary mask."""
    mask_array = np.array(mask)

    if len(mask_array.shape) == 3:
        mask_array = mask_array.mean(axis=2)

    y_coords, x_coords = np.where(mask_array > 127)

    if len(x_coords) == 0:
        return 0.5, 0.5

    cx = x_coords.mean() / mask_array.shape[1]
    cy = y_coords.mean() / mask_array.shape[0]

    return float(cx), float(cy)


class ICLRetriever:
    """
    High-level ICL retriever combining CLIP retrieval and ground-truth selection.
    """

    def __init__(
        self,
        clip_retriever: CLIPRetriever,
        database,
        task: str = "freeform",
    ):
        """
        Initialize ICL retriever.

        Args:
            clip_retriever: CLIP-based image retriever
            database: Dataset with images and crops
            task: Cropping task type
        """
        self.clip_retriever = clip_retriever
        self.database = database
        self.task = task

    def retrieve(
        self,
        query_image: Image.Image,
        S: int,
        T: int = 5,
        **kwargs,
    ) -> List[Dict]:
        """
        Retrieve ICL examples for a query image.

        Args:
            query_image: Query image
            S: Number of images to retrieve
            T: Number of crops per image
            **kwargs: Task-specific parameters (mask, mask_center, aspect_ratio)

        Returns:
            List of ICL examples
        """
        return retrieve_icl_examples(
            query_image=query_image,
            database=self.database,
            clip_retriever=self.clip_retriever,
            task=self.task,
            S=S,
            T=T,
            **kwargs,
        )
