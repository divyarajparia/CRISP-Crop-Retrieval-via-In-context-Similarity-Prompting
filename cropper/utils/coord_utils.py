"""
Coordinate utilities for Cropper.
Handles normalization, denormalization, and crop extraction.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def normalize_coords(
    coords: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    coord_range: Union[Tuple[int, int], str] = (1, 1000),
) -> Tuple[float, float, float, float]:
    """
    Normalize pixel coordinates to a specified range.

    Args:
        coords: (x1, y1, x2, y2) in pixel coordinates
        image_size: (width, height) of the image
        coord_range: Target range for normalization
            - (1, 1000) for free-form cropping
            - (0.0, 1.0) for subject-aware cropping
            - "pixel" to return pixel coordinates unchanged

    Returns:
        Normalized (x1, y1, x2, y2) coordinates
    """
    x1, y1, x2, y2 = coords
    w, h = image_size

    if coord_range == "pixel":
        return (x1, y1, x2, y2)

    if isinstance(coord_range, (tuple, list)):
        min_val, max_val = coord_range

        # Normalize to [0, 1] first
        x1_norm = x1 / w
        y1_norm = y1 / h
        x2_norm = x2 / w
        y2_norm = y2 / h

        # Scale to target range
        range_size = max_val - min_val
        x1_scaled = x1_norm * range_size + min_val
        y1_scaled = y1_norm * range_size + min_val
        x2_scaled = x2_norm * range_size + min_val
        y2_scaled = y2_norm * range_size + min_val

        return (x1_scaled, y1_scaled, x2_scaled, y2_scaled)

    return (x1, y1, x2, y2)


def denormalize_coords(
    coords: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    coord_range: Union[Tuple[int, int], str] = (1, 1000),
) -> Tuple[int, int, int, int]:
    """
    Denormalize coordinates to pixel values.

    Args:
        coords: Normalized (x1, y1, x2, y2) coordinates
        image_size: (width, height) of the image
        coord_range: Range used for normalization
            - (1, 1000) for free-form cropping
            - (0.0, 1.0) for subject-aware cropping
            - "pixel" if already in pixel coordinates

    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    x1, y1, x2, y2 = coords
    w, h = image_size

    if coord_range == "pixel":
        return (int(x1), int(y1), int(x2), int(y2))

    if isinstance(coord_range, (tuple, list)):
        min_val, max_val = coord_range
        range_size = max_val - min_val

        # Denormalize from range to [0, 1]
        x1_norm = (x1 - min_val) / range_size
        y1_norm = (y1 - min_val) / range_size
        x2_norm = (x2 - min_val) / range_size
        y2_norm = (y2 - min_val) / range_size

        # Scale to pixel coordinates
        x1_pixel = int(x1_norm * w)
        y1_pixel = int(y1_norm * h)
        x2_pixel = int(x2_norm * w)
        y2_pixel = int(y2_norm * h)

        return (x1_pixel, y1_pixel, x2_pixel, y2_pixel)

    return (int(x1), int(y1), int(x2), int(y2))


def validate_crop(
    crop: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    min_size: int = 10,
) -> Tuple[int, int, int, int]:
    """
    Validate and fix crop coordinates.

    Args:
        crop: (x1, y1, x2, y2) coordinates
        image_size: (width, height) of the image
        min_size: Minimum crop dimension

    Returns:
        Valid (x1, y1, x2, y2) coordinates
    """
    x1, y1, x2, y2 = crop
    w, h = image_size

    # Clamp to image bounds
    x1 = max(0, min(w - min_size, x1))
    y1 = max(0, min(h - min_size, y1))
    x2 = max(min_size, min(w, x2))
    y2 = max(min_size, min(h, y2))

    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x2 = min(w, x1 + min_size)
    if y2 <= y1:
        y2 = min(h, y1 + min_size)

    return (x1, y1, x2, y2)


def extract_crop(
    image: Image.Image,
    crop: Tuple[int, int, int, int],
    validate: bool = True,
) -> Image.Image:
    """
    Extract a crop from an image.

    Args:
        image: PIL Image
        crop: (x1, y1, x2, y2) pixel coordinates
        validate: Whether to validate crop coordinates

    Returns:
        Cropped PIL Image
    """
    if validate:
        crop = validate_crop(crop, image.size)

    return image.crop(crop)


def crop_from_normalized(
    image: Image.Image,
    coords: Tuple[float, float, float, float],
    coord_range: Union[Tuple[int, int], str] = (1, 1000),
) -> Image.Image:
    """
    Extract crop using normalized coordinates.

    Args:
        image: PIL Image
        coords: Normalized (x1, y1, x2, y2) coordinates
        coord_range: Coordinate normalization range

    Returns:
        Cropped PIL Image
    """
    pixel_coords = denormalize_coords(coords, image.size, coord_range)
    return extract_crop(image, pixel_coords)


def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1: (x1, y1, x2, y2) coordinates
        box2: (x1, y1, x2, y2) coordinates

    Returns:
        IoU score in [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_displacement(
    pred: Tuple[float, float, float, float],
    gt: Tuple[float, float, float, float],
    normalize: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Compute boundary displacement error between predicted and ground-truth boxes.

    Args:
        pred: Predicted (x1, y1, x2, y2) coordinates
        gt: Ground-truth (x1, y1, x2, y2) coordinates
        normalize: Whether to normalize by image size
        image_size: (width, height) for normalization

    Returns:
        Average L1 displacement
    """
    x1_p, y1_p, x2_p, y2_p = pred
    x1_g, y1_g, x2_g, y2_g = gt

    if normalize and image_size:
        w, h = image_size
        # Normalize to [0, 1]
        disp = (
            abs(x1_p / w - x1_g / w) +
            abs(y1_p / h - y1_g / h) +
            abs(x2_p / w - x2_g / w) +
            abs(y2_p / h - y2_g / h)
        ) / 4
    else:
        disp = (
            abs(x1_p - x1_g) +
            abs(y1_p - y1_g) +
            abs(x2_p - x2_g) +
            abs(y2_p - y2_g)
        ) / 4

    return float(disp)


def format_crop_for_prompt(
    crop: Union[Tuple, List],
    task: str,
    include_mos: bool = True,
) -> str:
    """
    Format a crop tuple as a string for VLM prompts.

    Args:
        crop: Crop tuple (with or without MOS)
        task: Task type for formatting
        include_mos: Whether to include MOS in output

    Returns:
        Formatted string like "(0.8, 100, 50, 800, 700)"
    """
    if task == "freeform":
        if len(crop) == 5 and include_mos:
            mos, x1, y1, x2, y2 = crop
            return f"({mos:.2f}, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
        elif len(crop) == 4:
            x1, y1, x2, y2 = crop
            return f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"

    elif task == "subject_aware":
        if len(crop) == 4:
            x1, y1, x2, y2 = crop
            return f"({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})"

    elif task == "aspect_ratio":
        if len(crop) == 4:
            x1, y1, x2, y2 = crop
            return f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"

    # Fallback
    return str(crop)


def get_center_crop(
    image_size: Tuple[int, int],
    aspect_ratio: Optional[float] = None,
    margin: float = 0.1,
) -> Tuple[int, int, int, int]:
    """
    Get a default center crop for fallback.

    Args:
        image_size: (width, height) of image
        aspect_ratio: Optional target aspect ratio
        margin: Margin from edges (fraction)

    Returns:
        (x1, y1, x2, y2) crop coordinates
    """
    w, h = image_size
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = margin_x
    y1 = margin_y
    x2 = w - margin_x
    y2 = h - margin_y

    if aspect_ratio:
        # Adjust to match aspect ratio
        crop_w = x2 - x1
        crop_h = y2 - y1
        current_ratio = crop_w / crop_h

        if current_ratio > aspect_ratio:
            # Too wide, reduce width
            new_w = int(crop_h * aspect_ratio)
            diff = crop_w - new_w
            x1 += diff // 2
            x2 -= diff - diff // 2
        else:
            # Too tall, reduce height
            new_h = int(crop_w / aspect_ratio)
            diff = crop_h - new_h
            y1 += diff // 2
            y2 -= diff - diff // 2

    return (x1, y1, x2, y2)
