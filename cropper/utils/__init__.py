"""Utility functions for Cropper."""

from .coord_utils import (
    normalize_coords,
    denormalize_coords,
    validate_crop,
    extract_crop,
    crop_from_normalized,
    compute_iou,
    compute_displacement,
    format_crop_for_prompt,
    get_center_crop,
)

from .visualization import (
    draw_crop_box,
    draw_multiple_crops,
    create_comparison_figure,
    visualize_iterative_refinement,
    save_result,
    create_grid,
)

__all__ = [
    # Coordinate utilities
    "normalize_coords",
    "denormalize_coords",
    "validate_crop",
    "extract_crop",
    "crop_from_normalized",
    "compute_iou",
    "compute_displacement",
    "format_crop_for_prompt",
    "get_center_crop",
    # Visualization
    "draw_crop_box",
    "draw_multiple_crops",
    "create_comparison_figure",
    "visualize_iterative_refinement",
    "save_result",
    "create_grid",
]
