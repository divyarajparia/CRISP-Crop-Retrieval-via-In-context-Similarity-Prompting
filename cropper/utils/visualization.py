"""
Visualization utilities for Cropper.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_crop_box(
    image: Image.Image,
    crop: Tuple[int, int, int, int],
    color: str = "red",
    width: int = 3,
    label: Optional[str] = None,
) -> Image.Image:
    """
    Draw a crop bounding box on an image.

    Args:
        image: PIL Image
        crop: (x1, y1, x2, y2) coordinates
        color: Box color
        width: Line width
        label: Optional label text

    Returns:
        Image with drawn box
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    x1, y1, x2, y2 = crop
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if label:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="white", font=font)

    return img_copy


def draw_multiple_crops(
    image: Image.Image,
    crops: List[Tuple[int, int, int, int]],
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    width: int = 2,
) -> Image.Image:
    """
    Draw multiple crop boxes on an image.

    Args:
        image: PIL Image
        crops: List of (x1, y1, x2, y2) coordinates
        colors: List of colors for each crop
        labels: List of labels for each crop
        width: Line width

    Returns:
        Image with drawn boxes
    """
    default_colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]

    if colors is None:
        colors = [default_colors[i % len(default_colors)] for i in range(len(crops))]

    img_copy = image.copy()

    for i, crop in enumerate(crops):
        color = colors[i] if i < len(colors) else "red"
        label = labels[i] if labels and i < len(labels) else None
        img_copy = draw_crop_box(img_copy, crop, color, width, label)

    return img_copy


def create_comparison_figure(
    original: Image.Image,
    crops: List[Tuple[Image.Image, str]],
    title: Optional[str] = None,
) -> Image.Image:
    """
    Create a comparison figure with original and cropped images.

    Args:
        original: Original image
        crops: List of (cropped_image, label) tuples
        title: Optional title

    Returns:
        Comparison figure as PIL Image
    """
    # Calculate dimensions
    n_cols = len(crops) + 1
    max_h = original.height
    total_w = original.width

    for crop_img, _ in crops:
        max_h = max(max_h, crop_img.height)
        total_w += crop_img.width

    # Add spacing
    spacing = 10
    total_w += spacing * (n_cols - 1)

    # Add title space
    title_h = 30 if title else 0
    label_h = 25

    # Create figure
    fig_h = max_h + title_h + label_h
    fig = Image.new("RGB", (total_w, fig_h), "white")
    draw = ImageDraw.Draw(fig)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    if title:
        draw.text((10, 5), title, fill="black", font=title_font)

    # Paste images
    x_offset = 0
    y_offset = title_h

    # Original
    fig.paste(original, (x_offset, y_offset))
    draw.text((x_offset, y_offset + original.height + 5), "Original", fill="black", font=font)
    x_offset += original.width + spacing

    # Crops
    for crop_img, label in crops:
        fig.paste(crop_img, (x_offset, y_offset))
        draw.text((x_offset, y_offset + crop_img.height + 5), label, fill="black", font=font)
        x_offset += crop_img.width + spacing

    return fig


def visualize_iterative_refinement(
    original: Image.Image,
    iterations: List[List[Tuple[int, int, int, int]]],
    scores: List[List[float]],
    final_crop: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Visualize the iterative refinement process.

    Args:
        original: Original image
        iterations: List of crop lists for each iteration
        scores: List of score lists for each iteration
        final_crop: Final selected crop

    Returns:
        Visualization figure
    """
    # Create a grid showing iterations
    n_iters = len(iterations)
    max_crops_per_iter = max(len(crops) for crops in iterations)

    # Calculate dimensions
    cell_w = 200
    cell_h = 200
    spacing = 5

    fig_w = (cell_w + spacing) * max_crops_per_iter + spacing
    fig_h = (cell_h + spacing) * (n_iters + 1) + 50  # +1 for final

    fig = Image.new("RGB", (fig_w, fig_h), "white")
    draw = ImageDraw.Draw(fig)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Draw iterations
    y_offset = 5

    for iter_idx, (iter_crops, iter_scores) in enumerate(zip(iterations, scores)):
        # Draw iteration label
        draw.text((5, y_offset), f"Iter {iter_idx}", fill="black", font=font)
        y_offset += 15

        x_offset = spacing

        for crop_idx, (crop, score) in enumerate(zip(iter_crops, iter_scores)):
            # Extract and resize crop
            x1, y1, x2, y2 = crop
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(original.width, x2)
            y2 = min(original.height, y2)

            if x2 > x1 and y2 > y1:
                crop_img = original.crop((x1, y1, x2, y2))
                crop_img.thumbnail((cell_w, cell_h - 20))
                fig.paste(crop_img, (x_offset, y_offset))

                # Draw score
                draw.text(
                    (x_offset, y_offset + crop_img.height + 2),
                    f"Score: {score:.3f}",
                    fill="black",
                    font=font,
                )

            x_offset += cell_w + spacing

        y_offset += cell_h + spacing

    # Draw final crop
    draw.text((5, y_offset), "Final", fill="green", font=font)
    y_offset += 15

    x1, y1, x2, y2 = final_crop
    if x2 > x1 and y2 > y1:
        final_img = original.crop((x1, y1, x2, y2))
        final_img.thumbnail((cell_w, cell_h - 20))
        fig.paste(final_img, (spacing, y_offset))

        # Draw box around final
        draw.rectangle(
            [spacing - 2, y_offset - 2, spacing + final_img.width + 2, y_offset + final_img.height + 2],
            outline="green",
            width=3,
        )

    return fig


def save_result(
    image: Image.Image,
    output_path: str,
    crop: Optional[Tuple[int, int, int, int]] = None,
    format: str = "JPEG",
):
    """
    Save an image or cropped result.

    Args:
        image: PIL Image
        output_path: Path to save to
        crop: Optional crop coordinates
        format: Image format
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if crop:
        image = image.crop(crop)

    image.save(output_path, format=format)


def create_grid(
    images: List[Image.Image],
    labels: Optional[List[str]] = None,
    cols: int = 4,
    cell_size: int = 256,
) -> Image.Image:
    """
    Create a grid of images.

    Args:
        images: List of PIL Images
        labels: Optional labels for each image
        cols: Number of columns
        cell_size: Size of each cell

    Returns:
        Grid image
    """
    n = len(images)
    rows = (n + cols - 1) // cols

    label_h = 25 if labels else 0
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + label_h)

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x = col * cell_size
        y = row * (cell_size + label_h)

        # Resize image to fit cell
        img_resized = img.copy()
        img_resized.thumbnail((cell_size, cell_size))

        # Center in cell
        paste_x = x + (cell_size - img_resized.width) // 2
        paste_y = y + (cell_size - img_resized.height) // 2

        grid.paste(img_resized, (paste_x, paste_y))

        # Draw label
        if labels and i < len(labels):
            draw.text((x + 5, y + cell_size), labels[i], fill="black", font=font)

    return grid
