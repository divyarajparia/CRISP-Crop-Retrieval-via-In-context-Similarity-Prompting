"""
Dataset loaders for GAICD, FCDB, and SACD datasets.
Implements PyTorch-style dataset classes for image cropping tasks.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GAICDDataset(Dataset):
    """
    GAICD Dataset for free-form image cropping.

    Contains 3,336 images:
    - Training: 2,636 images
    - Validation: 200 images
    - Test: 500 images

    Each image has ~90 annotated crops with MOS (mean opinion score).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        cache_embeddings: bool = True,
        embedding_cache_path: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Path to GAICD dataset root
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply to images
            cache_embeddings: Whether to cache CLIP embeddings
            embedding_cache_path: Path to save/load cached embeddings
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.cache_embeddings = cache_embeddings

        # Set up paths - GAICD has split subdirectories
        # Try both flat and split-based structures
        if (self.root_dir / "images" / split).exists():
            # Structure: images/train/, images/test/, images/val/
            self.image_dir = self.root_dir / "images" / split
            self.annotation_dir = self.root_dir / "annotations" / split
        else:
            # Flat structure: images/, annotations/
            self.image_dir = self.root_dir / "images"
            self.annotation_dir = self.root_dir / "annotations"

        self.split_file = self.root_dir / "splits" / f"{split}.txt"

        # Load image list
        self.image_ids = self._load_split()

        # Load all annotations
        self.annotations = self._load_annotations()

        # Embedding cache
        self.embedding_cache_path = embedding_cache_path or str(
            self.root_dir / "cache" / f"clip_embeddings_{split}.pkl"
        )
        self.embeddings = None
        if cache_embeddings and os.path.exists(self.embedding_cache_path):
            self._load_cached_embeddings()

    def _load_split(self) -> List[str]:
        """Load image IDs for the split."""
        if self.split_file.exists():
            with open(self.split_file, "r") as f:
                return [line.strip() for line in f.readlines() if line.strip()]

        # Fallback: load all images from image_dir (which may be split-specific)
        all_images = []
        if self.image_dir.exists():
            all_images = [
                f.stem for f in self.image_dir.glob("*.jpg")
            ] + [
                f.stem for f in self.image_dir.glob("*.png")
            ]

        if all_images:
            return sorted(all_images)

        raise FileNotFoundError(f"Could not find images in: {self.image_dir}")

    def _load_annotations(self) -> Dict[str, List[Tuple[float, int, int, int, int]]]:
        """
        Load crop annotations for all images.

        Returns:
            Dict mapping image_id to list of (MOS, x1, y1, x2, y2) tuples
        """
        annotations = {}

        for img_id in self.image_ids:
            ann_file = self.annotation_dir / f"{img_id}.txt"
            crops = []

            if ann_file.exists():
                with open(ann_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # GAICD format: x1 y1 x2 y2 score
                            x1, y1, x2, y2 = map(int, map(float, parts[0:4]))
                            mos = float(parts[4])
                            crops.append((mos, x1, y1, x2, y2))
            else:
                # Try alternative annotation format (JSON)
                json_file = self.annotation_dir / f"{img_id}.json"
                if json_file.exists():
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        for crop in data.get("crops", []):
                            mos = crop.get("mos", crop.get("score", 0.5))
                            x1 = crop.get("x1", crop.get("left", 0))
                            y1 = crop.get("y1", crop.get("top", 0))
                            x2 = crop.get("x2", crop.get("right", 100))
                            y2 = crop.get("y2", crop.get("bottom", 100))
                            crops.append((mos, x1, y1, x2, y2))

            # Sort by MOS score (highest first)
            crops.sort(key=lambda x: x[0], reverse=True)
            annotations[img_id] = crops

        return annotations

    def _load_cached_embeddings(self):
        """Load cached CLIP embeddings."""
        with open(self.embedding_cache_path, "rb") as f:
            self.embeddings = pickle.load(f)

    def save_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Save CLIP embeddings to cache."""
        os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)
        with open(self.embedding_cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an image and its crop annotations.

        Returns:
            Dict with keys:
                - 'image': PIL Image or transformed image
                - 'image_id': str
                - 'crops': List of (MOS, x1, y1, x2, y2) tuples
                - 'embedding': CLIP embedding (if cached)
        """
        img_id = self.image_ids[idx]

        # Load image
        img_path = self.image_dir / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = self.image_dir / f"{img_id}.png"

        image = Image.open(img_path).convert("RGB")

        # Get crops
        crops = self.annotations.get(img_id, [])

        # Apply transform
        if self.transform:
            image = self.transform(image)

        result = {
            "image": image,
            "image_id": img_id,
            "crops": crops,
            "image_path": str(img_path),
        }

        # Add embedding if cached
        if self.embeddings and img_id in self.embeddings:
            result["embedding"] = self.embeddings[img_id]

        return result

    def get_top_crops(self, img_id: str, T: int = 5) -> List[Tuple[float, int, int, int, int]]:
        """Get top-T crops by MOS score for an image."""
        crops = self.annotations.get(img_id, [])
        return crops[:T]


class FCDBDataset(Dataset):
    """
    FCDB Dataset for free-form and aspect-ratio-aware cropping.

    Contains 348 test images, each with a single user-annotated crop box.
    For aspect-ratio-aware cropping, the aspect ratio of the annotated box
    is used as the target ratio.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        cache_embeddings: bool = True,
        embedding_cache_path: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Path to FCDB dataset root
            transform: Optional transform to apply to images
            cache_embeddings: Whether to cache CLIP embeddings
            embedding_cache_path: Path to save/load cached embeddings
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_embeddings = cache_embeddings

        # Set up paths
        self.image_dir = self.root_dir / "images"
        self.annotation_file = self.root_dir / "cropping_testing_set.json"

        # Load annotations
        self.data = self._load_annotations()

        # Embedding cache
        self.embedding_cache_path = embedding_cache_path or str(
            self.root_dir / "cache" / "clip_embeddings.pkl"
        )
        self.embeddings = None
        if cache_embeddings and os.path.exists(self.embedding_cache_path):
            self._load_cached_embeddings()

    def _load_annotations(self) -> List[Dict]:
        """Load crop annotations from JSON file."""
        data = []

        # Build a map of flickr_photo_id prefix to actual filename
        image_files = {}
        if self.root_dir.exists():
            for img_path in self.root_dir.glob("*.jpg"):
                # Extract the flickr_photo_id (first part before underscore)
                flickr_id = img_path.stem.split("_")[0]
                image_files[flickr_id] = img_path.stem
                image_files[img_path.stem] = img_path.stem  # Also map full name

        if self.annotation_file.exists():
            with open(self.annotation_file, "r") as f:
                annotations = json.load(f)

            for item in annotations:
                # FCDB format: flickr_photo_id and crop as array [x1, y1, x2, y2]
                flickr_id = str(item.get("flickr_photo_id", ""))
                crop = item.get("crop", [0, 0, 100, 100])

                # Try to extract filename from URL
                url = item.get("url", "")
                if url:
                    filename = url.split("/")[-1].replace(".jpg", "")
                    img_id = filename
                else:
                    # Fall back to finding by flickr_id prefix
                    img_id = image_files.get(flickr_id, flickr_id)

                # Handle crop as array [x1, y1, x2, y2] or dict
                if isinstance(crop, list) and len(crop) >= 4:
                    x1, y1, x2, y2 = crop[0], crop[1], crop[2], crop[3]
                elif isinstance(crop, dict):
                    x1 = crop.get("x1", crop.get("left", 0))
                    y1 = crop.get("y1", crop.get("top", 0))
                    x2 = crop.get("x2", crop.get("right", 100))
                    y2 = crop.get("y2", crop.get("bottom", 100))
                else:
                    x1, y1, x2, y2 = 0, 0, 100, 100

                data.append({
                    "image_id": img_id,
                    "flickr_id": flickr_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })
        else:
            # Fallback: try to find images without annotations
            for img_path in self.root_dir.glob("*.jpg"):
                data.append({
                    "image_id": img_path.stem,
                    "x1": 0, "y1": 0, "x2": 100, "y2": 100,
                })

        return data

    def _load_cached_embeddings(self):
        """Load cached CLIP embeddings."""
        with open(self.embedding_cache_path, "rb") as f:
            self.embeddings = pickle.load(f)

    def save_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Save CLIP embeddings to cache."""
        os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)
        with open(self.embedding_cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an image and its crop annotation.

        Returns:
            Dict with keys:
                - 'image': PIL Image or transformed image
                - 'image_id': str
                - 'crop': (x1, y1, x2, y2) tuple
                - 'aspect_ratio': float (width/height of ground-truth crop)
                - 'embedding': CLIP embedding (if cached)
        """
        item = self.data[idx]
        img_id = item["image_id"]

        # Load image - FCDB has images directly in root_dir
        img_path = self.root_dir / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = self.root_dir / f"{img_id}.png"
        if not img_path.exists():
            # Try searching for file starting with flickr_id
            flickr_id = item.get("flickr_id", img_id.split("_")[0])
            for f in self.root_dir.glob(f"{flickr_id}_*.jpg"):
                img_path = f
                break

        image = Image.open(img_path).convert("RGB")

        # Get crop
        x1, y1 = item["x1"], item["y1"]
        x2, y2 = item["x2"], item["y2"]

        # Calculate aspect ratio
        crop_width = x2 - x1
        crop_height = y2 - y1
        aspect_ratio = crop_width / max(crop_height, 1)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        result = {
            "image": image,
            "image_id": img_id,
            "crop": (x1, y1, x2, y2),
            "aspect_ratio": aspect_ratio,
            "image_path": str(img_path),
        }

        # Add embedding if cached
        if self.embeddings and img_id in self.embeddings:
            result["embedding"] = self.embeddings[img_id]

        return result


class SACDDataset(Dataset):
    """
    SACD Dataset for subject-aware image cropping.

    Contains 2,906 images:
    - Training: 2,326 images
    - Validation: 290 images
    - Test: 290 images

    Each image has multiple subject masks and corresponding ground-truth crops.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        cache_embeddings: bool = True,
        embedding_cache_path: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Path to SACD dataset root
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply to images
            cache_embeddings: Whether to cache CLIP embeddings
            embedding_cache_path: Path to save/load cached embeddings
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.cache_embeddings = cache_embeddings

        # Set up paths
        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.annotation_dir = self.root_dir / "annotations"
        self.split_file = self.root_dir / "splits" / f"{split}.txt"

        # Load data
        self.data = self._load_data()

        # Embedding cache
        self.embedding_cache_path = embedding_cache_path or str(
            self.root_dir / "cache" / f"clip_embeddings_{split}.pkl"
        )
        self.embeddings = None
        if cache_embeddings and os.path.exists(self.embedding_cache_path):
            self._load_cached_embeddings()

    def _load_data(self) -> List[Dict]:
        """Load all data including images, masks, and crops."""
        data = []

        # Load image list from split file
        image_ids = []
        if self.split_file.exists():
            with open(self.split_file, "r") as f:
                image_ids = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Fallback: load all images
            if self.image_dir.exists():
                image_ids = [f.stem for f in self.image_dir.glob("*.jpg")]

        for img_id in image_ids:
            # Load annotation
            ann_file = self.annotation_dir / f"{img_id}.json"

            if ann_file.exists():
                with open(ann_file, "r") as f:
                    annotation = json.load(f)

                # Each annotation contains multiple subjects
                subjects = annotation.get("subjects", [annotation])

                for subj_idx, subject in enumerate(subjects):
                    mask_file = subject.get("mask_path", f"{img_id}_mask_{subj_idx}.png")
                    mask_path = self.mask_dir / mask_file

                    crop = subject.get("crop", {})

                    data.append({
                        "image_id": img_id,
                        "subject_idx": subj_idx,
                        "mask_path": str(mask_path),
                        "crop": (
                            crop.get("x1", 0),
                            crop.get("y1", 0),
                            crop.get("x2", 100),
                            crop.get("y2", 100),
                        ),
                        "mask_center": subject.get("mask_center", None),
                    })
            else:
                # Try to find masks directly
                mask_files = list(self.mask_dir.glob(f"{img_id}_mask_*.png"))
                for mask_idx, mask_path in enumerate(mask_files):
                    data.append({
                        "image_id": img_id,
                        "subject_idx": mask_idx,
                        "mask_path": str(mask_path),
                        "crop": (0, 0, 100, 100),  # Placeholder
                        "mask_center": None,
                    })

        return data

    def _load_cached_embeddings(self):
        """Load cached CLIP embeddings."""
        with open(self.embedding_cache_path, "rb") as f:
            self.embeddings = pickle.load(f)

    def save_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Save CLIP embeddings to cache."""
        os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)
        with open(self.embedding_cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        self.embeddings = embeddings

    def _compute_mask_center(self, mask: np.ndarray) -> Tuple[float, float]:
        """Compute the center point of a binary mask."""
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            return 0.5, 0.5  # Center of image if mask is empty

        cx = x_coords.mean() / mask.shape[1]
        cy = y_coords.mean() / mask.shape[0]
        return cx, cy

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an image, mask, and crop annotation.

        Returns:
            Dict with keys:
                - 'image': PIL Image or transformed image
                - 'image_id': str
                - 'mask': PIL Image (binary mask)
                - 'mask_center': (cx, cy) normalized center of mask
                - 'crop': (x1, y1, x2, y2) tuple
                - 'embedding': CLIP embedding (if cached)
        """
        item = self.data[idx]
        img_id = item["image_id"]

        # Load image
        img_path = self.image_dir / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = self.image_dir / f"{img_id}.png"

        image = Image.open(img_path).convert("RGB")

        # Load mask
        mask_path = Path(item["mask_path"])
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask)
        else:
            # Create a default center mask if not found
            mask_array = np.zeros((image.height, image.width), dtype=np.uint8)
            h, w = mask_array.shape
            mask_array[h//4:3*h//4, w//4:3*w//4] = 255
            mask = Image.fromarray(mask_array)

        # Compute mask center
        mask_center = item.get("mask_center")
        if mask_center is None:
            mask_center = self._compute_mask_center(mask_array)

        # Get crop
        crop = item["crop"]

        # Apply transform
        original_image = image
        if self.transform:
            image = self.transform(image)

        result = {
            "image": image,
            "original_image": original_image,
            "image_id": img_id,
            "subject_idx": item["subject_idx"],
            "mask": mask,
            "mask_center": mask_center,
            "crop": crop,
            "image_path": str(img_path),
        }

        # Add embedding if cached
        if self.embeddings and img_id in self.embeddings:
            result["embedding"] = self.embeddings[img_id]

        return result

    def get_subjects_for_image(self, img_id: str) -> List[Dict]:
        """Get all subjects (masks and crops) for a specific image."""
        return [item for item in self.data if item["image_id"] == img_id]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 4,
    collate_fn=None,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for a dataset."""

    def default_collate(batch):
        """Custom collate function that handles PIL Images."""
        return batch  # Return list of dicts

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or default_collate,
    )
