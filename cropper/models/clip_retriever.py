"""
CLIP-based prompt retrieval for Cropper.
Uses OpenCLIP with ViT-B/32 for image similarity.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CLIPRetriever:
    """
    CLIP-based retriever for in-context learning examples.
    Uses OpenCLIP ViT-B/32 matching the paper.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize CLIP retriever.

        Args:
            model_name: OpenCLIP model name
            pretrained: Pretrained weights to use
            device: Device to run model on
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.model = None
        self.preprocess = None
        self.tokenizer = None

        self._load_model()

        # Database embeddings
        self.database_embeddings = None
        self.database_ids = None

    def _load_model(self):
        """Load OpenCLIP model."""
        try:
            import open_clip

            logger.info(f"Loading OpenCLIP {self.model_name} ({self.pretrained})...")

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            logger.info("CLIP model loaded successfully")

        except ImportError:
            logger.error("OpenCLIP not installed. Install with: pip install open-clip-torch")
            raise
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single image to CLIP embedding.

        Args:
            image: PIL Image

        Returns:
            Normalized embedding vector
        """
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Batch encode multiple images.

        Args:
            images: List of PIL Images
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Array of normalized embeddings [N, D]
        """
        embeddings = []

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images")

        for i in iterator:
            batch = images[i:i + batch_size]

            # Preprocess batch
            batch_tensors = torch.stack([
                self.preprocess(img) for img in batch
            ]).to(self.device)

            # Encode
            batch_embeddings = self.model.encode_image(batch_tensors)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def build_database(
        self,
        dataset,
        cache_path: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        """
        Build embedding database from dataset.

        Args:
            dataset: Dataset with images
            cache_path: Path to save/load cached embeddings
            force_rebuild: Force rebuild even if cache exists
        """
        cache_path = Path(cache_path) if cache_path else None

        # Try to load from cache
        if cache_path and cache_path.exists() and not force_rebuild:
            logger.info(f"Loading embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.database_embeddings = data["embeddings"]
                self.database_ids = data["ids"]
            return

        logger.info(f"Building embedding database for {len(dataset)} images...")

        # Collect images and IDs
        images = []
        ids = []

        for i in tqdm(range(len(dataset)), desc="Loading images"):
            item = dataset[i]
            images.append(item["image"])
            ids.append(item["image_id"])

        # Encode all images
        self.database_embeddings = self.encode_images(images)
        self.database_ids = ids

        # Save to cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "embeddings": self.database_embeddings,
                    "ids": self.database_ids,
                }, f)
            logger.info(f"Saved embeddings to cache: {cache_path}")

    def retrieve_top_s(
        self,
        query_embedding: np.ndarray,
        S: int,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-S most similar images using cosine similarity.

        Args:
            query_embedding: Query image embedding
            S: Number of images to retrieve
            exclude_ids: List of image IDs to exclude (e.g., query image)

        Returns:
            List of (image_id, similarity_score) tuples
        """
        if self.database_embeddings is None:
            raise RuntimeError("Database not built. Call build_database first.")

        # Compute cosine similarities
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        similarities = self.database_embeddings @ query_embedding

        # Get top indices
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            img_id = self.database_ids[idx]

            # Skip excluded IDs
            if exclude_ids and img_id in exclude_ids:
                continue

            results.append((img_id, float(similarities[idx])))

            if len(results) >= S:
                break

        return results

    def select_ground_truth_freeform(
        self,
        dataset,
        retrieved_ids: List[str],
        T: int,
    ) -> List[Dict]:
        """
        Select top-T crops by MOS score for free-form cropping.

        Args:
            dataset: GAICD dataset
            retrieved_ids: List of retrieved image IDs
            T: Number of crops per image

        Returns:
            List of dicts with image_id and crops
        """
        results = []

        for img_id in retrieved_ids:
            crops = dataset.get_top_crops(img_id, T)
            results.append({
                "image_id": img_id,
                "crops": crops,  # List of (mos, x1, y1, x2, y2)
            })

        return results

    def select_ground_truth_subject_aware(
        self,
        dataset,
        retrieved_ids: List[str],
        query_mask_center: Tuple[float, float],
    ) -> List[Dict]:
        """
        Select crop with closest mask center for subject-aware cropping.

        Args:
            dataset: SACD dataset
            retrieved_ids: List of retrieved image IDs
            query_mask_center: (cx, cy) center of query mask

        Returns:
            List of dicts with image_id, mask_center, and crop
        """
        results = []

        for img_id in retrieved_ids:
            subjects = dataset.get_subjects_for_image(img_id)

            if not subjects:
                continue

            # Find subject with closest mask center
            best_subject = None
            best_distance = float("inf")

            for subject in subjects:
                center = subject.get("mask_center", (0.5, 0.5))
                distance = np.sqrt(
                    (center[0] - query_mask_center[0]) ** 2 +
                    (center[1] - query_mask_center[1]) ** 2
                )

                if distance < best_distance:
                    best_distance = distance
                    best_subject = subject

            if best_subject:
                results.append({
                    "image_id": img_id,
                    "mask_center": best_subject.get("mask_center", (0.5, 0.5)),
                    "crop": best_subject["crop"],
                })

        return results

    def select_ground_truth_aspect_ratio(
        self,
        dataset,
        retrieved_ids: List[str],
        target_aspect_ratio: float,
    ) -> List[Dict]:
        """
        Select crop with matching aspect ratio.

        Args:
            dataset: Dataset with aspect ratio annotations
            retrieved_ids: List of retrieved image IDs
            target_aspect_ratio: Target width/height ratio

        Returns:
            List of dicts with image_id and crop
        """
        results = []

        for img_id in retrieved_ids:
            # Get item from dataset
            item = None
            for i in range(len(dataset)):
                if dataset[i]["image_id"] == img_id:
                    item = dataset[i]
                    break

            if item is None:
                continue

            # For FCDB, each image has one crop
            crop = item.get("crop")
            if crop:
                results.append({
                    "image_id": img_id,
                    "crop": crop,
                    "aspect_ratio": item.get("aspect_ratio", 1.0),
                })

        return results


class FAISSRetriever:
    """
    FAISS-accelerated retriever for large datasets.
    """

    def __init__(
        self,
        clip_retriever: CLIPRetriever,
        index_type: str = "IVF",
        nlist: int = 100,
    ):
        """
        Initialize FAISS retriever.

        Args:
            clip_retriever: Base CLIP retriever
            index_type: FAISS index type ('flat', 'IVF')
            nlist: Number of clusters for IVF index
        """
        self.clip_retriever = clip_retriever
        self.index_type = index_type
        self.nlist = nlist
        self.index = None

    def build_index(self, force_rebuild: bool = False):
        """Build FAISS index from database embeddings."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed. Using numpy-based retrieval.")
            return

        if self.clip_retriever.database_embeddings is None:
            raise RuntimeError("Database not built. Build database first.")

        embeddings = self.clip_retriever.database_embeddings.astype("float32")
        d = embeddings.shape[1]

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)

        self.index.add(embeddings)
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def retrieve_top_s(
        self,
        query_embedding: np.ndarray,
        S: int,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Retrieve using FAISS index."""
        if self.index is None:
            # Fall back to numpy retrieval
            return self.clip_retriever.retrieve_top_s(query_embedding, S, exclude_ids)

        query = query_embedding.astype("float32").reshape(1, -1)

        # Retrieve more than S to handle exclusions
        k = min(S + len(exclude_ids or []) + 10, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for i, idx in enumerate(indices[0]):
            img_id = self.clip_retriever.database_ids[idx]

            if exclude_ids and img_id in exclude_ids:
                continue

            results.append((img_id, float(scores[0][i])))

            if len(results) >= S:
                break

        return results
