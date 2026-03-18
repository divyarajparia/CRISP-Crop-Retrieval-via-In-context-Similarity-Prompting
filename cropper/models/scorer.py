"""
Scorers for crop evaluation.
Implements VILA aesthetic scorer, CLIP content scorer, and area scorer.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BaseScorer(ABC):
    """Abstract base class for scorers."""

    @abstractmethod
    def score(self, image: Image.Image) -> float:
        """
        Score an image.

        Args:
            image: PIL Image to score

        Returns:
            Score in [0, 1] range
        """
        raise NotImplementedError

    def score_batch(self, images: List[Image.Image]) -> List[float]:
        """Score multiple images."""
        return [self.score(img) for img in images]


class VILAScorer(BaseScorer):
    """
    VILA-R aesthetic scorer.
    Falls back to NIMA or LAION aesthetic predictor if VILA-R is unavailable.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        require_vila: bool = False,
    ):
        """
        Initialize VILA scorer.

        Args:
            device: Device to run model on
            model_path: Path to VILA model weights (optional)
        """
        self.device = device
        self.model_path = model_path
        self.require_vila = require_vila
        self.model = None
        self.transform = None
        self.scorer_type = None

        self._load_model()

    def _load_model(self):
        """Try to load aesthetic scorers in order of preference."""
        logger.info("Initializing VILA scorer (require_vila=%s)", self.require_vila)
        # Try VILA-R first (best if available)
        if self._try_load_vila():
            logger.info("Aesthetic scorer backend selected: VILA-R")
            return

        if self.require_vila:
            raise RuntimeError(
                "VILA-R scorer is required for this run, but it could not be loaded. "
                "Install the required VILA dependencies and verify the VILA weights are available."
            )

        # Try LAION aesthetic predictor (we have pretrained weights)
        if self._try_load_laion():
            logger.warning("Aesthetic scorer backend selected: LAION fallback")
            return

        # Try NIMA (fallback, but has random head)
        if self._try_load_nima():
            logger.warning("Aesthetic scorer backend selected: NIMA fallback")
            return

        # Fall back to simple scoring
        logger.warning("No aesthetic scorer available. Using simple heuristics.")
        self.scorer_type = "heuristic"

    def _try_load_vila(self) -> bool:
        """Try to load VILA model using TensorFlow Hub."""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            logger.info("Attempting to load VILA model from TensorFlow Hub...")

            # Load VILA model from TFHub
            model_handle = 'https://tfhub.dev/google/vila/image/1'
            self.vila_model = hub.load(model_handle)
            self.vila_predict_fn = self.vila_model.signatures['serving_default']
            self.scorer_type = "vila"

            logger.info("VILA scorer loaded successfully from TensorFlow Hub!")
            return True

        except ImportError as e:
            logger.debug(f"TensorFlow/TFHub not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Could not load VILA from TFHub: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _try_load_nima(self) -> bool:
        """Try to load NIMA model."""
        try:
            from torchvision import models, transforms

            logger.info("Loading NIMA-based aesthetic scorer...")

            # Use a pretrained model as base
            # NIMA typically uses VGG16 or MobileNet
            self.model = models.mobilenet_v2(pretrained=True)

            # Modify for aesthetic scoring (1-10 scale)
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(self.model.last_channel, 10),
                torch.nn.Softmax(dim=1),
            )

            # Load pretrained NIMA weights if available
            # For now, use the pretrained backbone
            self.model = self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self.scorer_type = "nima"
            logger.info("NIMA scorer loaded successfully")
            return True

        except Exception as e:
            logger.debug(f"Could not load NIMA: {e}")
            return False

    def _try_load_laion(self) -> bool:
        """Try to load LAION aesthetic predictor with pretrained weights."""
        try:
            import open_clip
            from pathlib import Path

            logger.info("Loading LAION aesthetic predictor...")

            # LAION aesthetic predictor uses CLIP ViT-L/14
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained="openai",  # Use OpenAI weights for compatibility
            )

            self.model = model.to(self.device)
            self.model.eval()
            self.transform = preprocess

            # LAION aesthetic predictor MLP architecture (matches pretrained weights)
            # Input: 768 (CLIP ViT-L/14 embedding dim), Output: 1 (aesthetic score)
            # Architecture from: https://github.com/christophschuhmann/improved-aesthetic-predictor
            self.aesthetic_head = torch.nn.Sequential(
                torch.nn.Linear(768, 1024),   # layers.0
                torch.nn.ReLU(),              # layers.1 (no weights)
                torch.nn.Linear(1024, 128),   # layers.2
                torch.nn.ReLU(),              # layers.3 (no weights)
                torch.nn.Linear(128, 64),     # layers.4
                torch.nn.ReLU(),              # layers.5 (no weights)
                torch.nn.Linear(64, 16),      # layers.6
                torch.nn.Linear(16, 1),       # layers.7
            ).to(self.device)

            # Try to load pretrained weights
            weights_paths = [
                Path(__file__).parent.parent / "weights" / "aesthetic_predictor_v2.pth",
                Path("weights/aesthetic_predictor_v2.pth"),
                Path("/data1/es22btech11013/divya/AFCIL/divya/cv-project/cropper/weights/aesthetic_predictor_v2.pth"),
            ]

            weights_loaded = False
            for weights_path in weights_paths:
                if weights_path.exists():
                    logger.info(f"Loading aesthetic predictor weights from {weights_path}")
                    state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                    # Remap keys: "layers.X.weight" -> "X.weight" (for Sequential)
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = k.replace("layers.", "")
                        new_state_dict[new_key] = v
                    self.aesthetic_head.load_state_dict(new_state_dict)
                    weights_loaded = True
                    logger.info("Aesthetic predictor weights loaded successfully!")
                    break

            if not weights_loaded:
                logger.warning("Could not find aesthetic predictor weights. Skipping LAION scorer.")
                return False

            self.aesthetic_head.eval()
            self.scorer_type = "laion"
            logger.info(f"LAION aesthetic scorer loaded with pretrained weights")
            return True

        except Exception as e:
            logger.warning(f"Could not load LAION aesthetic predictor: {e}")
            return False

    @torch.no_grad()
    def score(self, image: Image.Image) -> float:
        """Score image aesthetics."""
        if self.scorer_type == "vila":
            return self._vila_score(image)

        elif self.scorer_type == "heuristic":
            return self._heuristic_score(image)

        elif self.scorer_type == "nima":
            return self._nima_score(image)

        elif self.scorer_type == "laion":
            return self._laion_score(image)

        return 0.5  # Default

    def _heuristic_score(self, image: Image.Image) -> float:
        """Simple heuristic-based aesthetic scoring."""
        # Consider various factors
        img_array = np.array(image)

        # Size factor - prefer reasonable sizes
        h, w = img_array.shape[:2]
        size_score = min(1.0, (h * w) / (1000 * 1000))

        # Contrast factor
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        contrast = gray.std() / 128.0
        contrast_score = min(1.0, contrast)

        # Saturation factor (prefer colorful images)
        if len(img_array.shape) == 3:
            sat = np.std(img_array, axis=2).mean() / 128.0
            saturation_score = min(1.0, sat)
        else:
            saturation_score = 0.5

        # Combined score
        score = (size_score + contrast_score + saturation_score) / 3.0
        return float(score)

    def _vila_score(self, image: Image.Image) -> float:
        """Score using VILA aesthetic predictor (TensorFlow Hub version)."""
        try:
            import tensorflow as tf
            from io import BytesIO

            # Convert PIL Image to bytes (VILA TFHub expects JPEG/PNG bytes)
            buffer = BytesIO()
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=95)
            image_bytes = buffer.getvalue()

            # Run inference
            prediction = self.vila_predict_fn(tf.constant(image_bytes))
            quality_score = prediction['predictions']

            # Extract score (already in [0, 1] range)
            score = float(quality_score.numpy().squeeze())
            score = max(0.0, min(1.0, score))

            return score

        except Exception as e:
            logger.warning(f"VILA scoring failed: {e}, falling back to LAION/heuristic")
            # Try LAION fallback if available
            if hasattr(self, 'aesthetic_head') and self.aesthetic_head is not None:
                return self._laion_score(image)
            return self._heuristic_score(image)

    def _nima_score(self, image: Image.Image) -> float:
        """Score using NIMA."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        # NIMA outputs distribution over 1-10
        # Compute expected value and normalize to [0, 1]
        scores = torch.arange(1, 11, dtype=torch.float32, device=self.device)
        expected = (output * scores).sum().item()
        normalized = (expected - 1) / 9.0

        return float(normalized)

    def _laion_score(self, image: Image.Image) -> float:
        """Score using LAION aesthetic predictor."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            # LAION predictor outputs ~1-10 scale, normalize to 0-1
            raw_score = self.aesthetic_head(features.float()).item()
            # Normalize: assume scores range from 1-10, map to 0-1
            score = (raw_score - 1.0) / 9.0
            score = max(0.0, min(1.0, score))

        return float(score)


class CLIPContentScorer(BaseScorer):
    """
    CLIP-based content preservation scorer.
    Measures cosine similarity between cropped and original image embeddings.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        """
        Initialize CLIP content scorer.

        Args:
            model_name: OpenCLIP model name
            pretrained: Pretrained weights
            device: Device to run model on
        """
        self.device = device
        self._load_model(model_name, pretrained)
        self.original_embedding = None

    def _load_model(self, model_name: str, pretrained: str):
        """Load CLIP model."""
        try:
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.model = self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Could not load CLIP model: {e}")
            self.model = None
            self.preprocess = None

    def set_original(self, original_image: Image.Image):
        """Set the original image for comparison."""
        if self.model is None:
            self.original_embedding = None
            return

        with torch.no_grad():
            img_tensor = self.preprocess(original_image).unsqueeze(0).to(self.device)
            self.original_embedding = self.model.encode_image(img_tensor)
            self.original_embedding = self.original_embedding / self.original_embedding.norm()

    @torch.no_grad()
    def score(self, image: Image.Image) -> float:
        """
        Score content preservation of a crop.

        Args:
            image: Cropped image

        Returns:
            Cosine similarity to original [0, 1]
        """
        if self.model is None or self.original_embedding is None:
            return 0.5  # Default

        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        crop_embedding = self.model.encode_image(img_tensor)
        crop_embedding = crop_embedding / crop_embedding.norm()

        similarity = (self.original_embedding @ crop_embedding.T).item()

        # Cosine similarity is in [-1, 1], map to [0, 1]
        return float((similarity + 1) / 2)


class AreaScorer(BaseScorer):
    """
    Area-based scorer.
    Measures the ratio of crop area to original image area.
    """

    def __init__(self, original_size: Optional[Tuple[int, int]] = None):
        """
        Initialize area scorer.

        Args:
            original_size: (width, height) of original image
        """
        self.original_size = original_size

    def set_original_size(self, size: Tuple[int, int]):
        """Set original image size."""
        self.original_size = size

    def score(self, image: Image.Image) -> float:
        """
        Score based on area ratio.

        Args:
            image: Cropped image

        Returns:
            Area ratio in [0, 1]
        """
        crop_area = image.width * image.height

        if self.original_size:
            orig_area = self.original_size[0] * self.original_size[1]
            return min(1.0, crop_area / orig_area)
        else:
            # Heuristic: prefer crops that are at least 10% of a typical image
            return min(1.0, crop_area / (1000 * 1000))

    def score_from_coords(
        self,
        crop: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
    ) -> float:
        """
        Score from crop coordinates.

        Args:
            crop: (x1, y1, x2, y2) coordinates
            image_size: (width, height) of original image

        Returns:
            Area ratio in [0, 1]
        """
        x1, y1, x2, y2 = crop
        crop_area = (x2 - x1) * (y2 - y1)
        orig_area = image_size[0] * image_size[1]

        return min(1.0, max(0.0, crop_area / orig_area))


class CombinedScorer:
    """
    Combined scorer that aggregates multiple scoring methods.
    """

    def __init__(
        self,
        scorers: Dict[str, BaseScorer],
        weights: Optional[Dict[str, float]] = None,
        task: str = "freeform",
    ):
        """
        Initialize combined scorer.

        Args:
            scorers: Dict of scorer name to scorer instance
            weights: Dict of scorer name to weight
            task: Task type for default weight selection
        """
        self.scorers = scorers
        self.task = task

        # Default weights based on task (from paper)
        if weights is None:
            if task == "freeform":
                # VILA + Area (Table 5)
                weights = {"vila": 1.0, "area": 1.0}
            elif task == "subject_aware":
                # VILA + Area (Table 14)
                weights = {"vila": 1.0, "area": 1.0}
            elif task == "aspect_ratio":
                # CLIP only (Table 16)
                weights = {"clip": 1.0}
            else:
                weights = {k: 1.0 for k in scorers.keys()}

        self.weights = weights

        # Normalize weights
        total = sum(self.weights.get(k, 0) for k in self.scorers.keys())
        if total > 0:
            self.weights = {k: self.weights.get(k, 0) / total for k in self.scorers.keys()}

    def set_original(self, original_image: Image.Image):
        """Set original image for all relevant scorers."""
        if "clip" in self.scorers:
            self.scorers["clip"].set_original(original_image)
        if "area" in self.scorers:
            self.scorers["area"].set_original_size(original_image.size)

    def score(self, crop_image: Image.Image) -> float:
        """
        Compute combined score for a crop.

        Args:
            crop_image: Cropped image

        Returns:
            Combined score in [0, 1]
        """
        total_score = 0.0

        for name, scorer in self.scorers.items():
            weight = self.weights.get(name, 0)
            if weight > 0:
                score = scorer.score(crop_image)
                total_score += weight * score

        return float(total_score)

    def score_batch(
        self,
        crop_images: List[Image.Image],
    ) -> List[float]:
        """Score multiple crops."""
        return [self.score(img) for img in crop_images]


def create_scorer(
    task: str = "freeform",
    device: str = "cuda",
    scorer_config: Optional[str] = None,
    require_exact_components: bool = False,
) -> CombinedScorer:
    """
    Factory function to create a scorer for a task.

    Args:
        task: Task type ('freeform', 'subject_aware', 'aspect_ratio')
        device: Device to run models on
        scorer_config: Scorer configuration string (e.g., "vila+area")
        require_exact_components: If True, fail instead of silently falling back
            when a requested scorer backend is unavailable.

    Returns:
        Combined scorer instance
    """
    scorers = {}

    # Parse config or use task defaults
    if scorer_config is None:
        if task in ["freeform", "subject_aware"]:
            scorer_config = "vila+area"
        else:
            scorer_config = "clip"

    config_parts = scorer_config.lower().split("+")
    logger.info(
        "Creating scorer for task=%s with config=%s (require_exact_components=%s)",
        task,
        scorer_config,
        require_exact_components,
    )

    if "vila" in config_parts:
        scorers["vila"] = VILAScorer(
            device=device,
            require_vila=require_exact_components,
        )

    if "clip" in config_parts:
        scorers["clip"] = CLIPContentScorer(device=device)

    if "area" in config_parts:
        scorers["area"] = AreaScorer()

    logger.info("Scorer components initialized: %s", ", ".join(sorted(scorers.keys())))

    return CombinedScorer(scorers, task=task)
