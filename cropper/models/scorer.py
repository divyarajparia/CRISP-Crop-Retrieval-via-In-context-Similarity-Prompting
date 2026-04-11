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
    def score(self, image: Image.Image, crop_box: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        Score an image.

        Args:
            image: PIL Image to score
            crop_box: Optional (x1, y1, x2, y2) of this crop in the original
                image's pixel space. Most scorers ignore this; the
                GaicdCalibrationScorer uses it to compute geometry features.

        Returns:
            Score in [0, 1] range
        """
        raise NotImplementedError

    def score_batch(
        self,
        images: List[Image.Image],
        crop_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[float]:
        """Score multiple images."""
        if crop_boxes is None:
            return [self.score(img) for img in images]
        return [self.score(img, box) for img, box in zip(images, crop_boxes)]


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
        """Try to load VILA-R model using JAX/Flax checkpoint."""
        try:
            from pathlib import Path
            import jax
            import jax.numpy as jnp

            logger.info("Attempting to load VILA-R model...")

            # Check for local checkpoint
            local_paths = [
                Path(__file__).parent.parent / "weights" / "vila_rank_tuned",
                Path("/data1/es22btech11013/divya/AFCIL/divya/cv-project/cropper/weights/vila_rank_tuned"),
            ]

            ckpt_dir = None
            for local_path in local_paths:
                if local_path.exists() and (local_path / "checkpoint_0").exists():
                    ckpt_dir = str(local_path)
                    break

            if ckpt_dir is None:
                logger.debug("VILA-R checkpoint not found")
                return False

            logger.info(f"Loading VILA-R from: {ckpt_dir}")

            # Import VILA model components
            from lingvo.core import py_utils
            from paxml import checkpoints
            from paxml import learners
            from paxml import tasks_lib
            from praxis import base_layer
            from praxis import optimizers
            from praxis import pax_fiddle
            from praxis import schedules

            # Import VILA config
            import sys
            vila_path = Path(__file__).parent / "vila"
            sys.path.insert(0, str(vila_path.parent))

            from models.vila import coca_vila
            from models.vila import coca_vila_configs

            NestedMap = py_utils.NestedMap

            # VILA constants
            _IMAGE_SIZE = 224
            _MAX_TEXT_LEN = 64
            _TEXT_VOCAB_SIZE = 64000

            # Build model
            coca_config = coca_vila_configs.CocaVilaConfig()
            coca_config.model_type = coca_vila.CoCaVilaRankBasedFinetune
            coca_config.decoding_max_len = _MAX_TEXT_LEN
            coca_config.text_vocab_size = _TEXT_VOCAB_SIZE
            model_p = coca_vila_configs.build_coca_vila_model(coca_config)
            model_p.model_dims = coca_config.model_dims
            model = model_p.Instantiate()

            # Keep this >=2 to avoid a zero denominator in VILA's
            # contrastive-loss setup during abstract initialization.
            # The official VILA example uses 4.
            dummy_batch_size = 4
            text_shape = (dummy_batch_size, 1, _MAX_TEXT_LEN)
            image_shape = (dummy_batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 3)
            input_specs = NestedMap(
                ids=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.int32),
                image=jax.ShapeDtypeStruct(shape=image_shape, dtype=jnp.float32),
                paddings=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
                labels=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
                regression_labels=jax.ShapeDtypeStruct(
                    shape=(dummy_batch_size, 10), dtype=jnp.float32
                ),
            )
            vars_weight_params = model.abstract_init_with_metadata(input_specs)

            # Learner for initialization
            learner_p = pax_fiddle.Config(learners.Learner)
            learner_p.name = 'learner'
            learner_p.optimizer = pax_fiddle.Config(
                optimizers.ShardedAdafactor,
                decay_method='adam',
                lr_schedule=pax_fiddle.Config(schedules.Constant),
            )
            learner = learner_p.Instantiate()

            # Load checkpoint
            train_state_global_shapes = tasks_lib.create_state_unpadded_shapes(
                vars_weight_params, discard_opt_states=False, learners=[learner]
            )
            model_states = checkpoints.restore_checkpoint(
                train_state_global_shapes, ckpt_dir
            )

            self.vila_model = model
            self.vila_model_states = model_states
            self.vila_image_size = _IMAGE_SIZE
            self.vila_max_text_len = _MAX_TEXT_LEN
            self.scorer_type = "vila"

            logger.info("VILA-R scorer loaded successfully!")
            return True

        except ImportError as e:
            logger.debug(f"VILA dependencies not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Could not load VILA-R: {e}")
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
    def score(self, image: Image.Image, crop_box: Optional[Tuple[int, int, int, int]] = None) -> float:
        """Score image aesthetics."""
        # crop_box is accepted for interface compatibility with BaseScorer; VILA
        # only looks at the cropped pixels.
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
        """Score using VILA-R aesthetic predictor (JAX/Flax version)."""
        try:
            import jax.numpy as jnp
            from praxis import base_layer
            from lingvo.core import py_utils

            NestedMap = py_utils.NestedMap

            # Preprocess image for VILA
            # VILA expects 224x224 RGB images normalized to [0, 1]
            # First resize with aspect-preserving crop
            pre_crop_size = 272
            img = image.resize((pre_crop_size, pre_crop_size), Image.LANCZOS)

            # Center crop to 224x224
            left = (pre_crop_size - self.vila_image_size) // 2
            top = (pre_crop_size - self.vila_image_size) // 2
            right = left + self.vila_image_size
            bottom = top + self.vila_image_size
            img = img.crop((left, top, right, bottom))

            # Convert to array
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.clip(img_array, 0.0, 1.0)

            # Add batch dimension
            img_batch = jnp.expand_dims(img_array, axis=0)

            # Create input batch
            input_batch = NestedMap(
                image=img_batch,
                ids=jnp.zeros((1, 1, self.vila_max_text_len), dtype=jnp.int32),
                paddings=jnp.zeros((1, 1, self.vila_max_text_len), dtype=jnp.int32),
            )

            # Run inference
            context_p = base_layer.JaxContext.HParams(do_eval=True)
            with base_layer.JaxContext(context_p):
                predictions = self.vila_model.apply(
                    {'params': self.vila_model_states.mdl_vars['params']},
                    input_batch,
                    method=self.vila_model.compute_predictions,
                )
                quality_scores = predictions['quality_scores']

            # Extract score (already in [0, 1] range). VILA may return shapes
            # like (1,) or (1, 1), so squeeze before conversion.
            score = float(np.asarray(quality_scores).squeeze())
            score = max(0.0, min(1.0, score))

            return score

        except Exception as e:
            logger.warning(f"VILA-R scoring failed: {e}, falling back to LAION/heuristic")
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
    def score(self, image: Image.Image, crop_box: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        Score content preservation of a crop.

        Args:
            image: Cropped image
            crop_box: Ignored; accepted for BaseScorer interface compatibility.

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

    def score(self, image: Image.Image, crop_box: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        Score based on area ratio.

        Args:
            image: Cropped image
            crop_box: Ignored; accepted for BaseScorer interface compatibility.

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
        if "gaicd_cal" in self.scorers:
            self.scorers["gaicd_cal"].set_original(original_image)

    def score(
        self,
        crop_image: Image.Image,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """
        Compute combined score for a crop.

        Args:
            crop_image: Cropped image
            crop_box: Optional (x1, y1, x2, y2) of this crop in the original
                image's pixel space. Forwarded to component scorers; only
                GaicdCalibrationScorer uses it (for geometry features).

        Returns:
            Combined score in [0, 1]
        """
        total_score = 0.0

        for name, scorer in self.scorers.items():
            weight = self.weights.get(name, 0)
            if weight > 0:
                score = scorer.score(crop_image, crop_box=crop_box)
                total_score += weight * score

        return float(total_score)

    def score_batch(
        self,
        crop_images: List[Image.Image],
        crop_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[float]:
        """Score multiple crops.

        If `crop_boxes` is provided it must be parallel to `crop_images`.
        Component scorers that implement a batched `score_batch` accepting
        boxes (currently just GaicdCalibrationScorer) get one fast call;
        scorers that don't fall back to per-image scoring.
        """
        if not crop_images:
            return []

        # Fast path: any component that can do a true batched forward + uses
        # boxes (gaicd_cal) gets called with score_batch directly. For all
        # other components we still loop per-image. This keeps existing
        # VILA / LAION / Area / CLIP behavior unchanged.
        n = len(crop_images)
        totals = [0.0] * n

        for name, scorer in self.scorers.items():
            weight = self.weights.get(name, 0)
            if weight <= 0:
                continue
            if name == "gaicd_cal" and hasattr(scorer, "score_batch"):
                comp_scores = scorer.score_batch(crop_images, crop_boxes=crop_boxes)
            else:
                if crop_boxes is None:
                    comp_scores = [scorer.score(img) for img in crop_images]
                else:
                    comp_scores = [
                        scorer.score(img, crop_box=box)
                        for img, box in zip(crop_images, crop_boxes)
                    ]
            for i, s in enumerate(comp_scores):
                totals[i] += weight * float(s)

        return [float(t) for t in totals]


def create_scorer(
    task: str = "freeform",
    device: str = "cuda",
    scorer_config: Optional[str] = None,
    require_exact_components: bool = False,
    weights: Optional[Dict[str, float]] = None,
    gaicd_cal_head_path: Optional[str] = None,
) -> CombinedScorer:
    """
    Factory function to create a scorer for a task.

    Args:
        task: Task type ('freeform', 'subject_aware', 'aspect_ratio')
        device: Device to run models on
        scorer_config: Scorer configuration string (e.g., "vila+area",
            "vila+area+gaicd_cal").
        require_exact_components: If True, fail instead of silently falling back
            when a requested scorer backend is unavailable.
        weights: Optional explicit per-component weights, e.g.
            {"vila": 0.3, "area": 0.0, "gaicd_cal": 0.7}. Only keys matching
            components actually built (per scorer_config) are forwarded;
            missing keys fall back to CombinedScorer's task defaults.
        gaicd_cal_head_path: Path to the trained GAICD calibration head
            checkpoint (.pkl). Required when "gaicd_cal" is in scorer_config.

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

    if "gaicd_cal" in config_parts:
        # Lazy import so installs without sklearn/lightgbm/open_clip can still
        # use vila+area scoring.
        from .gaicd_calibration_head import GaicdCalibrationScorer

        if not gaicd_cal_head_path:
            raise ValueError(
                "scorer_config contains 'gaicd_cal' but gaicd_cal_head_path "
                "was not provided. Set scorer.gaicd_cal_head_path in the "
                "config YAML or pass gaicd_cal_head_path= to create_scorer."
            )
        scorers["gaicd_cal"] = GaicdCalibrationScorer(
            head_path=gaicd_cal_head_path,
            device=device,
        )

    logger.info("Scorer components initialized: %s", ", ".join(sorted(scorers.keys())))

    forwarded_weights: Optional[Dict[str, float]] = None
    if weights is not None:
        forwarded_weights = {k: float(v) for k, v in weights.items() if k in scorers}
        logger.info("Scorer weights (pre-normalization): %s", forwarded_weights)

    return CombinedScorer(scorers, weights=forwarded_weights, task=task)
