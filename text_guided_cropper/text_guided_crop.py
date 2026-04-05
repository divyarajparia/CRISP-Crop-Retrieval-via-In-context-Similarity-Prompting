"""
Text-Guided Cropping (CRISP Task 2)
------------------------------------
Extends the Cropper pipeline with natural language intent for image cropping.

Instead of finding a single "best" aesthetic crop, the user provides a text
prompt (e.g. "focus on the person on the left", "wide cinematic crop",
"crop for Instagram square post") and the system:

  1. Encodes the text prompt with CLIP to build a text-conditioned query.
  2. Retrieves ICL examples from a database whose crops visually match the
     text-guided intent (CLIP image-similarity + text-crop re-ranking).
  3. Builds a VLM prompt that includes the user's text instruction alongside
     the retrieved example image-crop pairs.
  4. Calls Mantis-8B-Idefics2 (or any Idefics-compatible VLM) to propose R
     candidate crop boxes.
  5. Scores each candidate with a composite scorer (aesthetic proxy + CLIP
     text-image alignment + area preservation).
  6. Iteratively refines (L rounds) by feeding scores back to the VLM.
  7. Returns the best crop together with its score and bounding box.

References
----------
- Cropper (CVPR 2025) - Lee et al., Google DeepMind
- ClipCrop (ICCV workshop) - Zhong et al.
- CRISP Presentation v3 - Rajparia, Virani, Neeli
"""

from __future__ import annotations

import io
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CropBox:
    """A predicted crop bounding box with its predicted quality score."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0            # VLM-predicted MOS / confidence
    composite_score: float = 0.0  # after external scoring

    def to_pixel(
        self,
        width: int,
        height: int,
        coord_range: Tuple[float, float] = (1.0, 1000.0),
    ) -> "CropBox":
        """Convert normalised coordinates to pixel coordinates."""
        lo, hi = coord_range
        span = hi - lo
        x1 = int(round((self.x1 - lo) / span * width))
        y1 = int(round((self.y1 - lo) / span * height))
        x2 = int(round((self.x2 - lo) / span * width))
        y2 = int(round((self.y2 - lo) / span * height))
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return CropBox(x1, y1, x2, y2, self.score, self.composite_score)

    def area_ratio(self, width: int, height: int) -> float:
        w = max(0, self.x2 - self.x1)
        h = max(0, self.y2 - self.y1)
        return (w * h) / max(1, width * height)

    def apply(self, image: Image.Image) -> Image.Image:
        return image.crop((int(self.x1), int(self.y1),
                           int(self.x2), int(self.y2)))


@dataclass
class ICLExample:
    """One in-context learning example: image + its best crop box."""
    image: Image.Image
    crop: CropBox         # normalised coords
    image_path: Optional[str] = None


@dataclass
class TextGuidedCropResult:
    """Full result returned by TextGuidedCropper.crop()."""
    cropped_image: Image.Image
    crop_box: CropBox             # pixel coords
    composite_score: float
    all_candidates: List[CropBox] = field(default_factory=list)
    num_refinement_iters: int = 0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class CompositeScorer:
    """
    Scores a cropped region with three components:

    1. Aesthetic score  - lightweight proxy (aspect-ratio + centre-bias) or
                          VILA-R when available.
    2. Text-alignment   - CLIP cosine similarity between the crop's image
                          embedding and the user's text embedding.
                          THIS is the key Task-2 addition.
    3. Area preservation- ratio of crop area to original image area.

    Weights are configurable; defaults favour text alignment (0.5).
    """

    def __init__(
        self,
        clip_model=None,
        clip_preprocess=None,
        device: str = "cpu",
        vila_weight: float = 0.3,
        text_weight: float = 0.5,
        area_weight: float = 0.2,
    ):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.vila_weight = vila_weight
        self.text_weight = text_weight
        self.area_weight = area_weight
        self._text_features: Optional[torch.Tensor] = None

    def set_text_prompt(self, text_prompt: str) -> None:
        """Pre-encode the text prompt for fast per-crop scoring."""
        if self.clip_model is None:
            return
        import open_clip
        with torch.no_grad():
            tok = open_clip.tokenize([text_prompt]).to(self.device)
            feats = self.clip_model.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self._text_features = feats  # (1, D)

    def _aesthetic_proxy(self, crop: Image.Image) -> float:
        """
        Lightweight aesthetic proxy when VILA-R is unavailable.
        Prefers crops with aspect ratio near the golden ratio and
        a minimum meaningful size.
        """
        w, h = crop.size
        if w == 0 or h == 0:
            return 0.0
        aspect = w / h
        golden = 1.618
        aspect_score = 1.0 - min(abs(aspect - golden) / golden, 1.0)
        area_score = min(w * h / (320 * 320), 1.0)
        return 0.5 * aspect_score + 0.5 * area_score

    def _clip_text_score(self, crop: Image.Image) -> float:
        """CLIP cosine similarity between crop and text prompt in [0, 1]."""
        if self.clip_model is None or self._text_features is None:
            return 0.5
        try:
            img_t = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feats = self.clip_model.encode_image(img_t)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sim = (img_feats @ self._text_features.T).item()
            return (sim + 1.0) / 2.0   # [-1,1] -> [0,1]
        except Exception:
            return 0.5

    def score(
        self,
        crop_img: Image.Image,
        orig_width: int,
        orig_height: int,
        crop_box: CropBox,
    ) -> float:
        """Return composite score in [0, 1]."""
        aesthetic  = self._aesthetic_proxy(crop_img)
        text_sim   = self._clip_text_score(crop_img)
        area       = crop_box.area_ratio(orig_width, orig_height)
        # Penalise crops that are too small (<5%) or too large (>95%)
        area_score = 1.0 - abs(area - 0.5) * 2.0
        area_score = max(0.0, area_score)

        total = (self.vila_weight * aesthetic
                 + self.text_weight * text_sim
                 + self.area_weight * area_score)
        return float(np.clip(total, 0.0, 1.0))


# ---------------------------------------------------------------------------
# VLM interface
# ---------------------------------------------------------------------------

class VLMCropGenerator:
    """
    Wraps a Mantis-8B-Idefics2 (or any HuggingFace Idefics2-compatible) model
    for generating crop coordinates from interleaved image-text prompts.

    When ``model`` is None the class falls back to a heuristic generator so
    the entire pipeline can be exercised without GPU resources.
    """

    COORD_PATTERN = re.compile(
        r"\(?\s*"
        r"([\d.]+)\s*,\s*"
        r"([\d.]+)\s*,\s*"
        r"([\d.]+)\s*,\s*"
        r"([\d.]+)\s*"
        r"(?:,\s*([\d.]+))?"
        r"\s*\)?"
    )

    def __init__(
        self,
        model=None,
        processor=None,
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.05,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    def _parse_crops(
        self, text: str, n_expected: int, has_score: bool = True
    ) -> List[CropBox]:
        matches = self.COORD_PATTERN.findall(text)
        boxes: List[CropBox] = []
        for m in matches:
            nums = [float(v) for v in m if v != ""]
            if has_score and len(nums) == 5:
                s, x1, y1, x2, y2 = nums
            elif len(nums) == 4:
                x1, y1, x2, y2 = nums
                s = 0.75
            else:
                continue
            if x2 > x1 and y2 > y1:
                boxes.append(CropBox(x1, y1, x2, y2, score=s))
        return boxes[:n_expected]

    # ------------------------------------------------------------------
    def _heuristic_crops(
        self,
        image: Image.Image,
        text_prompt: str,
        n: int,
        coord_range: Tuple[float, float],
    ) -> List[CropBox]:
        """
        Generate n diverse crop proposals without a VLM.
        Uses keyword-based heuristics + slight scale/position variation.
        """
        lo, hi = coord_range
        span = hi - lo
        rng = np.random.default_rng(hash(text_prompt) % (2 ** 32))

        # Centre-of-mass heuristic from text keywords
        cx, cy = 0.5, 0.5
        p = text_prompt.lower()
        if "left"   in p: cx = 0.35
        if "right"  in p: cx = 0.65
        if "top"    in p: cy = 0.35
        if "bottom" in p: cy = 0.65

        # Aspect-ratio target from common keywords
        W, H = image.size
        aspect = W / H
        ar_map = {
            "square": 1.0,  "1:1": 1.0,
            "16:9": 16 / 9, "9:16": 9 / 16,
            "4:3": 4 / 3,   "3:4": 3 / 4,
            "2:1": 2.0,     "cinematic": 2.39,
            "portrait": 0.75, "landscape": 1.5,
        }
        for key, ar in ar_map.items():
            if key in p:
                aspect = ar
                break

        crops: List[CropBox] = []
        for scale in np.linspace(0.55, 0.90, n):
            if aspect >= 1.0:
                cw = scale
                ch = cw / aspect * (W / H)
            else:
                ch = scale
                cw = ch * aspect * (H / W)
            cw = float(np.clip(cw, 0.1, 1.0))
            ch = float(np.clip(ch, 0.1, 1.0))
            x1 = float(np.clip(cx - cw / 2 + rng.uniform(-0.04, 0.04),
                                0.0, 1.0 - cw))
            y1 = float(np.clip(cy - ch / 2 + rng.uniform(-0.04, 0.04),
                                0.0, 1.0 - ch))
            x2 = x1 + cw
            y2 = y1 + ch
            crops.append(CropBox(
                x1=lo + x1 * span,
                y1=lo + y1 * span,
                x2=lo + x2 * span,
                y2=lo + y2 * span,
                score=0.75,
            ))
        return crops

    # ------------------------------------------------------------------
    def _build_initial_prompt(
        self,
        query_image: Image.Image,
        text_prompt: str,
        icl_examples: List[ICLExample],
        n_crops: int,
        coord_range: Tuple[float, float],
        has_score: bool,
    ) -> Tuple[List[Dict], List[Any]]:
        """Build the interleaved message list for Idefics2-style VLMs."""
        lo, hi = coord_range
        images_list: List[Image.Image] = []

        coord_desc = (
            f"  x1, x2 = left/right positions normalised to [{lo:.0f}, {hi:.0f}]\n"
            f"  y1, y2 = top/bottom positions normalised to [{lo:.0f}, {hi:.0f}]\n"
        ) if hi > 2 else (
            f"  x1, x2 = left/right positions in [0, 1]\n"
            f"  y1, y2 = top/bottom positions in [0, 1]\n"
        )

        system_text = (
            "You are an expert image composition assistant. "
            "Crop images according to the user's explicit intent.\n"
            f'User intent: "{text_prompt}"\n\n'
            "Each crop region is represented as "
            + ("(s, x1, y1, x2, y2)" if has_score else "(x1, y1, x2, y2)")
            + " where:\n"
            + coord_desc
            + ("  s = MOS quality score in [0, 1]\n" if has_score else "")
            + "Example images and their best crops follow:\n"
        )

        content: List[Dict] = [{"type": "text", "text": system_text}]

        for ex in icl_examples:
            thumb = ex.image.copy()
            thumb.thumbnail((512, 512))
            images_list.append(thumb)
            content.append({"type": "image"})
            c = ex.crop
            if has_score:
                coord_str = (f"({c.score:.2f}, {c.x1:.0f}, {c.y1:.0f}, "
                             f"{c.x2:.0f}, {c.y2:.0f})")
            else:
                coord_str = (f"({c.x1:.4f}, {c.y1:.4f}, "
                             f"{c.x2:.4f}, {c.y2:.4f})")
            content.append({"type": "text", "text": coord_str + "\n"})

        query_thumb = query_image.copy()
        query_thumb.thumbnail((512, 512))
        images_list.append(query_thumb)
        content.append({"type": "image"})
        content.append({
            "type": "text",
            "text": (
                f'\nFor the user intent "{text_prompt}", '
                f"propose {n_crops} diverse crop candidates, "
                "best first, one per line."
            ),
        })

        messages = [{"role": "user", "content": content}]
        return messages, images_list

    # ------------------------------------------------------------------
    def _build_refinement_prompt(
        self,
        previous_messages: List[Dict],
        previous_images: List[Any],
        scored_candidates: List[CropBox],
        cropped_images: List[Image.Image],
        text_prompt: str,
        has_score: bool,
    ) -> Tuple[List[Dict], List[Any]]:
        """Append scored candidates and ask the VLM for an improved crop."""
        new_images = list(previous_images)
        new_content: List[Dict] = [
            {"type": "text",
             "text": "Previous crop candidates and their composite scores:\n"}
        ]
        for cb, ci in zip(scored_candidates, cropped_images):
            thumb = ci.copy()
            thumb.thumbnail((256, 256))
            new_images.append(thumb)
            new_content.append({"type": "image"})
            if has_score:
                coord_str = (f"({cb.x1:.0f}, {cb.y1:.0f}, "
                             f"{cb.x2:.0f}, {cb.y2:.0f})")
            else:
                coord_str = (f"({cb.x1:.4f}, {cb.y1:.4f}, "
                             f"{cb.x2:.4f}, {cb.y2:.4f})")
            new_content.append({
                "type": "text",
                "text": f"{coord_str}  Score: {cb.composite_score:.3f}\n",
            })

        new_content.append({
            "type": "text",
            "text": (
                f'Considering the user intent "{text_prompt}", '
                "propose a single improved crop that scores higher. "
                "Output one "
                + ("(s, x1, y1, x2, y2)" if has_score else "(x1, y1, x2, y2)")
                + " tuple."
            ),
        })

        new_messages = list(previous_messages) + [
            {"role": "user", "content": new_content}
        ]
        return new_messages, new_images

    # ------------------------------------------------------------------
    def generate(
        self,
        query_image: Image.Image,
        text_prompt: str,
        icl_examples: List[ICLExample],
        n_crops: int,
        coord_range: Tuple[float, float],
        has_score: bool = True,
    ) -> List[CropBox]:
        if self.model is None or self.processor is None:
            return self._heuristic_crops(
                query_image, text_prompt, n_crops, coord_range
            )

        messages, images_list = self._build_initial_prompt(
            query_image, text_prompt, icl_examples,
            n_crops, coord_range, has_score
        )
        try:
            prompt_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=prompt_text,
                images=images_list or None,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=(self.temperature > 0.01),
                    temperature=max(self.temperature, 0.01),
                )
            out_text = self.processor.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            crops = self._parse_crops(out_text, n_crops, has_score)
            if len(crops) < n_crops:
                crops += self._heuristic_crops(
                    query_image, text_prompt,
                    n_crops - len(crops), coord_range
                )
            return crops[:n_crops]
        except Exception as exc:
            logger.warning(
                f"VLM generation failed ({exc}); using heuristic fallback."
            )
            return self._heuristic_crops(
                query_image, text_prompt, n_crops, coord_range
            )

    # ------------------------------------------------------------------
    def refine(
        self,
        query_image: Image.Image,
        text_prompt: str,
        messages: List[Dict],
        images: List[Any],
        scored_candidates: List[CropBox],
        cropped_images: List[Image.Image],
        coord_range: Tuple[float, float],
        has_score: bool = True,
    ) -> CropBox:
        if self.model is None or self.processor is None:
            best = max(scored_candidates, key=lambda c: c.composite_score)
            lo, hi = coord_range
            eps = (hi - lo) * 0.015
            rng = np.random.default_rng()
            return CropBox(
                x1=float(np.clip(best.x1 + rng.uniform(-eps, eps), lo, hi)),
                y1=float(np.clip(best.y1 + rng.uniform(-eps, eps), lo, hi)),
                x2=float(np.clip(best.x2 + rng.uniform(-eps, eps), lo, hi)),
                y2=float(np.clip(best.y2 + rng.uniform(-eps, eps), lo, hi)),
                score=best.score,
            )

        new_messages, new_images = self._build_refinement_prompt(
            messages, images, scored_candidates, cropped_images,
            text_prompt, has_score
        )
        try:
            prompt_text = self.processor.apply_chat_template(
                new_messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=prompt_text,
                images=new_images or None,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )
            out_text = self.processor.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            crops = self._parse_crops(out_text, 1, has_score)
            if crops:
                return crops[0]
        except Exception as exc:
            logger.warning(
                f"VLM refinement failed ({exc}); returning best candidate."
            )
        return max(scored_candidates, key=lambda c: c.composite_score)


# ---------------------------------------------------------------------------
# ICL Retriever
# ---------------------------------------------------------------------------

class ICLRetriever:
    """
    Retrieves in-context learning examples from a database of images with
    annotated crops.

    Text-guided extension of the Cropper retrieval strategy:

      combined_score_i = (1 - w) * image_sim(query_img, db_img_i)
                       +       w * text_sim(text_prompt, crop_embed_i)

    where crop_embed_i is the CLIP image embedding of the *cropped* region
    of db_img_i.  This biases retrieval toward examples whose crops visually
    match the user's language intent.
    """

    def __init__(
        self,
        clip_model=None,
        clip_preprocess=None,
        device: str = "cpu",
    ):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

        self._db_images: List[Image.Image] = []
        self._db_crops: List[CropBox] = []
        self._db_paths: List[str] = []
        self._db_img_feats: Optional[torch.Tensor] = None
        self._db_crop_feats: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def add_to_database(
        self,
        images: List[Image.Image],
        crops: List[CropBox],
        paths: Optional[List[str]] = None,
    ) -> None:
        self._db_images.extend(images)
        self._db_crops.extend(crops)
        self._db_paths.extend(paths or [""] * len(images))
        self._db_img_feats = None
        self._db_crop_feats = None

    # ------------------------------------------------------------------
    def _encode_images(self, imgs: List[Image.Image]) -> torch.Tensor:
        if self.clip_model is None:
            return torch.zeros(len(imgs), 512)
        feats = []
        for img in imgs:
            t = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f = self.clip_model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    # ------------------------------------------------------------------
    def _encode_text(self, text: str) -> torch.Tensor:
        if self.clip_model is None:
            return torch.zeros(1, 512)
        import open_clip
        with torch.no_grad():
            tok = open_clip.tokenize([text]).to(self.device)
            feats = self.clip_model.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    # ------------------------------------------------------------------
    def _ensure_db_feats(self) -> None:
        if self._db_img_feats is None and self._db_images:
            self._db_img_feats = self._encode_images(self._db_images)

    # ------------------------------------------------------------------
    def _ensure_crop_feats(self) -> torch.Tensor:
        if self._db_crop_feats is not None:
            return self._db_crop_feats
        crop_imgs: List[Image.Image] = []
        for img, cb in zip(self._db_images, self._db_crops):
            w, h = img.size
            px = cb.to_pixel(w, h, coord_range=(1.0, 1000.0))
            try:
                crop_imgs.append(
                    img.crop((px.x1, px.y1, px.x2, px.y2))
                )
            except Exception:
                crop_imgs.append(img)
        self._db_crop_feats = self._encode_images(crop_imgs)
        return self._db_crop_feats

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_image: Image.Image,
        text_prompt: str,
        S: int = 10,
        text_rerank_weight: float = 0.5,
    ) -> List[ICLExample]:
        """
        Retrieve S examples most relevant to (query_image, text_prompt).
        """
        N = len(self._db_images)
        if N == 0:
            return []

        self._ensure_db_feats()
        query_feat = self._encode_images([query_image])   # (1, D)
        img_sim = (query_feat @ self._db_img_feats.T).squeeze(0)  # (N,)

        text_feat  = self._encode_text(text_prompt)       # (1, D)
        crop_feats = self._ensure_crop_feats()            # (N, D)
        txt_sim    = (text_feat @ crop_feats.T).squeeze(0)         # (N,)

        w1 = 1.0 - text_rerank_weight
        w2 = text_rerank_weight
        # Use numpy-style operations that work with both torch.Tensor and
        # plain numpy arrays; avoids float * Tensor ordering issues.
        combined = img_sim * w1 + txt_sim * w2
        sorted_idx = torch.argsort(combined, descending=True).tolist()
        top_idx = sorted_idx[:S]

        return [
            ICLExample(
                image=self._db_images[i],
                crop=self._db_crops[i],
                image_path=self._db_paths[i],
            )
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# Main TextGuidedCropper
# ---------------------------------------------------------------------------

class TextGuidedCropper:
    """
    Full text-guided cropping pipeline (CRISP Task 2).

    Pipeline
    --------
    1. [Retrieval]   Retrieve S ICL examples using CLIP image similarity +
                     text-crop re-ranking (the Task-2 key innovation).
    2. [Generation]  Call VLM with interleaved image-crop examples +
                     text prompt to propose R candidate crop boxes.
    3. [Scoring]     Score each candidate with the CompositeScorer
                     (aesthetic + CLIP text-alignment + area).
    4. [Refinement]  Repeat L times: feed scores back to VLM for improvement.
    5. [Selection]   Return the highest-scored crop.

    Parameters
    ----------
    vlm_model, vlm_processor :
        HuggingFace Idefics2-compatible model & processor (Mantis-8B-Idefics2).
        Pass ``None`` to use the heuristic fallback (no GPU required).
    clip_model, clip_preprocess :
        OpenCLIP model + preprocessing transform (ViT-B/32 recommended).
        Pass ``None`` to skip CLIP-based retrieval and text scoring.
    device : str
        ``"cuda"`` or ``"cpu"``.
    S : int
        Number of ICL examples to retrieve.
    R : int
        Number of candidate crops per VLM call.
    L : int
        Number of iterative refinement steps.
    text_rerank_weight : float
        Weight (0-1) for text-crop re-ranking vs. image similarity in retrieval.
    coord_range : tuple
        Normalisation range: (1, 1000) for free-form, (0, 1) for subject-aware.
    vila_weight, text_weight, area_weight : float
        Composite scorer component weights.
    """

    def __init__(
        self,
        vlm_model=None,
        vlm_processor=None,
        clip_model=None,
        clip_preprocess=None,
        device: Optional[str] = None,
        S: int = 10,
        R: int = 5,
        L: int = 2,
        text_rerank_weight: float = 0.5,
        coord_range: Tuple[float, float] = (1.0, 1000.0),
        vila_weight: float = 0.3,
        text_weight: float = 0.5,
        area_weight: float = 0.2,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.S = S
        self.R = R
        self.L = L
        self.text_rerank_weight = text_rerank_weight
        self.coord_range = coord_range

        if clip_model is not None:
            clip_model = clip_model.to(self.device)

        self.retriever = ICLRetriever(
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            device=self.device,
        )
        self.generator = VLMCropGenerator(
            model=vlm_model,
            processor=vlm_processor,
            device=self.device,
        )
        self.scorer = CompositeScorer(
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            device=self.device,
            vila_weight=vila_weight,
            text_weight=text_weight,
            area_weight=area_weight,
        )

    # ------------------------------------------------------------------
    def add_examples(
        self,
        images: List[Image.Image],
        crops: List[CropBox],
        paths: Optional[List[str]] = None,
    ) -> None:
        """Populate the retrieval database with image-crop pairs."""
        self.retriever.add_to_database(images, crops, paths)

    # ------------------------------------------------------------------
    def _score_candidates(
        self,
        candidates: List[CropBox],
        query_image: Image.Image,
    ) -> Tuple[List[CropBox], List[Image.Image]]:
        W, H = query_image.size
        scored: List[CropBox] = []
        cropped_imgs: List[Image.Image] = []
        for cb in candidates:
            px = cb.to_pixel(W, H, self.coord_range)
            try:
                crop_img = query_image.crop((px.x1, px.y1, px.x2, px.y2))
            except Exception:
                crop_img = query_image.copy()
            s = self.scorer.score(crop_img, W, H, px)
            scored.append(
                CropBox(cb.x1, cb.y1, cb.x2, cb.y2, cb.score,
                        composite_score=s)
            )
            cropped_imgs.append(crop_img)
        return scored, cropped_imgs

    # ------------------------------------------------------------------
    def crop(
        self,
        image: Image.Image,
        text_prompt: str,
        return_all_candidates: bool = False,
    ) -> TextGuidedCropResult:
        """
        Crop ``image`` according to ``text_prompt``.

        Example prompts
        ---------------
        - "Focus on the person on the left"
        - "Wide cinematic crop"
        - "Crop for Instagram square post"
        - "Emphasize the mountain, not the lake"
        - "Portrait orientation for mobile"

        Parameters
        ----------
        image : PIL.Image.Image
        text_prompt : str
        return_all_candidates : bool
            Populate TextGuidedCropResult.all_candidates if True.

        Returns
        -------
        TextGuidedCropResult
        """
        image = image.convert("RGB")

        # Pre-encode text for scorer
        self.scorer.set_text_prompt(text_prompt)

        # ---- 1. Retrieve ICL examples --------------------------------
        icl_examples = self.retriever.retrieve(
            query_image=image,
            text_prompt=text_prompt,
            S=self.S,
            text_rerank_weight=self.text_rerank_weight,
        )
        logger.info("Retrieved %d ICL examples.", len(icl_examples))

        has_score = self.coord_range[1] > 2.0  # True for [1,1000]

        # ---- 2. Initial generation ----------------------------------
        candidates = self.generator.generate(
            query_image=image,
            text_prompt=text_prompt,
            icl_examples=icl_examples,
            n_crops=self.R,
            coord_range=self.coord_range,
            has_score=has_score,
        )
        logger.info("Generated %d initial candidates.", len(candidates))

        # ---- 3. Score candidates ------------------------------------
        candidates, cropped_imgs = self._score_candidates(candidates, image)
        all_candidates = list(candidates) if return_all_candidates else []

        messages, msg_images = self.generator._build_initial_prompt(
            image, text_prompt, icl_examples,
            self.R, self.coord_range, has_score
        )

        # ---- 4. Iterative refinement --------------------------------
        for iteration in range(self.L):
            refined = self.generator.refine(
                query_image=image,
                text_prompt=text_prompt,
                messages=messages,
                images=msg_images,
                scored_candidates=candidates,
                cropped_images=cropped_imgs,
                coord_range=self.coord_range,
                has_score=has_score,
            )
            refined_list, refined_crops = self._score_candidates(
                [refined], image
            )
            refined = refined_list[0]
            logger.info(
                "Iter %d: refined score = %.4f",
                iteration + 1, refined.composite_score
            )
            candidates.append(refined)
            cropped_imgs.append(refined_crops[0])
            if return_all_candidates:
                all_candidates.append(refined)

        # ---- 5. Select best crop ------------------------------------
        best = max(candidates, key=lambda c: c.composite_score)
        W, H = image.size
        best_px = best.to_pixel(W, H, self.coord_range)
        final_crop = image.crop(
            (best_px.x1, best_px.y1, best_px.x2, best_px.y2)
        )

        return TextGuidedCropResult(
            cropped_image=final_crop,
            crop_box=best_px,
            composite_score=best.composite_score,
            all_candidates=all_candidates,
            num_refinement_iters=self.L,
        )

    # ------------------------------------------------------------------
    def crop_with_size(
        self,
        image: Image.Image,
        text_prompt: str,
        crop_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Backwards-compatible wrapper: crop then resize to ``crop_size`` (W, H).
        Drop-in replacement for the original sliding-window API.
        """
        result = self.crop(image, text_prompt)
        return result.cropped_image.resize(crop_size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
):
    """Load an OpenCLIP model and its preprocessing transform."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()
        return model, preprocess
    except ImportError:
        warnings.warn("open_clip not installed; CLIP features disabled.")
        return None, None


# ---------------------------------------------------------------------------
# VRAM guide
# ---------------------------------------------------------------------------
VRAM_GUIDE = """
┌─────────────────────────────────────────────────────────────────────────┐
│                  VLM VRAM Requirements (approximate)                    │
├──────────────────────────────────┬──────────┬──────────┬───────────────┤
│ Model                            │  fp16    │  4-bit   │  Fits 8 GB?   │
├──────────────────────────────────┼──────────┼──────────┼───────────────┤
│ TIGER-Lab/Mantis-8B-Idefics2     │ ~16 GB   │  ~5 GB ✓ │  4-bit only   │
│ HuggingFaceM4/idefics2-8b        │ ~16 GB   │  ~5 GB ✓ │  4-bit only   │
│ Qwen/Qwen2-VL-7B-Instruct        │ ~14 GB   │  ~5 GB ✓ │  4-bit only   │
│ Qwen/Qwen2-VL-2B-Instruct        │  ~4 GB ✓ │  ~2 GB ✓ │  YES (fp16)   │
│ HuggingFaceTB/SmolVLM-500M       │  ~1 GB ✓ │  ~1 GB ✓ │  YES (fp16)   │
│ llava-hf/llava-1.5-7b-hf         │ ~14 GB   │  ~5 GB ✓ │  4-bit only   │
└──────────────────────────────────┴──────────┴──────────┴───────────────┘
  For 8 GB VRAM, choose ONE of:
    A) quantize=4  with any 8B model  (pip install bitsandbytes)
    B) model_name="Qwen/Qwen2-VL-2B-Instruct"  (no quantization needed)
    C) model_name="HuggingFaceTB/SmolVLM-500M"  (smallest, fastest)
"""


def load_vlm_model(
    model_name: str = "TIGER-Lab/Mantis-8B-Idefics2",
    device: str = "cpu",
    use_fp16: bool = True,
    quantize: Optional[int] = None,
    print_vram_guide: bool = True,
):
    """
    Load a vision-language model for crop generation.

    Parameters
    ----------
    model_name : str
        Any HuggingFace VLM supporting multi-image interleaved input.
        Idefics2-based models (Mantis-8B-Idefics2, idefics2-8b) work
        natively.  Qwen2-VL and SmolVLM also work.
    device : str
        ``"cuda"`` or ``"cpu"``.
    use_fp16 : bool
        Load in float16 (halves VRAM vs float32). Ignored when ``quantize``
        is set.
    quantize : int or None
        ``4``  — 4-bit NF4 via bitsandbytes (~5 GB for 8B models). ← 8 GB GPU
        ``8``  — 8-bit LLM.int8 (~10 GB for 8B models).
        ``None`` — no quantization (use fp16/fp32 per ``use_fp16``).

        **For 8 GB VRAM:** pass ``quantize=4``.
        Requires:  ``pip install bitsandbytes accelerate``
    print_vram_guide : bool
        Print the VRAM reference table on first call.

    Returns
    -------
    (model, processor) or (None, None) on failure.
    The heuristic fallback is used automatically when (None, None) is returned.
    """
    if print_vram_guide:
        print(VRAM_GUIDE)

    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        load_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}

        if quantize in (4, 8):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for quantization.\n"
                    "Install with:  pip install bitsandbytes accelerate"
                )
            if quantize == 4:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info(
                    "4-bit NF4 quantization — expected VRAM ~5 GB for 8B models."
                )
            else:
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                logger.info(
                    "8-bit quantization — expected VRAM ~10 GB for 8B models."
                )
            load_kwargs["quantization_config"] = bnb_cfg
            # device_map="auto" lets accelerate spread layers across GPU/CPU
            load_kwargs["device_map"] = "auto"
        else:
            dtype = (
                torch.float16 if (use_fp16 and device != "cpu") else torch.float32
            )
            load_kwargs["torch_dtype"] = dtype

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)

        # Only call .to(device) when NOT using device_map (quantized models
        # are already placed by accelerate)
        if "device_map" not in load_kwargs:
            model = model.to(device)

        model.eval()
        logger.info("Loaded VLM: %s  (quantize=%s)", model_name, quantize)
        return model, processor

    except Exception as exc:
        warnings.warn(
            f"Could not load VLM '{model_name}': {exc}\n"
            "Running in heuristic-fallback mode (no VLM).\n"
            "Tip: for 8 GB VRAM use quantize=4 + pip install bitsandbytes accelerate\n"
            "  or use model_name='Qwen/Qwen2-VL-2B-Instruct' (fits in ~4 GB fp16)"
        )
        return None, None



def build_database_from_gaicd(
    gaicd_root: str,
    split: str = "train",
    max_images: int = 500,
    coord_range: Tuple[float, float] = (1.0, 1000.0),
) -> Tuple[List[Image.Image], List[CropBox]]:
    """
    Load images and best crop annotations from the GAICD dataset layout.

    Expected structure::

        <gaicd_root>/
            images/        (*.jpg)
            train.txt      (list of filenames)
            test.txt
            annotations/   (<stem>.txt  one crop per line: x1 y1 x2 y2 mos)

    Returns (images, crops) ready for TextGuidedCropper.add_examples().
    Missing files are skipped gracefully.
    """
    root = Path(gaicd_root)
    split_file = root / f"{split}.txt"
    images_dir = root / "images"
    ann_dir    = root / "annotations"

    if not root.exists():
        logger.warning("GAICD root '%s' not found; returning empty DB.",
                       gaicd_root)
        return [], []

    names = (
        split_file.read_text().strip().splitlines()
        if split_file.exists()
        else [p.name for p in sorted(images_dir.glob("*.jpg"))]
    )

    lo, hi = coord_range
    images: List[Image.Image] = []
    crops:  List[CropBox]     = []

    for name in names[:max_images]:
        img_path = images_dir / name
        ann_path = ann_dir / (Path(name).stem + ".txt")
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        best_crop: Optional[CropBox] = None
        best_mos = -1.0

        if ann_path.exists():
            for line in ann_path.read_text().strip().splitlines():
                try:
                    parts = list(map(float, line.split()))
                    if len(parts) == 5:
                        x1, y1, x2, y2, mos = parts
                    elif len(parts) == 4:
                        x1, y1, x2, y2 = parts
                        mos = 0.75
                    else:
                        continue
                    w, h = img.size
                    if x2 > hi:   # absolute pixel coords — normalise
                        span = hi - lo
                        x1 = lo + (x1 / w) * span
                        y1 = lo + (y1 / h) * span
                        x2 = lo + (x2 / w) * span
                        y2 = lo + (y2 / h) * span
                    if mos > best_mos:
                        best_mos  = mos
                        best_crop = CropBox(
                            x1, y1, x2, y2, score=min(mos / 5.0, 1.0)
                        )
                except Exception:
                    continue

        if best_crop is None:
            span   = hi - lo
            margin = 0.1 * span
            best_crop = CropBox(
                lo + margin, lo + margin,
                hi - margin, hi - margin,
                score=0.5,
            )

        images.append(img)
        crops.append(best_crop)

    logger.info(
        "Loaded %d images from GAICD (%s) at '%s'.",
        len(images), split, gaicd_root
    )
    return images, crops
