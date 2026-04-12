"""
Prompt builder for Cropper VLM prompts.
Builds exact prompts from the paper for all 3 cropping tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds VLM prompts for image cropping tasks.
    Supports free-form, subject-aware, and aspect-ratio-aware cropping.
    """

    # Free-form templates follow the wording and structure of Table 1 as closely
    # as possible from the paper text extraction, while remaining robust for Mantis.
    FREEFORM_INITIAL_TEMPLATE = """Localize the aesthetic part of the image. (s, x1, y1, x2, y2) represents the region. x1 and x2 are the left and right most positions, normalized into 1 to 1000, where 1 is the left and 1000 is the right. y1 and y2 are the top and bottom positions, normalized into 1 to 1000 where 1 is the top and 1000 is the bottom. s is MOS score. We provide several images here.
{examples}
{{Query image}}
Output {R} crops represented by (s, x1, y1, x2, y2). Only output crop tuples."""

    FREEFORM_REFINEMENT_TEMPLATE = """{initial_prompt}
{crop_feedback}
Propose similar crop that has high score. The region should be represented by (s, x1, y1, x2, y2). Only output the new crop tuples."""

    FREEFORM_REFINEMENT_RANKED_TEMPLATE = """{initial_prompt}
{crop_feedback}
Propose {R} variations of the best crop above. Adjust the boundaries to further improve aesthetic quality. Output (s, x1, y1, x2, y2). Only output crop tuples."""

    SUBJECT_AWARE_INITIAL_TEMPLATE = """Find visually appealing crop. Each region is represented by (x1, y1, x2, y2) coordinates. x1, x2 are the left and right most positions, normalized into 0 to 1, where 0 is the left and 1 is the right. y1, y2 are the top and bottom positions, normalized into 0 to 1 where 0 is the top and 1 is the bottom.
{examples}
{query_image}, ({cx}, {cy})"""

    SUBJECT_AWARE_REFINEMENT_TEMPLATE = """Localize aesthetic part of image. The region is represented by (x1,y1,x2,y2). x1, x2 are the left and right most positions, normalized into 0 to 1, where 0 is the left and 1 is the right. y1, y2 are the top and bottom positions, normalized into 0 to 1 where 0 is the top and 1 is the bottom. We provide several images here.
{crop_feedback}
Propose different crop. The region should be represented by (x1,y1,x2,y2). Output:"""

    ASPECT_RATIO_INITIAL_TEMPLATE = """Find visually appealing crop. Give the best crop in the form of a crop box and make sure the crop has certain width:height. Box is a 4-tuple defining the left, upper, right, and lower pixel coordinate in the form of (x1, y1, x2, y2). Here are some example images, its size, and crop w:h triplets and their corresponding crops.
{examples}
Now Give the best crop in the form of a crop box for the following image. Give {R} possible best crops.
{query_image}, size ({w}, {h}), crop ratio ({ratio})"""

    ASPECT_RATIO_REFINEMENT_TEMPLATE = """{initial_prompt}
Example Image: {query_image};
Crop ratio: {ratio}; Example output:
{crop_feedback}
Propose a different better crop with the given ratio. Output:"""

    def __init__(self, task: str = "freeform"):
        """
        Initialize prompt builder.

        Args:
            task: Cropping task type ('freeform', 'subject_aware', 'aspect_ratio')
        """
        self.task = task

    def build_initial_prompt(
        self,
        icl_examples: List[Dict],
        query_image: Image.Image,
        task_params: Optional[Dict] = None,
    ) -> Tuple[str, List[Image.Image]]:
        """
        Build the initial VLM prompt.

        Args:
            icl_examples: List of ICL examples, each with:
                - 'image': PIL Image
                - 'crops': List of crop tuples
                - For subject-aware: 'mask_center': (cx, cy)
            query_image: Query image to crop
            task_params: Task-specific parameters:
                - For aspect_ratio: 'aspect_ratio', 'R'
                - For subject_aware: 'mask_center'

        Returns:
            Tuple of (prompt_text, list_of_images)
        """
        task_params = task_params or {}
        logger.info(
            "Building %s initial prompt with %d ICL examples for query size=%s",
            self.task,
            len(icl_examples),
            query_image.size,
        )

        if self.task == "freeform":
            R = task_params.get("R", 6)
            return self._build_freeform_initial(
                icl_examples, query_image, R,
                visual_grounding=task_params.get("visual_grounding", False),
                visual_grounding_top_k=task_params.get("visual_grounding_top_k", None),
                task_params=task_params,
            )

        elif self.task == "subject_aware":
            mask_center = task_params.get("mask_center", (0.5, 0.5))
            return self._build_subject_aware_initial(icl_examples, query_image, mask_center)

        elif self.task == "aspect_ratio":
            aspect_ratio = task_params.get("aspect_ratio", 1.0)
            R = task_params.get("R", 6)
            return self._build_aspect_ratio_initial(icl_examples, query_image, aspect_ratio, R)

        else:
            raise ValueError(f"Unknown task: {self.task}")

    def build_refinement_prompt(
        self,
        initial_prompt: str,
        initial_images: List[Image.Image],
        crop_images: List[Image.Image],
        crop_coords: List[Tuple],
        scores: List[float],
        query_image: Image.Image,
        task_params: Optional[Dict] = None,
    ) -> Tuple[str, List[Image.Image]]:
        """
        Build the iterative refinement prompt.

        Args:
            initial_prompt: The initial prompt text
            initial_images: Images from the initial prompt
            crop_images: Cropped images from current candidates
            crop_coords: Coordinates of current candidates
            scores: Scores for current candidates
            query_image: Original query image
            task_params: Task-specific parameters

        Returns:
            Tuple of (prompt_text, list_of_images)
        """
        task_params = task_params or {}
        logger.info(
            "Building %s refinement prompt with %d crop candidates",
            self.task,
            len(crop_images),
        )

        if self.task == "freeform":
            return self._build_freeform_refinement(
                initial_prompt, initial_images, crop_images, crop_coords, scores,
                rank_anchored=task_params.get("rank_anchored", False),
                R=task_params.get("R", 6),
            )

        elif self.task == "subject_aware":
            return self._build_subject_aware_refinement(
                initial_prompt, initial_images, crop_images, crop_coords, scores
            )

        elif self.task == "aspect_ratio":
            aspect_ratio = task_params.get("aspect_ratio", 1.0)
            return self._build_aspect_ratio_refinement(
                initial_prompt, initial_images, crop_images, crop_coords, scores,
                query_image, aspect_ratio
            )

        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _build_freeform_initial(
        self,
        icl_examples: List[Dict],
        query_image: Image.Image,
        R: int,
        visual_grounding: bool = False,
        visual_grounding_top_k: Optional[int] = None,
        task_params: Optional[Dict] = None,
    ) -> Tuple[str, List[Image.Image]]:
        """Build free-form cropping initial prompt."""
        images = []
        example_lines = []

        # Lazy imports: normalize_coords maps pixel-space GT crops (as returned
        # by GAICDDataset after the row-major fix) into the 1..1000 coordinate
        # system the FREEFORM_INITIAL_TEMPLATE text declares to Mantis. Before
        # this normalization the prompt showed raw pixel numbers that only
        # coincidentally looked ~1-1000 because GAICD images are ~1024 wide.
        from utils.coord_utils import normalize_coords
        if visual_grounding:
            from utils.coord_utils import extract_crop

        for i, example in enumerate(icl_examples):
            ex_image = example["image"]
            images.append(ex_image)
            ex_size = ex_image.size  # (W, H) in pixels

            # Format crops as (s, x1, y1, x2, y2) in 1..1000 space
            crops = example.get("crops", [])
            crop_strs = []
            for crop in crops:
                if len(crop) == 5:
                    mos, x1, y1, x2, y2 = crop
                    # Normalize MOS to 0-1 range (GAICD uses 1-5 scale)
                    mos_norm = min(1.0, mos / 5.0) if mos > 1 else mos
                elif len(crop) == 4:
                    x1, y1, x2, y2 = crop
                    mos_norm = 0.80
                else:
                    continue
                xn1, yn1, xn2, yn2 = normalize_coords(
                    (x1, y1, x2, y2), ex_size, coord_range=(1, 1000)
                )
                crop_strs.append(
                    f"({mos_norm:.2f}, {int(round(xn1))}, {int(round(yn1))}, "
                    f"{int(round(xn2))}, {int(round(yn2))})"
                )

            crop_str = ", ".join(crop_strs)

            # Idea 1: visual crop grounding — append the top-1 crop image
            # immediately after the example image, and reference it via {crop N}.
            # Gated by visual_grounding_top_k: only the first K examples get a
            # crop image attached, so Mantis's total image count stays near
            # baseline density (attaching to all S=30 examples broke parsing).
            crop_image_appended = False
            within_top_k = (
                visual_grounding_top_k is None or i < visual_grounding_top_k
            )
            if visual_grounding and crops and within_top_k:
                try:
                    top_crop = crops[0]
                    if len(top_crop) == 5:
                        top_box = tuple(top_crop[1:5])
                    else:
                        top_box = tuple(top_crop[:4])
                    # Crops from GAICDDataset are already in pixel space after
                    # the row-major fix, so use extract_crop directly instead
                    # of crop_from_normalized (which would re-denormalize them).
                    crop_img = extract_crop(ex_image, top_box)
                    images.append(crop_img)
                    crop_image_appended = True
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "visual_grounding crop extraction failed for example %d: %s — falling back",
                        i, e,
                    )

            if crop_image_appended:
                example_lines.append(f"{{image {i+1}}}, {{crop {i+1}}}, {crop_str}")
            else:
                example_lines.append(f"{{image {i+1}}}, {crop_str}")

        # Add query image
        images.append(query_image)

        logger.info(
            "Freeform prompt contains %d example images and %d total GT crops",
            len(icl_examples),
            sum(len(example.get("crops", [])) for example in icl_examples),
        )

        examples_text = "\n".join(example_lines)
        prompt = self.FREEFORM_INITIAL_TEMPLATE.format(
            examples=examples_text,
            R=R,
        )

        # Idea 5: anti-bias instruction to discourage full-image crops
        if (task_params or {}).get("anti_bias_prompt", False):
            prompt += (
                " The crop should focus on the most aesthetically"
                " interesting region. It may be a sub-region of the"
                " image, not necessarily the entire image."
            )

        logger.debug("Freeform initial prompt length: %d chars", len(prompt))

        return prompt, images

    def _build_freeform_refinement(
        self,
        initial_prompt: str,
        initial_images: List[Image.Image],
        crop_images: List[Image.Image],
        crop_coords: List[Tuple],
        scores: List[float],
        rank_anchored: bool = False,
        R: int = 6,
    ) -> Tuple[str, List[Image.Image]]:
        """Build free-form cropping refinement prompt."""
        images = initial_images.copy()

        if not rank_anchored:
            feedback_lines = []
            for i, (crop_img, coords, score) in enumerate(zip(crop_images, crop_coords, scores)):
                images.append(crop_img)

                if len(coords) == 5:
                    mos, x1, y1, x2, y2 = coords
                else:
                    x1, y1, x2, y2 = coords
                    mos = score

                feedback_lines.append(
                    f"{{Cropped image {i+1}}} ({mos:.2f}, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}), Score is {{{score:.2f}}}"
                )

            crop_feedback = "\n".join(feedback_lines)

            prompt = self.FREEFORM_REFINEMENT_TEMPLATE.format(
                initial_prompt=initial_prompt,
                crop_feedback=crop_feedback,
            )

            return prompt, images

        # --- Rank-anchored path (Idea 2) ---
        # Sort triples by score descending and append images in sorted order
        # so that <image> tokens line up with {Cropped image i} labels.
        triples = list(zip(crop_images, crop_coords, scores))
        triples.sort(key=lambda t: t[2], reverse=True)

        feedback_lines = []
        for i, (crop_img, coords, score) in enumerate(triples):
            images.append(crop_img)

            if len(coords) == 5:
                mos, x1, y1, x2, y2 = coords
            else:
                x1, y1, x2, y2 = coords
                mos = score

            label = f"{{Cropped image {i+1}}}"
            line = (
                f"{label} ({mos:.2f}, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}), "
                f"Score {score:.2f}"
            )

            if i == 0:
                feedback_lines.append(f"Best crop so far — Score {score:.2f}:")
                feedback_lines.append(line)
                if len(triples) > 1:
                    feedback_lines.append("")
                    feedback_lines.append("Other candidates:")
            else:
                feedback_lines.append(line)

        crop_feedback = "\n".join(feedback_lines)

        prompt = self.FREEFORM_REFINEMENT_RANKED_TEMPLATE.format(
            initial_prompt=initial_prompt,
            crop_feedback=crop_feedback,
            R=R,
        )

        return prompt, images

    def _build_subject_aware_initial(
        self,
        icl_examples: List[Dict],
        query_image: Image.Image,
        mask_center: Tuple[float, float],
    ) -> Tuple[str, List[Image.Image]]:
        """Build subject-aware cropping initial prompt."""
        images = []
        example_lines = []

        for i, example in enumerate(icl_examples):
            images.append(example["image"])

            # Get mask center and crop
            ex_center = example.get("mask_center", (0.5, 0.5))
            crop = example.get("crop", (0.1, 0.1, 0.9, 0.9))

            if len(crop) == 4:
                x1, y1, x2, y2 = crop
            else:
                x1, y1, x2, y2 = crop[1:5] if len(crop) > 4 else (0.1, 0.1, 0.9, 0.9)

            example_lines.append(
                f"{{image {i+1}}} (({ex_center[0]:.2f}, {ex_center[1]:.2f}), {x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})"
            )

        images.append(query_image)

        logger.info(
            "Subject-aware prompt contains %d examples; query mask center=(%.3f, %.3f)",
            len(icl_examples),
            mask_center[0],
            mask_center[1],
        )

        examples_text = "\n".join(example_lines)
        prompt = self.SUBJECT_AWARE_INITIAL_TEMPLATE.format(
            examples=examples_text,
            query_image="{Query image}",
            cx=f"{mask_center[0]:.2f}",
            cy=f"{mask_center[1]:.2f}",
        )

        logger.debug("Subject-aware initial prompt length: %d chars", len(prompt))

        return prompt, images

    def _build_subject_aware_refinement(
        self,
        initial_prompt: str,
        initial_images: List[Image.Image],
        crop_images: List[Image.Image],
        crop_coords: List[Tuple],
        scores: List[float],
    ) -> Tuple[str, List[Image.Image]]:
        """Build subject-aware refinement prompt."""
        images = []  # Start fresh for refinement

        feedback_lines = []
        for i, (crop_img, coords, score) in enumerate(zip(crop_images, crop_coords, scores)):
            images.append(crop_img)

            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
            else:
                x1, y1, x2, y2 = 0.1, 0.1, 0.9, 0.9

            feedback_lines.append(
                f"{{Cropped image {i+1}}} Output: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})"
            )

        crop_feedback = "\n".join(feedback_lines)

        prompt = self.SUBJECT_AWARE_REFINEMENT_TEMPLATE.format(
            crop_feedback=crop_feedback,
        )

        return prompt, images

    def _build_aspect_ratio_initial(
        self,
        icl_examples: List[Dict],
        query_image: Image.Image,
        aspect_ratio: float,
        R: int,
    ) -> Tuple[str, List[Image.Image]]:
        """Build aspect-ratio-aware cropping initial prompt."""
        images = []
        example_lines = []

        for i, example in enumerate(icl_examples):
            ex_image = example["image"]
            images.append(ex_image)

            w, h = ex_image.size
            crop = example.get("crop", (0, 0, w, h))
            ex_ratio = example.get("aspect_ratio", 1.0)

            if len(crop) == 4:
                x1, y1, x2, y2 = crop
            else:
                x1, y1, x2, y2 = crop[1:5]

            example_lines.append(
                f"{{image {i+1}}}, size ({w}, {h}), crop ratio ({ex_ratio:.2f}), output ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            )

        images.append(query_image)
        qw, qh = query_image.size

        logger.info(
            "Aspect-ratio prompt contains %d examples; target ratio=%.4f; R=%d",
            len(icl_examples),
            aspect_ratio,
            R,
        )

        examples_text = "\n".join(example_lines)
        prompt = self.ASPECT_RATIO_INITIAL_TEMPLATE.format(
            examples=examples_text,
            query_image="{Query image}",
            w=qw,
            h=qh,
            ratio=f"{aspect_ratio:.2f}",
            R=R,
        )

        logger.debug("Aspect-ratio initial prompt length: %d chars", len(prompt))

        return prompt, images

    def _build_aspect_ratio_refinement(
        self,
        initial_prompt: str,
        initial_images: List[Image.Image],
        crop_images: List[Image.Image],
        crop_coords: List[Tuple],
        scores: List[float],
        query_image: Image.Image,
        aspect_ratio: float,
    ) -> Tuple[str, List[Image.Image]]:
        """Build aspect-ratio refinement prompt."""
        images = initial_images.copy()

        feedback_lines = []
        for i, (crop_img, coords, score) in enumerate(zip(crop_images, crop_coords, scores)):
            images.append(crop_img)

            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
            else:
                x1, y1, x2, y2 = 0, 0, 100, 100

            feedback_lines.append(
                f"{{Cropped image {i+1}}} ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            )

        crop_feedback = "\n".join(feedback_lines)

        prompt = self.ASPECT_RATIO_REFINEMENT_TEMPLATE.format(
            initial_prompt=initial_prompt,
            query_image="{Query image}",
            ratio=f"{aspect_ratio:.2f}",
            crop_feedback=crop_feedback,
        )

        return prompt, images


def format_prompt_for_mantis(
    text_prompt: str,
    images: List[Image.Image],
) -> str:
    """
    Format prompt for Mantis-8B-Idefics2 model.

    Args:
        text_prompt: Text prompt with {image N} placeholders
        images: List of images

    Returns:
        Formatted prompt with proper image tokens
    """
    # Replace {image N} placeholders with <image> tokens
    formatted = text_prompt

    # Replace all image references. Order matters only insofar as the images
    # list must list tensors in the same order they appear textually; the
    # actual <image> tokens are positional in Mantis. We use a wide range so
    # all kinds of placeholders ({image N}, {crop N}, {Cropped image N}) get
    # substituted regardless of how many were emitted.
    for i in range(1, 200):
        formatted = formatted.replace(f"{{image {i}}}", "<image>")
        formatted = formatted.replace(f"{{crop {i}}}", "<image>")
        formatted = formatted.replace(f"{{Cropped image {i}}}", "<image>")

    formatted = formatted.replace("{Query image}", "<image>")

    return formatted
