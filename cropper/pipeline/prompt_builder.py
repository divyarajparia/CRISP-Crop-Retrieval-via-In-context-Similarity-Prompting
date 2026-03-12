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

    # Prompt templates optimized for Mantis/Idefics2
    FREEFORM_INITIAL_TEMPLATE = """Task: Find aesthetic crop regions in images.
Format: (score, x1, y1, x2, y2) where score is 0.0-1.0, coordinates are 1-1000.

Here are example images with their best crops:
{examples}

Now analyze the LAST image shown and output 5 aesthetic crop regions in the same format.
Only output the coordinates, nothing else:"""

    FREEFORM_REFINEMENT_TEMPLATE = """{initial_prompt}
{crop_feedback}
Propose similar crop that has high score. The region should be represented by (s, x1, y1, x2, y2)."""

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
            return self._build_freeform_initial(icl_examples, query_image)

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
                initial_prompt, initial_images, crop_images, crop_coords, scores
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
    ) -> Tuple[str, List[Image.Image]]:
        """Build free-form cropping initial prompt."""
        images = []
        example_lines = []

        for i, example in enumerate(icl_examples):
            images.append(example["image"])

            # Format crops as (s, x1, y1, x2, y2)
            crops = example.get("crops", [])
            crop_strs = []
            for crop in crops:
                if len(crop) == 5:
                    mos, x1, y1, x2, y2 = crop
                    # Normalize MOS to 0-1 range (GAICD uses 1-5 scale)
                    mos_norm = min(1.0, mos / 5.0) if mos > 1 else mos
                    crop_strs.append(f"({mos_norm:.2f}, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
                elif len(crop) == 4:
                    x1, y1, x2, y2 = crop
                    crop_strs.append(f"(0.80, {int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")

            crop_str = ", ".join(crop_strs)
            example_lines.append(f"Image {i+1}: {crop_str}")

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
    ) -> Tuple[str, List[Image.Image]]:
        """Build free-form cropping refinement prompt."""
        images = initial_images.copy()

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

    # Replace all image references
    for i in range(len(images)):
        formatted = formatted.replace(f"{{image {i+1}}}", "<image>")

    formatted = formatted.replace("{Query image}", "<image>")

    for i in range(1, 100):  # Clean up any remaining
        formatted = formatted.replace(f"{{Cropped image {i}}}", "<image>")

    return formatted
