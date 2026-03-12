"""
VLM Wrapper for Mantis-8B-Idefics2 and other Vision-Language Models.
Supports multi-image input for in-context learning.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""

    @abstractmethod
    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        temperature: float = 0.05,
        max_new_tokens: int = 256,
        num_outputs: int = 1,
    ) -> str:
        """
        Generate text output given images and a prompt.

        Args:
            images: List of PIL Images to include in the prompt
            prompt: Text prompt (may include image placeholders)
            temperature: Sampling temperature
            max_new_tokens: Maximum number of tokens to generate
            num_outputs: Number of outputs to generate

        Returns:
            Generated text
        """
        raise NotImplementedError

    @abstractmethod
    def parse_crops(
        self,
        output_text: str,
        task: str,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple]:
        """
        Parse crop coordinates from VLM output text.

        Args:
            output_text: Raw text output from VLM
            task: Cropping task type ('freeform', 'subject_aware', 'aspect_ratio')
            image_size: (width, height) of the original image for pixel coordinates

        Returns:
            List of crop tuples. Format depends on task:
            - freeform: (mos, x1, y1, x2, y2)
            - subject_aware: (x1, y1, x2, y2)
            - aspect_ratio: (x1, y1, x2, y2)
        """
        raise NotImplementedError


class MantisVLM(BaseVLM):
    """
    Wrapper for Mantis-8B-Idefics2 from HuggingFace.
    Model: TIGER-Lab/Mantis-8B-Idefics2

    This model supports multi-image input, which is critical for ICL.
    """

    def __init__(
        self,
        model_name: str = "TIGER-Lab/Mantis-8B-Idefics2",
        device: str = "cuda",
        use_fp16: bool = True,
        max_images: int = 10,
    ):
        """
        Initialize Mantis VLM.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            use_fp16: Whether to use float16 precision
            max_images: Maximum number of images in context
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.max_images = max_images

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the Mantis model and processor."""
        try:
            from transformers import Idefics2ForConditionalGeneration, AutoProcessor

            logger.info(f"Loading {self.model_name}...")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # Load model with appropriate precision
            dtype = torch.float16 if self.use_fp16 else torch.float32

            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )

            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")

        except ImportError as e:
            logger.warning(f"Could not import required libraries: {e}")
            logger.info("Falling back to mock VLM for testing")
            self.model = None
            self.processor = None

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to mock VLM for testing")
            self.model = None
            self.processor = None

    def _prepare_inputs(
        self,
        images: List[Image.Image],
        prompt: str,
    ) -> Dict:
        """Prepare inputs for the model."""
        if self.processor is None:
            return {"prompt": prompt, "images": images}

        # Build messages in chat format for Idefics2/Mantis
        # Each image is represented as {"type": "image"}
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        return inputs

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        temperature: float = 0.05,
        max_new_tokens: int = 256,
        num_outputs: int = 1,
    ) -> str:
        """Generate text output given images and a prompt."""

        # Limit number of images
        if len(images) > self.max_images:
            logger.warning(
                f"Number of images ({len(images)}) exceeds max ({self.max_images}). "
                "Truncating to max_images."
            )
            images = images[:self.max_images]

        if self.model is None:
            # Mock response for testing
            return self._mock_generate(prompt, len(images))

        try:
            inputs = self._prepare_inputs(images, prompt)

            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    num_return_sequences=num_outputs,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            # Decode output
            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0]

            # Extract assistant's response - try multiple patterns
            for marker in ["Assistant:", "assistant:", "ASSISTANT:", "<assistant>", "\nAssistant"]:
                if marker in generated_text:
                    generated_text = generated_text.split(marker)[-1].strip()
                    break

            # Also try to extract from end of output (after last user message)
            if "User:" in generated_text or "user:" in generated_text:
                parts = generated_text.replace("user:", "User:").split("User:")
                generated_text = parts[-1].strip()

            logger.debug(f"VLM raw output: {repr(generated_text[:200])}")
            return generated_text

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return ""

    def _mock_generate(self, prompt: str, num_images: int) -> str:
        """Generate mock response for testing without model."""
        import random

        # Determine task type from prompt
        if "MOS" in prompt or "aesthetic" in prompt.lower():
            # Free-form cropping response
            crops = []
            for _ in range(5):
                mos = round(random.uniform(0.5, 1.0), 2)
                x1 = random.randint(1, 300)
                y1 = random.randint(1, 300)
                x2 = random.randint(x1 + 200, 900)
                y2 = random.randint(y1 + 200, 900)
                crops.append(f"({mos}, {x1}, {y1}, {x2}, {y2})")
            return ", ".join(crops)

        elif "mask" in prompt.lower() or "subject" in prompt.lower():
            # Subject-aware cropping response
            x1 = round(random.uniform(0.1, 0.3), 2)
            y1 = round(random.uniform(0.1, 0.3), 2)
            x2 = round(random.uniform(0.7, 0.9), 2)
            y2 = round(random.uniform(0.7, 0.9), 2)
            return f"({x1}, {y1}, {x2}, {y2})"

        else:
            # Aspect-ratio or generic response
            crops = []
            for _ in range(5):
                x1 = random.randint(50, 200)
                y1 = random.randint(50, 200)
                x2 = random.randint(x1 + 300, 800)
                y2 = random.randint(y1 + 300, 800)
                crops.append(f"({x1}, {y1}, {x2}, {y2})")
            return ", ".join(crops)

    def parse_crops(
        self,
        output_text: str,
        task: str,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple]:
        """
        Parse crop coordinates from VLM output text.

        Handles various output formats:
        - (mos, x1, y1, x2, y2) for free-form
        - (x1, y1, x2, y2) for subject-aware and aspect-ratio
        - Variable whitespace
        - Missing/extra parentheses
        - Out-of-range coordinates (clamped)
        """
        crops = []

        if not output_text:
            return crops

        # Clean up the text
        text = output_text.strip()

        # Pattern for free-form: (mos, x1, y1, x2, y2) or [mos, x1, y1, x2, y2]
        if task == "freeform":
            pattern = r'[\(\[]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\)\]]'
            matches = re.findall(pattern, text)

            for match in matches:
                try:
                    mos = float(match[0])
                    x1 = int(float(match[1]))
                    y1 = int(float(match[2]))
                    x2 = int(float(match[3]))
                    y2 = int(float(match[4]))

                    # Clamp coordinates to valid range [1, 1000]
                    x1 = max(1, min(1000, x1))
                    y1 = max(1, min(1000, y1))
                    x2 = max(1, min(1000, x2))
                    y2 = max(1, min(1000, y2))

                    # Ensure x2 > x1 and y2 > y1
                    if x2 <= x1:
                        x2 = min(1000, x1 + 100)
                    if y2 <= y1:
                        y2 = min(1000, y1 + 100)

                    # Clamp MOS to [0, 1]
                    mos = max(0.0, min(1.0, mos))

                    crops.append((mos, x1, y1, x2, y2))

                except (ValueError, IndexError):
                    continue

        # Pattern for subject-aware: (x1, y1, x2, y2) or [x1, y1, x2, y2] normalized 0-1
        elif task == "subject_aware":
            pattern = r'[\(\[]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\)\]]'
            matches = re.findall(pattern, text)

            for match in matches:
                try:
                    x1 = float(match[0])
                    y1 = float(match[1])
                    x2 = float(match[2])
                    y2 = float(match[3])

                    # Clamp to [0, 1]
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))

                    # Ensure x2 > x1 and y2 > y1
                    if x2 <= x1:
                        x2 = min(1.0, x1 + 0.1)
                    if y2 <= y1:
                        y2 = min(1.0, y1 + 0.1)

                    crops.append((x1, y1, x2, y2))

                except (ValueError, IndexError):
                    continue

        # Pattern for aspect-ratio: (x1, y1, x2, y2) or [x1, y1, x2, y2] in pixels
        elif task == "aspect_ratio":
            pattern = r'[\(\[]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\)\]]'
            matches = re.findall(pattern, text)

            for match in matches:
                try:
                    x1 = int(float(match[0]))
                    y1 = int(float(match[1]))
                    x2 = int(float(match[2]))
                    y2 = int(float(match[3]))

                    # Clamp to image size if provided
                    if image_size:
                        w, h = image_size
                        x1 = max(0, min(w, x1))
                        y1 = max(0, min(h, y1))
                        x2 = max(0, min(w, x2))
                        y2 = max(0, min(h, y2))

                    # Ensure x2 > x1 and y2 > y1
                    if x2 <= x1:
                        x2 = x1 + 100
                    if y2 <= y1:
                        y2 = y1 + 100

                    crops.append((x1, y1, x2, y2))

                except (ValueError, IndexError):
                    continue

        return crops


class Idefics2VLM(BaseVLM):
    """
    Alternative wrapper using Idefics2 directly.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceM4/idefics2-8b",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the Idefics2 model."""
        try:
            from transformers import Idefics2ForConditionalGeneration, AutoProcessor

            logger.info(f"Loading {self.model_name}...")

            dtype = torch.float16 if self.use_fp16 else torch.float32

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
            self.model.eval()

        except Exception as e:
            logger.error(f"Error loading Idefics2: {e}")
            self.model = None
            self.processor = None

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        temperature: float = 0.05,
        max_new_tokens: int = 256,
        num_outputs: int = 1,
    ) -> str:
        """Generate text using Idefics2."""
        if self.model is None:
            return ""

        # Build messages in chat format
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}],
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text

    def parse_crops(
        self,
        output_text: str,
        task: str,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple]:
        """Use the same parsing logic as MantisVLM."""
        # Reuse MantisVLM parsing
        return MantisVLM.parse_crops(self, output_text, task, image_size)


def create_vlm(
    model_type: str = "mantis",
    **kwargs,
) -> BaseVLM:
    """
    Factory function to create a VLM wrapper.

    Args:
        model_type: One of 'mantis', 'idefics2'
        **kwargs: Additional arguments for the model

    Returns:
        VLM wrapper instance
    """
    if model_type.lower() == "mantis":
        return MantisVLM(**kwargs)
    elif model_type.lower() == "idefics2":
        return Idefics2VLM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
