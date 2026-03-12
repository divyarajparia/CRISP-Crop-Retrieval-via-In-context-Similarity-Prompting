"""
Cropper: Vision-Language Model for Image Cropping through In-Context Learning

A training-free image cropping framework using VLMs with in-context learning.
Replication of the CVPR 2025 paper using open-source VLMs (Mantis-8B-Idefics2).
"""

from .pipeline.cropper import Cropper, create_cropper

__version__ = "0.1.0"
__all__ = ["Cropper", "create_cropper"]
