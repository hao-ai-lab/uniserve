"""Multimodal processor registry."""
from .base import MultimodalDataItem, MultimodalProcessor
from .registry import (
    build_image_input_stage,
    get_processor_for_descriptor,
    get_processor_for_model,
    register_processor,
)

__all__ = [
    "MultimodalDataItem",
    "MultimodalProcessor",
    "build_image_input_stage",
    "get_processor_for_descriptor",
    "get_processor_for_model",
    "register_processor",
]
