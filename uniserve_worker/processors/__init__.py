"""Multimodal processor registry."""
from .base import MultimodalDataItem, MultimodalProcessor
from .registry import (
    get_image_pipeline_for_descriptor,
    get_image_pipeline_for_model,
    get_processor_for_descriptor,
    get_processor_for_model,
    register_processor,
)

__all__ = [
    "MultimodalDataItem",
    "MultimodalProcessor",
    "get_image_pipeline_for_descriptor",
    "get_image_pipeline_for_model",
    "get_processor_for_descriptor",
    "get_processor_for_model",
    "register_processor",
]
