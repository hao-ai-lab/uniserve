"""Multimodal processor registry."""
from .base import MultimodalDataItem, MultimodalProcessor
from .registry import get_processor_for_model, register_processor

__all__ = [
    "MultimodalDataItem",
    "MultimodalProcessor",
    "get_processor_for_model",
    "register_processor",
]
