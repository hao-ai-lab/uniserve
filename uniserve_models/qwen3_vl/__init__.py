"""Qwen3-VL vision tower, multimodal text encoding and checkpoint names.

This package is not a catalog model. A composing model package (MiniMax H3)
builds ``VisionConfig`` with ``read_vision_config`` from its checkpoint's
``vision_config``, composes ``TextEncoder`` from a ``uniserve_models.qwen3``
decoder with the checkpoint's M-RoPE sections, computes M-RoPE positions
with ``rope_index``, and loads both towers with ``weights.assignments``.
"""

from . import weights
from .config import PixelConfig, VisionConfig, read_vision_config
from .encoder import TextEncoder, VisionEncoder
from .positions import rope_index
from .vision import Attention, PatchMerger, TransformerLayer, VisionTower

__all__ = [
    "Attention",
    "PatchMerger",
    "PixelConfig",
    "TextEncoder",
    "TransformerLayer",
    "VisionConfig",
    "VisionEncoder",
    "VisionTower",
    "read_vision_config",
    "rope_index",
    "weights",
]
