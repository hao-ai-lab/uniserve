"""Qwen3-VL vision tower, language model, prompt encoding and checkpoint names.

This package is not a catalog model. A composing model package (MiniMax H3)
reads ``VisionConfig`` with ``read_vision_config`` from its checkpoint's
``vision_config``, builds ``Encoder`` over a ``Transformer`` configured with
the checkpoint's M-RoPE sections, and loads both towers with
``weights.assignments``. ``Encoder.positions`` computes a prompt's M-RoPE
coordinates (``rope_index``).
"""

from . import weights
from .config import PixelConfig, VisionConfig, read_vision_config
from .encoder import Encoder, VisionEncoder
from .positions import rope_index
from .transformer import Transformer
from .vision import Attention, PatchMerger, TransformerLayer, VisionTower

__all__ = [
    "Attention",
    "Encoder",
    "PatchMerger",
    "PixelConfig",
    "Transformer",
    "TransformerLayer",
    "VisionConfig",
    "VisionEncoder",
    "VisionTower",
    "read_vision_config",
    "rope_index",
    "weights",
]
