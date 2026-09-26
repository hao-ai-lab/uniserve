"""DiffusionGemma numerical model and checkpoint definitions.

``uniserve_models.loading`` selects this package for
``DiffusionGemmaForBlockDiffusion`` checkpoints and reads the names exported
here as its model package contract: configuration normalization, the model
class, its entry points, checkpoint sources and mappings, precision
presets, and ``image_processor``, the Gemma-4 image preprocessing that
serving and Python callers apply before ``Model.vision_encoder``.
"""

from .config import (
    Config,
    DiffusionConfig,
    LayerAttention,
    TextConfig,
    VisionConfig,
    config_sources,
    read_config,
)
from .model import Model, entry_points
from .precision import checkpoint_precision, precisions
from .processing import image_processor
from .weights import checkpoint_mappings, checkpoint_sources

# Block diffusion has no classifier-free-guidance prompt framing.
flow_prompt = None

__all__ = [
    "config_sources",
    "image_processor",
    "flow_prompt",
    "Config",
    "DiffusionConfig",
    "LayerAttention",
    "TextConfig",
    "VisionConfig",
    "read_config",
    "Model",
    "entry_points",
    "checkpoint_sources",
    "checkpoint_mappings",
    "precisions",
    "checkpoint_precision",
]
