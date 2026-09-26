"""DiffusionGemma numerical model and checkpoint definitions.

``uniserve_models.loading`` selects this package for
``DiffusionGemmaForBlockDiffusion`` checkpoints and reads the names exported
here as its model package contract: configuration normalization, the model
class, its entry points, checkpoint sources and mappings, and precision
presets. ``processing`` holds the Gemma-4 image preprocessing Python callers
apply before ``Model.vision_encoder``.
"""

from . import processing
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
from .weights import checkpoint_mappings, checkpoint_sources

# No shared image transform descriptor expresses the Gemma-4 budget-fitted
# resize, so the loading contract declares no worker-side image
# preprocessing; ``processing`` implements it for Python callers. Block
# diffusion has no classifier-free-guidance prompt framing.
image_processor = None
flow_prompt = None

__all__ = [
    "config_sources",
    "image_processor",
    "flow_prompt",
    "processing",
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
