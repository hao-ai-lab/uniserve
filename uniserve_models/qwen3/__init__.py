"""Qwen3 numerical models and checkpoint definitions.

``uniserve_models.loading`` selects this package for ``Qwen3ForCausalLM`` and
``Qwen3MoeForCausalLM`` checkpoints and reads the names exported here as its
model package contract: configuration normalization, the model class, its
entry points, checkpoint sources and mappings, and precision presets. The
MiniMax H3 text encoder also reuses ``Config``, ``Transformer`` and
``weights.parameter_sources``.
"""

from .config import Config, config_sources, read_config
from .model import Model, entry_points
from .transformer import Attention, MoE, Transformer, TransformerLayer
from .weights import (
    checkpoint_mappings,
    checkpoint_precision,
    checkpoint_sources,
    precisions,
)

# Qwen3 is text-only: the loading contract reads ``None`` as no image
# preprocessing and no flow prompt.
image_processor = None
flow_prompt = None

__all__ = [
    "config_sources",
    "image_processor",
    "flow_prompt",
    "Config",
    "read_config",
    "Model",
    "entry_points",
    "Attention",
    "MoE",
    "Transformer",
    "TransformerLayer",
    "checkpoint_sources",
    "checkpoint_mappings",
    "precisions",
    "checkpoint_precision",
]
