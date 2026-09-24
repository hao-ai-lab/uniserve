"""SenseNova U1 numerical models and checkpoint definitions.

``uniserve_models.loading`` selects this package for ``NEOChatModel``
checkpoints and reads the names exported here as its model package contract:
configuration normalization, the model class, its entry points, checkpoint
sources and mappings, precision presets, and the caller-side image processor
and flow prompt.
"""

from .config import Config, TransformerConfig, config_sources, read_config
from .denoiser import Denoiser
from .inputs import DenoiserInput, ImageConditioning
from .model import Model, entry_points
from .processing import flow_prompt, image_processor
from .transformer import Transformer, TransformerLayer
from .weights import (
    checkpoint_mappings,
    checkpoint_precision,
    checkpoint_sources,
    precisions,
)

__all__ = [
    "config_sources",
    "image_processor",
    "flow_prompt",
    "Config",
    "TransformerConfig",
    "read_config",
    "ImageConditioning",
    "DenoiserInput",
    "Denoiser",
    "Model",
    "entry_points",
    "Transformer",
    "TransformerLayer",
    "checkpoint_sources",
    "checkpoint_mappings",
    "precisions",
    "checkpoint_precision",
]
