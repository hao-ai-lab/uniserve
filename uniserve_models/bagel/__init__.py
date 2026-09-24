"""BAGEL numerical models and checkpoint definitions.

BAGEL is a unified understanding and generation model. One mixture of
transformers (MoT) backbone carries a text expert for language tokens and a
flow expert for image latents; a SigLIP tower encodes input images; and a
FLUX autoencoder maps between pixels and the latents the flow expert
denoises. ``uniserve_models.loading`` selects this package for checkpoints
declaring ``BagelForConditionalGeneration``.
"""

from .config import Config, TransformerConfig, config_sources, read_config
from .denoiser import Denoiser
from .inputs import DenoiserInput
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
