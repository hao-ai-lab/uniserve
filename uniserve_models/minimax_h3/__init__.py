"""MiniMax H3 numerical models and checkpoint definitions.

H3 generates a video with a stereo audio track from a text prompt and
optional image, video and audio conditions. The package composes
independently placeable components (see ``entry_points``): the Qwen text
encoder, one denoiser per DiT partition the checkpoint holds (each with its
token refiner), the video decoder with its RGB post-processor, and the audio
decoder. ``read_config`` recognizes the checkpoint layout and normalizes its
JSON sidecars, and ``checkpoint_mappings`` maps its tensors onto the
modules.
"""

from .attention import Dense, Sparse
from .conditioning import Conditioner, RefinerBlock, TokenRefiner
from .config import (
    Config,
    DenoiserConfig,
    DenseAttention,
    DmdLadder,
    PddGrid,
    SparseAttention,
    TransformerConfig,
    UniformGrid,
    config_sources,
    read_config,
)
from .decoding import AudioDecoder, VideoDecoder
from .denoiser import Denoiser
from .encoder import TextEncoder, TextEncoderConfig
from .encoding import AudioEncoder, VideoEncoder
from .inputs import AttentionInput, DenoiserInput, DenoiserSize, SequenceInput
from .model import Model, entry_points
from .modulation import OutputNorm, TimestepEmbedding
from .output import VideoPostprocessor
from .packing import DensePacking, TilePacking
from .precision import checkpoint_precision, precisions, weight_config
from .transformer import StepProjection, Transformer, TransformerLayer
from .weights import checkpoint_mappings, checkpoint_sources

# The model loader reads these package attributes. H3 accepts no image inputs
# and runs without classifier-free guidance, so it supplies neither an image
# processor nor guidance prompt framing.
image_processor = None
flow_prompt = None

__all__ = [
    "config_sources",
    "image_processor",
    "flow_prompt",
    "Model",
    "Config",
    "Dense",
    "Sparse",
    "Conditioner",
    "RefinerBlock",
    "TokenRefiner",
    "DenoiserConfig",
    "DenseAttention",
    "DmdLadder",
    "PddGrid",
    "SparseAttention",
    "TransformerConfig",
    "UniformGrid",
    "AudioDecoder",
    "AudioEncoder",
    "VideoDecoder",
    "VideoEncoder",
    "Denoiser",
    "TextEncoder",
    "TextEncoderConfig",
    "AttentionInput",
    "DenoiserInput",
    "DenoiserSize",
    "SequenceInput",
    "OutputNorm",
    "TimestepEmbedding",
    "VideoPostprocessor",
    "DensePacking",
    "TilePacking",
    "StepProjection",
    "Transformer",
    "TransformerLayer",
    "read_config",
    "checkpoint_sources",
    "checkpoint_mappings",
    "entry_points",
    "precisions",
    "checkpoint_precision",
    "weight_config",
]
