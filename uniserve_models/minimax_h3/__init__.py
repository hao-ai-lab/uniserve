"""MiniMax H3 numerical models and checkpoint definitions.

H3 generates a video with a stereo audio track from a text prompt and
optional image, video and audio conditions. The package composes
independently placeable components (see ``entry_points``): the Qwen text
encoder, one denoiser per DiT partition the checkpoint holds (each with its
token refiner), the video decoder with its RGB post-processor, and the audio
decoder. ``read_config`` recognizes the base release or a FastH3 export and
normalizes its JSON sidecars, ``base_checkpoint`` names the base revision a
FastH3 export reads its omitted components from, and
``checkpoint_mappings`` maps the tensors onto the modules.
"""

from .attention import Dense, SegmentSparse, Sparse
from .checkpoint import base_checkpoint
from .conditioning import Conditioner, RefinerBlock, TokenRefiner
from .config import (
    Config,
    DenoiserConfig,
    DenseAttention,
    SparseAttention,
    TransformerConfig,
    config_sources,
    read_config,
)
from .decoding import AudioDecoder, VideoDecoder
from .denoiser import Denoiser
from .encoder import TextEncoderConfig, text_encoder
from .encoding import AudioEncoder, VideoEncoder
from .inputs import (
    AttentionInput,
    DenoiserInput,
    DenoiserSize,
    SegmentInput,
    SequenceInput,
)
from .model import Model, entry_points
from .modulation import OutputNorm, TimestepEmbedding
from .output import VideoPostprocessor
from .packing import DensePacking, SegmentPacking, TilePacking
from .precision import checkpoint_precision, precisions, weight_config
from .transformer import StepProjection, Transformer, TransformerLayer
from .weights import checkpoint_mappings, checkpoint_sources

# The model loader reads these package attributes. H3 accepts no image inputs
# and runs without classifier-free guidance, so it supplies neither an image
# processor nor guidance prompt framing.
image_processor = None
flow_prompt = None

__all__ = [
    "base_checkpoint",
    "config_sources",
    "image_processor",
    "flow_prompt",
    "Model",
    "Config",
    "Dense",
    "SegmentSparse",
    "Sparse",
    "Conditioner",
    "RefinerBlock",
    "TokenRefiner",
    "DenoiserConfig",
    "DenseAttention",
    "SparseAttention",
    "TransformerConfig",
    "AudioDecoder",
    "AudioEncoder",
    "VideoDecoder",
    "VideoEncoder",
    "Denoiser",
    "TextEncoderConfig",
    "AttentionInput",
    "DenoiserInput",
    "DenoiserSize",
    "SegmentInput",
    "SequenceInput",
    "OutputNorm",
    "TimestepEmbedding",
    "VideoPostprocessor",
    "DensePacking",
    "SegmentPacking",
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
    "text_encoder",
    "weight_config",
]
