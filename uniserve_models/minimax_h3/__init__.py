"""MiniMax H3 numerical models and checkpoint definitions.

H3 generates a video with a stereo audio track from a text prompt. The
package composes four independently placeable components (see
``entry_points``): the Qwen text encoder, the denoiser with its token
refiner, the video decoder with its RGB post-processor, and the audio
decoder. ``read_config`` normalizes the checkpoint's JSON sidecars and
``checkpoint_mappings`` maps its tensors onto the modules.
"""

from .attention import Attention
from .conditioning import Conditioner, RefinerBlock, TokenRefiner
from .config import (
    Config,
    DiffusionConfig,
    TransformerConfig,
    config_sources,
    read_config,
)
from .decoding import AudioDecoder, VideoDecoder
from .denoiser import Denoiser
from .encoder import TextEncoder, TextEncoderConfig
from .inputs import AttentionInput, DenoiserInput, DenoiserSize
from .model import Model, entry_points
from .modulation import OutputNorm, TimestepEmbedding
from .output import VideoPostprocessor
from .packing import Packing
from .precision import checkpoint_precision, precisions, weight_config
from .transformer import Transformer, TransformerLayer
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
    "Attention",
    "Conditioner",
    "RefinerBlock",
    "TokenRefiner",
    "DiffusionConfig",
    "TransformerConfig",
    "AudioDecoder",
    "VideoDecoder",
    "Denoiser",
    "TextEncoder",
    "TextEncoderConfig",
    "AttentionInput",
    "DenoiserInput",
    "DenoiserSize",
    "OutputNorm",
    "TimestepEmbedding",
    "VideoPostprocessor",
    "Packing",
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
