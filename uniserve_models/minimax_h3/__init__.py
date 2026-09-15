"""MiniMax H3 numerical models and checkpoint definitions."""

from types import MappingProxyType

from torch import nn

from uniserve.loading import checkpoint
from uniserve.model import EntryPoint

from .attention import Attention
from .conditioning import Conditioner, RefinerBlock, TokenRefiner
from .config import Config, DiffusionConfig, TransformerConfig, read_config
from .decoding import AudioDecoder, VideoDecoder
from .diffusion import Denoiser
from .encoder import TextEncoder, TextEncoderConfig
from .inputs import AttentionInput, DenoiserInput, DenoiserSize
from .modulation import OutputNorm, TimestepEmbedding
from .output import VideoPostprocessor
from .packing import Packing
from .precision import precisions, weight_config
from .transformer import Transformer, TransformerLayer
from .weights import checkpoint_mappings

__all__ = [
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
    "entry_paths",
    "precisions",
    "weight_config",
]


class Model(nn.Module):
    """Compose shared numerical capabilities without retaining execution owners."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config.text_encoder)
        self.denoiser = Denoiser(config.denoiser, config.diffusion)
        self.video_decoder = VideoDecoder(config.video_decoder, frame_size=config.output.frame_size)
        self.audio_decoder = AudioDecoder(
            config.audio_decoder, sample_rate=config.output.sample_rate
        )
        self.video_postprocessor = VideoPostprocessor(
            frame_size=config.output.frame_size, frame_rate=config.output.frame_rate
        )


checkpoint_sources = (
    checkpoint.Config("denoiser", "transformer", module_path="denoiser"),
    checkpoint.Config("text_encoder", "text_encoder", module_path="text_encoder"),
    checkpoint.Config("video_decoder", "vae", module_path="video_decoder"),
    checkpoint.Config("audio_decoder", "audio_vae", module_path="audio_decoder"),
)

# Catalog paths preserve the IPC's logical components while Python exposes
# ordinary capability methods on independently placeable numerical modules.
entry_paths = MappingProxyType(
    {
        "text_encoder": "text_encoder.encode",
        "denoiser": "denoiser.forward",
        "video_decoder": "video_decoder.decode",
        "audio_decoder": "audio_decoder.decode",
        "output": "video_postprocessor.forward",
    }
)


def entry_points(config: Config):
    return MappingProxyType(
        {
            "text_encoder": (EntryPoint("encode", groups=("tp",)),),
            "denoiser": (
                EntryPoint("conditioner.encode", stage="first", groups=("tp",)),
                EntryPoint("forward", groups=("tp", "sp", "pp", "cp", "ulysses")),
            ),
            "video_decoder": (EntryPoint("decode"),),
            "audio_decoder": (EntryPoint("decode"),),
            "video_postprocessor": (EntryPoint("forward"),),
        }
    )
