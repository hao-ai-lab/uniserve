"""MiniMax H3 capability composition and numerical entry points."""

from __future__ import annotations

from types import MappingProxyType

from torch import nn

from uniserve.model import EntryPoint

from .config import Config
from .decoding import AudioDecoder, VideoDecoder
from .denoiser import Denoiser
from .encoder import TextEncoder
from .output import VideoPostprocessor


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


# IPC component names select methods on independently placeable numerical modules.
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
