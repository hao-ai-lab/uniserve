"""MiniMax H3 capability composition and numerical entry points."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from torch import nn

from uniserve.model import ComponentEntry, EntryPoint

from .config import Config
from .decoding import AudioDecoder, VideoDecoder
from .denoiser import Denoiser
from .encoder import TextEncoder
from .output import VideoPostprocessor


class Model(nn.Module):
    """Compose shared numerical capabilities without retaining execution owners."""  # noqa: E501

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config.text_encoder)
        self.denoiser = Denoiser(config.denoiser, config.diffusion)
        self.video_decoder = VideoDecoder(
            config.video_decoder, frame_size=config.output.frame_size
        )
        self.audio_decoder = AudioDecoder(
            config.audio_decoder, sample_rate=config.output.sample_rate
        )
        self.video_postprocessor = VideoPostprocessor(
            frame_size=config.output.frame_size,
            frame_rate=config.output.frame_rate,
        )


# IPC component names select methods on independently placeable numerical
# modules.
def entry_points(config: Config) -> Mapping[str, ComponentEntry]:
    """Declare each IPC entry's owning component and its callable stages."""
    return MappingProxyType(
        {
            "text_encoder": ComponentEntry(
                "text_encoder", (EntryPoint("encode", groups=("tp",)),)
            ),
            # The conditioner runs on the first pipeline stage only, the stage
            # whose ``Denoiser.forward`` scatters refined text into the packed
            # token rows.
            "denoiser": ComponentEntry(
                "denoiser",
                (
                    EntryPoint(
                        "conditioner.encode", stage="first", groups=("tp",)
                    ),
                    EntryPoint(
                        "forward", groups=("tp", "sp", "pp", "cp", "ulysses")
                    ),
                ),
            ),
            # The rank that reconstructs a media unit also converts it to RGB,
            # so the decoder and its post-processor are one component. The
            # entry owns two sibling modules, which the empty component path
            # names.
            "video_decoder": ComponentEntry(
                "",
                (
                    EntryPoint("video_decoder.decode"),
                    EntryPoint("video_postprocessor.forward"),
                ),
            ),
            "audio_decoder": ComponentEntry(
                "audio_decoder", (EntryPoint("decode"),)
            ),
        }
    )
