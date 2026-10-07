"""MiniMax H3 capability composition and numerical entry points."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from torch import nn

from uniserve.model import ComponentEntry, EntryPoint

from .config import Config
from .decoding import AudioDecoder, VideoDecoder
from .denoiser import Denoiser
from .encoder import text_encoder
from .encoding import AudioEncoder, VideoEncoder
from .output import VideoPostprocessor


class Model(nn.Module):
    """Compose shared numerical capabilities without retaining execution owners.

    Each DiT partition the checkpoint holds is its own denoising component:
    ``denoiser`` (the ``transformer`` partition) and ``reference_denoiser``
    (``transformer_ref``); a deployment places one of them. The video
    decoder reconstructs the latent order those denoisers publish, so a
    checkpoint's denoisers must share one attention kind. The video and audio
    encoders turn condition pixels and soundtracks into the latent rows the
    denoisers condition on.
    """

    denoiser: Denoiser | None
    reference_denoiser: Denoiser | None

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.text_encoder = text_encoder(config.text_encoder)
        for name in ("denoiser", "reference_denoiser"):
            denoiser = config.denoisers.get(name)
            setattr(
                self, name, None if denoiser is None else Denoiser(denoiser)
            )
        kinds = {
            type(denoiser.attention) for denoiser in config.denoisers.values()
        }
        if len(kinds) != 1:
            raise ValueError(
                "an H3 checkpoint's denoisers must share one attention kind"
            )
        attention = next(iter(config.denoisers.values())).attention
        self.video_decoder = VideoDecoder(config.video_vae, attention=attention)
        self.audio_decoder = AudioDecoder(
            config.audio_vae, sample_rate=config.output.sample_rate
        )
        self.video_encoder = VideoEncoder(config.video_vae)
        self.audio_encoder = AudioEncoder(
            config.audio_vae, sample_rate=config.output.sample_rate
        )
        self.video_postprocessor = VideoPostprocessor(
            frame_rate=config.output.frame_rate
        )


def _denoiser_entry(name: str) -> ComponentEntry:
    # The conditioner runs on the first pipeline stage only, the stage whose
    # ``Denoiser.forward`` writes refined text into the packed token rows.
    return ComponentEntry(
        name,
        (
            EntryPoint("conditioner.encode", stage="first", groups=("tp",)),
            EntryPoint("forward", groups=("tp", "sp", "pp", "cp", "ulysses")),
        ),
    )


# IPC component names select methods on independently placeable numerical
# modules.
def entry_points(config: Config) -> Mapping[str, ComponentEntry]:
    """Declare each IPC entry's owning component and its callable stages."""
    return MappingProxyType(
        {
            # The conditioner reads prompts with their vision tokens spliced
            # in, so its vision tower is part of the same component.
            "text_encoder": ComponentEntry(
                "text_encoder",
                (
                    EntryPoint("encode", groups=("tp",)),
                    EntryPoint("vision.encode", groups=("tp",)),
                ),
            ),
            **{name: _denoiser_entry(name) for name in config.denoisers},
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
            # Both condition encoders run where condition latents are made;
            # like the video decoder, the entry owns two sibling modules.
            "latent_encoder": ComponentEntry(
                "",
                (
                    EntryPoint("video_encoder.encode"),
                    EntryPoint("audio_encoder.encode"),
                ),
            ),
        }
    )
