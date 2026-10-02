"""H3 latent unpacking for the video and audio decoders.

The denoiser publishes each modality's final latent in its packed order:
video rows in raster order under dense attention and tile-major under sparse
attention (``packing.video_order``), and audio rows channel-major, all frames
of the first stereo channel before the second. The decoders here convert a
window of that latent into the native VAE input layout at the request's
canvas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from uniserve.media import video
from uniserve.model import AudioDecoder as BaseAudioDecoder
from uniserve.model import VideoDecoder as BaseVideoDecoder
from uniserve.tensors import BufferConfig, OutputLayout

from . import audio_vae, video_vae
from .config import Attention
from .output import frame_slices
from .packing import (
    FPS,
    audio_latent_frames,
    latent_raster,
    unpatchify_video_into,
    video_latent_frames,
    video_order,
)


class VideoDecoder(BaseVideoDecoder):
    """Unpack packed H3 video latents into each native seven-frame VAE window.

    ``attention`` is the attention kind of the denoisers whose latents this
    decoder reconstructs; it fixes their packed video row order.
    """

    def __init__(self, config: video_vae.Config, *, attention: Attention):
        super().__init__(video_vae.Model(config))
        self.config = config
        self.attention = attention

    def frame_slices(self, num_frames: int) -> tuple[slice, ...]:
        return frame_slices(num_frames)

    def output_layout(self, size: video.Config) -> Mapping[str, OutputLayout]:
        units = len(self.frame_slices(size.num_frames))
        # Each unit decodes one native 25-frame RGB window at the output raster.
        shape = (units, 1, 3, 25, size.frame.height, size.frame.width)
        return {
            "video": OutputLayout(
                shape,
                torch.float16,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }

    def workspace_buffers(
        self, size: video.Config
    ) -> Mapping[str, BufferConfig]:
        self.frame_slices(size.num_frames)
        height, width = latent_raster(size.frame)
        channels = self.config.latent_channels
        # One seven-latent-frame window: the NCTHW decoder input and the
        # raster patch rows gathered for it.
        return {
            "video_input": BufferConfig(
                (1, channels, 7, height, width), torch.float32
            ),
            "reconstruction_tokens": BufferConfig(
                (7 * (height // 2) * (width // 2), channels * 4), torch.float32
            ),
        }

    def constant_buffers(
        self, size: video.Config
    ) -> Mapping[str, BufferConfig]:
        self.frame_slices(size.num_frames)
        height, width = latent_raster(size.frame)
        return {
            "video_raster_order": BufferConfig(
                (
                    video_latent_frames(size.num_frames)
                    * (height // 2)
                    * (width // 2),
                ),
                torch.int64,
            )
        }

    @torch.inference_mode()
    def prepare_constants(
        self, size: video.Config, *, out: Mapping[str, torch.Tensor]
    ) -> None:
        configs = self.constant_buffers(size)
        if out.keys() != configs.keys():
            raise ValueError("video decoding requires its raster-order indices")
        target, config = (
            out["video_raster_order"],
            configs["video_raster_order"],
        )
        if target.shape != config.shape or target.dtype != config.dtype:
            raise ValueError(
                "video raster-order indices have incompatible shape or dtype"
            )
        # The argsort maps each raster row to its position among the packed
        # video rows.
        target.copy_(
            torch.argsort(
                video_order(
                    self.attention,
                    num_frames=size.num_frames,
                    canvas=size.frame,
                )
            )
        )

    def unpack_latents(self, latent, frames, size, *, constants, workspace):
        target, tokens = (
            workspace["video_input"],
            workspace["reconstruction_tokens"],
        )
        height, width = target.shape[-2:]
        tokens_per_frame = (height // 2) * (width // 2)
        shape = (
            video_latent_frames(size.num_frames) * tokens_per_frame,
            self.config.latent_channels * 4,
        )
        if latent.shape != shape:
            raise ValueError(
                f"video decoder requires complete final latent tokens "
                f"with shape {shape}"
            )
        # Unit k decodes latent frames [5k, 5k + 7): five frames that advance
        # the timeline and two trailing frames that the next unit's window
        # also reads. Gathering the window's raster rows from the packed
        # latent yields them in raster order.
        unit = frames.start // 17
        start = unit * 5 * tokens_per_frame
        indices = constants["video_raster_order"][
            start : start + 7 * tokens_per_frame
        ]
        torch.index_select(latent, 0, indices, out=tokens)
        unpatchify_video_into(
            tokens,
            target,
            frames=7,
            height=height,
            width=width,
            channels=self.config.latent_channels,
        )
        return target


class AudioDecoder(BaseAudioDecoder):
    """Convert H3 channel-major latents to interleaved stereo PCM."""

    def __init__(self, config: audio_vae.Config, *, sample_rate: int):
        super().__init__(audio_vae.Model(config), sample_rate=sample_rate)
        self.config = config

    def output_layout(self, num_samples: int) -> Mapping[str, OutputLayout]:
        self.latent_frames(num_samples)
        return {
            "audio": OutputLayout(
                (num_samples, 2),
                torch.int16,
                (slice(0, num_samples), slice(0, 2)),
                variable_axes=(0,),
            )
        }

    def latent_frames(self, num_samples: int) -> int:
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError(
                "audio duration must contain a positive sample count"
            )
        return math.ceil(num_samples / self.latent_rate)

    def track_samples(self, num_frames: int, frame_rate: int) -> int:
        # The denoiser generates audio_latent_frames(num_frames) latents per
        # channel (packing.py), and each decodes to latent_rate samples.
        if frame_rate != FPS:
            raise ValueError(f"H3 audio accompanies {FPS} fps video")
        return audio_latent_frames(num_frames) * self.latent_rate

    @property
    def latent_rate(self) -> int:
        return math.prod(self.config.encoder_rates)

    def latent_halo(self) -> int:
        return audio_vae.receptive_field(self.config)

    def workspace_buffers(
        self, latent_frames: int
    ) -> Mapping[str, BufferConfig]:
        if type(latent_frames) is not int or latent_frames < 1:
            raise ValueError(
                "audio reconstruction requires positive latent frames"
            )
        return {
            "audio_latents": BufferConfig(
                (2, self.config.latent_channels, latent_frames), torch.float32
            )
        }

    def unpack_latents(self, latent, num_samples, *, window, workspace):
        frames, channels = (
            self.latent_frames(num_samples),
            self.config.latent_channels,
        )
        if latent.shape != (2 * frames, channels):
            raise ValueError(
                "audio decoder requires one complete stereo latent timeline"
            )
        span = window.stop - window.start
        backing = workspace["audio_latents"]
        if (
            backing.ndim != 3
            or backing.shape[:2] != (2, channels)
            or backing.shape[2] < span
        ):
            raise ValueError(
                "audio workspace must cover both channel-major latent sequences"
            )
        # Repack the window's [2 * frames, channels] token rows into the VAE's
        # channel-major [stereo, channels, frames] layout.
        inputs = backing[:, :, :span]
        inputs.copy_(
            latent.reshape(2, frames, channels)[:, window, :].permute(0, 2, 1)
        )
        return inputs
