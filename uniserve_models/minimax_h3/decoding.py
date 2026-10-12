"""H3 latent unpacking for the video and audio decoders.

The denoiser publishes each modality's final latent in its packed order:
video rows in raster order under dense attention and tile-major under sparse
attention (``packing.video_order``), and audio rows channel-major, all frames
of the first stereo channel before the second. The decoders here convert a
clip of that latent into the native VAE input layout at the request's
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
from .config import Attention, DenseAttention
from .output import video_clips
from .packing import (
    audio_latent_frames,
    latent_raster,
    unpatchify_video_into,
    video_latent_frames,
    video_order,
)


class VideoDecoder(BaseVideoDecoder):
    """Unpack packed H3 video latents into each clip's seven-frame VAE input.

    Every clip decodes seven latent frames into a native 25-frame clip output
    at the output raster, so a clip's decode depends on the raster alone.
    ``attention`` is the attention kind of the
    denoisers whose latents this decoder reconstructs; it fixes their packed
    video row order.
    """

    def __init__(self, config: video_vae.Config, *, attention: Attention):
        super().__init__(video_vae.Model(config))
        self.config = config
        self.attention = attention

    def clips(self, num_frames: int) -> tuple[slice, ...]:
        return video_clips(num_frames)

    def output_layout(self, size: video.Config) -> Mapping[str, OutputLayout]:
        count = len(self.clips(size.num_frames))
        # Each clip decodes one native 25-frame RGB clip output at the output
        # raster.
        shape = (count, 1, 3, 25, size.frame.height, size.frame.width)
        return {
            "video": OutputLayout(
                shape,
                torch.float16,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }

    def clip_output(self, size: video.Config, clip: slice) -> video.Config:
        if clip not in self.clips(size.num_frames):
            raise ValueError("video clip must be one of the video's clips")
        return video.Config(25, size.frame)

    def clip_input(self, output: video.Config) -> BufferConfig:
        if output.num_frames != 25:
            raise ValueError("an H3 clip decodes 25 native frames")
        height, width = latent_raster(output.frame)
        # The NCTHW decoder input of seven latent frames.
        return BufferConfig(
            (1, self.config.latent_channels, 7, height, width), torch.float32
        )

    def unpack_latents(self, latent, clip, size, *, out):
        config = self.clip_input(self.clip_output(size, clip))
        if tuple(out.shape) != config.shape or out.dtype != config.dtype:
            raise ValueError("video clip input has an incompatible layout")
        height, width = latent_raster(size.frame)
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

        # Clip k decodes latent frames [5k, 5k + 7): five frames that advance
        # the timeline and two trailing frames that the next clip's input
        # also reads. Their raster rows are [start, stop).
        number = clip.start // 17
        start = number * 5 * tokens_per_frame
        stop = start + 7 * tokens_per_frame
        if isinstance(self.attention, DenseAttention):
            # Dense packing keeps the raster order.
            tokens = latent[start:stop]
        else:
            # Tile and segment packing order the rows tile-major, which
            # depends on the frame count; the argsort maps each raster row to
            # its position among the packed rows.
            positions = torch.argsort(
                video_order(
                    self.attention,
                    num_frames=size.num_frames,
                    canvas=size.frame,
                )
            )[start:stop]
            if latent.device.type == "cuda":
                # A pinned source keeps the transfer asynchronous; a pageable
                # one would wait for the work already queued on the stream.
                positions = positions.pin_memory().to(
                    latent.device, non_blocking=True
                )
            tokens = latent.index_select(0, positions)
        unpatchify_video_into(
            tokens,
            out,
            frames=7,
            height=height,
            width=width,
            channels=self.config.latent_channels,
        )


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
        # channel on the H3 timeline (packing.py), whose frame rate the video
        # postprocessor reports; each latent decodes to latent_rate samples.
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
