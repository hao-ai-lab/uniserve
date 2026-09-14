"""H3 latent packing and temporal reconstruction components."""

from __future__ import annotations

import math

import torch
from torch import nn

from uniserve.model.batch import TensorOutput
from uniserve.model.media import DecodeWindow, VideoSize
from uniserve.model.tensors import TensorViews
from uniserve.tensors import BufferConfig, OutputLayout
from uniserve_models.minimax_h3.audio_vae import AudioDecoderConfig, MiniMaxH3AudioVAE
from uniserve_models.minimax_h3.layout import validate_frames
from uniserve_models.minimax_h3.output import decode_windows
from uniserve_models.minimax_h3.packing import (
    audio_latent_frames,
    build_packed_layout,
    unpatchify_video_into,
    video_latent_frames,
)
from uniserve_models.minimax_h3.video_vae import MiniMaxH3VideoVAE, VideoDecoderConfig


class VideoDecoder(nn.Module):
    """Decode complete packed video latents through the resident native VAE.

    Configuration is retained on nonresident ranks for global output sizing.
    The optional native module is the only owner of learned decoder weights.
    """

    def __init__(self, config: VideoDecoderConfig, native: MiniMaxH3VideoVAE | None) -> None:
        super().__init__()
        self.config = config
        self.native = native

    def output_layout(self, size: VideoSize, *, units: int) -> dict[str, OutputLayout]:
        validate_frames(size.frames)
        if not 1 <= units <= len(decode_windows(size.frames)):
            raise ValueError("video output units must lie within the complete video")
        return {
            "video": OutputLayout(
                (units, 1, 3, 25, self.config.height, self.config.width),
                torch.float16,
                variable_axes=(0,),
            )
        }

    def workspace_buffers(self, size: VideoSize) -> dict[str, BufferConfig]:
        validate_frames(size.frames)
        return {
            "video_input": BufferConfig((1, 24, 7, 48, 84), torch.float32),
            "reconstruction_rows": BufferConfig((7 * 24 * 42, 96), torch.float32),
        }

    def constant_buffers(self, size: VideoSize) -> dict[str, BufferConfig]:
        validate_frames(size.frames)
        return {
            "video_raster_order": BufferConfig(
                (video_latent_frames(size.frames) * 24 * 42,), torch.int64
            )
        }

    @torch.inference_mode()
    def prepare_constants(self, size: VideoSize, *, out: TensorViews) -> None:
        configs = self.constant_buffers(size)
        if out.keys() != configs.keys():
            raise ValueError("video decoding requires its raster-order indices")
        target = out["video_raster_order"]
        config = configs["video_raster_order"]
        if tuple(target.shape) != config.shape or target.dtype != config.dtype:
            raise ValueError("video raster-order indices have incompatible shape or dtype")
        packed = build_packed_layout(
            text_rows=64, num_frames=size.frames, audio_frames=audio_latent_frames(size.frames)
        )
        target.copy_(torch.argsort(packed.video_raster_indices))

    @torch.inference_mode()
    def decode(
        self,
        latents: tuple[torch.Tensor, ...],
        size: VideoSize,
        windows: tuple[DecodeWindow, ...],
        *,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        if self.native is None:
            raise ValueError("this partition does not participate in video decoding")
        legal = decode_windows(size.frames)
        if (
            not latents
            or len(windows) != len(latents)
            or any(window not in legal for window in windows)
        ):
            raise ValueError("video decoding requires explicitly assigned legal windows")
        rows = video_latent_frames(size.frames) * 24 * 42
        if any(tuple(value.shape) != (rows, 96) for value in latents):
            raise ValueError("video decoder requires complete final latent rows")
        values = []
        for latent, window in zip(latents, windows, strict=True):
            start = window.latent_start * 24 * 42
            indices = constants["video_raster_order"][start : start + 7 * 24 * 42]
            torch.index_select(latent, 0, indices, out=scratch["reconstruction_rows"])
            unpatchify_video_into(
                scratch["reconstruction_rows"],
                scratch["video_input"],
                frames=7,
                height=48,
                width=84,
            )
            value = self.native(scratch["video_input"]).unsqueeze(0)
            # Native graph bindings may reuse one output address. Preserve each
            # independent logical result before the next decoder invocation.
            values.append(value.clone() if len(latents) > 1 else value)
        return TensorOutput(
            {"video": tuple(values)},
            {"video": tuple(OutputLayout(tuple(value.shape), value.dtype) for value in values)},
        )


class AudioDecoder(nn.Module):
    """Reconstruct channel-major latent rows into interleaved stereo PCM.

    The caller supplies the target sample count. Video frame rates and request
    duration are not part of the audio network configuration or computation.
    """

    def __init__(self, config: AudioDecoderConfig, native: MiniMaxH3AudioVAE | None) -> None:
        super().__init__()
        self.config = config
        self.native = native

    def output_layout(self, samples: int) -> dict[str, OutputLayout]:
        if samples < 1:
            raise ValueError("audio output requires a positive sample count")
        return {"audio": OutputLayout((samples, 2), torch.int16, variable_axes=(0,))}

    def latent_frames(self, samples: int) -> int:
        """Return the native encoder timeline needed to cover PCM samples."""

        if samples < 1:
            raise ValueError("audio duration must contain samples")
        return math.ceil(samples / math.prod(self.config.encoder_rates))

    def workspace_buffers(self, latent_frames: int) -> dict[str, BufferConfig]:
        if latent_frames < 1:
            raise ValueError("audio reconstruction requires positive latent frames")
        return {
            "audio_latents": BufferConfig(
                (2, self.config.latent_channels, latent_frames), torch.float32
            )
        }

    @torch.inference_mode()
    def decode(
        self, latents: tuple[torch.Tensor, ...], *, samples: int, scratch: TensorViews
    ) -> TensorOutput:
        if self.native is None:
            raise ValueError("this partition does not participate in audio decoding")
        if not latents or samples < 1:
            raise ValueError("audio decoding requires latent rows and a positive sample count")
        inputs = scratch["audio_latents"]
        if inputs.ndim != 3 or inputs.shape[:2] != (2, self.config.latent_channels):
            raise ValueError("audio scratch must represent two channel-major latent sequences")
        frames = inputs.shape[-1]
        if any(
            tuple(value.shape) != (2 * frames, self.config.latent_channels) for value in latents
        ):
            raise ValueError("audio decoder requires one complete stereo latent")
        values = []
        for latent in latents:
            inputs.copy_(latent.view(2, frames, self.config.latent_channels).permute(0, 2, 1))
            decoded = self.native(inputs)
            if decoded.shape[0] < samples:
                raise RuntimeError("audio decoder returned less than the requested duration")
            value = decoded[:samples]
            values.append(value.clone() if len(latents) > 1 else value)
        return TensorOutput(
            {"audio": tuple(values)},
            {"audio": tuple(OutputLayout(tuple(value.shape), value.dtype) for value in values)},
        )
