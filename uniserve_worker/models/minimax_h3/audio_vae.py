"""Checkpoint-exact resident MiniMax H3 audio autoencoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layout import PROFILE_AUDIO_RATE, PROFILE_FPS, H3ComputeInputs

__all__ = ["MiniMaxH3AudioVAE"]


class MiniMaxH3AudioVAE(nn.Module):
    """Encode reference stereo and decode generated H3 audio latents."""

    vae: Any
    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(self, vae: nn.Module) -> None:
        """Bind a pretrained decoder and materialize its latent normalization statistics."""

        super().__init__()
        self.vae = vae.float()
        if not hasattr(vae, "decode"):
            raise TypeError("MiniMax H3 audio VAE does not expose decode")
        mean = self.vae.config.latents_mean
        std = self.vae.config.latents_std
        if mean is None or std is None or len(mean) != 32 or len(std) != 32:
            raise ValueError("MiniMax H3 audio VAE must declare 32-channel latent statistics")
        self.register_buffer(
            "latents_mean",
            torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 32, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 32, 1),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Identify the execution device from the resident decoder parameters."""

        return next(self.vae.parameters()).device

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode prepared 32-kHz stereo `[2, samples]` to `[2, 32, time]`.

        The caller owns resampling and duration limits. Channels are independent
        mono examples for the checkpoint VAE. Ref2VA uses the posterior mode,
        not a sample, and normalizes in FP32 without visual noise augmentation.
        """

        if waveform.ndim != 2 or waveform.shape[0] != 2 or waveform.shape[1] < 1:
            raise ValueError("H3 reference audio must have shape [2, samples] with samples > 0")
        if not waveform.is_floating_point():
            raise ValueError("H3 reference audio must be a floating-point waveform")
        if not hasattr(self.vae, "encode"):
            raise TypeError("MiniMax H3 audio VAE does not expose encode")
        samples = waveform.to(device=self.device, dtype=torch.float32)
        latents = self.vae.encode(samples[:, None]).latent_dist.mode().float()
        if latents.ndim != 3 or latents.shape[:2] != (2, 32) or latents.shape[2] < 1:
            raise RuntimeError("MiniMax H3 audio encoder returned invalid stereo geometry")
        return (latents - self.latents_mean) / self.latents_std

    def _decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        """Denormalize audio latents and convert decoder output to interleaved signed-16 stereo."""

        latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        latents = latents * self.latents_std + self.latents_mean
        decoded = self.vae.decode(latents).sample.float()
        if decoded.ndim != 3 or decoded.shape[:2] != (2, 1):
            raise RuntimeError("MiniMax H3 audio decoder returned invalid stereo geometry")
        # PyAV accepts interleaved signed-16 stereo as [samples, channels].
        return (
            decoded[:, 0]
            .transpose(0, 1)
            .clamp_(-1.0, 1.0)
            .mul_(32767.0)
            .round_()
            .to(torch.int16)
            .contiguous()
        )

    @torch.inference_mode()
    def forward(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        """Decode `[2, 32, time]` normalized latents into interleaved stereo PCM16 samples."""

        if normalized_latents.ndim != 3 or normalized_latents.shape[:2] != (2, 32):
            raise ValueError("the H3 audio latent must have shape [2, 32, time]")
        return self._decode(normalized_latents)

    def prepare_input(
        self,
        execution: H3ComputeInputs,
        latents: torch.Tensor,
        cursor: int,
        max_units: int,
    ) -> torch.Tensor:
        """Pack complete channel-major audio rows into decoder layout."""

        layout, scratch = execution.layout, execution.media
        if (
            cursor != 0
            or max_units != 1
            or tuple(latents.shape) != (int(layout.packed.audio_indices.numel()), 32)
        ):
            raise ValueError("audio decoder requires one complete stereo latent")
        scratch.audio_latents.copy_(
            latents.view(2, layout.packed.audio_frames, 32).permute(0, 2, 1)
        )
        return scratch.audio_latents

    @staticmethod
    def logical_output(execution: H3ComputeInputs, value: torch.Tensor) -> torch.Tensor:
        """Trim decoded PCM to the exact requested video duration."""

        samples = round(execution.layout.frame_count * PROFILE_AUDIO_RATE / PROFILE_FPS)
        if value.shape[0] < samples:
            raise RuntimeError("audio decoder returned less than the video duration")
        return value[:samples]

    def warmup_input(self, audio_frames: int) -> torch.Tensor:
        """Create one representative latent input on the decoder device."""

        return torch.zeros((2, 32, audio_frames), dtype=torch.float32, device=self.device)
