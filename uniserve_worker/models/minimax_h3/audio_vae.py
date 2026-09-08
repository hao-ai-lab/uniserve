"""Checkpoint-exact resident MiniMax H3 audio decoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = ["MiniMaxH3AudioVAE"]


class MiniMaxH3AudioVAE(nn.Module):
    """Decodes H3 audio latents into bounded stereo PCM waveforms."""

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
