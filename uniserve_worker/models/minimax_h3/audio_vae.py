"""Checkpoint-exact resident MiniMax H3 audio decoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...nn.vae.decoder import LatentDecoder

__all__ = ["MiniMaxH3AudioVAE"]


class MiniMaxH3AudioVAE(LatentDecoder):
    """Decodes H3 audio latents into bounded stereo PCM waveforms."""

    latent_shape = (2, 32, None)
    vae: Any
    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(self, vae: nn.Module) -> None:
        """Compose an audio decoder with its configured latent normalization statistics."""

        super().__init__()
        self.vae = vae.float()
        if not hasattr(vae, "decode"):
            raise TypeError("MiniMax H3 audio VAE does not expose decode")
        mean = self.vae.config.latents_mean
        std = self.vae.config.latents_std
        if mean is None or std is None or len(mean) != 32 or len(std) != 32:
            raise ValueError("MiniMax H3 audio VAE must declare 32-channel latent statistics")
        # Keep constant values when parameter storage is deferred. The public
        # loader stages graph buffers after materializing the learned modules.
        statistics_device = "cpu" if self.device.type == "meta" else self.device
        self.register_buffer(
            "latents_mean",
            torch.tensor(mean, dtype=torch.float32, device=statistics_device).view(1, 32, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32, device=statistics_device).view(1, 32, 1),
            persistent=False,
        )

    def _reconstruct(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert native decoder output to interleaved signed-16 stereo."""

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
