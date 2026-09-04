"""Checkpoint-exact resident MiniMax H3 audio decoder."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["MiniMaxH3AudioVAE"]


class MiniMaxH3AudioVAE(nn.Module):
    """Decodes H3 audio latents into bounded stereo PCM waveforms."""

    def __init__(self, vae: nn.Module) -> None:
        """Bind a pretrained decoder and materialize its latent normalization statistics."""

        super().__init__()
        self.vae = vae.float()
        if not hasattr(vae, "decode"):
            raise TypeError("MiniMax H3 audio VAE does not expose decode")
        mean = vae.config.latents_mean
        std = vae.config.latents_std
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

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: torch.device,
        local_files_only: bool = False,
    ) -> "MiniMaxH3AudioVAE":
        """Load the checkpoint audio decoder in FP32 on the target device."""

        try:
            from diffusers import AutoencoderKLMiniMaxH3Audio
        except ImportError as error:
            raise RuntimeError("MiniMax H3 requires the diffusers audio VAE runtime") from error
        vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            checkpoint,
            subfolder="audio_vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(device=device, dtype=torch.float32)
        return cls(vae)

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
    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        """Decode `[2, 32, time]` normalized latents into interleaved stereo PCM16 samples."""

        if normalized_latents.ndim != 3 or normalized_latents.shape[:2] != (2, 32):
            raise ValueError("the H3 audio latent must have shape [2, 32, time]")
        return self._decode(normalized_latents)

    @torch.inference_mode()
    def warmup_decoder(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        """Exercise the same bounded decode path used for request audio reconstruction."""

        if normalized_latents.ndim != 3 or normalized_latents.shape[:2] != (2, 32):
            raise ValueError("the H3 audio latent must have shape [2, 32, time]")
        return self._decode(normalized_latents)
