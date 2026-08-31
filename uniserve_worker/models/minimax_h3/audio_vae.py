"""Checkpoint-exact resident MiniMax H3 audio decoder."""

from __future__ import annotations

import torch
from torch import nn

from ...execution.fixed_graph import FixedShapeGraphCache

__all__ = ["MiniMaxH3AudioVAE"]


class MiniMaxH3AudioVAE(nn.Module):
    def __init__(self, vae: nn.Module) -> None:
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
        self.decode_graphs: FixedShapeGraphCache[torch.Tensor] | None = None
        self.register_buffer("decode_graph_storage", None, persistent=False)

    def configure_graph_cache(self, capacity: int, *, max_latent_frames: int) -> None:
        self.decode_graphs = FixedShapeGraphCache(self.device, capacity=capacity)
        if int(max_latent_frames) < 1:
            raise ValueError("audio graph latent capacity must be positive")
        self.decode_graph_storage = torch.empty(
            (2, 32, int(max_latent_frames)),
            dtype=torch.float32,
            device=self.device,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: torch.device,
        local_files_only: bool = False,
    ) -> "MiniMaxH3AudioVAE":
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
        return next(self.vae.parameters()).device

    def _decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
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
        if normalized_latents.ndim != 3 or normalized_latents.shape[:2] != (2, 32):
            raise ValueError("the H3 audio latent must have shape [2, 32, time]")
        if self.decode_graphs is None or self.decode_graph_storage is None:
            raise RuntimeError("the H3 audio decoder graph has not been captured")
        key = tuple(int(value) for value in normalized_latents.shape)
        if normalized_latents.shape[-1] > self.decode_graph_storage.shape[-1]:
            raise ValueError("audio decoder input exceeds the configured graph capacity")
        graph_input = self.decode_graph_storage[..., : normalized_latents.shape[-1]]
        graph_input.copy_(normalized_latents)
        return self.decode_graphs.execute(
            key,
            lambda: self._decode(graph_input),
            warmup=lambda: self._decode(graph_input),
        )

    @torch.inference_mode()
    def capture_decoder(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        if normalized_latents.ndim != 3 or normalized_latents.shape[:2] != (2, 32):
            raise ValueError("the H3 audio latent must have shape [2, 32, time]")
        return self.decode(normalized_latents)
