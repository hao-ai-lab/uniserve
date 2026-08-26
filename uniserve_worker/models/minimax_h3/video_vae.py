"""Fixed-profile temporal video decode over the checkpoint MiniMax H3 VAE."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = ["MiniMaxH3VideoVAE"]


class MiniMaxH3VideoVAE(nn.Module):
    """Own one resident FP32 checkpoint VAE and decode one finalized unit."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae.float()
        required = (
            "_decode_clip",
            "tokens_chunk_size",
            "token_overlap",
            "frame_pre_padding",
            "frame_overlap",
        )
        missing = [name for name in required if not hasattr(vae, name)]
        if missing:
            raise TypeError(f"MiniMax H3 video VAE is missing {missing!r}")
        mean = tuple(float(value) for value in vae.config.latents_mean)
        std = tuple(float(value) for value in vae.config.latents_std)
        if len(mean) != 24 or len(std) != 24:
            raise ValueError("MiniMax H3 video VAE must declare 24-channel latent statistics")
        self.register_buffer(
            "latents_mean",
            torch.tensor(mean, dtype=torch.float32, device=self.device).view(
                1, 24, 1, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32, device=self.device).view(
                1, 24, 1, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor(
                (0.485, 0.456, 0.406),
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(
                (0.229, 0.224, 0.225),
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: torch.device,
        local_files_only: bool = False,
    ) -> "MiniMaxH3VideoVAE":
        try:
            from diffusers import AutoencoderKLMiniMaxH3
        except ImportError as error:
            raise RuntimeError("MiniMax H3 requires the diffusers VAE runtime") from error
        vae = AutoencoderKLMiniMaxH3.from_pretrained(
            checkpoint,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(device=device, dtype=torch.float32)
        return cls(vae)

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def _decode_segment(self, latents: torch.Tensor) -> torch.Tensor:
        span = int(self.vae.tokens_chunk_size) + int(self.vae.token_overlap)
        clip = self.vae._decode_clip(latents[:, :, :span])
        return clip[:, :, int(self.vae.frame_pre_padding) :]

    @torch.inference_mode()
    def decode_unit(
        self,
        normalized_latents: torch.Tensor,
        unit: int,
        previous_overlap: torch.Tensor | None,
        *,
        final_unit: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one RGB24 unit and the decoder overlap for its successor."""

        if normalized_latents.shape != (1, 24, 7, 48, 84):
            raise ValueError(
                "an H3 video decode unit must have shape [1, 24, 7, 48, 84]"
            )
        latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        latents = latents * self.latents_std + self.latents_mean
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            segment = self._decode_segment(latents)
        body_frames = (
            int(self.vae.tokens_chunk_size)
            * int(self.vae.temporal_compression_ratio)
            - int(self.vae.frame_pre_padding)
        )
        body = segment[:, :, :body_frames]
        if previous_overlap is not None:
            body = self.vae._blend(
                previous_overlap,
                body,
                int(self.vae.frame_overlap),
                dim=-3,
            )
        next_overlap = segment[
            :, :, body_frames + int(self.vae.frame_pre_padding) :
        ].contiguous()
        if final_unit:
            body = torch.cat((body, next_overlap[:, :, :5]), dim=2)
        pixels = (body.float() * self.pixel_std + self.pixel_mean).clamp_(0.0, 1.0)
        rgb24 = (
            pixels[0]
            .permute(1, 2, 3, 0)
            .mul_(255.0)
            .round_()
            .to(torch.uint8)
            .contiguous()
        )
        return rgb24, next_overlap[:, :, :5].contiguous()

    def compile_decoder(self) -> None:
        self.vae.decoder = torch.compile(self.vae.decoder, fullgraph=True)
