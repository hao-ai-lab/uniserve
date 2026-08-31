"""Fixed-profile temporal video decode over the checkpoint MiniMax H3 VAE."""

from __future__ import annotations

import torch
from torch import nn

from ...execution.fixed_graph import FixedShapeGraphCache
from ...nn.quant.nvfp4 import replace_nvfp4_linears
from .precision import VideoVAELinearPrecision

__all__ = ["MiniMaxH3VideoVAE"]


class MiniMaxH3VideoVAE(nn.Module):
    """Own one resident checkpoint VAE and decode temporal segments."""

    def __init__(self, vae: nn.Module, *, linear_precision: VideoVAELinearPrecision) -> None:
        super().__init__()
        self.vae = vae.float()
        self.linear_precision = linear_precision
        self.autocast_dtype = (
            torch.float16 if linear_precision == "fp16" else torch.bfloat16
        )
        if linear_precision == "nvfp4":
            replaced = replace_nvfp4_linears(self.vae.decoder)
            if replaced != 217:
                raise RuntimeError(
                    f"MiniMax H3 video decoder expected 217 aligned linear layers, got {replaced}"
                )
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
            torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 24, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 24, 1, 1, 1),
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
        self.decode_graphs: FixedShapeGraphCache[torch.Tensor] | None = None
        self.register_buffer("decode_graph_input", None, persistent=False)

    def configure_graph_cache(self, capacity: int) -> None:
        self.decode_graphs = FixedShapeGraphCache(self.device, capacity=capacity)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: torch.device,
        local_files_only: bool = False,
        linear_precision: VideoVAELinearPrecision,
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
        return cls(vae, linear_precision=linear_precision)

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def _decode_segment(self, latents: torch.Tensor) -> torch.Tensor:
        span = int(self.vae.tokens_chunk_size) + int(self.vae.token_overlap)
        clip = self._decode_spatial_tiles(latents[:, :, :span])
        return clip[:, :, int(self.vae.frame_pre_padding) :].contiguous()

    def _decode_spatial_tiles(self, latents: torch.Tensor) -> torch.Tensor:
        if not bool(self.vae.use_tiling):
            return self.vae.decoder(self.vae.post_quant_conv(latents))

        ratio = int(self.vae.spatial_compression_ratio)
        height = int(latents.shape[-2]) * ratio
        width = int(latents.shape[-1]) * ratio
        y_indices, y_lengths, y_overlaps = self.vae._split_tiles(
            height,
            int(self.vae.tile_sample_min_height),
            int(self.vae.tile_sample_min_overlap_height),
        )
        x_indices, x_lengths, x_overlaps = self.vae._split_tiles(
            width,
            int(self.vae.tile_sample_min_width),
            int(self.vae.tile_sample_min_overlap_width),
        )
        tiles = torch.cat(
            tuple(
                latents[
                    ...,
                    y_pos // ratio : y_pos // ratio + y_length // ratio,
                    x_pos // ratio : x_pos // ratio + x_length // ratio,
                ]
                for y_pos, y_length in zip(y_indices, y_lengths, strict=True)
                for x_pos, x_length in zip(x_indices, x_lengths, strict=True)
            ),
            dim=0,
        )
        decoded = self.vae.decoder(self.vae.post_quant_conv(tiles))
        flat_tiles = decoded.split(1, dim=0)
        columns = len(x_indices)
        rows = [
            list(flat_tiles[start : start + columns])
            for start in range(0, len(flat_tiles), columns)
        ]
        return self.vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _decode_normalized_segment(
        self,
        normalized_latents: torch.Tensor,
    ) -> torch.Tensor:
        latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        latents = latents * self.latents_std + self.latents_mean
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.device.type == "cuda",
        ):
            return self._decode_segment(latents).to(torch.float16)

    @torch.inference_mode()
    def decode_segment(
        self,
        normalized_latents: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_latents.shape != (1, 24, 7, 48, 84):
            raise ValueError("an H3 video decode unit must have shape [1, 24, 7, 48, 84]")
        if self.decode_graphs is None or self.decode_graph_input is None:
            raise RuntimeError("the H3 video decoder graph has not been captured")
        self.decode_graph_input.copy_(normalized_latents)
        return self.decode_graphs.execute(
            tuple(int(value) for value in normalized_latents.shape),
            lambda: self._decode_normalized_segment(self.decode_graph_input),
            warmup=lambda: self._decode_normalized_segment(self.decode_graph_input),
        )

    @torch.inference_mode()
    def assemble_segment(
        self,
        segment: torch.Tensor,
        previous_overlap: torch.Tensor | None,
        *,
        final_unit: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Join one decoded segment and return RGB24 frames plus its successor overlap."""

        if segment.shape != (1, 3, 25, 768, 1344):
            raise ValueError("an H3 decoded video segment must have shape [1, 3, 25, 768, 1344]")
        body_frames = int(self.vae.tokens_chunk_size) * int(
            self.vae.temporal_compression_ratio
        ) - int(self.vae.frame_pre_padding)
        body = segment[:, :, :body_frames]
        if previous_overlap is not None:
            body = self.vae._blend(
                previous_overlap,
                body,
                int(self.vae.frame_overlap),
                dim=-3,
            )
        next_overlap = segment[:, :, body_frames + int(self.vae.frame_pre_padding) :].contiguous()
        if final_unit:
            body = torch.cat((body, next_overlap[:, :, :5]), dim=2)
        pixels = (body.float() * self.pixel_std + self.pixel_mean).clamp_(0.0, 1.0)
        rgb24 = pixels[0].permute(1, 2, 3, 0).mul_(255.0).round_().to(torch.uint8).contiguous()
        return rgb24, next_overlap[:, :, :5].contiguous()

    def compile_decoder(self) -> None:
        self.vae.decoder = torch.compile(self.vae.decoder, fullgraph=True)

    @torch.inference_mode()
    def capture_decoder(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        """Warm and capture the fixed-shape segment decoder."""

        if normalized_latents.shape != (1, 24, 7, 48, 84):
            raise ValueError("an H3 video decode unit must have shape [1, 24, 7, 48, 84]")
        self.decode_graph_input = torch.empty_like(normalized_latents, device=self.device)
        self.decode_graph_input.copy_(normalized_latents)
        if self.decode_graphs is None:
            raise RuntimeError("the H3 video decoder graph cache is not configured")
        return self.decode_graphs.execute(
            tuple(int(value) for value in normalized_latents.shape),
            lambda: self._decode_normalized_segment(self.decode_graph_input),
            warmup=lambda: self._decode_normalized_segment(self.decode_graph_input),
        )
