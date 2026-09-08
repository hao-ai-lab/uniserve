"""Fixed-profile temporal video decode over the checkpoint MiniMax H3 VAE."""

from __future__ import annotations

import torch
from torch import nn

from ...nn.quant.config import LinearPrecision
from .video_vae_decoder import MiniMaxH3VideoDecoder

__all__ = ["MiniMaxH3VideoVAE"]


class MiniMaxH3VideoVAE(nn.Module):
    """Own one resident checkpoint VAE and decode temporal segments."""

    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(self, vae: MiniMaxH3VideoDecoder, *, linear_precision: LinearPrecision) -> None:
        """Prepare one resident video decoder with fixed precision and normalization buffers."""

        super().__init__()
        self.vae = vae
        self.linear_precision = linear_precision
        self.autocast_dtype = torch.float16 if linear_precision == "fp16" else torch.bfloat16

        # Channel-wise latent statistics invert checkpoint normalization before
        # reconstruction; pixel statistics restore the decoder's RGB domain.
        mean = (
            0.858090341091156,
            -0.9606591463088989,
            1.0661640167236328,
            -0.5090325474739075,
            -0.2727581858634949,
            -1.3675414323806763,
            -0.2553254961967468,
            -0.26907554268836975,
            -0.5376840829849243,
            -0.0464097298681736,
            0.6657370328903198,
            0.19690127670764923,
            -0.5460608005523682,
            -0.4035342037677765,
            -0.23683024942874908,
            0.25928452610969543,
            -0.30133944749832153,
            0.211341992020607,
            -1.1206848621368408,
            0.3581933379173279,
            -0.04225143790245056,
            0.2604829967021942,
            0.22864092886447906,
            0.7056031823158264,
        )
        std = (
            1.2223774194717407,
            1.2767263650894165,
            1.6831774711608887,
            1.7549455165863037,
            1.5636216402053833,
            2.194143533706665,
            0.9653137922286987,
            1.0569885969161987,
            0.841948926448822,
            0.7729952931404114,
            1.8955937623977661,
            0.946841835975647,
            0.7996809482574463,
            0.44988900423049927,
            0.7197399735450745,
            0.6936293244361877,
            2.961095094680786,
            2.7694199085235596,
            3.0496184825897217,
            2.1088054180145264,
            3.276226282119751,
            3.1627357006073,
            2.2816812992095947,
            2.6127843856811523,
        )
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

    @property
    def device(self) -> torch.device:
        """Return the device that owns the decoder's learned parameters."""

        return next(self.vae.parameters()).device

    def _decode_segment(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one temporal latent segment and remove its prepended overlap frames."""

        span = int(self.vae.tokens_chunk_size) + int(self.vae.token_overlap)
        clip = self._decode_spatial_tiles(latents[:, :, :span])
        return clip[:, :, int(self.vae.frame_pre_padding) :].contiguous()

    def _decode_spatial_tiles(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one latent clip directly or tile and blend it across spatial overlap regions."""

        if not bool(self.vae.use_tiling):
            return self.vae(self.vae.post_quant_conv(latents))

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
        # Decode all spatial tiles as one batch, then blend them back into the
        # full-resolution temporal segment using the decoder's overlap contract.
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
        decoded = self.vae(self.vae.post_quant_conv(tiles))
        flat_tiles = decoded.split(1, dim=0)
        columns = len(x_indices)
        rows = [
            list(flat_tiles[start : start + columns])
            for start in range(0, len(flat_tiles), columns)
        ]
        return self.vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    def forward(
        self,
        normalized_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Denormalize one latent segment and decode it through the spatial tiling path."""

        if normalized_latents.shape != (1, 24, 7, 48, 84):
            raise ValueError("an H3 video decode unit must have shape [1, 24, 7, 48, 84]")
        latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        latents = latents * self.latents_std + self.latents_mean
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.device.type == "cuda",
        ):
            return self._decode_segment(latents).to(torch.float16)
