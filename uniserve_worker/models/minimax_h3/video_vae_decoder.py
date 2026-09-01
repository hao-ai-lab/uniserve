# SPDX-License-Identifier: Apache-2.0
"""UniServe-owned MiniMax H3 video VAE decoder."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["MiniMaxH3VideoDecoder"]


class _RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dimension: int = 48,
        theta: float = 100.0,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dimension % 6:
            raise ValueError("H3 video VAE rotary width must divide three axes")
        inverse = 1.0 / theta ** torch.arange(
            0,
            1,
            6 / dimension,
            dtype=torch.float32,
            device=device,
        )
        self.register_buffer("inv_freq", inverse, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * positions[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


def _apply_rotary(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    width = cosine.shape[-1]
    rotary, passthrough = value[..., :width], value[..., width:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cosine + rotated * sine, passthrough), dim=-1)


class _Attention(nn.Module):
    def __init__(self, width: int = 2048, heads: int = 32, head_dim: int = 64) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=False)
        self.to_q = nn.Linear(width, width, bias=True)
        self.to_k = nn.Linear(width, width, bias=True)
        self.to_v = nn.Linear(width, width, bias=True)
        self.to_out = nn.ModuleList((nn.Linear(width, width, bias=True), nn.Dropout(0.0)))

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        query = self.to_q(hidden).view(batch, sequence, self.heads, self.head_dim)
        key = self.to_k(hidden).view(batch, sequence, self.heads, self.head_dim)
        value = self.to_v(hidden).view(batch, sequence, self.heads, self.head_dim)
        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)
        cosine, sine = (tensor.to(query.dtype) for tensor in rotary)
        query = _apply_rotary(query, cosine, sine)
        key = _apply_rotary(key, cosine, sine)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        return self.to_out[0](attended.transpose(1, 2).reshape(batch, sequence, -1))


class _SwiGLU(nn.Module):
    def __init__(self, width: int = 2048, intermediate: int = 8192) -> None:
        super().__init__()
        self.proj = nn.Linear(width, intermediate * 2, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(hidden).chunk(2, dim=-1)
        return value * F.silu(gate)


class _FeedForward(nn.Module):
    def __init__(self, width: int = 2048, intermediate: int = 8192) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            (_SwiGLU(width, intermediate), nn.Dropout(0.0), nn.Linear(intermediate, width))
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            hidden = layer(hidden)
        return hidden


class _TransformerBlock(nn.Module):
    def __init__(self, width: int = 2048) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-5, elementwise_affine=True)
        self.attn = _Attention(width)
        self.scale1 = nn.Parameter(torch.empty(width))
        self.norm2 = nn.RMSNorm(width, eps=1e-5, elementwise_affine=True)
        self.ff = _FeedForward(width)
        self.scale2 = nn.Parameter(torch.empty(width))

    def forward(
        self,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        normalized = self.norm1(hidden.float()).to(hidden.dtype)
        hidden = hidden + self.attn(normalized, rotary) * self.scale1
        normalized = self.norm2(hidden.float()).to(hidden.dtype)
        return hidden + self.ff(normalized) * self.scale2


class MiniMaxH3VideoDecoder(nn.Module):
    """Checkpoint-defined 36-layer ViT decoder and latent-channel projection."""

    spatial_compression_ratio = 16
    temporal_compression_ratio = 4
    tokens_chunk_size = 5
    token_overlap = 2
    frame_pre_padding = 3
    frame_overlap = 5
    use_tiling = True
    tile_sample_min_height = 256
    tile_sample_min_width = 256
    tile_sample_min_overlap_height = 64
    tile_sample_min_overlap_width = 64

    def __init__(
        self,
        *,
        parameter_device: torch.device | str = "meta",
        buffer_device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        width = 2048
        with torch.device(parameter_device):
            self.post_quant_conv = nn.Conv3d(24, 24, kernel_size=1)
            self.decoder = nn.Module()
            self.decoder.proj_in = nn.Linear(24, width)
            self.decoder.register_tokens = nn.Parameter(torch.empty(1, 4, width))
            self.decoder.transformer_blocks = nn.ModuleList(
                _TransformerBlock(width) for _ in range(36)
            )
            self.decoder.norm_out = nn.LayerNorm(width, eps=1e-5)
            self.decoder.proj_out = nn.Linear(width, 3 * 4 * 16 * 16)
        self.decoder.rope = _RotaryEmbedding(device=buffer_device)

    @staticmethod
    def _split_tiles(
        length: int,
        tile_size: int,
        minimum_overlap: int,
    ) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= length:
            return [0], [length], []
        tile_count = math.ceil(length / tile_size)
        while tile_size * tile_count - minimum_overlap * (tile_count - 1) < length:
            tile_count += 1
        overlaps = [minimum_overlap] * (tile_count - 1)
        remaining = tile_size * tile_count - sum(overlaps) - length
        for index in range(remaining // 16):
            overlaps[index % (tile_count - 1)] += 16
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + tile_size - overlap)
        return starts, [tile_size] * tile_count, overlaps

    @staticmethod
    def _blend(
        previous: torch.Tensor,
        current: torch.Tensor,
        extent: int,
        dim: int,
    ) -> torch.Tensor:
        extent = min(previous.shape[dim], current.shape[dim], extent)
        positions = torch.arange(extent, device=current.device, dtype=current.dtype)
        shape = [1] * current.ndim
        shape[dim] = extent
        previous_weight = (1 - positions / extent).view(shape)
        current_weight = (positions / extent).view(shape)
        previous_slice = [slice(None)] * current.ndim
        current_slice = [slice(None)] * current.ndim
        previous_slice[dim] = slice(-extent, None)
        current_slice[dim] = slice(0, extent)
        blended = (
            previous[tuple(previous_slice)] * previous_weight
            + current[tuple(current_slice)] * current_weight
        )
        if extent == current.shape[dim]:
            return blended
        remainder = [slice(None)] * current.ndim
        remainder[dim] = slice(extent, None)
        return torch.cat((blended, current[tuple(remainder)]), dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        assembled_rows: list[torch.Tensor] = []
        for row_index, row in enumerate(tiles):
            assembled: list[torch.Tensor] = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self._blend(
                        tiles[row_index - 1][column_index],
                        tile,
                        height_overlaps[row_index - 1],
                        -2,
                    )
                if column_index:
                    tile = self._blend(
                        row[column_index - 1],
                        tile,
                        width_overlaps[column_index - 1],
                        -1,
                    )
                if row_index + 1 < len(tiles):
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index + 1 < len(row):
                    tile = tile[..., :, : -width_overlaps[column_index]]
                assembled.append(tile)
            assembled_rows.append(torch.cat(assembled, dim=-1))
        return torch.cat(assembled_rows, dim=-2)

    def forward(self, projected_latents: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = projected_latents.shape
        hidden = projected_latents.permute(0, 2, 3, 4, 1).reshape(
            batch,
            frames * height * width,
            channels,
        )
        hidden = self.decoder.proj_in(hidden)
        patch_count = hidden.shape[1]
        registers = self.decoder.register_tokens.expand(batch, -1, -1)
        hidden = torch.cat((hidden, registers, torch.zeros_like(hidden[:, :1])), dim=1)
        axes = tuple(
            2.0 * (torch.arange(0.5, size, device=hidden.device, dtype=torch.float32) / size) - 1.0
            for size in (frames, height, width)
        )
        positions = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, 2)
        positions = positions.unsqueeze(0).expand(batch, -1, -1)
        suffix = positions.new_zeros((batch, 5, 3))
        rotary = self.decoder.rope(torch.cat((positions, suffix), dim=1))
        for block in self.decoder.transformer_blocks:
            hidden = block(hidden, rotary)
        hidden = self.decoder.proj_out(self.decoder.norm_out(hidden))[:, :patch_count]
        hidden = hidden.view(batch, frames, height, width, 3, 4, 16, 16)
        return (
            hidden.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .contiguous()
            .reshape(batch, 3, frames * 4, height * 16, width * 16)
        )
