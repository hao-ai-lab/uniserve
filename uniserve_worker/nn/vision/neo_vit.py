"""Shared NEO-ViT embedding tower."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..rope import RotaryEmbedding, apply_rotary_emb
from .patching import build_abs_positions_from_grid_hw

__all__ = [
    'NeoVitConfig',
    'NeoVitEncoder',
]


@dataclass(frozen=True)
class NeoVitConfig:
    hidden_size: int = 1024
    llm_hidden_size: int = 1024
    downsample_ratio: float = 0.5
    patch_size: int = 16
    num_channels: int = 3
    rope_theta_vision: float = 10000.0


class NeoVitEncoder(nn.Module):
    """Conv patch + dense downsample NEO vision encoder with 2D RoPE."""

    def __init__(self, cfg: NeoVitConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_size)
        llm_hidden = int(cfg.llm_hidden_size)
        self.downsample_factor = max(1, int(round(1 / float(cfg.downsample_ratio))))
        self.patch_size = int(cfg.patch_size)
        channels = int(cfg.num_channels)
        self.num_channels = channels
        theta = float(cfg.rope_theta_vision)
        if hidden % 4 != 0:
            raise ValueError("NEO-ViT hidden_size must be divisible by 4")
        self.patch_embedding = nn.Conv2d(
            channels,
            hidden,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            hidden,
            llm_hidden,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()
        # Per-axis rotary frequencies; the rotation convention is passed as data
        # to the shared RoPE helper.
        # Reuse the shared RotaryEmbedding so the inv_freq / cos-sin construction has a
        # single owner; dim is hidden//2 because the head is split into x/y halves.
        self.rope = RotaryEmbedding(dim=hidden // 2, theta=theta)

    def forward(self, pixels: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 2:
            pixels = pixels.view(-1, self.num_channels, self.patch_size, self.patch_size)
        if pixels.ndim != 4:
            raise ValueError("NeoVitEncoder expects flattened patches or NCHW patch pixels")
        # ``grid_hw`` carries the (h, w) of each image and drives the 2D RoPE
        # positions and dense downsample; require the (B, 2) shape so the
        # downstream ``[0][0]`` / ``[:, 0]`` indexing is well-defined.
        if grid_hw.ndim != 2 or grid_hw.shape[1] != 2:
            raise ValueError(
                f"NeoVitEncoder expects grid_hw of shape (B, 2), got {tuple(grid_hw.shape)}"
            )
        patch_embeds = self.gelu(self.patch_embedding(pixels)).view(-1, self.patch_embedding.out_channels)
        patch_embeds = self._apply_2d_rope(patch_embeds.float(), grid_hw).to(dtype=patch_embeds.dtype)
        return self._dense_downsample(patch_embeds, grid_hw)

    def _apply_2d_rope(self, patch_embeds: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        abs_x, abs_y = build_abs_positions_from_grid_hw(grid_hw, device=patch_embeds.device)
        half = patch_embeds.shape[-1] // 2
        x_cos, x_sin = self.rope.cos_sin_1d(abs_x)
        y_cos, y_sin = self.rope.cos_sin_1d(abs_y)
        x_part = apply_rotary_emb(
            patch_embeds[..., :half],
            x_cos,
            x_sin,
            rotation="interleaved",
        )
        y_part = apply_rotary_emb(
            patch_embeds[..., half:],
            y_cos,
            y_sin,
            rotation="interleaved",
        )
        return torch.cat([x_part, y_part], dim=-1)

    def _dense_downsample(self, patch_embeds: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        shapes = grid_hw.tolist()
        if not shapes:
            return patch_embeds.new_empty((0, self.dense_embedding.out_channels))
        # Conv2d acts independently per batch element, so when every image shares
        # the same (h, w) grid (the common batched-serving case) the per-image
        # Python loop is equivalent to a single batched conv over (N, C, h, w).
        # This collapses N kernel launches into one and is bit-identical to the
        # per-image path; only the heterogeneous grid case keeps the loop.
        h0, w0 = int(shapes[0][0]), int(shapes[0][1])
        if all(int(h) == h0 and int(w) == w0 for h, w in shapes):
            n = len(shapes)
            if n * h0 * w0 != patch_embeds.shape[0]:
                raise ValueError("grid_hw does not cover all NEO-ViT patch embeddings")
            image = patch_embeds.view(n, h0, w0, -1).permute(0, 3, 1, 2)
            dense = self.dense_embedding(image).permute(0, 2, 3, 1)
            return dense.reshape(-1, self.dense_embedding.out_channels)
        out = []
        cursor = 0
        for h_raw, w_raw in shapes:
            h = int(h_raw)
            w = int(w_raw)
            image = patch_embeds[cursor:cursor + h * w].view(1, h, w, -1).permute(0, 3, 1, 2)
            dense = self.dense_embedding(image).permute(0, 2, 3, 1).reshape(-1, self.dense_embedding.out_channels)
            out.append(dense)
            cursor += h * w
        if cursor != patch_embeds.shape[0]:
            raise ValueError("grid_hw does not cover all NEO-ViT patch embeddings")
        return torch.cat(out, dim=0) if out else patch_embeds.new_empty((0, self.dense_embedding.out_channels))
