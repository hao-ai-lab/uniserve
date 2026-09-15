"""2D position embedding helpers shared by image-capable models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "get_1d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed",
    "get_flattened_position_ids_extrapolate",
    "PositionEmbedding",
]


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Create sine-then-cosine features for arbitrary one-dimensional positions."""

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Concatenate independent height and width sinusoidal grid embeddings."""

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    *,
    cls_token: bool = False,
    extra_tokens: int = 0,
    pe_interpolation: float = 1.0,
) -> np.ndarray:
    """Create a square 2D sinusoidal table with optional leading special-token rows."""

    grid_h = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid_w = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid_axes = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid_axes, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_flattened_position_ids_extrapolate(
    img_h: int,
    img_w: int,
    patch_size: int,
    max_num_patches_per_side: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Flatten patch coordinates into ids on a fixed maximum-width position grid."""

    nph, npw = int(img_h) // int(patch_size), int(img_w) // int(patch_size)
    coords_h = torch.arange(0, nph, device=device)
    coords_w = torch.arange(0, npw, device=device)
    return (coords_h[:, None] * int(max_num_patches_per_side) + coords_w).flatten()


class PositionEmbedding(nn.Module):
    """Learned position rows indexed on a fixed two-dimensional grid."""

    def __init__(self, grid_size: tuple[int, int], hidden_size: int):
        super().__init__()
        if len(grid_size) != 2 or any(type(size) is not int or size < 1 for size in grid_size):
            raise ValueError("position grid requires two positive integer dimensions")
        self.grid_size = grid_size
        self.weight = nn.Parameter(torch.empty(grid_size[0] * grid_size[1], hidden_size))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.weight[positions]
