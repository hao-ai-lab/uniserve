"""2D position embedding helpers shared by image-capable models."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "get_flattened_position_ids_extrapolate",
    "PositionEmbedding",
]


def get_flattened_position_ids_extrapolate(
    img_h: int,
    img_w: int,
    patch_size: int,
    max_num_patches_per_side: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Flatten patch coordinates into ids on a fixed maximum-width position
    grid.
    """  # noqa: D205
    nph, npw = int(img_h) // int(patch_size), int(img_w) // int(patch_size)
    coords_h = torch.arange(0, nph, device=device)
    coords_w = torch.arange(0, npw, device=device)
    return (
        coords_h[:, None] * int(max_num_patches_per_side) + coords_w
    ).flatten()


class PositionEmbedding(nn.Module):
    """Learned position rows indexed on a fixed two-dimensional grid."""

    def __init__(self, grid_size: tuple[int, int], hidden_size: int):
        super().__init__()
        if len(grid_size) != 2 or any(
            type(size) is not int or size < 1 for size in grid_size
        ):
            raise ValueError(
                "position grid requires two positive integer dimensions"
            )
        self.grid_size = grid_size
        self.weight = nn.Parameter(
            torch.empty(grid_size[0] * grid_size[1], hidden_size)
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.weight[positions]
