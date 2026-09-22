"""2D position embedding helpers shared by image-capable models."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "build_abs_positions_from_grid_hw",
    "get_flattened_position_ids_extrapolate",
    "PositionEmbedding",
]


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor,
    *,
    device=None,
    total: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-patch x/y coordinates for one or more image grids.

    ``total`` is the number of patches across all grids. The caller takes it
    from a static tensor shape, so no value is read back from ``grid_hw`` and
    the reduction stays capturable inside a CUDA graph.
    """
    device = device or grid_hw.device
    grid_hw = grid_hw.to(device)
    batch = grid_hw.shape[0]
    heights = grid_hw[:, 0]
    widths = grid_hw[:, 1]
    counts = heights * widths
    total = int(total)

    patch_to_sample = torch.repeat_interleave(
        torch.arange(batch, device=device), counts, output_size=total
    )
    patch_id = torch.arange(total, device=device)
    # ``counts.new_zeros(1)`` allocates the leading zero directly on-device;
    # ``torch.tensor([0], device=device)`` would stage a host tensor and copy
    # it, which is rejected mid CUDA-graph capture.
    offsets = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)
    patch_id = patch_id - offsets[patch_to_sample]

    width_per_patch = widths[patch_to_sample]
    abs_x = patch_id % width_per_patch
    abs_y = patch_id // width_per_patch
    return abs_x, abs_y


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
