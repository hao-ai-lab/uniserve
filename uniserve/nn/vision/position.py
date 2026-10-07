"""2D position embedding helpers shared by image-capable models."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "build_abs_positions_from_grid_hw",
    "get_flattened_position_ids_extrapolate",
    "merged_grid_coordinates",
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


def merged_grid_coordinates(
    height: int,
    width: int,
    merge: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(rows, columns)`` of a patch grid in merge-block order.

    Encoders that merge ``merge x merge`` neighbouring patches into one output
    token (Qwen-VL) serialize a ``height x width`` patch grid block by block:
    blocks in raster order, and the ``merge x merge`` patches of each block in
    raster order within it, so every merged token reads ``merge**2``
    consecutive rows. Both results are int64 ``[height * width]`` tensors.

    Raises:
        ValueError: ``merge`` does not divide both grid sides.
    """
    if min(height, width, merge) < 1 or height % merge or width % merge:
        raise ValueError("merged patch grids require whole merge blocks")
    rows = torch.arange(height, device=device)[:, None].expand(height, width)
    columns = torch.arange(width, device=device)[None, :].expand(height, width)

    # [blocks_h, merge, blocks_w, merge] -> [blocks_h, blocks_w, merge, merge]
    shape = (height // merge, merge, width // merge, merge)
    return (
        rows.reshape(shape).transpose(1, 2).flatten(),
        columns.reshape(shape).transpose(1, 2).flatten(),
    )


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

    def interpolate(
        self, height: int, width: int, *, merge: int = 1
    ) -> torch.Tensor:
        """Resample the learned grid onto a ``height x width`` patch grid.

        Grid corners align with the patch grid's corners: patch row ``i``
        samples table row ``i * (rows - 1) / (height - 1)`` (``linspace``),
        and likewise for columns. Each patch blends its four neighbouring
        table entries bilinearly with FP32 weights, so the result is FP32
        ``[height * width, hidden]`` in ``merged_grid_coordinates`` order.
        Coordinates are evaluated on the table's device, where the
        fractional weights are taken as ``linspace`` rounds them there.

        Raises:
            ValueError: ``merge`` does not divide both grid sides.
        """
        device = self.weight.device
        table_rows, table_columns = self.grid_size

        # Corner-aligned sample coordinates, truncated to their lower table
        # neighbour; the upper neighbour clamps at the last table entry.
        sampled = []
        for size, extent in ((height, table_rows), (width, table_columns)):
            coordinate = torch.linspace(0, extent - 1, size, device=device)
            lower = coordinate.int()
            upper = (lower + 1).clamp(max=extent - 1)
            sampled.append((lower, upper, coordinate - lower))
        (top, bottom, vertical), (left, right, horizontal) = sampled

        # [4, height, width] table indices and bilinear weights of the
        # top-left, top-right, bottom-left and bottom-right neighbours.
        indices = torch.stack(
            (
                top[:, None] * table_columns + left[None, :],
                top[:, None] * table_columns + right[None, :],
                bottom[:, None] * table_columns + left[None, :],
                bottom[:, None] * table_columns + right[None, :],
            )
        )
        weights = torch.stack(
            (
                (1 - vertical)[:, None] * (1 - horizontal)[None, :],
                (1 - vertical)[:, None] * horizontal[None, :],
                vertical[:, None] * (1 - horizontal)[None, :],
                vertical[:, None] * horizontal[None, :],
            )
        )

        rows, columns = merged_grid_coordinates(
            height, width, merge, device=device
        )
        order = rows * width + columns
        indices, weights = indices.flatten(1)[:, order], weights.flatten(1)
        # Table rows promote to FP32 against the weights; the four
        # neighbours are summed in FP32.
        return (self.weight[indices] * weights[:, order, None]).sum(0)
