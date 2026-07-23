"""Prefix-bound helpers for visible-end attention."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _torch():
    import torch

    return torch


def _validate_visible_end(visible_end: torch.Tensor) -> tuple[int, int]:
    torch = _torch()
    if visible_end.dtype != torch.int32:
        raise TypeError(f"visible_end must be int32, got {visible_end.dtype}")
    if visible_end.ndim != 2:
        raise ValueError(
            f"visible_end must be 2D (batch, seqlen_q), got shape {tuple(visible_end.shape)}"
        )
    return int(visible_end.shape[0]), int(visible_end.shape[1])


def compute_prefix_bounds(
    visible_end: torch.Tensor,
    *,
    q_tile_size: int,
) -> torch.Tensor:
    """Reduce ``visible_end[batch, seqlen_q]`` to per-Q-tile min/max bounds."""

    torch = _torch()
    batch, seqlen_q = _validate_visible_end(visible_end)
    q_tile_size = int(q_tile_size)
    if q_tile_size <= 0:
        raise ValueError("q_tile_size must be positive")
    num_q_tiles = (seqlen_q + q_tile_size - 1) // q_tile_size
    out = torch.zeros(
        (batch, num_q_tiles, 2),
        dtype=torch.int32,
        device=visible_end.device,
    )
    for tile in range(num_q_tiles):
        start = tile * q_tile_size
        end = min(start + q_tile_size, seqlen_q)
        values = visible_end[:, start:end]
        out[:, tile, 0] = values.min(dim=-1).values
        out[:, tile, 1] = values.max(dim=-1).values
    return out.contiguous()


def compute_prefix_bounds_varlen(
    visible_end: torch.Tensor,
    seqlens_q: torch.Tensor,
    *,
    q_tile_size: int,
    num_q_tiles: int | None = None,
) -> torch.Tensor:
    """Reduce padded ``visible_end`` rows using per-batch query lengths."""

    torch = _torch()
    batch, max_q = _validate_visible_end(visible_end)
    if seqlens_q.ndim != 1 or int(seqlens_q.shape[0]) != batch:
        raise ValueError("seqlens_q must be a 1D tensor with one entry per batch row")
    q_tile_size = int(q_tile_size)
    if q_tile_size <= 0:
        raise ValueError("q_tile_size must be positive")
    seqlens_q = seqlens_q.to(device=visible_end.device, dtype=torch.int32)
    max_tiles = int(num_q_tiles) if num_q_tiles is not None else 0
    if max_tiles <= 0:
        max_tiles = (max_q + q_tile_size - 1) // q_tile_size
    if seqlens_q.device.type == "cpu":
        lengths = tuple(int(value) for value in seqlens_q.tolist())
        if any(length < 0 or length > max_q for length in lengths):
            raise ValueError(f"query lengths must be within visible_end width {max_q}")
    if batch == 0 or max_tiles == 0:
        return torch.zeros(
            (batch, max_tiles, 2),
            dtype=torch.int32,
            device=visible_end.device,
        )

    tiled_width = max_tiles * q_tile_size
    values = visible_end[:, :tiled_width]
    if tiled_width > max_q:
        values = torch.nn.functional.pad(values, (0, tiled_width - max_q))
    values = values.reshape(batch, max_tiles, q_tile_size)
    positions = torch.arange(tiled_width, device=visible_end.device, dtype=torch.int32)
    valid = positions.reshape(1, max_tiles, q_tile_size) < seqlens_q.reshape(batch, 1, 1)
    has_values = valid.any(dim=-1)
    minimum = torch.where(valid, values, torch.iinfo(torch.int32).max).amin(dim=-1)
    maximum = torch.where(valid, values, torch.iinfo(torch.int32).min).amax(dim=-1)
    zeros = torch.zeros_like(minimum)
    return torch.stack(
        (torch.where(has_values, minimum, zeros), torch.where(has_values, maximum, zeros)),
        dim=-1,
    ).contiguous()


__all__ = ["compute_prefix_bounds", "compute_prefix_bounds_varlen"]
