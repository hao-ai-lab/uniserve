"""Reduces per-query visible-end limits into query-tile bounds."""

from __future__ import annotations

import torch


def _validate_visible_end(visible_end: torch.Tensor) -> tuple[int, int]:
    """Validate the visible-end matrix and return its batch and query extents."""

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
    """Return per-query-tile minimum and maximum visible-end values."""

    batch, seqlen_q = _validate_visible_end(visible_end)
    seqlens_q = torch.full(
        (batch,),
        seqlen_q,
        dtype=torch.int32,
        device=visible_end.device,
    )
    return compute_prefix_bounds_varlen(
        visible_end,
        seqlens_q,
        q_tile_size=q_tile_size,
    )


def compute_prefix_bounds_varlen(
    visible_end: torch.Tensor,
    seqlens_q: torch.Tensor,
    *,
    q_tile_size: int,
    num_q_tiles: int | None = None,
) -> torch.Tensor:
    """Return tile bounds while excluding padding beyond each query length."""

    batch, max_q = _validate_visible_end(visible_end)
    if seqlens_q.ndim != 1 or int(seqlens_q.shape[0]) != batch:
        raise ValueError("seqlens_q must be a 1D tensor with one entry per batch row")
    q_tile_size = int(q_tile_size)
    if q_tile_size <= 0:
        raise ValueError("q_tile_size must be positive")
    seqlens_q = seqlens_q.to(device=visible_end.device, dtype=torch.int32)
    if num_q_tiles is not None:
        max_tiles = int(num_q_tiles)
    else:
        # The longest logical row defines output shape; padded query width does not.
        max_len = int(seqlens_q.max().item()) if batch else 0
        max_tiles = (max_len + q_tile_size - 1) // q_tile_size
    if max_tiles < 0:
        raise ValueError("num_q_tiles must be non-negative")
    if batch == 0 or max_tiles == 0:
        return torch.zeros(
            (batch, max_tiles, 2),
            dtype=torch.int32,
            device=visible_end.device,
        )
    # CPU lengths can be rejected eagerly. Device lengths remain bounded by the
    # validity mask so positions outside the padded matrix never affect a tile.
    if seqlens_q.device.type == "cpu":
        lengths = tuple(int(value) for value in seqlens_q.tolist())
        if any(length < 0 or length > max_q for length in lengths):
            raise ValueError(f"query lengths must be within visible_end width {max_q}")
    tiled_width = max_tiles * q_tile_size
    if tiled_width <= max_q:
        values = visible_end[:, :tiled_width]
    else:
        values = torch.nn.functional.pad(visible_end, (0, tiled_width - max_q))
    values = values.reshape(batch, max_tiles, q_tile_size)
    positions = torch.arange(
        tiled_width, device=visible_end.device, dtype=torch.int32
    ).reshape(1, max_tiles, q_tile_size)
    valid = positions < seqlens_q.reshape(batch, 1, 1)
    has_values = valid.any(dim=-1)
    int32 = torch.iinfo(torch.int32)
    minimum = torch.where(valid, values, values.new_full((), int32.max)).amin(dim=-1)
    maximum = torch.where(valid, values, values.new_full((), int32.min)).amax(dim=-1)
    zeros = torch.zeros_like(minimum)
    return torch.stack(
        (
            torch.where(has_values, minimum, zeros),
            torch.where(has_values, maximum, zeros),
        ),
        dim=-1,
    ).contiguous()


def prefix_block_sparsity(
    visible_end: torch.Tensor,
    *,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    max_key_length: int,
    query_tile: int,
    key_tile: int,
    variable_length: bool,
):
    """Describe fully visible and masked KV tiles using the CuTe sparse ABI.

    Prefix minima identify tiles requiring no element mask; maxima bound the
    tiles that need the per-query mask. Storage is bounded by host geometry,
    while lengths, counts and offsets remain live device values under replay.
    Variable-length indices have each sequence's actual KV-tile row stride.
    """

    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    batch, width = _validate_visible_end(visible_end)
    query_tiles = (width + query_tile - 1) // query_tile
    key_tiles = (max_key_length + key_tile - 1) // key_tile
    if query_lengths.shape != (batch,) or key_lengths.shape != (batch,):
        raise ValueError("prefix sparsity requires one query and key length per sequence")
    bounds = compute_prefix_bounds_varlen(
        visible_end, query_lengths, q_tile_size=query_tile, num_q_tiles=query_tiles
    ).clamp_min(0)
    bounds = torch.minimum(bounds, key_lengths[:, None, None])
    full = bounds[..., 0] // key_tile
    partial = (bounds[..., 1] + key_tile - 1) // key_tile - full
    columns = torch.arange(key_tiles, device=visible_end.device, dtype=torch.int32)
    if not variable_length:
        return BlockSparseTensorsTorch(
            partial[:, None].contiguous(),
            (full[..., None] + columns)[:, None].contiguous(),
            full[:, None].contiguous(),
            columns.expand(batch, 1, query_tiles, key_tiles).contiguous(),
            block_size=(query_tile, key_tile),
        )

    q_tiles = (query_lengths + query_tile - 1) // query_tile
    k_tiles = (key_lengths + key_tile - 1) // key_tile
    cumulative_tiles = torch.nn.functional.pad(q_tiles.cumsum(0, dtype=torch.int32), (1, 0))
    cumulative_indices = torch.nn.functional.pad(
        (q_tiles * k_tiles).cumsum(0, dtype=torch.int32), (1, 0)
    )
    positions = torch.arange(batch * query_tiles, device=visible_end.device, dtype=torch.int32)
    owners = torch.searchsorted(cumulative_tiles[1:], positions, right=True).clamp_max(batch - 1)
    local_tile = positions - cumulative_tiles[owners]
    source = owners * query_tiles + local_tile.clamp(0, query_tiles - 1)
    valid = positions < cumulative_tiles[-1]
    full_counts = torch.where(valid, full.reshape(-1)[source], 0)
    mask_counts = torch.where(valid, partial.reshape(-1)[source], 0)

    positions = torch.arange(
        batch * query_tiles * key_tiles, device=visible_end.device, dtype=torch.int32
    )
    owners = torch.searchsorted(cumulative_indices[1:], positions, right=True).clamp_max(batch - 1)
    local_index = positions - cumulative_indices[owners]
    stride = k_tiles[owners].clamp_min(1)
    local_tile = torch.div(local_index, stride, rounding_mode="floor").clamp(0, query_tiles - 1)
    columns = local_index.remainder(stride)
    first_masked = full.reshape(-1)[owners * query_tiles + local_tile]
    return BlockSparseTensorsTorch(
        mask_counts[None].contiguous(),
        (first_masked + columns)[None].contiguous(),
        full_counts[None].contiguous(),
        columns[None].contiguous(),
        cu_total_m_blocks=cumulative_tiles,
        cu_block_idx_offsets=cumulative_indices,
        block_size=(query_tile, key_tile),
    )


__all__ = ["compute_prefix_bounds", "compute_prefix_bounds_varlen", "prefix_block_sparsity"]
