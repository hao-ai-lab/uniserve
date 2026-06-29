"""Small tensor helpers for paged KV addressing."""
from __future__ import annotations

import torch

__all__ = [
    'write_locations',
    'decode_write_locations',
    'paged_kv_write',
]


def write_locations(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map absolute token positions to (page id, in-page offset).

    ``positions`` is ``[batch, n]`` (one column per token being written).
    Returns ``page_ids`` and ``offsets`` of the same shape, both int64. Page ids
    are gathered from ``block_table`` rows; an out-of-range page slot or physical
    page id is left to surface as an index error rather than validated here.
    """

    page_size = max(1, int(page_size))
    positions = positions.to(dtype=torch.int64)
    page_slots = torch.div(positions, page_size, rounding_mode="floor")
    offsets = torch.remainder(positions, page_size)
    page_ids = block_table.gather(1, page_slots)
    return page_ids.to(dtype=torch.int64), offsets.to(dtype=torch.int64)


def decode_write_locations(
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return page ids and offsets for one-token paged decode writes."""

    page_ids, offsets = write_locations(block_table, cache_seqlens.unsqueeze(1), page_size)
    return page_ids.squeeze(1), offsets.squeeze(1)


def paged_kv_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_ids: torch.Tensor,
    offsets: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    *,
    cast: bool = False,
) -> None:
    """Scatter current K/V into the paged cache at ``(page_ids, offsets)``.

    ``k_cache``/``v_cache`` are ``[pages, page_size, heads, dim]``;
    ``page_ids``/``offsets`` and ``k_current``/``v_current`` share the same
    leading addressing shape (``[batch, n]`` and ``[batch, n, heads, dim]``, or
    a flat ``[N]`` and ``[N, heads, dim]``). When ``cast`` is set, the source is
    cast to the cache dtype before writing; otherwise the source dtype must match
    the cache. Out-of-range indices are left to surface as a device index error.
    """

    page_size = int(k_cache.shape[1])
    num_pages = int(k_cache.shape[0])
    heads = int(k_cache.shape[2])
    head_dim = int(k_cache.shape[3])
    flat_index = (page_ids * page_size + offsets).reshape(-1)
    k_flat = k_cache.view(num_pages * page_size, heads, head_dim)
    v_flat = v_cache.view(num_pages * page_size, heads, head_dim)
    k_src = k_current.reshape(-1, heads, head_dim)
    v_src = v_current.reshape(-1, heads, head_dim)
    if cast:
        k_src = k_src.to(dtype=k_cache.dtype)
        v_src = v_src.to(dtype=v_cache.dtype)
    k_flat.index_copy_(0, flat_index, k_src)
    v_flat.index_copy_(0, flat_index, v_src)
