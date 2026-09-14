"""Small tensor helpers for paged KV addressing."""

from __future__ import annotations

import torch

from uniserve.runtime.triton import triton_available

__all__ = [
    "write_locations",
    "decode_write_locations",
    "paged_kv_write",
]

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


_TRITON_KV_WRITE_BLOCK = 256


if triton is not None:

    @triton.jit
    def _paged_kv_write_kernel(
        k_cache_ptr,
        v_cache_ptr,
        locations_ptr,
        offsets_ptr,
        k_src_ptr,
        v_src_ptr,
        num_pages: tl.constexpr,
        page_size: tl.constexpr,
        row_width: tl.constexpr,
        k_row_stride: tl.constexpr,
        v_row_stride: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Scatter contiguous key and value rows into physical page-offset locations."""

        row = tl.program_id(0)
        columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
        location = tl.load(locations_ptr + row)
        if offsets_ptr is None:
            # Runtime locations already encode the physical token slot. Zero
            # and negative values are non-writing rows, including graph padding.
            cache_row = location
            persists = location > 0
            valid_address = location < num_pages * page_size
        else:
            page_offset = tl.load(offsets_ptr + row)
            cache_row = location * page_size + page_offset
            persists = location >= 0
            valid_address = (location < num_pages) & (page_offset >= 0)
            valid_address &= page_offset < page_size
        tl.device_assert((~persists) | valid_address, "paged KV write index out of bounds")

        k_source_offsets = row * k_row_stride + columns
        v_source_offsets = row * v_row_stride + columns
        cache_offsets = cache_row * row_width + columns
        mask = persists & valid_address & (columns < row_width)
        k = tl.load(k_src_ptr + k_source_offsets, mask=mask, other=0.0)
        v = tl.load(v_src_ptr + v_source_offsets, mask=mask, other=0.0)
        tl.store(k_cache_ptr + cache_offsets, k, mask=mask)
        tl.store(v_cache_ptr + cache_offsets, v, mask=mask)


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


def _triton_paged_kv_write_eligible(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    locations: torch.Tensor,
    offsets: torch.Tensor | None,
    k_src: torch.Tensor,
    v_src: torch.Tensor,
) -> bool:
    """Return whether paged KV inputs satisfy the fused Triton scatter contract."""

    addresses = (locations,) if offsets is None else (locations, offsets)
    tensors = (k_cache, v_cache, *addresses, k_src, v_src)
    if (
        triton is None
        or torch.is_grad_enabled()
        or not all(tensor.is_cuda for tensor in tensors)
        or len({tensor.device for tensor in tensors}) != 1
        or not triton_available(k_cache.device)
    ):
        return False
    if any(address.dtype not in (torch.int32, torch.int64) for address in addresses):
        return False
    if (
        k_cache.shape != v_cache.shape
        or k_cache.dtype != v_cache.dtype
        or k_src.dtype != k_cache.dtype
        or v_src.dtype != v_cache.dtype
    ):
        return False
    if not all(tensor.is_contiguous() for tensor in (k_cache, v_cache, *addresses)):
        return False
    head_dim = int(k_src.shape[2])
    row_width = int(k_src.shape[1]) * head_dim
    if not all(
        int(tensor.stride(2)) == 1
        and int(tensor.stride(1)) == head_dim
        and int(tensor.stride(0)) >= row_width
        for tensor in (k_src, v_src)
    ):
        return False
    num_rows = int(locations.numel())
    return (
        num_rows > 0
        and (offsets is None or int(offsets.numel()) == num_rows)
        and int(k_src.shape[0]) == num_rows
        and int(v_src.shape[0]) == num_rows
        and int(k_src.shape[1] * k_src.shape[2]) > 0
    )


def _triton_paged_kv_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    locations: torch.Tensor,
    offsets: torch.Tensor | None,
    k_src: torch.Tensor,
    v_src: torch.Tensor,
) -> None:
    """Launch the fused paged KV scatter over flattened token rows."""

    num_pages, page_size, heads, head_dim = (int(dim) for dim in k_cache.shape)
    row_width = heads * head_dim
    grid = (int(locations.numel()), triton.cdiv(row_width, _TRITON_KV_WRITE_BLOCK))
    _paged_kv_write_kernel[grid](
        k_cache,
        v_cache,
        locations,
        offsets,
        k_src,
        v_src,
        num_pages,
        page_size,
        row_width,
        int(k_src.stride(0)),
        int(v_src.stride(0)),
        _TRITON_KV_WRITE_BLOCK,
        num_warps=4,
        debug=True,
    )


def paged_kv_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    locations: torch.Tensor,
    offsets: torch.Tensor | None,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    *,
    cast: bool = False,
) -> None:
    """Scatter current K/V into physical token slots or page/offset addresses.

    ``k_cache``/``v_cache`` are ``[pages, page_size, heads, dim]``. With
    ``offsets=None``, locations are encoded physical token slots and nonpositive
    entries mask writes. Otherwise, locations are page IDs, negative page IDs
    mask writes, and offsets select the token within each page. Address columns
    and K/V share their leading shape: ``[N]`` with ``[N, heads, dim]``, or
    ``[batch, n]`` with ``[batch, n, heads, dim]``. When ``cast`` is set, sources
    are converted to the cache dtype; otherwise their dtype must match. Callers
    authorize the write intervals; out-of-range addresses are index errors.
    """

    page_size = int(k_cache.shape[1])
    num_pages = int(k_cache.shape[0])
    heads = int(k_cache.shape[2])
    head_dim = int(k_cache.shape[3])
    k_flat = k_cache.view(num_pages * page_size, heads, head_dim)
    v_flat = v_cache.view(num_pages * page_size, heads, head_dim)
    k_src = k_current.reshape(-1, heads, head_dim)
    v_src = v_current.reshape(-1, heads, head_dim)
    locations = locations.reshape(-1)
    offsets = None if offsets is None else offsets.reshape(-1)
    if _triton_paged_kv_write_eligible(
        k_cache,
        v_cache,
        locations,
        offsets,
        k_src,
        v_src,
    ):
        _triton_paged_kv_write(k_cache, v_cache, locations, offsets, k_src, v_src)
        return
    persists = locations > 0 if offsets is None else locations >= 0
    selected = torch.nonzero(persists, as_tuple=False).reshape(-1)
    if int(selected.numel()) == 0:
        return
    locations = locations.index_select(0, selected)
    offsets = None if offsets is None else offsets.index_select(0, selected)
    k_src = k_src.index_select(0, selected)
    v_src = v_src.index_select(0, selected)
    if cast:
        k_src = k_src.to(dtype=k_cache.dtype)
        v_src = v_src.to(dtype=v_cache.dtype)
    flat_index = locations if offsets is None else locations * page_size + offsets
    k_flat.index_copy_(0, flat_index, k_src)
    v_flat.index_copy_(0, flat_index, v_src)
