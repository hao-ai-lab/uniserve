"""Paged K/V scatter, FP8 block rescaling and block initialization.

Caches store ``[blocks, tokens, heads, dim]`` rows. Writes address physical
token slots ``block * block_size + token``; ``-1`` skips a token.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

_WRITE_BLOCK = 256


if triton is not None:

    @triton.jit
    def _paged_kv_write_kernel(
        k_cache_ptr,
        v_cache_ptr,
        slots_ptr,
        k_src_ptr,
        v_src_ptr,
        k_initialized_ptr,
        v_initialized_ptr,
        num_pages: tl.constexpr,
        page_size: tl.constexpr,
        row_width: tl.constexpr,
        head_dim: tl.constexpr,
        k_strides: tl.constexpr,
        v_strides: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Scatter borrowed key and value rows into physical token slots."""
        # Grid: (token rows, ceil(row_width / block)). Each program scatters a
        # block-wide slice of one token's flattened [heads, head_dim] row.
        row = tl.program_id(0)
        columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
        location = tl.load(slots_ptr + row)
        # -1 excludes the token from both payload and initialization state.
        # Physical block zero is an ordinary, writable numerical block.
        cache_row = location
        persists = location >= 0
        valid_address = location < num_pages * page_size
        tl.device_assert(location >= -1, "paged KV write index below -1")
        tl.device_assert(
            (~persists) | valid_address, "paged KV write index out of bounds"
        )

        # Decompose flat row columns into (head, dim) for strided source reads.
        heads = columns // head_dim
        dimensions = columns % head_dim
        k_source_offsets = (
            row * k_strides[0]
            + heads * k_strides[1]
            + dimensions * k_strides[2]
        )
        v_source_offsets = (
            row * v_strides[0]
            + heads * v_strides[1]
            + dimensions * v_strides[2]
        )
        cache_offsets = cache_row * row_width + columns
        mask = persists & valid_address & (columns < row_width)
        k = tl.load(k_src_ptr + k_source_offsets, mask=mask, other=0.0)
        v = tl.load(v_src_ptr + v_source_offsets, mask=mask, other=0.0)
        tl.store(k_cache_ptr + cache_offsets, k, mask=mask)
        tl.store(v_cache_ptr + cache_offsets, v, mask=mask)
        if k_initialized_ptr is not None:
            # Every writing token commits the same true bit, independently of
            # head width. The block's payload and flag share stream ordering.
            if tl.program_id(1) == 0:
                block = cache_row // page_size
                tl.store(
                    k_initialized_ptr + block, 1, mask=persists & valid_address
                )
                tl.store(
                    v_initialized_ptr + block, 1, mask=persists & valid_address
                )

    @triton.jit
    def _rescale_fp8_kernel(
        values,
        old_scales,
        new_scales,
        initialized,
        width: tl.constexpr,
        tile: tl.constexpr,
        compute_dtype: tl.constexpr,
    ):
        # values: [blocks, width] flattened FP8 codes; scales, flags: [blocks].
        block = tl.program_id(0)
        old = tl.load(old_scales + block)
        new = tl.load(new_scales + block)
        active = tl.load(initialized + block) & (new > old)
        # Untouched and non-growing blocks perform only the metadata reads. A
        # growing block keeps its old scale until every encoded element is
        # read.
        if active:
            for start in range(0, width, tile):
                indices = start + tl.arange(0, tile)
                value = tl.load(
                    values + block * width + indices, indices < width, other=0.0
                ).to(tl.float32)
                decoded = (value * old).to(compute_dtype).to(tl.float32)
                encoded = tl.maximum(-448.0, tl.minimum(448.0, decoded / new))
                tl.store(
                    values + block * width + indices, encoded, indices < width
                )


if triton is not None:

    @triton.jit
    def _fill_kernel(
        tensors,
        widths: tl.constexpr,
        values: tl.constexpr,
        tiles: tl.constexpr,
        start,
        block: tl.constexpr,
    ):
        # Grid axis 0 walks the per-field 1024-element tiles of one block,
        # concatenated across fields; axis 1 indexes blocks along the leading
        # axis, offset by ``start``. Each tensor is [block, *widths[field]].
        tile = tl.program_id(0)
        page = start + tl.program_id(1)
        first: tl.constexpr = 0
        for field in tl.static_range(len(widths)):
            if tile >= first and tile < first + tiles[field]:
                offsets = (tile - first) * block + tl.arange(0, block)
                tl.store(
                    tensors[field]
                    + page.to(tl.int64) * widths[field]
                    + offsets,
                    values[field],
                    offsets < widths[field],
                )
            first += tiles[field]


def can_run_paged_kv_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    k_source: torch.Tensor,
    v_source: torch.Tensor,
) -> bool:
    """Return whether contiguous CUDA caches and slot rows fit the scatter.

    Sources are ``[rows, heads, dim]`` views with arbitrary strides and the
    cache dtype.
    """
    tensors = (k_cache, v_cache, slots, k_source, v_source)
    return (
        launchable(k_cache.device)
        and not torch.is_grad_enabled()
        and all(tensor.device == k_cache.device for tensor in tensors)
        and slots.dtype in (torch.int32, torch.int64)
        and k_cache.shape == v_cache.shape
        and k_cache.dtype == v_cache.dtype
        and k_source.dtype == k_cache.dtype
        and v_source.dtype == v_cache.dtype
        and all(tensor.is_contiguous() for tensor in (k_cache, v_cache, slots))
        and slots.numel() > 0
        and int(k_source.shape[0]) == slots.numel()
        and int(v_source.shape[0]) == slots.numel()
        and int(k_source.shape[1] * k_source.shape[2]) > 0
    )


def paged_kv_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    k_source: torch.Tensor,
    v_source: torch.Tensor,
    initialized: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    """Scatter source rows into their slots and mark written blocks.

    Device assertions reject slots below ``-1`` or beyond the cache.
    """
    num_pages, page_size, heads, head_dim = (int(dim) for dim in k_cache.shape)
    row_width = heads * head_dim
    with torch.cuda.device(k_cache.device):
        _paged_kv_write_kernel[
            (int(slots.numel()), triton.cdiv(row_width, _WRITE_BLOCK))
        ](
            k_cache,
            v_cache,
            slots,
            k_source,
            v_source,
            None if initialized is None else initialized[0],
            None if initialized is None else initialized[1],
            num_pages,
            page_size,
            row_width,
            head_dim,
            k_source.stride(),
            v_source.stride(),
            _WRITE_BLOCK,
            num_warps=4,
            debug=True,
        )


def rescale_fp8_blocks(
    values: torch.Tensor,
    old: torch.Tensor,
    new: torch.Tensor,
    initialized: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Re-encode initialized E4M3 blocks whose scale grows from old to new.

    Decoded values round through the logical ``dtype`` before re-encoding.
    Scale tensors are read-only; the caller commits new scales afterward.
    """
    width = values[0].numel() if values.shape[0] else 0
    if not width:
        return
    with torch.cuda.device(values.device):
        _rescale_fp8_kernel[(values.shape[0],)](
            values,
            old,
            new,
            initialized,
            width,
            triton.next_power_of_2(min(width, 1024)),
            {
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float32: tl.float32,
            }[dtype],
            num_warps=4,
        )


def can_fill_blocks(tensors: tuple[torch.Tensor, ...]) -> bool:
    """Return whether contiguous CUDA fields of fillable dtypes fit."""
    return (
        bool(tensors)
        and launchable(tensors[0].device)
        and all(
            tensor.device == tensors[0].device
            and tensor.is_contiguous()
            and tensor.dtype
            in {
                torch.bool,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            }
            for tensor in tensors
        )
    )


def fill_blocks(
    tensors: tuple[torch.Tensor, ...],
    values: tuple[int, ...],
    widths: tuple[int, ...],
    start: int,
    stop: int,
) -> None:
    """Fill leading-axis blocks ``[start, stop)`` of every field in one launch.

    ``widths`` holds each field's elements per block.
    """
    tiles = tuple((width + 1023) // 1024 for width in widths)
    with torch.cuda.device(tensors[0].device):
        _fill_kernel[(sum(tiles), stop - start)](
            tensors, widths, values, tiles, start, 1024
        )
