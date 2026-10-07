"""Paged K/V scatter, FP8 block rescaling and block initialization.

Caches store ``[blocks, tokens, heads, dim]`` rows. Writes address physical
token slots ``block * block_size + token``; ``-1`` skips a token.

These kernels back ``uniserve.cache.paged.paged_kv_write`` and
``uniserve.cache._fp8.rescale_``; ``uniserve_kernels.attention.paged`` also
calls the scatter body from its own kernel. The scatter has an eligibility
check (``unsupported_paged_kv_write``); ``paged_kv_write`` raises on CUDA
when it reports a reason. Every launch selects the device of its tensors,
independently of the calling thread's current CUDA device. Block initialization
backs ``uniserve.runtime._block_fill.BlockFill``; its eligibility check
selects native fills for supported contiguous CUDA fields.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton, unsupported_operands

# Elements of one flattened ``heads * dim`` cache row per scatter program.
# ``uniserve_kernels.attention.paged`` passes the same width as a literal.
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
        # ``num_pages`` and ``page_size`` are the cache's block count and
        # tokens per block; ``block_size`` is this kernel's tile width
        # (``_WRITE_BLOCK``), unrelated to the cache block size. Shapes and
        # source strides are constexpr, so each distinct cache shape or
        # source stride tuple compiles its own specialization.
        #
        # Grid: (token rows, ceil(row_width / block_size)). Each program
        # scatters a ``block_size``-wide slice of one token's flattened
        # [heads, head_dim] row. The caches are contiguous, so a slot indexes
        # a [num_pages * page_size, row_width] view directly.
        row = tl.program_id(0)
        columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
        location = tl.load(slots_ptr + row)
        # -1 excludes the token from both payload and initialization state.
        # Physical block zero is an ordinary, writable numerical block.
        cache_row = location
        persists = location >= 0
        valid_address = location < num_pages * page_size
        # Triton compiles device asserts only in debug mode, so both launchers
        # (``paged_kv_write`` and ``uniserve_kernels.attention.paged``) pass
        # ``debug=True``.
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
        # A ``None`` flag pointer is a compile-time constant, so this branch
        # disappears when the caller tracks no initialization state.
        if k_initialized_ptr is not None:
            # Only the first column program of each token row stores flags,
            # so a row commits its block's flag once regardless of how many
            # programs span the head width. Rows sharing a block all store the
            # same true value. Payload and flags are written by one launch, so
            # later work on the same stream observes both.
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
        # values: [blocks, width] contiguous E4M3 codes rewritten in place;
        # old_scales, new_scales and initialized: [blocks]. One program owns
        # one block and walks it in ``tile``-element steps.
        block = tl.program_id(0)
        old = tl.load(old_scales + block)
        new = tl.load(new_scales + block)
        active = tl.load(initialized + block) & (new > old)

        # Uninitialized and non-growing blocks perform only the metadata
        # reads. Scales are never written here: every element of a growing
        # block decodes with its old scale, and the caller commits the new
        # scales after this launch.
        if active:
            for start in range(0, width, tile):
                indices = start + tl.arange(0, tile)
                value = tl.load(
                    values + block * width + indices, indices < width, other=0.0
                ).to(tl.float32)
                # Decode, round through the logical cache dtype, re-encode
                # against the grown scale and clamp to the E4M3 finite range
                # (+/-448); the store converts the FP32 result to E4M3.
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
        # Grid axis 0 walks the per-field ``block``-element tiles of one cache
        # block, concatenated across fields; axis 1 indexes cache blocks along
        # the leading axis, offset by ``start``. Field ``i`` is a contiguous
        # tensor of ``widths[i]`` elements per cache block. Widths, values
        # and tile counts are constexpr; ``start`` is a runtime argument.
        tile = tl.program_id(0)
        page = start + tl.program_id(1)
        # The unrolled field loop selects the one field whose tile range
        # contains this program; ``first`` accumulates each field's starting
        # tile at compile time.
        first: tl.constexpr = 0
        for field in tl.static_range(len(widths)):
            if tile >= first and tile < first + tiles[field]:
                offsets = (tile - first) * block + tl.arange(0, block)
                # The cache block index widens to int64 before scaling by the
                # field width, so offsets past 2**31 elements do not overflow.
                tl.store(
                    tensors[field]
                    + page.to(tl.int64) * widths[field]
                    + offsets,
                    values[field],
                    offsets < widths[field],
                )
            first += tiles[field]


#: Cache dtypes a scatter may convert floating sources into; the store
#: rounds to nearest even like ``Tensor.to``.
_CONVERTIBLE = (torch.float16, torch.bfloat16, torch.float32)


def unsupported_paged_kv_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    k_source: torch.Tensor,
    v_source: torch.Tensor,
    *,
    cast: bool = False,
) -> str | None:
    """Return why the scatter cannot write these rows, or ``None``.

    Caches are contiguous ``[blocks, block_size, heads, dim]`` tensors of one
    shape and dtype, slots a contiguous int32/int64 vector, and sources
    ``[rows, heads, dim]`` views with arbitrary strides and one row per
    slot. Sources share the cache dtype unless ``cast`` requests the
    conversion into a float16, bfloat16 or float32 cache. Slot values are
    not inspected here; the kernel asserts their bounds.
    """
    reason = unsupported_operands(k_cache, v_cache, slots, k_source, v_source)
    if reason is not None:
        return reason
    if slots.dtype not in (torch.int32, torch.int64):
        return f"slot dtype {slots.dtype} is not int32 or int64"
    if k_cache.shape != v_cache.shape or k_cache.dtype != v_cache.dtype:
        return "key and value caches differ in shape or dtype"
    if not all(tensor.is_contiguous() for tensor in (k_cache, v_cache, slots)):
        return "a cache or the slot vector is not contiguous"
    if k_source.ndim != 3 or any(
        int(source.shape[0]) != slots.numel() for source in (k_source, v_source)
    ):
        return "sources are not [rows, heads, dim] with one row per slot"
    if k_source.dtype != k_cache.dtype or v_source.dtype != v_cache.dtype:
        if not cast:
            return "source dtypes differ from the cache dtype"
        if k_cache.dtype not in _CONVERTIBLE:
            return (
                f"conversion into a {k_cache.dtype} cache has no kernel; "
                "encoded caches write their own codes"
            )
    return None


def paged_kv_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    k_source: torch.Tensor,
    v_source: torch.Tensor,
    initialized: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    """Scatter source rows into their slots and mark written blocks.

    Launch only after ``unsupported_paged_kv_write`` accepts the same
    tensors. Sources of another floating dtype convert on store.

    Args:
        k_cache: Contiguous ``[blocks, block_size, heads, dim]`` key cache,
            written in place.
        v_cache: Value cache with the key cache's shape and dtype.
        slots: Contiguous flat int32 or int64 slots, one per source row;
            ``-1`` skips the row.
        k_source: ``[rows, heads, dim]`` key rows with arbitrary strides.
        v_source: ``[rows, heads, dim]`` value rows.
        initialized: Optional key and value ``[blocks]`` bool flags; every
            block that receives a row is set to ``True``.

    Device assertions reject slots below ``-1`` or beyond the cache. A failed
    assertion surfaces asynchronously as a CUDA error rather than a Python
    exception at this call.
    """
    num_pages, page_size, heads, head_dim = (int(dim) for dim in k_cache.shape)
    row_width = heads * head_dim
    if slots.numel() == 0 or row_width == 0:
        return
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

    Args:
        values: Contiguous ``[blocks, ...]`` E4M3 codes, rewritten in place
            for blocks where ``initialized`` is set and ``new > old``.
        old: ``[blocks]`` resident dequantization scales.
        new: ``[blocks]`` scales the re-encoded blocks will use.
        initialized: ``[blocks]`` bool flags; unset blocks are left alone.
        dtype: Logical cache dtype; float16, bfloat16 or float32. Other
            dtypes raise ``KeyError`` unless ``values`` is empty.
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

    ``widths`` holds each field's elements per block and ``values`` each
    field's fill value, converted to the field dtype on store. The caller
    must have passed ``tensors`` to ``can_fill_blocks`` and owns bounds
    validation of the interval.
    """
    # The tile count and the kernel's ``block`` argument share the 1024
    # element tile width.
    tiles = tuple((width + 1023) // 1024 for width in widths)
    with torch.cuda.device(tensors[0].device):
        _fill_kernel[(sum(tiles), stop - start)](
            tensors, widths, values, tiles, start, 1024
        )
