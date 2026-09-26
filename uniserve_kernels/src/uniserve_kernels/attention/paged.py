"""Native paged-attention length columns with an optional fused K/V write.

The TensorRT-LLM attention provider needs each sequence's complete key
length (prefix plus current queries) and its cumulative offsets as device
columns. One launch derives both from the live query and prefix columns and,
when :func:`uniserve_kernels.cache.unsupported_paged_kv_write` accepts the
call's K/V write, also scatters those rows into the paged cache. The caller
is ``uniserve.runtime.backends.attention._paged_inputs``.
"""

from __future__ import annotations

import torch

from uniserve_kernels.cache import _paged_kv_write_kernel
from uniserve_kernels.triton import tl, triton

if triton is not None:

    @triton.jit(do_not_specialize=["batch_size"])
    def _prepare_kernel(
        queries,
        prefixes,
        query_offsets,
        prefix_offsets,
        lengths,
        offsets,
        key_cache,
        value_cache,
        indices,
        key,
        value,
        key_initialized,
        value_initialized,
        sequence_strides: tl.constexpr,
        batch_size: tl.int32,
        sequence_block: tl.constexpr,
        num_pages: tl.constexpr,
        page_size: tl.constexpr,
        row_width: tl.constexpr,
        head_dim: tl.constexpr,
        key_strides: tl.constexpr,
        value_strides: tl.constexpr,
    ):
        # Grid: (write rows + 1, column blocks). The last program along axis
        # zero, in column block zero, builds both columns; the leading
        # programs, launched only for a fused write, run the cache write's
        # own scatter body, which keeps its device assertions and
        # initialized-block flags. ``batch_size`` is excluded from Triton's
        # integer specialization, so the batch count affects the compiled
        # variant only through the power-of-two ``sequence_block``.
        if tl.program_id(0) == tl.num_programs(0) - 1:
            if tl.program_id(1) == 0:
                # queries/prefixes are per-sequence length columns; offsets are
                # cumulative over the batch with one extra leading zero entry.
                rows = tl.arange(0, sequence_block)
                q = tl.load(
                    queries + rows * sequence_strides[0], rows < batch_size, 0
                )
                p = tl.load(
                    prefixes + rows * sequence_strides[1], rows < batch_size, 0
                )
                tl.store(lengths + rows, q + p, rows < batch_size)

                q_offset = tl.load(
                    query_offsets + rows * sequence_strides[2],
                    rows <= batch_size,
                    0,
                )
                p_offset = tl.load(
                    prefix_offsets + rows * sequence_strides[3],
                    rows <= batch_size,
                    0,
                )
                # Prefix and query offsets are cumulative in the same sequence
                # order, so their sum is the cumulative complete key length.
                tl.store(
                    offsets + rows, q_offset + p_offset, rows <= batch_size
                )
        elif indices is not None:
            _paged_kv_write_kernel(
                key_cache,
                value_cache,
                indices,
                key,
                value,
                key_initialized,
                value_initialized,
                num_pages,
                page_size,
                row_width,
                head_dim,
                key_strides,
                value_strides,
                256,
            )


def prepare(
    columns: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    *,
    batch_size: int,
    write: tuple | None,
) -> None:
    """Build complete key lengths and offsets, and optionally scatter K/V.

    The launch enqueues on the current stream of ``lengths``'s device and
    runs with Triton ``debug=True``, so out-of-range slots trip the scatter's
    device assertions.

    Args:
        columns: Per-sequence query and prefix lengths (``batch_size``
            entries each) followed by their cumulative offsets
            (``batch_size + 1`` entries each, starting at zero); any stride.
        lengths: Contiguous output receiving ``query + prefix`` for each
            sequence.
        offsets: Contiguous output receiving the ``batch_size + 1``
            cumulative complete key offsets.
        batch_size: Number of sequences.
        write: ``None``, or ``(key_cache, value_cache, slots, key, value,
            key_initialized, value_initialized, block_size)`` for a fused
            cache write that
            :func:`uniserve_kernels.cache.unsupported_paged_kv_write` accepts:
            contiguous ``[pages, block_size, heads, head_dim]`` caches, one
            slot per ``[heads, head_dim]`` source row (``-1`` skips a row),
            and the per-block initialized flags that written blocks set.
    """
    if write is None:
        key_cache = value_cache = slots = key = value = None
        key_initialized = value_initialized = None
        rows = width = pages = block_size = head_dim = 0
        key_strides = value_strides = (0, 0, 0)
    else:
        (
            key_cache,
            value_cache,
            slots,
            key,
            value,
            key_initialized,
            value_initialized,
            block_size,
        ) = write
        # One scatter program row per source token, each covering the
        # flattened ``heads * head_dim`` row in 256-element column blocks.
        rows, width = key.shape[0], key.shape[1] * key.shape[2]
        pages, head_dim = key_cache.shape[0], key.shape[2]
        key_strides, value_strides = key.stride(), value.stride()

    with torch.cuda.device(lengths.device):
        _prepare_kernel[(rows + 1, max(1, triton.cdiv(width, 256)))](
            *columns,
            lengths,
            offsets,
            key_cache,
            value_cache,
            slots,
            key,
            value,
            key_initialized,
            value_initialized,
            tuple(column.stride(0) for column in columns),
            batch_size,
            # Covers the ``batch_size + 1`` offset entries.
            triton.next_power_of_2(batch_size + 1),
            pages,
            block_size,
            width,
            head_dim,
            key_strides,
            value_strides,
            num_warps=4,
            debug=True,
        )
