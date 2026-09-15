"""Native paged-attention columns and optional fused dense K/V scatter."""

import torch
import triton
import triton.language as tl

from uniserve.runtime.paged_kv_math import (
    _paged_kv_write_kernel,
    _triton_paged_kv_write_eligible,
)


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
    # One additional CTA prepares both native columns. Counts follow the live
    # launch rather than specializing every ragged batch encountered in eager
    # execution. Scatter CTAs preserve update()'s checks and initialization.
    if tl.program_id(0) == tl.num_programs(0) - 1:
        if tl.program_id(1) == 0:
            # queries/prefixes are per-sequence length columns; offsets are
            # cumulative over the batch with one extra leading zero entry.
            rows = tl.arange(0, sequence_block)
            q = tl.load(queries + rows * sequence_strides[0], rows < batch_size, 0)
            p = tl.load(prefixes + rows * sequence_strides[1], rows < batch_size, 0)
            tl.store(lengths + rows, q + p, rows < batch_size)

            q_offset = tl.load(query_offsets + rows * sequence_strides[2], rows <= batch_size, 0)
            p_offset = tl.load(prefix_offsets + rows * sequence_strides[3], rows <= batch_size, 0)
            # Prefix and query offsets are cumulative in the same sequence
            # order, so their sum is the cumulative complete key length.
            tl.store(offsets + rows, q_offset + p_offset, rows <= batch_size)
    elif indices is not None:
        _paged_kv_write_kernel(
            key_cache,
            value_cache,
            indices,
            None,
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


def prepare(state, key, value, batch, *, lengths, offsets):
    """Read live sequence columns and commit this call's optional cache write.

    Dense contiguous backing can share a launch with metadata generation.
    Other numerical representations use the state's ordinary write operation.
    The outputs borrow context workspace and are valid until its next use.
    """

    indices = batch.write_indices
    fused = False

    if indices is not None:
        if state is None:
            raise RuntimeError("attention cache update requires bound prefix state")
        state._validate_update(key, value, indices)
        fused = _triton_paged_kv_write_eligible(state.key, state.value, indices, None, key, value)
        if not fused:
            state.update(key, value, indices=indices)

    # The last CTA builds the metadata columns; the leading CTAs scatter the
    # current K/V rows into the cache when the write was fused into this launch.
    rows = key.shape[0] if fused else 0
    width = key.shape[1] * key.shape[2] if fused else 0
    columns = (
        batch.queries.values,
        batch.prefixes.values,
        batch.queries.offsets,
        batch.prefixes.offsets,
    )

    with torch.cuda.device(lengths.device):
        _prepare_kernel[(rows + 1, max(1, triton.cdiv(width, 256)))](
            *columns,
            lengths,
            offsets,
            state.key if fused else None,
            state.value if fused else None,
            indices if fused else None,
            key if fused else None,
            value if fused else None,
            state.initialized["key"] if fused else None,
            state.initialized["value"] if fused else None,
            tuple(column.stride(0) for column in columns),
            batch.queries.batch_size,
            triton.next_power_of_2(batch.queries.batch_size + 1),
            state.key.shape[0] if fused else 0,
            state.block_size if fused else 0,
            width,
            key.shape[2] if fused else 0,
            key.stride() if fused else (0, 0, 0),
            value.stride() if fused else (0, 0, 0),
            num_warps=4,
            debug=True,
        )
