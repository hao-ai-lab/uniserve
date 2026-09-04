"""Request-pool staging for fixed-capacity decode execution buffers.

The Triton kernel gathers live request rows, expands their paged-cache tables,
derives each token's cache write location, and initializes inactive capacity in
one launch. This keeps graph-replayed decode inputs internally consistent while
the scheduler changes the set of live request slots.
"""

from __future__ import annotations

import torch

from ..backends.triton import triton_available

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def _gather_request_decode_inputs_kernel(
        request_pool_indices,
        request_page_tables,
        request_cache_lengths,
        request_tokens,
        request_positions,
        input_ids,
        positions,
        block_tables,
        cache_lengths,
        kv_lengths,
        query_lengths,
        decode_page_ids,
        decode_page_offsets,
        rows,
        page_table_group_stride: tl.constexpr,
        page_table_row_stride: tl.constexpr,
        page_table_column_stride: tl.constexpr,
        request_token_stride: tl.constexpr,
        request_position_stride: tl.constexpr,
        block_table_row_stride: tl.constexpr,
        max_rows: tl.constexpr,
        table_width: tl.constexpr,
        group_id: tl.constexpr,
        page_size: tl.constexpr,
        block: tl.constexpr,
    ):
        """Gather block-table cells and per-row decode scalars by request slot."""

        # The flattened launch covers both the two-dimensional block table and
        # the one-dimensional scalar buffers. Each program handles whichever
        # domains contain its offsets.
        offsets = tl.program_id(0) * block + tl.arange(0, block)

        # Map active output rows through the request pool into the selected KV
        # group. Stores span the full graph capacity, zeroing inactive rows.
        table_elements = max_rows * table_width
        table_mask = offsets < table_elements
        table_rows = offsets // table_width
        columns = offsets - table_rows * table_width
        live_table = table_mask & (table_rows < rows)
        table_slots = tl.load(
            request_pool_indices + table_rows,
            mask=live_table,
            other=0,
        )
        table_values = tl.load(
            request_page_tables
            + group_id * page_table_group_stride
            + table_slots * page_table_row_stride
            + columns * page_table_column_stride,
            mask=live_table,
            other=0,
        )
        tl.store(
            block_tables + table_rows * block_table_row_stride + columns,
            table_values,
            mask=table_mask,
        )

        # Gather the next token and its sequence state for every live row. The
        # masked load defaults also initialize unused scalar capacity to stable
        # graph inputs during the same launch.
        scalar_mask = offsets < max_rows
        live_rows = scalar_mask & (offsets < rows)
        slots = tl.load(request_pool_indices + offsets, mask=live_rows, other=0)
        cache = tl.load(
            request_cache_lengths + slots,
            mask=live_rows,
            other=0,
        )
        tokens = tl.load(
            request_tokens + slots * request_token_stride,
            mask=live_rows,
            other=1,
        )
        token_positions = tl.load(
            request_positions + slots * request_position_stride,
            mask=live_rows,
            other=0,
        )

        # The current cache length is the append position: its quotient selects
        # a physical page and its remainder selects the offset within that page.
        page_slots = cache // page_size
        write_pages = tl.load(
            request_page_tables
            + group_id * page_table_group_stride
            + slots * page_table_row_stride
            + page_slots * page_table_column_stride,
            mask=live_rows & (page_slots < table_width),
            other=0,
        )
        tl.store(request_pool_indices + offsets, slots, mask=scalar_mask)
        tl.store(input_ids + offsets, tokens, mask=scalar_mask)
        tl.store(positions + offsets, token_positions, mask=scalar_mask)
        tl.store(cache_lengths + offsets, cache, mask=scalar_mask)
        tl.store(kv_lengths + offsets, cache + 1, mask=scalar_mask)
        tl.store(query_lengths + offsets, 1, mask=scalar_mask)
        tl.store(decode_page_ids + offsets, write_pages, mask=scalar_mask)
        tl.store(decode_page_offsets + offsets, cache % page_size, mask=scalar_mask)


def gather_request_decode_inputs(
    *,
    request_pool_indices: torch.Tensor,
    request_page_tables: torch.Tensor,
    request_cache_lengths: torch.Tensor,
    request_tokens: torch.Tensor,
    request_positions: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    cache_lengths: torch.Tensor,
    kv_lengths: torch.Tensor,
    query_lengths: torch.Tensor,
    decode_page_ids: torch.Tensor,
    decode_page_offsets: torch.Tensor,
    rows: int,
    group_id: int,
    page_size: int,
) -> None:
    """Gather live request state into fixed-capacity decode buffers in place.

    ``request_pool_indices[:rows]`` identifies scheduler-owned request slots.
    The function materializes their selected KV-group page tables, next-token
    ids and positions, cache/query lengths, and physical append locations.
    Output rows beyond ``rows`` are initialized for safe fixed-shape graph
    replay. Every tensor must reside on the same CUDA device.
    """

    # A single-device contract lets the fused kernel dereference every input
    # directly and prevents partially staged graph inputs.
    tensors = (
        request_pool_indices,
        request_page_tables,
        request_cache_lengths,
        request_tokens,
        request_positions,
        input_ids,
        positions,
        block_tables,
        cache_lengths,
        kv_lengths,
        query_lengths,
        decode_page_ids,
        decode_page_offsets,
    )
    device = request_pool_indices.device
    if device.type != "cuda" or any(value.device != device for value in tensors):
        raise ValueError("request-indexed decode staging requires one CUDA device")
    if triton is None or not triton_available(device):
        raise RuntimeError("request-indexed decode staging requires Triton")

    # ``request_pool_indices`` defines scalar output capacity; ``rows`` selects
    # the live prefix populated from request-owned state.
    row_count = int(rows)
    max_rows = int(request_pool_indices.numel())
    if row_count < 1 or row_count > max_rows:
        raise ValueError("request-indexed decode row count exceeds staging capacity")
    if request_page_tables.ndim != 3 or block_tables.ndim != 2:
        raise ValueError("request-indexed decode page tables must be rank three and two")

    # The graph's dense table may expose a narrower page horizon than the
    # request pool, but it must hold every output row and the chosen KV group.
    width = int(block_tables.shape[1])
    if (
        int(block_tables.shape[0]) != max_rows
        or int(request_page_tables.shape[2]) < width
        or int(group_id) < 0
        or int(group_id) >= int(request_page_tables.shape[0])
        or int(page_size) < 1
    ):
        raise ValueError("request-indexed decode geometry is invalid")
    scalar_outputs = (
        input_ids,
        positions,
        cache_lengths,
        kv_lengths,
        query_lengths,
        decode_page_ids,
        decode_page_offsets,
    )
    if any(int(value.numel()) < max_rows for value in scalar_outputs):
        raise ValueError("request-indexed decode scalar buffers are undersized")

    # Size the grid for the larger flattened output domain so the one launch
    # always covers both block-table cells and per-row scalars.
    block = 256
    _gather_request_decode_inputs_kernel[
        (triton.cdiv(max(max_rows, int(block_tables.numel())), block),)
    ](
        *tensors,
        rows=row_count,
        page_table_group_stride=int(request_page_tables.stride(0)),
        page_table_row_stride=int(request_page_tables.stride(1)),
        page_table_column_stride=int(request_page_tables.stride(2)),
        request_token_stride=int(request_tokens.stride(0)),
        request_position_stride=int(request_positions.stride(0)),
        block_table_row_stride=int(block_tables.stride(0)),
        max_rows=max_rows,
        table_width=width,
        group_id=int(group_id),
        page_size=int(page_size),
        block=block,
        num_warps=4,
    )


__all__ = ["gather_request_decode_inputs"]
