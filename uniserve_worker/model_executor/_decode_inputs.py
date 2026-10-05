"""Gather resident request state into fixed-capacity decode input buffers.

The Triton kernel gathers live request rows, expands the unit tables of every
numerical block table, derives each token's cache write location per table,
and initializes inactive capacity in one launch. This keeps graph-replayed
decode inputs internally consistent while the scheduler changes the set of
live request slots.

`TokenBuffers` in `uniserve_worker.model_executor.input_buffers` uses it for
a call without prepared attention whose rows are all request-indexed decode
rows, when the request state is resident on the lane's CUDA device. That
state comes from `BlockTables` (unit tables, start pages, table shapes and
verified cache lengths) and `DecodeState` (next tokens and logical lengths),
both indexed by request slot.
"""

from __future__ import annotations

import torch
from uniserve_kernels.triton import launchable

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

if triton is not None:
    # Request counts and staged widths change during serving. Keeping them as
    # unspecialized runtime values means an arrival does not load another
    # kernel variant.
    @triton.jit(do_not_specialize=["rows", "columns"])
    def _gather_request_decode_inputs_kernel(
        request_pool_indices,
        request_unit_tables,
        request_start_pages,
        table_shapes,
        request_cache_lengths,
        request_tokens,
        request_positions,
        input_ids,
        positions,
        block_tables,
        start_pages,
        cache_lengths,
        query_lengths,
        query_offsets,
        prefix_offsets,
        write_indices,
        rows,
        unit_table_stride: tl.constexpr,
        unit_row_stride: tl.constexpr,
        request_width: tl.constexpr,
        start_group_stride: tl.constexpr,
        request_token_stride: tl.constexpr,
        request_position_stride: tl.constexpr,
        position_axis_stride: tl.constexpr,
        position_axes: tl.constexpr,
        block_table_stride: tl.constexpr,
        block_table_row_stride: tl.constexpr,
        start_page_stride: tl.constexpr,
        write_stride: tl.constexpr,
        max_rows: tl.constexpr,
        row_block: tl.constexpr,
        columns,
        block: tl.constexpr,
    ):
        """Gather one table's cells and per-row decode scalars.

        Axis 1 of the grid selects the numerical table. On it, program 0
        writes the table's start pages and write addresses, and on table 0
        also every shared per-row scalar column; the other programs copy the
        table's cells. Their outputs are disjoint, and the only input a
        program stores to (``request_pool_indices`` past ``rows``, on table
        0) is never loaded by another program, which reads slots only for
        live rows, so no global barrier is needed.
        """
        table = tl.program_id(1)

        # [group, page tokens, window or -1] of this table.
        group = tl.load(table_shapes + table * 3)
        page_tokens = tl.load(table_shapes + table * 3 + 1)
        window = tl.load(table_shapes + table * 3 + 2)
        units = request_unit_tables + table * unit_table_stride

        if tl.program_id(0) == 0:
            offsets = tl.arange(0, row_block)
            scalar_mask = offsets < max_rows
            live_rows = scalar_mask & (offsets < rows)
            slots = tl.load(
                request_pool_indices + offsets, mask=live_rows, other=0
            )
            cache = tl.load(
                request_cache_lengths + slots, mask=live_rows, other=0
            )
            installed = tl.load(
                request_start_pages + group * start_group_stride + slots,
                mask=live_rows,
                other=0,
            )

            # A windowed table stages pages from the first one the decode
            # query's window reaches; a full table from page zero.
            first = tl.where(
                window >= 0, tl.maximum(cache - window, 0) // page_tokens, 0
            )
            tl.store(
                start_pages + table * start_page_stride + offsets,
                tl.where(live_rows, first, 0),
                mask=scalar_mask,
            )

            # Cache length identifies the append page and its token offset.
            # Inactive rows and unavailable pages keep the public -1 sentinel.
            column = cache // page_tokens - installed
            write_units = tl.load(
                units + slots * unit_row_stride + column,
                mask=live_rows & (column >= 0) & (column < request_width),
                other=-1,
            )
            writes = tl.where(
                write_units >= 0,
                write_units.to(tl.int64) * page_tokens + cache % page_tokens,
                -1,
            )
            tl.store(
                write_indices + table * write_stride + offsets,
                writes,
                mask=scalar_mask,
            )

            if table == 0:
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
                # Padded rows are redirected to slot 0, the permanent
                # inactive slot of `BlockTables` and `DecodeState`.
                tl.store(
                    request_pool_indices + offsets,
                    0,
                    mask=scalar_mask & ~live_rows,
                )
                tl.store(input_ids + offsets, tokens, mask=scalar_mask)

                # Only the first axis carries the token position; the
                # spatial axes of three-axis multimodal positions stay zero
                # for text tokens.
                for axis in tl.static_range(position_axes):
                    tl.store(
                        positions + axis * position_axis_stride + offsets,
                        token_positions if axis == 0 else 0,
                        mask=scalar_mask,
                    )
                tl.store(cache_lengths + offsets, cache, mask=scalar_mask)
                tl.store(query_lengths + offsets, 1, mask=scalar_mask)

                # Exclusive prefix sums over the full capacity: every row,
                # including padding, has one query token, and padded rows add
                # no cached tokens.
                tl.store(query_offsets, 0)
                tl.store(prefix_offsets, 0)
                tl.store(
                    query_offsets + offsets + 1, offsets + 1, mask=scalar_mask
                )
                tl.store(
                    prefix_offsets + offsets + 1,
                    tl.cumsum(cache),
                    mask=scalar_mask,
                )
        else:
            table_offsets = (tl.program_id(0) - 1) * block + tl.arange(0, block)

            # Map active output rows through the request pool into the
            # table's installed units, from the row's first staged page.
            # Stores span the full row capacity, zeroing inactive rows.
            table_elements = max_rows * columns
            table_mask = table_offsets < table_elements
            table_rows = table_offsets // columns
            cells = table_offsets - table_rows * columns
            live_table = table_mask & (table_rows < rows)
            # Names differ from the scalar branch's: Triton requires a name
            # bound in both branches of a runtime condition to keep one type.
            table_slots = tl.load(
                request_pool_indices + table_rows,
                mask=live_table,
                other=0,
            )
            cell_cache = tl.load(
                request_cache_lengths + table_slots, mask=live_table, other=0
            )
            cell_installed = tl.load(
                request_start_pages + group * start_group_stride + table_slots,
                mask=live_table,
                other=0,
            )
            cell_first = tl.where(
                window >= 0,
                tl.maximum(cell_cache - window, 0) // page_tokens,
                0,
            )
            source = cell_first - cell_installed + cells
            table_values = tl.load(
                units + table_slots * unit_row_stride + source,
                mask=live_table & (source >= 0) & (source < request_width),
                other=0,
            )
            tl.store(
                block_tables
                + table * block_table_stride
                + table_rows * block_table_row_stride
                + cells,
                table_values,
                mask=table_mask,
            )


def gather_request_decode_inputs(
    *,
    request_pool_indices: torch.Tensor,
    request_unit_tables: torch.Tensor,
    request_start_pages: torch.Tensor,
    table_shapes: torch.Tensor,
    request_cache_lengths: torch.Tensor,
    request_tokens: torch.Tensor,
    request_positions: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    start_pages: torch.Tensor,
    cache_lengths: torch.Tensor,
    query_lengths: torch.Tensor,
    query_offsets: torch.Tensor,
    prefix_offsets: torch.Tensor,
    write_indices: torch.Tensor,
    rows: int,
    columns: int,
) -> None:
    """Gather live request state into fixed-capacity decode buffers in place.

    ``request_pool_indices[:rows]`` identifies scheduler-owned request slots.
    The function materializes, for every numerical table ``t`` described by
    ``table_shapes[t] = (group, page_tokens, window or -1)``, the slots'
    units from their first staged page into ``block_tables[t]``, that page
    into ``start_pages[t]`` and the append address into
    ``write_indices[t]``; and, shared by all tables, next-token ids and
    position axes, cache/query lengths and offsets. A row's first staged
    page of a windowed table is the first page its window reaches,
    ``max(0, cache - window) // page_tokens``; of a full table, page zero.
    ``request_unit_tables[t, slot]`` holds the slot's units from its
    installed start page ``request_start_pages[group, slot]`` on. Each table
    stages ``columns`` cells per row. Output rows beyond ``rows`` are
    initialized for safe fixed-shape graph replay: token id 1, zero
    positions, cache length and start page, one query token, a zeroed table
    row, and the write-index sentinel -1. The function also overwrites
    ``request_pool_indices[rows:]`` with slot 0.

    Every tensor must reside on the same CUDA device. The kernel honors the
    table and slot strides of ``request_unit_tables``, the group stride of
    ``request_start_pages``, the first-axis strides of ``request_tokens`` and
    ``request_positions``, the table and row strides of ``block_tables``, the
    table strides of ``start_pages`` and ``write_indices``, and the axis
    stride of ``positions``; every other dimension is indexed with unit
    stride, which is checked for the columns of ``positions`` and the unit
    tables.

    Raises:
        ValueError: When the tensors are not on one CUDA device, ``rows`` is
            outside ``[1, request_pool_indices.numel()]``, a table has the
            wrong rank, count or capacity, ``columns`` is not positive,
            ``positions`` is not one or three unit-stride axes, or an output
            buffer is smaller than the capacity requires.
        RuntimeError: When Triton is unavailable or cannot launch on the
            device.
    """
    # The fused kernel dereferences every tensor directly, so all must share
    # one CUDA device. All validation precedes the launch, so a rejected call
    # stages nothing.
    tensors = (
        request_pool_indices,
        request_unit_tables,
        request_start_pages,
        table_shapes,
        request_cache_lengths,
        request_tokens,
        request_positions,
        input_ids,
        positions,
        block_tables,
        start_pages,
        cache_lengths,
        query_lengths,
        query_offsets,
        prefix_offsets,
        write_indices,
    )
    device = request_pool_indices.device
    if device.type != "cuda" or any(
        value.device != device for value in tensors
    ):
        raise ValueError("request-indexed decode requires one CUDA device")
    if triton is None or not launchable(device):
        raise RuntimeError("request-indexed decode requires Triton")

    # ``request_pool_indices`` defines scalar output capacity; ``rows`` selects
    # the live prefix populated from request-owned state.
    row_count = int(rows)
    max_rows = int(request_pool_indices.numel())
    if row_count < 1 or row_count > max_rows:
        raise ValueError(
            "request-indexed decode row count exceeds buffer capacity"
        )
    if (
        request_unit_tables.ndim != 3
        or request_start_pages.ndim != 2
        or block_tables.ndim != 3
        or start_pages.ndim != 2
        or write_indices.ndim != 2
    ):
        raise ValueError("request-indexed decode tables have the wrong rank")

    # Every staged table must hold every output row and ``columns`` cells.
    tables = int(request_unit_tables.shape[0])
    if (
        table_shapes.shape != (tables, 3)
        or table_shapes.dtype != torch.int32
        or not table_shapes.is_contiguous()
        or request_unit_tables.stride(2) != 1
        or tuple(block_tables.shape[:2]) != (tables, max_rows)
        or int(block_tables.shape[2]) < int(columns)
        or block_tables.stride(2) != 1
        or int(start_pages.shape[0]) != tables
        or int(start_pages.shape[1]) < max_rows
        or int(write_indices.shape[0]) != tables
        or int(write_indices.shape[1]) < max_rows
        or int(columns) < 1
    ):
        raise ValueError(
            "request-indexed decode table capacity or shape is invalid"
        )
    if (
        positions.ndim != 2
        or positions.shape[0] not in (1, 3)
        or positions.stride(1) != 1
    ):
        raise ValueError(
            "decode positions require one or three contiguous token axes"
        )
    scalar_outputs = (
        input_ids,
        positions[0],
        cache_lengths,
        query_lengths,
    )
    if any(int(value.numel()) < max_rows for value in scalar_outputs):
        raise ValueError("request-indexed decode scalar buffers are undersized")
    if min(query_offsets.numel(), prefix_offsets.numel()) < max_rows + 1:
        raise ValueError("request-indexed decode offset buffers are undersized")

    # Per table, one program produces the scalar columns over a power-of-two
    # row block (``tl.arange`` extents must be powers of two) masked to
    # ``max_rows``; the remaining programs copy ``block`` table cells each.
    block = 256
    grid = (1 + triton.cdiv(max_rows * int(columns), block), tables)
    _gather_request_decode_inputs_kernel[grid](
        *tensors,
        rows=row_count,
        unit_table_stride=int(request_unit_tables.stride(0)),
        unit_row_stride=int(request_unit_tables.stride(1)),
        request_width=int(request_unit_tables.shape[2]),
        start_group_stride=int(request_start_pages.stride(0)),
        request_token_stride=int(request_tokens.stride(0)),
        request_position_stride=int(request_positions.stride(0)),
        position_axis_stride=int(positions.stride(0)),
        position_axes=int(positions.shape[0]),
        block_table_stride=int(block_tables.stride(0)),
        block_table_row_stride=int(block_tables.stride(1)),
        start_page_stride=int(start_pages.stride(0)),
        write_stride=int(write_indices.stride(0)),
        max_rows=max_rows,
        row_block=triton.next_power_of_2(max_rows),
        columns=int(columns),
        block=block,
        num_warps=4,
    )


__all__ = ["gather_request_decode_inputs"]
