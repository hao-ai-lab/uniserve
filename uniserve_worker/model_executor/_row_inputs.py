"""Gather attention columns for multi-token request rows on device.

One Triton launch fills, for every numerical block table, the staged units,
first staged pages and per-token cache write addresses of a call's rows from
the request slots' resident tables (``BlockTables``), together with the
call's shared query and prefix lengths and offsets. It extends the decode
gather of ``_decode_inputs`` to rows of any length, prefill chunks, canvas
readouts and canvas steps alike, the way vLLM derives its slot mapping on
the device from its block table, positions and query starts
(``vllm/v1/worker/block_table.py``, ``compute_slot_mapping``). The host
supplies only per-row scalars, each table's staged page count per row and
each token's row.

``AttentionBuffers.stage_rows`` in ``uniserve_worker.model_executor.
input_buffers`` builds the host columns and the attention batch around it.
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

# Sections of the int64 host column array, each ``row_stride`` long: slot,
# prefix, query, writes, query offsets and prefix offsets (``rows + 1``
# entries each), then one section of staged page counts per table, then
# each token's row.
ROW_SECTIONS = 6

if triton is not None:
    # Row, token, cell counts and the staged width change every call;
    # keeping them as unspecialized runtime values means a new count loads
    # no other kernel variant.
    @triton.jit(do_not_specialize=["rows", "tokens", "cell_programs", "width"])
    def _gather_request_rows_kernel(
        columns_ptr,
        request_unit_tables,
        request_start_pages,
        table_shapes,
        block_tables,
        start_pages,
        cache_lengths,
        query_lengths,
        query_offsets,
        prefix_offsets,
        write_indices,
        rows,
        tokens,
        cell_programs,
        row_stride: tl.constexpr,
        token_section: tl.constexpr,
        width,
        unit_table_stride: tl.constexpr,
        unit_row_stride: tl.constexpr,
        request_width: tl.constexpr,
        start_group_stride: tl.constexpr,
        block_table_stride: tl.constexpr,
        block_table_row_stride: tl.constexpr,
        start_page_stride: tl.constexpr,
        write_stride: tl.constexpr,
        row_block: tl.constexpr,
        block: tl.constexpr,
    ):
        """Stage one table's rows, cells or tokens, by program.

        Axis 1 selects the numerical table. Program 0 writes the table's
        first staged pages and, on table 0, the shared lengths and offsets;
        programs ``1..cell_programs`` copy ``block`` table cells each; the
        rest write ``block`` tokens' cache addresses each. Every program
        reads only the host columns and the resident request tables, and
        their outputs are disjoint.
        """
        table = tl.program_id(1)
        program = tl.program_id(0)

        # [group, page tokens, window or -1] of this table.
        group = tl.load(table_shapes + table * 3)
        page_tokens = tl.load(table_shapes + table * 3 + 1)
        window = tl.load(table_shapes + table * 3 + 2)
        units = request_unit_tables + table * unit_table_stride
        slots_ptr = columns_ptr
        prefixes_ptr = columns_ptr + row_stride
        queries_ptr = columns_ptr + 2 * row_stride
        writes_ptr = columns_ptr + 3 * row_stride
        query_offsets_ptr = columns_ptr + 4 * row_stride
        prefix_offsets_ptr = columns_ptr + 5 * row_stride
        pages_ptr = columns_ptr + (6 + table) * row_stride

        if program == 0:
            offsets = tl.arange(0, row_block)
            row_live = offsets < rows
            row_prefix = tl.load(prefixes_ptr + offsets, mask=row_live, other=0)
            # A windowed table stages from the first page the row's first
            # query reaches back to; a full table from page zero.
            row_first = tl.where(
                window >= 0,
                tl.maximum(row_prefix - window, 0) // page_tokens,
                0,
            )
            tl.store(
                start_pages + table * start_page_stride + offsets,
                row_first,
                mask=row_live,
            )
            if table == 0:
                row_query = tl.load(
                    queries_ptr + offsets, mask=row_live, other=0
                )
                tl.store(cache_lengths + offsets, row_prefix, mask=row_live)
                tl.store(query_lengths + offsets, row_query, mask=row_live)
                bounds = offsets < rows + 1
                tl.store(
                    query_offsets + offsets,
                    tl.load(query_offsets_ptr + offsets, mask=bounds, other=0),
                    mask=bounds,
                )
                tl.store(
                    prefix_offsets + offsets,
                    tl.load(prefix_offsets_ptr + offsets, mask=bounds, other=0),
                    mask=bounds,
                )
        elif program <= cell_programs:
            cells = (program - 1) * block + tl.arange(0, block)
            cell_row = cells // width
            cell = cells - cell_row * width
            live = cell_row < rows
            slot = tl.load(slots_ptr + cell_row, mask=live, other=0)
            prefix = tl.load(prefixes_ptr + cell_row, mask=live, other=0)
            pages = tl.load(pages_ptr + cell_row, mask=live, other=0)
            installed = tl.load(
                request_start_pages + group * start_group_stride + slot,
                mask=live,
                other=0,
            )
            first = tl.where(
                window >= 0,
                tl.maximum(prefix - window, 0) // page_tokens,
                0,
            )
            # Row ``slot`` of the resident table holds the units of pages
            # ``installed..``; cells past the row's staged pages are zero.
            source = first - installed + cell
            staged = live & (cell < pages)
            values = tl.load(
                units + slot * unit_row_stride + source,
                mask=staged & (source >= 0) & (source < request_width),
                other=0,
            )
            tl.store(
                block_tables
                + table * block_table_stride
                + cell_row * block_table_row_stride
                + cell,
                values,
                mask=live,
            )
        else:
            token = (program - 1 - cell_programs) * block + tl.arange(0, block)
            live = token < tokens
            token_row = tl.load(
                columns_ptr + token_section + token, mask=live, other=0
            )
            slot = tl.load(slots_ptr + token_row, mask=live, other=0)
            prefix = tl.load(prefixes_ptr + token_row, mask=live, other=0)
            start = tl.load(query_offsets_ptr + token_row, mask=live, other=0)
            writes = tl.load(writes_ptr + token_row, mask=live, other=0)
            installed = tl.load(
                request_start_pages + group * start_group_stride + slot,
                mask=live,
                other=0,
            )
            # Token ``token`` of row ``token_row`` sits at position prefix +
            # token - start and appends to that page's unit; a read-only row's
            # tokens address no cache slot (-1).
            position = prefix + token - start
            column = position // page_tokens - installed
            unit = tl.load(
                units + slot * unit_row_stride + column,
                mask=live
                & (writes != 0)
                & (column >= 0)
                & (column < request_width),
                other=-1,
            )
            address = tl.where(
                (writes != 0) & (unit >= 0),
                unit.to(tl.int64) * page_tokens + position % page_tokens,
                -1,
            )
            tl.store(
                write_indices + table * write_stride + token,
                address,
                mask=live,
            )


def gather_request_rows(
    *,
    columns: torch.Tensor,
    rows: int,
    tokens: int,
    width: int,
    request_unit_tables: torch.Tensor,
    request_start_pages: torch.Tensor,
    table_shapes: torch.Tensor,
    block_tables: torch.Tensor,
    start_pages: torch.Tensor,
    cache_lengths: torch.Tensor,
    query_lengths: torch.Tensor,
    query_offsets: torch.Tensor,
    prefix_offsets: torch.Tensor,
    write_indices: torch.Tensor,
) -> None:
    """Stage every table's columns of ``rows`` request rows.

    ``columns`` is the device copy of the host column array (sections of
    ``row_columns_stride`` elements, see ``ROW_SECTIONS``): per row its
    request slot, prefix length,
    query length and whether it writes the cache; the exclusive query and
    prefix offsets (``rows + 1`` each); each table's staged page count per
    row; and each of the ``tokens`` tokens' row. For table ``t``, described
    by ``table_shapes[t] = (group, page_tokens, window or -1)``, the launch
    writes rows ``0..rows`` of ``block_tables[t]`` (``width`` cells, zero
    past a row's staged pages) from the first page each row stages (page
    zero, or for a windowed table ``max(0, prefix - window) //
    page_tokens``) into ``start_pages[t]``, and each token's append address
    into ``write_indices[t]`` (-1 for rows that do not write); and, shared by
    all tables, the lengths and offsets. ``request_unit_tables[t, slot]``
    holds the slot's units from its installed start page
    ``request_start_pages[group, slot]`` on.

    Every tensor must reside on one device with unit innermost stride; the
    caller validates the rows against the installed tables
    (``attention.row_tables``). On a device where Triton launches, one
    kernel stages every table; elsewhere equivalent tensor operations do.

    Raises:
        ValueError: When the tensors are not on one device or the counts are
            outside the columns' capacity.
    """
    tensors = (
        columns,
        request_unit_tables,
        request_start_pages,
        table_shapes,
        block_tables,
        start_pages,
        cache_lengths,
        query_lengths,
        query_offsets,
        prefix_offsets,
        write_indices,
    )
    device = columns.device
    if any(value.device != device for value in tensors):
        raise ValueError("request row preparation requires one device")

    count, token_count = int(rows), int(tokens)
    tables = int(request_unit_tables.shape[0])
    stride = row_columns_stride(block_tables.shape[1])
    if (
        count < 1
        or count > int(block_tables.shape[1])
        or token_count < 1
        or int(width) < 1
        or int(width) > int(block_tables.shape[2])
        or token_count > int(write_indices.shape[1])
        or int(columns.numel())
        < row_columns_size(
            int(block_tables.shape[1]), tables, int(write_indices.shape[1])
        )
    ):
        raise ValueError("request rows exceed their buffer capacity")

    if triton is None or device.type != "cuda" or not launchable(device):
        _gather_rows(
            columns,
            request_unit_tables,
            request_start_pages,
            table_shapes,
            block_tables,
            start_pages,
            cache_lengths,
            query_lengths,
            query_offsets,
            prefix_offsets,
            write_indices,
            rows=count,
            tokens=token_count,
            width=int(width),
            stride=stride,
        )
        return

    block = 256
    cell_programs = triton.cdiv(count * int(width), block)
    grid = (1 + cell_programs + triton.cdiv(token_count, block), tables)
    _gather_request_rows_kernel[grid](
        *tensors,
        rows=count,
        tokens=token_count,
        cell_programs=cell_programs,
        row_stride=stride,
        token_section=(ROW_SECTIONS + tables) * stride,
        width=int(width),
        unit_table_stride=int(request_unit_tables.stride(0)),
        unit_row_stride=int(request_unit_tables.stride(1)),
        request_width=int(request_unit_tables.shape[2]),
        start_group_stride=int(request_start_pages.stride(0)),
        block_table_stride=int(block_tables.stride(0)),
        block_table_row_stride=int(block_tables.stride(1)),
        start_page_stride=int(start_pages.stride(0)),
        write_stride=int(write_indices.stride(0)),
        row_block=triton.next_power_of_2(stride),
        block=block,
        num_warps=4,
    )


def _gather_rows(
    columns,
    request_unit_tables,
    request_start_pages,
    table_shapes,
    block_tables,
    start_pages,
    cache_lengths,
    query_lengths,
    query_offsets,
    prefix_offsets,
    write_indices,
    *,
    rows,
    tokens,
    width,
    stride,
):
    """Stage the kernel's outputs with tensor operations, table by table."""
    tables = int(request_unit_tables.shape[0])
    request_width = int(request_unit_tables.shape[2])
    slots = columns[:rows]
    prefix = columns[stride : stride + rows]
    query = columns[2 * stride : 2 * stride + rows]
    writes = columns[3 * stride : 3 * stride + rows] != 0
    token_section = (ROW_SECTIONS + tables) * stride
    owner = columns[token_section : token_section + tokens]
    starts = columns[4 * stride : 4 * stride + rows + 1]
    position = prefix[owner] + torch.arange(tokens, device=columns.device)
    position -= starts[owner]
    cells = torch.arange(width, device=columns.device)

    for table, (group, page_tokens, window) in enumerate(table_shapes.tolist()):
        first = (
            (prefix - window).clamp(min=0) // page_tokens
            if window >= 0
            else torch.zeros_like(prefix)
        )
        start_pages[table, :rows] = first
        installed = request_start_pages[group, slots].to(torch.int64)
        pages = columns[(ROW_SECTIONS + table) * stride :][:rows]

        source = (first - installed)[:, None] + cells[None]
        staged = (
            (cells[None] < pages[:, None])
            & (source >= 0)
            & (source < request_width)
        )
        units = request_unit_tables[
            table, slots[:, None], source.clamp(0, request_width - 1)
        ]
        block_tables[table, :rows, :width] = torch.where(
            staged, units, torch.zeros_like(units)
        )

        column = position // page_tokens - installed[owner]
        valid = writes[owner] & (column >= 0) & (column < request_width)
        unit = request_unit_tables[
            table, slots[owner], column.clamp(0, request_width - 1)
        ].to(torch.int64)
        write_indices[table, :tokens] = torch.where(
            valid & (unit >= 0),
            unit * page_tokens + position % page_tokens,
            torch.full_like(unit, -1),
        )

    cache_lengths[:rows] = prefix
    query_lengths[:rows] = query
    query_offsets[: rows + 1] = starts
    prefix_offsets[: rows + 1] = columns[5 * stride : 5 * stride + rows + 1]


def row_columns_stride(max_rows: int) -> int:
    """Return the length of one row section: every row and its end offset."""
    return int(max_rows) + 1


def row_columns_size(max_rows: int, tables: int, max_tokens: int) -> int:
    """Return the int64 elements the host column array of a call needs."""
    return (ROW_SECTIONS + int(tables)) * row_columns_stride(max_rows) + int(
        max_tokens
    )


__all__ = [
    "ROW_SECTIONS",
    "gather_request_rows",
    "row_columns_size",
    "row_columns_stride",
]
