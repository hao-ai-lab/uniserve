"""Construct explicit attention inputs from scheduler-assigned unit tables.

The engine's scheduler assigns each request slot its KV units per cache
group; the worker's ``BlockTables`` hold the installed group tables. A group
whose logical page spans ``units_per_page`` units owns that many numerical
block tables (see ``uniserve.runtime.prefix_cache``), so one call's attention
is an ``AttentionBatch`` with one entry per numerical table. Every entry
shares the call's query and prefix lengths; entries differ in their pages,
page size, write addresses and, for a history-windowed group, the first page
each row stages.

This module validates one homogeneous group of ``AttentionRow`` values
against the installed tables (``row_tables``) and selects the pages each
numerical table stages (``table_pages``); ``input_buffers.AttentionBuffers.
stage_rows`` stages those pages from the resident tables on the device. For
callers that pass a prepared batch it also builds the host-side attention
batch (``from_tables``: ``PagedInput`` entries for rows that append to the
cache, ``SegmentedInput`` entries for read-only prefix/current calls).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from uniserve.math import ceil_div
from uniserve.nn.attention import (
    AttentionBatch,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
    paged_append,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor.input_batch import AttentionRow
from uniserve_worker.storage.block_tables import GroupTable

if TYPE_CHECKING:
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.kv_cache import KVCacheManager


@dataclass(frozen=True, slots=True)
class TablePages:
    """The pages one numerical block table stages for every row of a call.

    Attributes:
        block_size: Tokens per page of the table's cache group.
        windowed: Whether the group keeps a history window, so rows carry
            their first staged page; a full-history table starts every row
            at page zero.
        start_pages: Each row's first staged logical page.
        rows: Each row's units of pages ``start_pages[row]..``.
    """

    block_size: int
    windowed: bool
    start_pages: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]

    @property
    def width(self) -> int:
        """Return the most pages any row stages, at least one."""
        return max(1, *map(len, self.rows))


def row_tables(
    tasks: tuple[AttentionRow, ...],
    *,
    tables: BlockTables | None,
    cache: KVCacheManager | None,
) -> tuple[tuple[GroupTable, ...], ...]:
    """Validate scheduler cache extents and resolve each row's group tables.

    Args:
        tasks: Rows of one attention call.
        tables: Installed request block tables.
        cache: Resident KV storage, which authorizes each row's write interval.

    Returns:
        Each row's installed table of every cache group, in group order.

    Raises:
        WorkerError: An ``invalid_descriptor`` error when ``tasks`` is empty,
            storage or tables are absent, a length is out of range, a request
            slot lacks an installed group table, a row's resulting length
            exceeds its allocated capacity, or a windowed row would read a
            retired page. Errors from ``cache.require_writable`` for a
            writing row propagate: a resource error when the interval
            overlaps an export or an import destination, or
            ``invalid_descriptor`` for an interval outside its table.
    """
    if not tasks:
        raise invalid_descriptor("attention metadata requires forward rows")
    if tables is None or cache is None:
        raise invalid_descriptor(
            "paged attention requires resident KV storage and request tables"
        )

    result = []
    for task in tasks:
        query, prefix = int(task.query_tokens), int(task.seq_len)
        if query < 1 or prefix < 0:
            raise invalid_descriptor("forward attention lengths are invalid")

        # Only rows that write KV extend the cached sequence; read-only rows
        # need capacity for their prefix alone.
        resulting = prefix + (query if task.write_kv else 0)
        if resulting > tables.allocated_length(task.request_pool_idx):
            raise invalid_descriptor(
                "forward row exceeds its scheduler block table"
            )

        groups = tuple(
            tables.table(task.request_pool_idx, group)
            for group in range(len(tables.groups))
        )
        for table in groups:
            window = table.shape.window
            if (
                window is not None
                and first_page(table, prefix) < table.start_page
            ):
                raise invalid_descriptor(
                    "attention row reads retired window pages"
                )
            if task.write_kv:
                cache.require_writable(table, start=prefix, length=query)
        result.append(groups)
    return tuple(result)


def first_page(table: GroupTable, prefix: int) -> int:
    """Return the first page a row's queries after ``prefix`` read.

    A query at position ``p`` reads keys at positions ``p - window`` through
    ``p``, so the first query of the row, at ``prefix``, reaches back
    farthest. A full-history table reads from page zero.
    """
    window = table.shape.window
    if window is None:
        return 0
    return max(0, prefix - window) // table.shape.page_tokens


def table_pages(
    tables: Sequence[Sequence[GroupTable]],
    *,
    prefix_lengths: Sequence[int],
    query_lengths: Sequence[int],
) -> tuple[TablePages, ...]:
    """Select the pages every numerical table stages for each row.

    ``tables`` holds each row's table of every group, in group order. A
    full-history table stages every installed page of the row. A windowed
    table stages the pages from the first one the row's window reaches
    through the page holding its last query token, or through its last
    installed page when that comes first; at most ``ceil((window + query) /
    page_tokens) + 1`` pages.

    Returns:
        One ``TablePages`` per numerical table in table order: every group's
        unit positions, group-major.
    """
    if not tables:
        raise ValueError("attention tables require at least one row")
    result = []
    for group, first in enumerate(tables[0]):
        shape = first.shape
        for position in range(shape.units_per_page):
            starts, rows = [], []
            for row, prefix, query in zip(
                tables, prefix_lengths, query_lengths, strict=True
            ):
                table = row[group]
                units = table.row(position)
                start = first_page(table, int(prefix))
                if shape.window is not None:
                    end = min(
                        table.end_page,
                        ceil_div(int(prefix) + int(query), shape.page_tokens),
                    )
                    first_column = start - table.start_page
                    units = units[
                        first_column : max(first_column, end - table.start_page)
                    ]
                starts.append(start)
                rows.append(units)
            result.append(
                TablePages(
                    shape.page_tokens,
                    shape.window is not None,
                    tuple(starts),
                    tuple(rows),
                )
            )
    return tuple(result)


def from_tables(
    pages: Sequence[TablePages],
    *,
    query_lengths: Sequence[int],
    prefix_lengths: Sequence[int],
    causal: Sequence[bool],
    write: Sequence[bool],
) -> AttentionBatch:
    """Build a paged append batch or a read-only prefix/current batch.

    All row arguments align by row, and ``pages`` holds every numerical
    table in table order. When any row writes, each entry is a
    ``PagedInput`` whose rows with ``write`` false keep their query
    positions but carry write index -1, the convention for "no cache write"
    that graph padding in ``graph_inputs.pad_text`` also relies on. When no
    row writes, each entry is a ``SegmentedInput`` in which each query sees
    its prefix and its whole current segment; that form requires every row
    to be noncausal. Tensors are built on the host for transfer to the device.
    Every entry shares one pair of query and prefix lengths.

    Raises:
        ValueError: If a read-only call has a causal row or no host query
            lengths, or if the attention input constructors reject the pages
            and lengths.
    """
    query_lengths, prefix_lengths = tuple(query_lengths), tuple(prefix_lengths)
    queries = SequenceLengths.from_lengths(query_lengths, device="cpu")
    prefixes = SequenceLengths.from_lengths(prefix_lengths, device="cpu")

    if any(write):
        # Rows that do not write keep their query positions but address no
        # cache token.
        skipped = np.repeat(
            np.logical_not(np.asarray(write, dtype=bool)),
            np.asarray(query_lengths, dtype=np.int64),
        )
        entries = {}
        for number, table in enumerate(pages):
            start_pages = table.start_pages if table.windowed else None
            blocks, writes = paged_append(
                table.rows,
                query_lengths=query_lengths,
                prefix_lengths=prefix_lengths,
                block_size=table.block_size,
                start_pages=start_pages,
            )
            writes[skipped] = -1
            # Every table reads the same query domain and prefixes.
            entries[number] = PagedInput(
                queries,
                prefixes,
                BlockTable(
                    torch.from_numpy(blocks),
                    table.block_size,
                    None
                    if start_pages is None
                    else torch.tensor(start_pages, dtype=torch.int32),
                    start_pages,
                ),
                torch.from_numpy(writes),
                tuple(causal),
            )
        return AttentionBatch(entries, queries)

    if any(causal):
        raise ValueError(
            "read-only prefix/current calls require noncausal current sequences"
        )
    maximum = queries.maximum
    if maximum is None:
        raise ValueError("read-only calls require host query lengths")

    # Read-only rows write no KV (no write indices) and see their whole
    # current segment, so the per-position visibility end, [rows, max query],
    # is each row's own query length and the current segment is fully
    # visible.
    visible = queries.values[:, None].expand(-1, maximum)
    entries = {
        number: SegmentedInput(
            queries,
            prefixes,
            _host_table(table),
            None,
            visible,
            True,
        )
        for number, table in enumerate(pages)
    }
    return AttentionBatch(entries, queries)


def _host_table(pages: TablePages) -> BlockTable:
    """Pack one table's rows into a zero-padded host ``BlockTable``.

    The table is ``[rows, width]`` int32; short rows are zero-padded and
    their prefix lengths bound the valid span.
    """
    # Rows fill a NumPy array, one slice assignment per row, rather than one
    # tensor construction per row.
    host = np.zeros((len(pages.rows), pages.width), dtype=np.int32)
    for index, row in enumerate(pages.rows):
        host[index, : len(row)] = row
    table = torch.from_numpy(host)
    if not pages.windowed:
        return BlockTable(table, pages.block_size)
    return BlockTable(
        table,
        pages.block_size,
        torch.tensor(pages.start_pages, dtype=torch.int32),
        pages.start_pages,
    )
