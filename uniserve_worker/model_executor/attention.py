"""Construct explicit attention inputs from scheduler-assigned unit tables.

The engine's scheduler assigns each request slot its KV units per cache
group; the worker's ``BlockTables`` hold the installed group tables. A group
whose logical page spans ``units_per_page`` units owns that many numerical
block tables (see ``uniserve.runtime.kv_cache``), so one call's attention
is an ``AttentionBatch`` with one entry per numerical table. Every entry
shares the call's query and prefix lengths; entries differ in their pages,
block size, write addresses and, for a history-windowed group, the first page
each row stages.

This module validates one homogeneous group of ``AttentionRow`` values
against the installed tables (``row_tables``) and selects the pages each
numerical table stages (``table_pages``); ``input_buffers.AttentionBuffers.
stage_rows`` stages those pages from the resident tables on the device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from uniserve.math import ceil_div
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
    """Resolve each row's installed group tables and authorize its writes.

    The scheduler allocates each request's units, retires its window pages
    and bounds its lengths; rows are taken as stated.

    Args:
        tasks: Rows of one attention call.
        tables: Installed request block tables.
        cache: Resident KV storage, which authorizes each row's write interval.

    Returns:
        Each row's installed table of every cache group, in group order.

    Raises:
        WorkerError: An ``invalid_descriptor`` error when ``tasks`` is empty,
            storage or tables are absent, or a request slot lacks an installed
            group table. Errors from ``cache.require_writable`` for a
            writing row propagate: a resource error when the interval
            overlaps a publication or an import destination, or
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
        groups = tuple(
            tables.table(task.request_pool_idx, group)
            for group in range(len(tables.groups))
        )
        if task.write_kv:
            for table in groups:
                cache.require_writable(
                    table,
                    start=int(task.seq_len),
                    length=int(task.query_tokens),
                )
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
    return max(0, prefix - window) // table.shape.block_size


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
    block_size) + 1`` pages.

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
                        ceil_div(int(prefix) + int(query), shape.block_size),
                    )
                    first_column = start - table.start_page
                    units = units[
                        first_column : max(first_column, end - table.start_page)
                    ]
                starts.append(start)
                rows.append(units)
            result.append(
                TablePages(
                    shape.block_size,
                    shape.window is not None,
                    tuple(starts),
                    tuple(rows),
                )
            )
    return tuple(result)
