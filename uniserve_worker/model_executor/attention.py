"""Construct explicit attention inputs from scheduler-assigned unit tables.

The engine's scheduler assigns each request slot its KV units per cache
group; the worker's ``BlockTables`` hold the installed group tables. A group
whose logical page spans ``units_per_page`` units owns that many numerical
block tables (see ``uniserve.runtime.prefix_cache``), so one call's attention
is an ``AttentionBatch`` with one entry per numerical table. Every entry
shares the call's query and prefix lengths; entries differ in their pages,
page size, write addresses and, for a history-windowed group, the first page
each row selects.

Rust selects visible pages and checks cache writes for one homogeneous batch.
``AttentionBuffers.gather_rows`` gathers units and addresses from resident
GPU tables. ``from_tables`` constructs host tensors for warmup and direct
numerical callers using the same native page selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from uniserve.nn.attention import (
    AttentionBatch,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
    paged_append,
)
from uniserve_worker._uniserve_ipc import TablePages as TablePages
from uniserve_worker._uniserve_ipc import table_pages as table_pages
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor.input_batch import AttentionRow

if TYPE_CHECKING:
    from uniserve_worker.storage.kv_cache import KVCacheManager


def prepare_attention(
    tasks: tuple[AttentionRow, ...],
    *,
    cache: KVCacheManager | None,
) -> tuple[TablePages, ...]:
    """Borrow selected KV pages after native capacity and write checks.

    A writing row extends its prefix by the query length. A read-only row
    consumes only its existing prefix. Windowed reads must remain within
    installed pages; writes must not overlap a transfer's retained range.
    """
    if cache is None:
        raise invalid_descriptor("paged attention requires resident KV storage")

    return cache.prepare_attention(
        tuple(
            (row.request_pool_idx, row.seq_len, row.query_tokens, row.write_kv)
            for row in tasks
        )
    )


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
    rows = pages.rows
    host = np.zeros((len(rows), pages.width), dtype=np.int32)
    for index, row in enumerate(rows):
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
