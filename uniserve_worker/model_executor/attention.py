"""Construct explicit attention inputs from scheduler-assigned page tables.

The engine's scheduler assigns each request slot its KV pages; the worker's
``BlockTables`` hold the installed tables. This module validates one
homogeneous group of ``AttentionRow`` values against those tables and builds
the host-side attention input (``PagedInput`` for rows that append to the
cache, ``SegmentedInput`` for read-only prefix/current calls) that
``uniserve_worker.model_executor.input_buffers`` then stages into its fixed
device backing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from uniserve.math import bucketed_length
from uniserve.nn.attention import (
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor.input_batch import AttentionRow

if TYPE_CHECKING:
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.kv_cache import KVCacheManager


def cache_pages(
    tasks: tuple[AttentionRow, ...],
    *,
    tables: BlockTables | None,
    cache: KVCacheManager | None,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Validate scheduler cache extents and resolve each row's physical pages.

    Args:
        tasks: Rows of one attention call; all must share one KV group.
        tables: Installed request block tables.
        cache: Resident KV storage, which authorizes each row's write interval.

    Returns:
        Each row's installed page table, and the block-table width to stage:
        the longest row rounded up to a power of two and capped at the tables'
        ``max_blocks_per_request``.

    Raises:
        WorkerError: An ``invalid_descriptor`` error when ``tasks`` is empty,
            storage or tables are absent, rows mix KV groups, a length is out
            of range, a request slot has no installed table, or a row's
            resulting length exceeds its allocated capacity. Errors from
            ``cache.require_writable`` for a writing row propagate: a
            resource error when the interval overlaps a publication or an
            import destination, or ``invalid_descriptor`` for invalid pages.
    """
    if not tasks:
        raise invalid_descriptor("attention metadata requires forward rows")
    if tables is None or cache is None:
        raise invalid_descriptor(
            "paged attention requires resident KV storage and request tables"
        )
    groups = {int(task.group_id) for task in tasks}
    if len(groups) != 1:
        raise invalid_descriptor("one attention call cannot mix KV groups")
    group_id = groups.pop()

    query_lens = tuple(int(task.query_tokens) for task in tasks)
    prefix_lens = tuple(int(task.seq_len) for task in tasks)
    if any(length < 1 for length in query_lens) or any(
        length < 0 for length in prefix_lens
    ):
        raise invalid_descriptor("forward attention lengths are invalid")

    pages = tuple(
        tables.pages(task.request_pool_idx, group_id) for task in tasks
    )
    capacities = tuple(
        tables.allocated_length(task.request_pool_idx) for task in tasks
    )
    # Only rows that write KV extend the cached sequence; read-only rows need
    # capacity for their prefix alone.
    for task, prefix, query, capacity, row_pages in zip(
        tasks, prefix_lens, query_lens, capacities, pages, strict=True
    ):
        resulting = prefix + (query if task.write_kv else 0)
        if resulting > capacity:
            raise invalid_descriptor(
                "forward row exceeds its scheduler block table"
            )
        if task.write_kv:
            cache.require_writable(
                row_pages, group=group_id, start=prefix, length=query
            )

    # Power-of-two widths keep graph shapes few, but the table capacity need
    # not be a power of two. Resident and staged tables are exactly
    # ``max_blocks_per_request`` wide, so padding stops at that bound.
    width = min(
        bucketed_length(max(1, max(map(len, pages)))),
        tables.max_blocks_per_request,
    )
    return pages, width


def from_blocks(
    *, pages, query_lengths, prefix_lengths, block_size, causal, write
):
    """Build a paged append input or a read-only prefix/current input.

    All arguments align by row. When any row writes, the result is a
    ``PagedInput`` whose rows with ``write`` false keep their query positions
    but carry write index -1, the convention for "no cache write" that graph
    padding in ``graph_inputs.pad_text`` also relies on. When no row writes,
    the result is a ``SegmentedInput`` in which each query sees its prefix
    and its whole current segment; that form requires every row to be
    noncausal. Tensors are built on the host for later staging.

    Raises:
        ValueError: If a read-only call has a causal row or no host query
            lengths, if a partially writing call has no write indices, or if
            the attention input constructors reject the pages and lengths.
    """
    if any(write):
        result = PagedInput.from_blocks(
            blocks=tuple(tuple(row) for row in pages),
            query_lengths=query_lengths,
            prefix_lengths=prefix_lengths,
            block_size=block_size,
            causal=causal,
            device="cpu",
        )
        if not all(write):
            writes = result.write_indices
            if writes is None:
                raise ValueError("paged appends require cache write addresses")
            offset = 0
            for length, enabled in zip(query_lengths, write, strict=True):
                if not enabled:
                    writes[offset : offset + length].fill_(-1)
                offset += length
        return result

    queries = SequenceLengths.from_lengths(query_lengths, device="cpu")
    prefixes = SequenceLengths.from_lengths(prefix_lengths, device="cpu")
    # [rows, max pages] int32; short rows are zero-padded and their prefix
    # lengths bound the valid span.
    table = torch.zeros(
        (len(pages), max(1, max(map(len, pages)))), dtype=torch.int32
    )
    for index, row in enumerate(pages):
        table[index, : len(row)] = torch.tensor(row, dtype=torch.int32)
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
    return SegmentedInput(
        queries,
        prefixes,
        BlockTable(table, block_size),
        None,
        queries.values[:, None].expand(-1, maximum),
        True,
    )


def columns(tasks, *, tables, cache):
    """Build the attention input for one homogeneous group of forward rows."""
    pages, _width = cache_pages(tasks, tables=tables, cache=cache)
    return from_blocks(
        pages=pages,
        query_lengths=tuple(task.query_tokens for task in tasks),
        prefix_lengths=tuple(task.seq_len for task in tasks),
        block_size=cache.info.block_size,
        causal=tuple(task.causal for task in tasks),
        write=tuple(task.write_kv for task in tasks),
    )
