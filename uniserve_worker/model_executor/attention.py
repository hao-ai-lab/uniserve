"""Construct explicit attention inputs from scheduler-assigned page tables."""

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
    """Validate scheduler cache extents and return physical pages and bounded.

    width.
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

    # The last shape bucket can end at a non-power-of-two context capacity.
    # Both resident and staged tables own that exact bound; shape padding
    # must not invent columns beyond their scheduler-visible page bounds.
    width = min(
        bucketed_length(max(1, max(map(len, pages)))),
        tables.max_blocks_per_request,
    )
    return pages, width


def from_blocks(
    *, pages, query_lengths, prefix_lengths, block_size, causal, write
):
    """Build ordinary paged appends or a read-only prefix/current attention.

    call.
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

    # Read-only rows write no KV and see their whole current segment, so the
    # per-position visibility end is simply each row's own query length.
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
