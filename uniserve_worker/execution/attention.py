"""Derive row-aligned forward tensors from scheduler tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from uniserve.attention.inputs import physical_columns
from uniserve.attention.metadata import AttentionMetadata, AttentionMode
from uniserve.attention.selection import AttentionSelection
from uniserve.math import bucketed_length
from uniserve.runtime.kv_cache import KVCacheConfig
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.protocol.batch import ForwardMode

from .rows import ForwardRow

if TYPE_CHECKING:
    from ..runtime.block_tables import BlockTables
    from ..runtime.cache_manager import CacheManager


def cache_pages(
    tasks: tuple[ForwardRow, ...], *, tables: BlockTables | None, cache: CacheManager | None
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Validate scheduler cache extents and return physical pages and bounded width."""

    if not tasks:
        raise invalid_descriptor("attention metadata requires forward rows")
    if tables is None or cache is None:
        raise invalid_descriptor("paged attention requires resident KV storage and request tables")
    groups = {int(task.group_id) for task in tasks}
    if len(groups) != 1:
        raise invalid_descriptor("one attention call cannot mix KV groups")
    group_id = groups.pop()
    query_lens = tuple(int(task.query_tokens) for task in tasks)
    prefix_lens = tuple(int(task.seq_len) for task in tasks)
    if any(length < 1 for length in query_lens) or any(length < 0 for length in prefix_lens):
        raise invalid_descriptor("forward attention lengths are invalid")
    pages = tuple(tables.pages(task.request_pool_idx, group_id) for task in tasks)
    capacities = tuple(tables.allocated_length(task.request_pool_idx) for task in tasks)
    for task, prefix, query, capacity, row_pages in zip(
        tasks, prefix_lens, query_lens, capacities, pages, strict=True
    ):
        resulting = prefix + (query if task.write_kv else 0)
        if resulting > capacity:
            raise invalid_descriptor("forward row exceeds its scheduler block table")
        if task.write_kv:
            cache.require_writable(row_pages, group=group_id, start=prefix, length=query)
    # The last shape bucket can end at a non-power-of-two context capacity.
    # Both resident and staged tables own that exact bound; shape padding
    # must not invent columns beyond their scheduler-visible page geometry.
    width = min(
        bucketed_length(max(1, max(map(len, pages)))),
        tables.max_blocks_per_request,
    )
    return pages, width


def columns(
    tasks: tuple[ForwardRow, ...],
    *,
    tables: BlockTables | None,
    cache: CacheManager | None,
    packed: bool,
) -> AttentionMetadata:
    """Build packed attention mode, sequence, cache, position, and route tensors for forward rows."""

    pages, width = cache_pages(tasks, tables=tables, cache=cache)
    assert cache is not None
    query_lens = tuple(task.query_tokens for task in tasks)
    prefix_lens = tuple(task.seq_len for task in tasks)
    causal_rows = tuple(task.causal for task in tasks)
    pure_decode = all(
        task.forward_mode is ForwardMode.DECODE
        and task.token_ids is not None
        and task.query_tokens == 1
        for task in tasks
    )
    try:
        return physical_columns(
            pages=pages,
            prefix_lens=prefix_lens,
            query_lens=query_lens,
            causal_rows=causal_rows,
            write_rows=tuple(task.write_kv for task in tasks),
            positions=tuple(
                task.attention_indexes
                if task.attention_indexes is not None
                else _three_axis_positions(task.positions, task.query_tokens)
                for task in tasks
            ),
            token_rows=tuple(task.token_ids is not None for task in tasks),
            text_local_indices=tuple(task.text_local_indices for task in tasks),
            width=width,
            block_size=cache.cache.page_size,
            packed=packed,
            decode=pure_decode,
        )
    except ValueError as error:
        raise invalid_descriptor(str(error)) from error


def _three_axis_positions(positions: torch.Tensor | None, query: int) -> torch.Tensor:
    """Validate and normalize optional positions to three axes by query row."""

    if positions is None:
        return torch.zeros((3, query), dtype=torch.long)
    if positions.ndim == 1 and int(positions.numel()) == query:
        return torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
    if positions.ndim == 2 and tuple(positions.shape) == (3, query):
        return positions
    raise invalid_descriptor("row positions cannot be lowered to three-axis attention indexes")


__all__ = ["columns"]


def supports_flow_attention(
    selection: AttentionSelection,
    geometry: KVCacheConfig,
    pool: CacheManager,
    device: torch.device,
) -> bool:
    """Return whether the selected backend can execute the model's flow-attention geometry."""

    if pool.cache.is_quantized:
        return False
    head_dim = int(geometry.head_dim)
    for provider in selection.providers:
        if provider.can_bind(
            AttentionMode.PACKED,
            head_dim=head_dim,
            block_size=pool.cache.page_size,
            device=device,
        ):
            return True
    return False
