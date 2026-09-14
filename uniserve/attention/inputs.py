"""Construct numerical attention inputs from explicit pages, lengths and positions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from uniserve.attention.metadata import AttentionMetadata, AttentionMode, ExpertRoute, RouteSpan
from uniserve.math import bucketed_length


def physical_columns(
    *,
    pages: Sequence[Sequence[int]],
    prefix_lens: tuple[int, ...],
    query_lens: tuple[int, ...],
    causal_rows: tuple[bool, ...],
    write_rows: tuple[bool, ...],
    positions: tuple[torch.Tensor, ...],
    token_rows: tuple[bool, ...],
    text_local_indices: tuple[tuple[int, ...], ...],
    width: int,
    block_size: int,
    packed: bool = False,
    decode: bool = False,
) -> AttentionMetadata:
    """Build CPU tensor metadata from explicit numerical page and row inputs."""

    block_table = torch.zeros((len(query_lens), width), dtype=torch.int32)
    for row, row_pages in enumerate(pages):
        if row_pages:
            block_table[row, : len(row_pages)] = torch.tensor(row_pages, dtype=torch.int32)
    out_cache_loc = _output_locations(
        pages,
        prefix_lens,
        query_lens,
        write_rows,
        block_size,
    )
    common = AttentionMetadata(
        attention_mode=AttentionMode.PAGED_VARLEN,
        prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
        has_cache_writes=any(write_rows),
        block_table=block_table,
        prefix_lens_cpu=prefix_lens,
        query_lens_cpu=query_lens,
        causal_rows_cpu=causal_rows,
        causal=len(set(causal_rows)) == 1 and causal_rows[0],
    )
    if packed and not decode:
        return _packed_columns(
            common,
            positions,
            token_rows,
            text_local_indices,
            query_lens,
            prefix_lens,
            width,
            block_size,
            causal_rows,
        )
    seq_lens = tuple(prefix + query for prefix, query in zip(prefix_lens, query_lens, strict=True))
    return replace(
        common,
        attention_mode=AttentionMode.PAGED_DECODE if decode else AttentionMode.PAGED_VARLEN,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu=seq_lens,
        max_seqlen_k=width * block_size,
        cu_seqlens_q=None if decode else _cumulative(query_lens),
        cu_seqlens_k=None if decode else _cumulative(seq_lens),
        output_indices=None
        if decode
        else torch.tensor(
            tuple(sum(query_lens[: index + 1]) - 1 for index in range(len(query_lens))),
            dtype=torch.int64,
        ),
        max_seqlen_q=0 if decode else max(query_lens),
    )


def dense_columns(row_count: int, query_lens: Sequence[int]) -> AttentionMetadata:
    """Build cumulative query offsets and maximum lengths for dense packed rows."""

    lengths = tuple(int(value) for value in query_lens)
    return AttentionMetadata(
        attention_mode=AttentionMode.DENSE,
        prefix_lens=torch.zeros(row_count, dtype=torch.int32),
        query_lens=torch.tensor(lengths, dtype=torch.int32),
        out_cache_loc=torch.zeros(sum(lengths), dtype=torch.int64),
        has_cache_writes=False,
        prefix_lens_cpu=(0,) * row_count,
        query_lens_cpu=lengths,
        seq_lens_cpu=lengths,
        causal_rows_cpu=(),
    )


def _packed_columns(
    common: AttentionMetadata,
    positions: tuple[torch.Tensor, ...],
    token_rows: tuple[bool, ...],
    text_local_indices: tuple[tuple[int, ...], ...],
    query_lens: tuple[int, ...],
    prefix_lens: tuple[int, ...],
    width: int,
    block_size: int,
    causal_rows: tuple[bool, ...],
) -> AttentionMetadata:
    """Build packed attention boundaries, cache tables, write locations, and route spans."""

    max_query = bucketed_length(max(query_lens))
    visible = torch.zeros((len(query_lens), max_query), dtype=torch.int32)
    indexes: list[torch.Tensor] = []
    spans: list[RouteSpan] = []
    offset = 0

    def append_span(route: ExpertRoute, count: int) -> None:
        """Append rows to the packed route, coalescing adjacent spans with the same expert."""

        nonlocal offset
        if count < 1:
            return
        if spans and spans[-1].route is route:
            previous = spans[-1]
            spans[-1] = RouteSpan(route, previous.token_start, previous.token_count + count)
        else:
            spans.append(RouteSpan(route, offset, count))
        offset += count

    for row, query in enumerate(query_lens):
        visible[row, :query] = (
            torch.arange(1, query + 1, dtype=torch.int32) if causal_rows[row] else query
        )
        position = positions[row]
        if tuple(position.shape) != (3, query):
            raise ValueError("packed attention indexes must have shape [3, query]")
        indexes.append(position.to(device="cpu", dtype=torch.long))
        if token_rows[row]:
            append_span(ExpertRoute.TEXT, query)
            continue
        local_text = tuple(int(value) for value in text_local_indices[row])
        if local_text != tuple(sorted(set(local_text))) or any(
            value < 0 or value >= query for value in local_text
        ):
            raise ValueError("packed text-local indexes are invalid")
        cursor = 0
        local_index = 0
        while local_index < len(local_text):
            run_start = local_text[local_index]
            append_span(ExpertRoute.FLOW, run_start - cursor)
            run_end = run_start + 1
            local_index += 1
            while local_index < len(local_text) and local_text[local_index] == run_end:
                run_end += 1
                local_index += 1
            append_span(ExpertRoute.TEXT, run_end - run_start)
            cursor = run_end
        append_span(ExpertRoute.FLOW, query - cursor)
    return replace(
        common,
        attention_mode=AttentionMode.PACKED,
        attention_indexes=torch.cat(indexes, dim=1),
        route_spans=tuple(spans),
        visible_end=visible,
        cu_seqlens_q=_cumulative(query_lens),
        max_seqlen_q=max_query,
        max_seqlen_k=width * block_size,
        fully_visible=not any(causal_rows),
        seq_lens_cpu=tuple(
            prefix + query for prefix, query in zip(prefix_lens, query_lens, strict=True)
        ),
    )


def _output_locations(
    pages: Sequence[Sequence[int]],
    prefix_lens: Sequence[int],
    query_lens: Sequence[int],
    write_rows: Sequence[bool],
    block_size: int,
) -> torch.Tensor:
    """Map newly computed query rows to physical page and offset destinations."""

    values: list[int] = []
    for row_pages, prefix, query, write in zip(
        pages, prefix_lens, query_lens, write_rows, strict=True
    ):
        for offset in range(query):
            if not write:
                values.append(0)
                continue
            position = prefix + offset
            page_slot, page_offset = divmod(position, block_size)
            if page_slot >= len(row_pages):
                raise ValueError("KV output location exceeds its block table")
            values.append(int(row_pages[page_slot]) * block_size + page_offset)
    return torch.tensor(values, dtype=torch.int64)


def _cumulative(lengths: Sequence[int]) -> torch.Tensor:
    """Build cumulative row offsets from host lengths."""

    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


__all__ = ["physical_columns", "dense_columns"]
