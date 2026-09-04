"""Derive row-aligned forward tensors from scheduler tables."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch

from uniserve_worker.execution.batch import RunKind
from uniserve_worker.execution.forward_batch import AttentionMode, ExpertRoute, RouteSpan
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.foundation.math import bucketed_length

from .rows import ForwardRow


def columns(runtime, tasks: tuple[ForwardRow, ...]) -> dict[str, object]:
    """Build packed attention mode, sequence, cache, position, and route tensors for forward rows."""

    if not tasks:
        raise invalid_descriptor("attention metadata requires forward rows")
    groups = {int(task.group_id) for task in tasks}
    if len(groups) != 1:
        raise invalid_descriptor("one attention call cannot mix KV groups")
    group_id = groups.pop()
    query_lens = tuple(int(task.query_tokens) for task in tasks)
    seq_lens = tuple(int(task.seq_len) for task in tasks)
    if any(length < 1 for length in query_lens) or any(length < 0 for length in seq_lens):
        raise invalid_descriptor("forward attention lengths are invalid")
    causal_rows = tuple(bool(task.causal) for task in tasks)
    pure_decode = all(
        task.operation.kind is RunKind.AR_DECODE
        and task.token_ids is not None
        and task.query_tokens == 1
        for task in tasks
    )
    binding = _binding_identity(tasks)
    pages = tuple(
        runtime.req_to_token_pool.pages(task.request_pool_idx, group_id) for task in tasks
    )
    capacities = tuple(
        runtime.req_to_token_pool.allocated_length(task.request_pool_idx) for task in tasks
    )
    for task, prefix, query, capacity in zip(tasks, seq_lens, query_lens, capacities, strict=True):
        resulting = prefix + (query if task.write_kv else 0)
        if resulting > capacity:
            raise invalid_descriptor("forward row exceeds its scheduler block table")
    width = bucketed_length(max(1, max(map(len, pages))))
    if pure_decode and all(task.request_indexed_decode for task in tasks):
        states = runtime.runtime_states
        if (
            states is not None
            and states.device.type == "cuda"
            and runtime.req_to_token_pool.page_tables.device == states.device
        ):
            return {
                "forward_mode": AttentionMode.REQUEST_INDEXED_DECODE,
                "seq_lens_cpu": seq_lens,
                "query_lens_cpu": query_lens,
                "kv_lens_cpu": tuple(length + 1 for length in seq_lens),
                "causal_rows_cpu": causal_rows,
                "causal": len(set(causal_rows)) == 1 and causal_rows[0],
                "group_id": group_id,
                "binding": binding,
                "request_page_tables": runtime.req_to_token_pool.page_tables,
                "request_cache_lengths": runtime.req_to_token_pool.verified_lens,
                "request_tokens": states.future_input_tokens[:, 0],
                "request_positions": states.logical_lengths,
                "table_width": width,
                "page_size": runtime.cache_pool.block_size,
            }
    block_table = torch.zeros((len(tasks), width), dtype=torch.int32)
    for row, row_pages in enumerate(pages):
        if row_pages:
            block_table[row, : len(row_pages)] = torch.tensor(row_pages, dtype=torch.int32)
    out_cache_loc = _output_locations(
        pages,
        seq_lens,
        query_lens,
        tuple(bool(task.write_kv) for task in tasks),
        runtime.cache_pool.block_size,
    )
    common: dict[str, object] = {
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32),
        "query_lens": torch.tensor(query_lens, dtype=torch.int32),
        "out_cache_loc": out_cache_loc,
        "has_cache_writes": any(bool(task.write_kv) for task in tasks),
        "block_table": block_table,
        "seq_lens_cpu": seq_lens,
        "query_lens_cpu": query_lens,
        "causal_rows_cpu": causal_rows,
        "causal": len(set(causal_rows)) == 1 and causal_rows[0],
        "group_id": group_id,
        "binding": binding,
    }
    if runtime.model.tensorized_mixed and not pure_decode:
        return {**common, **_packed_columns(tasks, query_lens, seq_lens, width, runtime)}
    kv_lens = tuple(prefix + query for prefix, query in zip(seq_lens, query_lens, strict=True))
    common["kv_lens"] = torch.tensor(kv_lens, dtype=torch.int32)
    common["kv_lens_cpu"] = kv_lens
    common["max_seqlen_k"] = width * runtime.cache_pool.block_size
    if pure_decode:
        common["forward_mode"] = AttentionMode.PAGED_DECODE
        return common
    common.update(
        forward_mode=AttentionMode.PAGED_VARLEN,
        cu_seqlens_q=_cumulative(query_lens),
        cu_seqlens_k=_cumulative(kv_lens),
        output_indices=torch.tensor(
            tuple(sum(query_lens[: index + 1]) - 1 for index in range(len(query_lens))),
            dtype=torch.int64,
        ),
        max_seqlen_q=max(query_lens),
    )
    return common


def dense_columns(row_count: int, query_lens: Sequence[int]) -> dict[str, object]:
    """Build cumulative query offsets and maximum lengths for dense packed rows."""

    lengths = tuple(int(value) for value in query_lens)
    return {
        "forward_mode": AttentionMode.DENSE,
        "seq_lens": torch.zeros(row_count, dtype=torch.int32),
        "query_lens": torch.tensor(lengths, dtype=torch.int32),
        "out_cache_loc": torch.zeros(sum(lengths), dtype=torch.int64),
        "has_cache_writes": False,
        "seq_lens_cpu": (0,) * row_count,
        "query_lens_cpu": lengths,
        "kv_lens_cpu": lengths,
        "causal_rows_cpu": (),
    }


def _packed_columns(
    tasks: tuple[ForwardRow, ...],
    query_lens: tuple[int, ...],
    seq_lens: tuple[int, ...],
    width: int,
    runtime,
) -> dict[str, object]:
    """Build packed attention boundaries, cache tables, write locations, and route spans."""

    max_query = bucketed_length(max(query_lens))
    visible = torch.zeros((len(tasks), max_query), dtype=torch.int32)
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

    for row, (task, query) in enumerate(zip(tasks, query_lens, strict=True)):
        visible[row, :query] = (
            torch.arange(1, query + 1, dtype=torch.int32) if task.causal else query
        )
        position = task.attention_indexes
        if position is None:
            position = _three_axis_positions(task.positions, query)
        if tuple(position.shape) != (3, query):
            raise invalid_descriptor("packed attention indexes must have shape [3, query]")
        indexes.append(position.to(device="cpu", dtype=torch.long))
        if task.token_ids is not None:
            append_span(ExpertRoute.TEXT, query)
            continue
        local_text = tuple(int(value) for value in task.text_local_indices)
        if local_text != tuple(sorted(set(local_text))) or any(
            value < 0 or value >= query for value in local_text
        ):
            raise invalid_descriptor("packed text-local indexes are invalid")
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
    return {
        "forward_mode": AttentionMode.PACKED,
        "attention_indexes": torch.cat(indexes, dim=1),
        "route_spans": tuple(spans),
        "visible_end": visible,
        "cu_seqlens_q": _cumulative(query_lens),
        "max_seqlen_q": max_query,
        "max_seqlen_k": width * runtime.cache_pool.block_size,
        "fully_visible": all(not task.causal for task in tasks),
        "kv_lens_cpu": seq_lens,
    }


def _output_locations(
    pages: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    query_lens: Sequence[int],
    write_rows: Sequence[bool],
    block_size: int,
) -> torch.Tensor:
    """Map newly computed query rows to physical page and offset destinations."""

    values: list[int] = []
    for row_pages, prefix, query, write in zip(
        pages, seq_lens, query_lens, write_rows, strict=True
    ):
        for offset in range(query):
            if not write:
                values.append(0)
                continue
            position = prefix + offset
            page_slot, page_offset = divmod(position, block_size)
            if page_slot >= len(row_pages):
                raise invalid_descriptor("KV output location exceeds its block table")
            values.append(int(row_pages[page_slot]) * block_size + page_offset)
    return torch.tensor(values, dtype=torch.int64)


def _three_axis_positions(positions: torch.Tensor | None, query: int) -> torch.Tensor:
    """Validate and normalize optional positions to three axes by query row."""

    if positions is None:
        return torch.zeros((3, query), dtype=torch.long)
    if positions.ndim == 1 and int(positions.numel()) == query:
        return torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
    if positions.ndim == 2 and tuple(positions.shape) == (3, query):
        return positions
    raise invalid_descriptor("row positions cannot be lowered to three-axis attention indexes")


def _cumulative(lengths: Sequence[int]) -> torch.Tensor:
    """Build cumulative row offsets from host lengths."""

    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


def _binding_identity(tasks: Sequence[ForwardRow]) -> int:
    """Return a shared attention binding when every forward row agrees."""

    hasher = hashlib.blake2b(digest_size=8)
    for task in tasks:
        hasher.update(int(task.operation.request_key.request_id).to_bytes(8, "little"))
        hasher.update(int(task.operation.request_key.epoch).to_bytes(8, "little"))
        hasher.update(int(task.operation.op_id).to_bytes(8, "little"))
    return int.from_bytes(hasher.digest(), "little")


__all__ = ["columns", "dense_columns"]
