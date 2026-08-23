"""Attention-plan construction for packed and paged forward rows."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import cast

import torch

from uniserve_worker.batch import WorkVariant
from uniserve_worker.execution.forward_batch import (
    AttentionSelection,
    ExpertRoute,
    KvView,
    PackedAttentionPlan,
    PagedDecodePlan,
    PagedVarlenPlan,
    RequestIndexedDecodePlan,
    RouteSpan,
)
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.foundation.math import bucketed_length
from uniserve_worker.runtime.cache_pool import CacheBatchView, CacheRow

from .rows import ForwardRow


def plan(
    runtime,
    tasks: tuple[ForwardRow, ...],
) -> tuple[
    KvView,
    PagedDecodePlan | RequestIndexedDecodePlan | PagedVarlenPlan | PackedAttentionPlan,
]:
    pure_token_decode = all(
        task.operation.work.variant is WorkVariant.TOKEN_DECODE
        and task.token_ids is not None
        and task.query_tokens == 1
        for task in tasks
    )
    if runtime.model.tensorized_mixed and not pure_token_decode:
        return _packed_plan(runtime, tasks)
    query_lens = tuple(task.query_tokens for task in tasks)
    if any(task.entry is None for task in tasks):
        raise RuntimeError("paged attention task has no aligned KV entry")
    view = CacheBatchView.from_validated_rows(
        runtime.cache_pool,
        tuple(cast(CacheRow, task.entry) for task in tasks),
        query_lengths=query_lens,
    )
    kv_lens = tuple(base + query for base, query in zip(view.base_lens, query_lens, strict=True))
    causal_values = {bool(task.causal) for task in tasks}
    if len(causal_values) != 1:
        raise invalid_descriptor("paged attention rows must share causal semantics")
    causal = causal_values.pop()
    binding = _binding_identity(tasks)
    if all(task.request_indexed_decode for task in tasks):
        states = runtime.runtime_states
        groups = {cast(CacheRow, task.entry).group_id for task in tasks}
        if (
            states is not None
            and states.device.type == "cuda"
            and runtime.cache_pool.request_page_tables.device == states.device
            and len(groups) == 1
        ):
            request_decode = RequestIndexedDecodePlan(
                backends=selection(
                    runtime,
                ),
                request_page_tables=runtime.cache_pool.request_page_tables,
                request_cache_lengths=states.valid_cache_lengths,
                request_tokens=states.future_input_tokens[:, 0],
                request_positions=states.logical_lengths,
                group_id=groups.pop(),
                page_size=int(view.block_size),
                table_width=int(view.block_table_width),
                cache_seqlens_cpu=tuple(view.base_lens),
                kv_seqlens_cpu=kv_lens,
                query_lens_cpu=query_lens,
                causal=causal,
                binding=binding,
            )
            return view, request_decode
    host = torch.device("cpu")
    block_table = view.block_table(host)
    cache_seqlens = view.cache_seqlens(host)
    context_capacity = int(block_table.shape[1]) * int(view.block_size)
    if all(query == 1 for query in query_lens):
        page_ids = _stage_ints(
            tuple(
                task.entry.block_table[task.entry.length // view.block_size]
                for task in tasks
                if task.entry is not None
            ),
            dtype=torch.int32,
        )
        page_offsets = _stage_ints(
            tuple(cast(CacheRow, task.entry).length % view.block_size for task in tasks),
            dtype=torch.int32,
        )
        decode_attention = PagedDecodePlan(
            backends=selection(
                runtime,
            ),
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            kv_seqlens=_stage_ints(
                kv_lens,
                dtype=torch.int32,
            ),
            query_lens=_stage_ints(
                (1,) * len(tasks),
                dtype=torch.int32,
            ),
            cache_seqlens_cpu=tuple(view.base_lens),
            kv_seqlens_cpu=kv_lens,
            query_lens_cpu=query_lens,
            decode_page_ids=page_ids,
            decode_page_offsets=page_offsets,
            max_context_len=context_capacity,
            causal=causal,
            binding=binding,
        )
        return view, decode_attention
    cu_q = _cumulative(query_lens)
    cu_k = _cumulative(kv_lens)
    varlen_attention = PagedVarlenPlan(
        backends=selection(
            runtime,
        ),
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        query_lens=_stage_ints(
            query_lens,
            dtype=torch.int32,
        ),
        kv_seqlens=_stage_ints(
            kv_lens,
            dtype=torch.int32,
        ),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        output_indices=_stage_ints(
            tuple(sum(query_lens[: index + 1]) - 1 for index in range(len(query_lens))),
            dtype=torch.int64,
        ),
        cache_seqlens_cpu=tuple(view.base_lens),
        query_lens_cpu=query_lens,
        kv_seqlens_cpu=kv_lens,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=context_capacity,
        max_context_len=context_capacity,
        causal=causal,
        binding=binding,
    )
    return view, varlen_attention


def _packed_plan(
    runtime,
    tasks: tuple[ForwardRow, ...],
) -> tuple[KvView, PackedAttentionPlan]:
    if any(task.entry is None for task in tasks):
        raise RuntimeError("packed attention task has no aligned KV row")
    view = CacheBatchView(
        runtime.cache_pool,
        tuple(cast(CacheRow, task.entry) for task in tasks),
        tuple(task.query_tokens for task in tasks),
        tuple(task.write_kv for task in tasks),
    )
    query_lens = tuple(task.query_tokens for task in tasks)
    base_lens = view.base_lens
    key_lens = tuple(base + query for base, query in zip(base_lens, query_lens, strict=True))
    # The kernel checks ``visible_end`` against ``max_seqlen_q``, so the query
    # bound and the tensor it sizes are bucketed together: one executable then
    # serves a range of chunk widths instead of one per exact width. Positions
    # past a row's own query length stay zero, the padding this plan already
    # uses for rows shorter than the widest one.
    max_query = bucketed_length(max(query_lens))
    visible = torch.zeros((len(tasks), max_query), dtype=torch.int32)
    index_parts: list[torch.Tensor] = []
    route_spans: list[RouteSpan] = []
    offset = 0

    def append_span(route: ExpertRoute, count: int) -> None:
        nonlocal offset
        if count < 1:
            return
        if route_spans and route_spans[-1].route is route:
            previous = route_spans[-1]
            route_spans[-1] = RouteSpan(
                route,
                previous.token_start,
                previous.token_count + count,
            )
        else:
            route_spans.append(RouteSpan(route, offset, count))
        offset += count

    for row, (task, base, query) in enumerate(zip(tasks, base_lens, query_lens, strict=True)):
        if task.causal:
            visible[row, :query] = torch.arange(
                base + 1,
                base + query + 1,
                dtype=torch.int32,
            )
        else:
            visible[row, :query] = base + query
        indexes = task.attention_indexes
        if indexes is None:
            indexes = _three_axis_positions(task.positions, query)
        if tuple(indexes.shape) != (3, query):
            raise invalid_descriptor("packed attention indexes must have shape [3, query]")
        index_parts.append(indexes.to(device="cpu", dtype=torch.long))
        if task.token_ids is not None:
            append_span(ExpertRoute.TEXT, query)
        else:
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
    host = torch.device("cpu")
    write_page_ids, write_page_offsets, write_token_indices = view.write_plan(host)
    page_table = view.block_table(host)
    context_capacity = int(page_table.shape[1]) * int(view.block_size)
    attention = PackedAttentionPlan(
        backends=selection(
            runtime,
        ),
        indexes=torch.cat(index_parts, dim=1),
        route_spans=tuple(route_spans),
        visible_end=visible,
        cu_seqlens_q=_cumulative(query_lens),
        page_table=page_table,
        seqused_k=torch.tensor(key_lens, dtype=torch.int32),
        write_page_ids=write_page_ids,
        write_page_offsets=write_page_offsets,
        write_token_indices=write_token_indices,
        max_seqlen_q=max_query,
        max_seqlen_k=context_capacity,
        use_prefix_bounds=True,
        fully_visible=all(not task.causal for task in tasks),
        binding=_binding_identity(tasks),
        query_lens_cpu=query_lens,
        key_lens_cpu=key_lens,
    )
    return view, attention


def selection(runtime) -> AttentionSelection:
    if runtime.attention is None:
        raise RuntimeError("model route has no attention selection")
    return runtime.attention


def _three_axis_positions(positions: torch.Tensor | None, query: int) -> torch.Tensor:
    if positions is None:
        return torch.zeros((3, query), dtype=torch.long)
    if positions.ndim == 1 and int(positions.numel()) == query:
        return torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
    if positions.ndim == 2 and tuple(positions.shape) == (3, query):
        return positions
    raise invalid_descriptor("row positions cannot be lowered to three-axis attention indexes")


def _stage_ints(
    values: Sequence[int],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(tuple(int(value) for value in values), dtype=dtype)


def _cumulative(
    lengths: Sequence[int],
) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return _stage_ints(
        values,
        dtype=torch.int32,
    )


def _binding_identity(tasks: Sequence[ForwardRow]) -> int:
    digest = hashlib.sha256(b"uniserve-forward-binding\0")
    for task in tasks:
        digest.update(task.phase.value.encode("ascii"))
        digest.update(task.kind.encode("ascii"))
        digest.update(task.query_tokens.to_bytes(8, "little"))
    return int.from_bytes(digest.digest()[:8], "little")
