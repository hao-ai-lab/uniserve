"""Bounded CUDA graph executables over fixed and exact-shape forward batches."""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, cast

import torch

from uniserve_worker.execution.forward_batch import (
    AttentionMode,
    AttentionSelection,
    ForwardBatch,
    ForwardOutput,
    TokenSelection,
    packed_tensor_views,
)
from uniserve_worker.execution.lane import verify_graph_context
from uniserve_worker.foundation.math import bucketed_length
from uniserve_worker.models.runtime import CacheGeometry
from uniserve_worker.runtime.cache_pool import CachePool

logger = logging.getLogger(__name__)
TOKEN_CONTINUATION_BIT = 1 << 31


class GraphExecutionError(RuntimeError):
    """A configured CUDA graph bucket could not execute safely."""


class _GraphMiss(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GraphGreedyOutput:
    request_pool_indices: torch.Tensor
    tokens: torch.Tensor
    valid: torch.Tensor
    active: torch.Tensor
    finish: torch.Tensor
    continuation: torch.Tensor
    tagged_tokens: torch.Tensor
    completion: torch.Tensor


@dataclass(frozen=True, slots=True)
class GraphRun:
    output: ForwardOutput
    path: str
    row_count: int
    padded_row_count: int
    greedy: GraphGreedyOutput | None = None


@dataclass(slots=True)
class _GraphState:
    graph: torch.cuda.CUDAGraph
    batch: ForwardBatch
    output: ForwardOutput
    greedy: GraphGreedyOutput | None
    published: tuple[ForwardOutput, ...]
    releases: tuple[Callable[[], None], ...]
    startup_resident: bool
    signature: tuple[object, ...]
    publish_cursor: int = 0
    batch_leaves: tuple[torch.Tensor, ...] = ()
    plan_leaves: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True, slots=True)
class _DecodeGeometry:
    bucket: int
    width: int
    padding: int


@dataclass(frozen=True, slots=True)
class _PrefillGeometry:
    token_bucket: int
    row_bucket: int
    width: int
    padding: int
    maximum_padding: int
    max_query_len: int
    max_key_len: int


class CudaGraphRunner:
    """Own the immutable CUDA graph set for one execution partition."""

    def __init__(
        self,
        *,
        enabled: bool,
        prefill_enabled: bool,
        cache: CacheGeometry,
        cache_pool: CachePool,
        attention: AttentionSelection,
        block_size: int,
        weight_digest: str,
        memory_budget_bytes: int,
        decode_batch_sizes: tuple[int, ...] = (),
        decode_predicates: torch.Tensor | None = None,
        decode_context_blocks: int = 0,
        packed_context_blocks: int = 0,
        prefill_token_sizes: tuple[int, ...] = (),
        prefill_row_sizes: tuple[int, ...] = (8, 16),
        stream: torch.cuda.Stream | None = None,
        expected_context: int | None = None,
        expected_resident_executables: int | None = None,
        output_slot_count: int = 2,
    ) -> None:
        if not weight_digest or block_size < 1 or memory_budget_bytes < 0 or output_slot_count < 1:
            raise ValueError("graph-store identity and geometry are invalid")
        if decode_predicates is not None and (
            decode_predicates.ndim != 1 or decode_predicates.dtype is not torch.bool
        ):
            raise ValueError("decode predicate state must be a boolean row vector")
        self.enabled = bool(enabled)
        self.prefill_enabled = bool(prefill_enabled)
        self.cache = cache
        self.cache_pool = cache_pool
        self.attention = attention
        self.block_size = int(block_size)
        self.weight_digest = str(weight_digest)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.decode_batch_sizes = tuple(
            sorted({int(value) for value in decode_batch_sizes if int(value) > 0})
        )
        self.decode_predicates = decode_predicates
        self.decode_context_blocks = max(0, int(decode_context_blocks))
        self.packed_context_blocks = max(0, int(packed_context_blocks))
        self.prefill_token_sizes = tuple(
            sorted({int(value) for value in prefill_token_sizes if int(value) > 0})
        )
        self.prefill_row_sizes = tuple(
            sorted({int(value) for value in prefill_row_sizes if int(value) > 1})
        )
        self.captures = 0
        self._states: dict[tuple[object, ...], _GraphState] = {}
        self._equivalence_checks: list[tuple[str, torch.Tensor]] = []
        self._warmed: set[tuple[object, ...]] = set()
        self._warmed_exact: set[tuple[object, ...]] = set()
        self._covered_exact: set[tuple[object, ...]] = set()
        self._next_binding = 1
        self._device: torch.device | None = None
        self._pool_handle: Any = None
        self._sealed = False
        self._stream = stream
        self._expected_context = expected_context
        self._expected_resident_executables = (
            None
            if expected_resident_executables is None
            else max(0, int(expected_resident_executables))
        )
        self._output_slot_count = int(output_slot_count)

    @property
    def resident_bytes(self) -> int:
        return _private_pool_bytes(self._device)

    def bind_partition(
        self,
        stream: torch.cuda.Stream | None,
        expected_context: int | None,
    ) -> None:
        if self._warmed or self._states or self._sealed:
            raise GraphExecutionError("CUDA graph runner was bound after startup began")
        self._stream = stream
        self._expected_context = expected_context

    def complete_startup(self) -> None:
        """Verify and seal the configured bucket set before request admission."""

        if self._sealed:
            return
        self._warmed.difference_update(self._warmed_exact)
        self._covered_exact.difference_update(self._warmed_exact)
        self._warmed_exact.clear()
        if self._warmed:
            raise GraphExecutionError("startup left configured graph buckets uncaptured")
        if (
            self._expected_resident_executables is not None
            and len(self._states) != self._expected_resident_executables
        ):
            raise GraphExecutionError(
                "resident CUDA graph count does not match the physical executable catalog: "
                f"resident={len(self._states)} "
                f"expected={self._expected_resident_executables} "
                f"families={self._resident_family_counts()!r}"
            )
        self._complete_equivalence_checks()
        if self.resident_bytes > self.memory_budget_bytes:
            raise GraphExecutionError("captured graph residency exceeds its startup budget")
        self._sealed = True

    def _resident_family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key in self._states:
            family = str(key[0]) if key else "unknown"
            if family == "exact" and len(key) > 1:
                family = f"exact_{key[1]}"
                signature = key[-1]
                if isinstance(signature, tuple) and len(signature) > 3:
                    token_rows = signature[2]
                    flow_rows = signature[3]
                    if token_rows and flow_rows:
                        family = "exact_decode_flow"
                    elif flow_rows:
                        family = "exact_flow"
                    elif token_rows:
                        query_lens = signature[4] if len(signature) > 4 else ()
                        selections = signature[5] if len(signature) > 5 else ()
                        if query_lens and all(int(value) == 1 for value in query_lens):
                            family = "exact_decode"
                        elif selections and all(
                            value == TokenSelection.HIDDEN.value for value in selections
                        ):
                            family = "exact_prefix"
                        else:
                            family = "exact_prefill"
            counts[family] = counts.get(family, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def startup_signature(self) -> tuple[object, ...]:
        return (
            self.decode_batch_sizes,
            self.prefill_token_sizes,
            self.prefill_row_sizes,
            tuple(sorted((repr(key) for key in self._states))),
            self.captures,
            self._sealed,
        )

    def execute(
        self,
        key: Hashable,
        batch: ForwardBatch,
        forward: Callable[[ForwardBatch], ForwardOutput],
        *,
        eligible: bool,
        borrow_output: bool = False,
    ) -> GraphRun:
        rows = batch.row_count
        if not eligible or not self.enabled or not _cuda_batch(batch):
            return GraphRun(self._eager(batch, forward), "eager", rows, rows)
        if batch.forward_mode is AttentionMode.PAGED_VARLEN and (
            not self.prefill_enabled
            or any(
                selection is not TokenSelection.LAST_LOGITS for selection in batch.token_selections
            )
        ):
            return GraphRun(self._eager(batch, forward), "eager", rows, rows)
        if batch.forward_mode is AttentionMode.PACKED and not self.prefill_enabled:
            return GraphRun(self._eager(batch, forward), "eager", rows, rows)
        if batch.forward_mode is AttentionMode.PACKED and _quantized_kv(self):
            return GraphRun(self._eager(batch, forward), "eager", rows, rows)
        try:
            _graph_provider(self.attention, batch.forward_mode)
        except _GraphMiss:
            return GraphRun(self._eager(batch, forward), "eager", rows, rows)

        decode = _decode_geometry(
            batch,
            self.decode_batch_sizes,
            self.block_size,
            self.decode_context_blocks,
        )
        prefill = (
            None
            if decode is not None
            else _prefill_geometry(
                batch,
                self.prefill_token_sizes,
                self.block_size,
                self.prefill_row_sizes,
                self.decode_context_blocks,
            )
        )
        if decode is not None:
            execution = _pad_decode_batch(batch, decode, self.block_size)
            state_key = _decode_signature(batch, decode)
            signature = state_key
            padded_rows = decode.bucket
            startup_resident = True
        elif prefill is not None:
            execution = _pad_prefill_batch(batch, prefill)
            state_key = _prefill_signature(batch, prefill)
            signature = state_key
            padded_rows = prefill.row_bucket
            startup_resident = True
        else:
            execution = _normalize_exact_batch(
                batch,
                context_blocks=self.packed_context_blocks,
                block_size=self.block_size,
            )
            signature = _exact_signature(execution)
            direct_key = ("exact", *_direct_key(key))
            state_key = (*direct_key, signature)
            padded_rows = rows
            startup_resident = False

        if not startup_resident and not self._sealed:
            self._covered_exact.add(state_key)
        state = self._states.get(state_key)
        if state is None:
            covered = startup_resident or state_key in self._covered_exact
            if self._sealed:
                if covered:
                    raise GraphExecutionError("configured CUDA graph bucket is not resident")
                return GraphRun(self._eager(execution, forward), "eager", rows, padded_rows)
            if state_key not in self._warmed:
                self._warmed.add(state_key)
                if not startup_resident:
                    self._warmed_exact.add(state_key)
                eager_output = self._eager(execution, forward)
                return GraphRun(
                    _trim_output(eager_output, rows),
                    "graph_fallback",
                    rows,
                    padded_rows,
                    _trim_greedy(
                        _greedy_decode(execution, eager_output, self.decode_predicates),
                        rows,
                    ),
                )
            if self.memory_budget_bytes == 0:
                raise GraphExecutionError("configured CUDA graph residency has no memory budget")
            try:
                eager_output = self._eager(execution, forward)
                eager = self._snapshot_output(_trim_output(eager_output, rows))
                force_finish = (
                    None
                    if execution.decode_force_finish is None
                    else execution.decode_force_finish.detach().clone(
                        memory_format=torch.preserve_format
                    )
                )
                state = self._capture(
                    execution,
                    forward,
                    startup_resident=startup_resident,
                    signature=signature,
                )
                self._states[state_key] = state
                self._warmed.remove(state_key)
                self._warmed_exact.discard(state_key)
                self.captures += 1
                if force_finish is not None:
                    execution.decode_force_finish.copy_(force_finish)
                self._replay(state, execution)
                self._queue_equivalence_check(
                    eager,
                    _trim_output(state.output, rows),
                    label=repr(state_key),
                )
                context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
                with context:
                    graph_greedy = _trim_greedy(
                        _greedy_decode_values(
                            state.batch,
                            state.output,
                            self.decode_predicates,
                            force_finish,
                            clear_force_finish=False,
                        ),
                        rows,
                    )
                    self._queue_greedy_equivalence_check(
                        graph_greedy,
                        _trim_greedy(state.greedy, rows),
                        label=repr(state_key),
                    )
            except Exception as error:
                removed = self._states.pop(state_key, None)
                if removed is not None:
                    _release_state(removed)
                raise GraphExecutionError("configured CUDA graph capture failed") from error
            output = (
                _trim_output(state.output, rows)
                if borrow_output
                else self._publish_output(state, rows)
            )
            return GraphRun(
                output,
                "graph_capture",
                rows,
                padded_rows,
                _trim_greedy(state.greedy, rows),
            )
        if state.signature != signature:
            raise GraphExecutionError("configured CUDA graph physical shape changed")
        try:
            self._replay(state, execution)
        except Exception as error:
            raise GraphExecutionError("CUDA graph replay failed") from error
        output = (
            _trim_output(state.output, rows) if borrow_output else self._publish_output(state, rows)
        )
        return GraphRun(
            output,
            "graph_replay",
            rows,
            padded_rows,
            _trim_greedy(state.greedy, rows),
        )

    def close(self) -> None:
        states = tuple(self._states.values())
        self._states.clear()
        self._warmed.clear()
        self._warmed_exact.clear()
        self._covered_exact.clear()
        self._equivalence_checks.clear()
        for state in states:
            _release_state(state)

    def invalidate(self, weight_digest: str) -> None:
        """Retire every executable captured against the previous weight identity."""

        if not weight_digest:
            raise ValueError("CUDA graph invalidation requires a weight digest")
        self.close()
        self.weight_digest = str(weight_digest)
        self._sealed = False

    def _capture(
        self,
        batch: ForwardBatch,
        forward: Callable[[ForwardBatch], ForwardOutput],
        *,
        startup_resident: bool,
        signature: tuple[object, ...],
    ) -> _GraphState:
        self._device = _batch_device(batch)
        static = _graph_batch(
            batch,
            self._next_binding,
            own_inputs=not startup_resident,
        )
        self._next_binding += 1
        releases = self._prepare_attention(static, batch, capture=True)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            if self._pool_handle is None:
                self._pool_handle = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self._pool_handle, stream=self._stream):
                output = forward(static)
                greedy = _greedy_decode(static, output, self.decode_predicates)
            if not isinstance(output, ForwardOutput):
                raise TypeError("captured model call did not return ForwardOutput")
            graph.instantiate()
            verify_graph_context(graph, self._expected_context)
            context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
            with context:
                published = tuple(
                    ForwardOutput(tuple(torch.empty_like(value) for value in output.values))
                    for _ in range(self._output_slot_count)
                )
            return _GraphState(
                graph,
                static,
                output,
                greedy,
                published,
                releases,
                startup_resident,
                signature,
                batch_leaves=tuple(_tensor_leaves(static)),
                plan_leaves=tuple(_attention_tensor_leaves(static)),
            )
        except Exception:
            for release in reversed(releases):
                release()
            reset = getattr(graph, "reset", None)
            if callable(reset):
                reset()
            raise

    def _eager(
        self,
        batch: ForwardBatch,
        forward: Callable[[ForwardBatch], ForwardOutput],
    ) -> ForwardOutput:
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context:
            return forward(batch)

    def _replay(self, state: _GraphState, execution: ForwardBatch) -> None:
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context:
            if not state.startup_resident:
                _copy_into_leaves(state.batch_leaves, execution, "forward")
            else:
                _copy_into_leaves(
                    state.plan_leaves,
                    tuple(_attention_tensor_leaves(execution)),
                    "attention",
                )
            self._prepare_attention(state.batch, execution, capture=False)
            state.graph.replay()

    def _publish_output(self, state: _GraphState, rows: int) -> ForwardOutput:
        published = state.published[state.publish_cursor]
        state.publish_cursor = (state.publish_cursor + 1) % len(state.published)
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context:
            for destination, source in zip(
                published.values[:rows], state.output.values[:rows], strict=True
            ):
                destination.copy_(source)
        return ForwardOutput(published.values[:rows])

    def _snapshot_output(self, output: ForwardOutput) -> ForwardOutput:
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context:
            return _clone_output(output)

    def _queue_equivalence_check(
        self,
        reference: ForwardOutput,
        candidate: ForwardOutput,
        *,
        label: str,
    ) -> None:
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context:
            check = _equivalence_check(reference, candidate)
        self._equivalence_checks.append((label, check))

    def _queue_greedy_equivalence_check(
        self,
        reference: GraphGreedyOutput | None,
        candidate: GraphGreedyOutput | None,
        *,
        label: str,
    ) -> None:
        if reference is None or candidate is None:
            if reference is not candidate:
                raise GraphExecutionError("CUDA graph greedy output availability changed")
            return
        for field in fields(GraphGreedyOutput):
            expected = getattr(reference, field.name)
            actual = getattr(candidate, field.name)
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise GraphExecutionError("CUDA graph greedy output geometry changed")
            self._equivalence_checks.append(
                (f"{label}:{field.name}", torch.eq(expected, actual).all())
            )

    def _complete_equivalence_checks(self) -> None:
        if not self._equivalence_checks:
            return
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context:
            complete = torch.stack(tuple(check for _, check in self._equivalence_checks)).all()
        if not bool(complete.item()):
            failed = tuple(
                label for label, check in self._equivalence_checks if not bool(check.item())
            )
            raise GraphExecutionError(
                "CUDA graph output differs from direct execution for buckets: " + ", ".join(failed)
            )
        self._equivalence_checks.clear()

    def _prepare_attention(
        self,
        static_batch: ForwardBatch,
        live_batch: ForwardBatch,
        *,
        capture: bool,
    ) -> tuple[Callable[[], None], ...]:
        static = static_batch
        live = live_batch
        if static.forward_mode is not live.forward_mode:
            raise _GraphMiss("attention form changed for a graph bucket")
        if static.forward_mode in {AttentionMode.DENSE, AttentionMode.PACKED}:
            return ()
        if static.forward_mode not in {AttentionMode.PAGED_DECODE, AttentionMode.PAGED_VARLEN}:
            return ()
        prepared = _live_attention(static, live)
        backend = _graph_provider(self.attention, static.forward_mode)
        key_cache, _value_cache = self.cache_pool.layer_cache(0, static.group_id)
        q_dtype = key_cache.dtype
        kv_dtype = key_cache.dtype
        releases: list[Callable[[], None]] = []
        if static.forward_mode is AttentionMode.PAGED_DECODE:
            prepare = getattr(backend, "prepare_paged_decode_cuda_graph", None)
            if callable(prepare):
                prepare(
                    static.binding,
                    prepared,
                    batch_size=int(cast(torch.Tensor, static.block_table).shape[0]),
                    max_indices=max(1, int(cast(torch.Tensor, static.block_table).numel())),
                    num_q_heads=int(self.cache.num_attention_heads),
                    num_kv_heads=int(self.cache.num_kv_heads),
                    head_dim=int(self.cache.head_dim),
                    page_size=self.block_size,
                    q_dtype=q_dtype,
                    kv_dtype=kv_dtype,
                )
                if capture:
                    release = getattr(backend, "release_paged_decode_graph_binding", None)
                    if callable(release):
                        releases.append(_release_call(release, static.binding))
            return tuple(releases)
        if static.forward_mode is not AttentionMode.PAGED_VARLEN:
            return ()
        bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
        prepare = getattr(backend, "prepare_paged_prefill_cuda_graph", None)
        if callable(bind) and callable(prepare):
            if capture:
                bind(
                    static.binding,
                    static,
                    device=cast(torch.Tensor, static.block_table).device,
                )
                release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
                if callable(release):
                    releases.append(_release_call(release, static.binding))
            prepare(
                static.binding,
                prepared,
                num_q_heads=int(self.cache.num_attention_heads),
                num_kv_heads=int(self.cache.num_kv_heads),
                head_dim=int(self.cache.head_dim),
                page_size=self.block_size,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                causal=static.causal,
            )
        return tuple(releases)


def _decode_geometry(
    batch: ForwardBatch,
    batch_sizes: tuple[int, ...],
    block_size: int,
    context_blocks: int,
) -> _DecodeGeometry | None:
    if batch.forward_mode is not AttentionMode.PAGED_DECODE or not batch_sizes:
        return None
    rows = batch.row_count
    bucket = next((value for value in batch_sizes if value >= rows), None)
    if bucket is None or batch.input_ids is None or batch.positions is None:
        return None
    if (
        batch.token_row_indices != tuple(range(rows))
        or batch.query_lens_cpu != (1,) * rows
        or int(batch.input_ids.numel()) != rows
        or int(batch.positions.shape[-1]) != rows
        or len(set(batch.token_selections)) != 1
    ):
        return None
    if batch.block_table is None:
        return None
    live_width = int(batch.block_table.shape[1])
    if context_blocks > 0 and live_width > context_blocks:
        return None
    reserved_width = max(1, (int(bucket) + int(block_size) - 1) // int(block_size))
    return _DecodeGeometry(
        bucket=int(bucket),
        width=max(live_width, int(context_blocks), reserved_width),
        padding=int(bucket) - rows,
    )


def _prefill_geometry(
    batch: ForwardBatch,
    token_sizes: tuple[int, ...],
    block_size: int,
    row_sizes: tuple[int, ...],
    context_blocks: int,
) -> _PrefillGeometry | None:
    if batch.forward_mode is not AttentionMode.PAGED_VARLEN or not token_sizes:
        return None
    rows = batch.row_count
    row_bucket = next((value for value in row_sizes if value > rows), None)
    query_lens = tuple(int(value) for value in batch.query_lens_cpu)
    if (
        row_bucket is None
        or (rows >= row_sizes[0] and all(value == 1 for value in query_lens))
        or batch.input_ids is None
        or batch.positions is None
        or batch.token_row_indices != tuple(range(rows))
        or len(query_lens) != rows
        or any(value < 1 for value in query_lens)
        or int(batch.input_ids.numel()) != sum(query_lens)
        or int(batch.positions.shape[-1]) != sum(query_lens)
    ):
        return None
    live_tokens = sum(query_lens)
    token_bucket = next((value for value in token_sizes if value >= live_tokens), None)
    if token_bucket is None:
        return None
    if batch.block_table is None:
        return None
    live_width = int(batch.block_table.shape[1])
    if context_blocks > 0 and live_width > context_blocks:
        return None
    previous = max((value for value in token_sizes if value < token_bucket), default=0)
    maximum_padding = int(token_bucket) - int(previous)
    reserved_width = max(1, (maximum_padding + int(block_size) - 1) // int(block_size))
    width = max(live_width, int(context_blocks), reserved_width)
    return _PrefillGeometry(
        token_bucket=int(token_bucket),
        row_bucket=int(row_bucket),
        width=width,
        padding=int(token_bucket) - live_tokens,
        maximum_padding=maximum_padding,
        max_query_len=bucketed_length(int(token_bucket)),
        max_key_len=width * int(block_size),
    )


def _pad_decode_batch(
    batch: ForwardBatch,
    geometry: _DecodeGeometry,
    block_size: int,
) -> ForwardBatch:
    bucket = geometry.bucket
    input_ids = _fixed_view(cast(torch.Tensor, batch.input_ids), (bucket,))
    positions = _expand_token_axis(cast(torch.Tensor, batch.positions), bucket)
    input_embeddings = (
        None
        if batch.input_embeddings is None
        else _fixed_view(
            batch.input_embeddings,
            (bucket, int(batch.input_embeddings.shape[1])),
        )
    )
    embedding_mask = (
        None if batch.embedding_mask is None else _fixed_view(batch.embedding_mask, (bucket,))
    )
    if batch.block_table is None or batch.kv_lens is None:
        raise _GraphMiss("paged decode has incomplete tensors")
    return replace(
        batch,
        row_count=bucket,
        req_pool_indices=_fixed_view(batch.req_pool_indices, (bucket,)),
        seq_lens=_fixed_view(batch.seq_lens, (bucket,)),
        query_lens=_fixed_view(batch.query_lens, (bucket,)),
        out_cache_loc=_fixed_view(batch.out_cache_loc, (bucket,)),
        block_table=_fixed_view(batch.block_table, (bucket, geometry.width)),
        kv_lens=_fixed_view(batch.kv_lens, (bucket,)),
        seq_lens_cpu=(*batch.seq_lens_cpu, *(0 for _ in range(geometry.padding))),
        kv_lens_cpu=(*batch.kv_lens_cpu, *(1 for _ in range(geometry.padding))),
        query_lens_cpu=(*batch.query_lens_cpu, *(1 for _ in range(geometry.padding))),
        max_seqlen_k=geometry.width * int(block_size),
        decode_force_finish=(
            None
            if batch.decode_force_finish is None
            else _fixed_view(batch.decode_force_finish, (bucket,))
        ),
        token_row_indices=tuple(range(bucket)),
        input_ids=input_ids,
        input_embeddings=input_embeddings,
        embedding_mask=embedding_mask,
        positions=positions,
        token_selections=(batch.token_selections[0],) * bucket,
    )


def _pad_prefill_batch(batch: ForwardBatch, geometry: _PrefillGeometry) -> ForwardBatch:
    live_rows = batch.row_count
    dummy_rows = geometry.row_bucket - live_rows
    input_ids = _fixed_view(cast(torch.Tensor, batch.input_ids), (geometry.token_bucket,))
    positions = _expand_token_axis(cast(torch.Tensor, batch.positions), geometry.token_bucket)
    input_embeddings = (
        None
        if batch.input_embeddings is None
        else _fixed_view(
            batch.input_embeddings,
            (geometry.token_bucket, int(batch.input_embeddings.shape[1])),
        )
    )
    embedding_mask = (
        None
        if batch.embedding_mask is None
        else _fixed_view(batch.embedding_mask, (geometry.token_bucket,))
    )
    if (
        batch.block_table is None
        or batch.kv_lens is None
        or batch.cu_seqlens_q is None
        or batch.cu_seqlens_k is None
        or batch.output_indices is None
    ):
        raise _GraphMiss("paged prefill has incomplete tensors")
    block_table = _fixed_view(
        batch.block_table,
        (geometry.row_bucket, geometry.width),
    )
    cache_seqlens = _fixed_view(batch.seq_lens, (geometry.row_bucket,))
    query_lens = _fixed_view(batch.query_lens, (geometry.row_bucket,))
    kv_seqlens = _fixed_view(batch.kv_lens, (geometry.row_bucket,))
    cu_seqlens_q = _fixed_view(batch.cu_seqlens_q, (geometry.row_bucket + 1,))
    cu_seqlens_k = _fixed_view(batch.cu_seqlens_k, (geometry.row_bucket + 1,))
    output_indices = _fixed_view(batch.output_indices, (geometry.row_bucket,))
    cache_seqlens[live_rows:].zero_()
    query_lens[live_rows:].zero_()
    kv_seqlens[live_rows:].zero_()
    if geometry.padding:
        query_lens[live_rows : live_rows + 1].fill_(geometry.padding)
        kv_seqlens[live_rows : live_rows + 1].fill_(geometry.padding)
    cu_seqlens_q[live_rows + 1 :].fill_(geometry.token_bucket)
    padded_kv_tokens = sum(int(value) for value in batch.kv_lens_cpu) + geometry.padding
    cu_seqlens_k[live_rows + 1 :].fill_(padded_kv_tokens)
    output_indices[live_rows:].zero_()
    if geometry.padding:
        output_indices[live_rows : live_rows + 1].fill_(geometry.token_bucket - 1)
    dummy_query_lens = (geometry.padding, *(0 for _ in range(dummy_rows - 1)))
    return replace(
        batch,
        row_count=geometry.row_bucket,
        req_pool_indices=_fixed_view(
            batch.req_pool_indices,
            (geometry.row_bucket,),
        ),
        seq_lens=cache_seqlens,
        query_lens=query_lens,
        out_cache_loc=_fixed_view(batch.out_cache_loc, (geometry.token_bucket,)),
        block_table=block_table,
        kv_lens=kv_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        output_indices=output_indices,
        seq_lens_cpu=(*batch.seq_lens_cpu, *(0 for _ in range(dummy_rows))),
        query_lens_cpu=(*batch.query_lens_cpu, *dummy_query_lens),
        kv_lens_cpu=(*batch.kv_lens_cpu, *dummy_query_lens),
        max_seqlen_q=geometry.max_query_len,
        max_seqlen_k=geometry.max_key_len,
        token_row_indices=tuple(range(geometry.row_bucket)),
        input_ids=input_ids,
        input_embeddings=input_embeddings,
        embedding_mask=embedding_mask,
        positions=positions,
        token_selections=(batch.token_selections[0],) * geometry.row_bucket,
    )


def _decode_signature(batch: ForwardBatch, geometry: _DecodeGeometry) -> tuple[object, ...]:
    return (
        "paged_decode_bucket",
        batch.phase.value,
        geometry.bucket,
        geometry.width,
        batch.token_selections[0].value,
        _batch_tensor_signature(batch),
        bool(batch.causal),
        bool(batch.has_cache_writes),
        _owner_signature(batch),
    )


def _prefill_signature(batch: ForwardBatch, geometry: _PrefillGeometry) -> tuple[object, ...]:
    return (
        "paged_prefill_bucket",
        batch.phase.value,
        geometry.row_bucket,
        geometry.token_bucket,
        geometry.width,
        geometry.max_query_len,
        geometry.max_key_len,
        batch.token_selections[0].value,
        _batch_tensor_signature(batch),
        bool(batch.causal),
        bool(batch.has_cache_writes),
        _owner_signature(batch),
    )


def _batch_tensor_signature(batch: ForwardBatch) -> tuple[object, ...]:
    assert batch.input_ids is not None and batch.positions is not None
    return (
        str(batch.input_ids.dtype),
        batch.input_ids.device.type,
        batch.positions.ndim,
        int(batch.positions.shape[0]) if batch.positions.ndim == 2 else 1,
        str(batch.positions.dtype),
        batch.input_embeddings is not None,
        None if batch.input_embeddings is None else str(batch.input_embeddings.dtype),
    )


def _owner_signature(batch: ForwardBatch) -> tuple[object, ...]:
    return (
        type(batch.mesh).__qualname__,
        type(batch.output).__qualname__,
    )


def _direct_key(key: Hashable) -> tuple[object, ...]:
    if isinstance(key, tuple):
        return tuple(key)
    return (key,)


def _exact_signature(batch: ForwardBatch) -> tuple[object, ...]:
    return (
        batch.phase.value,
        batch.row_count,
        batch.token_row_indices,
        batch.flow_row_indices,
        batch.forward_mode.value,
        batch.query_lens_cpu,
        tuple(value.value for value in batch.token_selections),
        batch.flow_image_tokens,
        batch.flow_heights,
        batch.flow_widths,
        tuple(value is not None for value in batch.flow_conditioning),
        tuple(_tensor_signature(value) for value in _tensor_leaves(batch)),
        batch.route_spans,
        batch.causal_rows_cpu,
        bool(batch.has_cache_writes),
        batch.max_seqlen_q,
        batch.max_seqlen_k,
        _owner_signature(batch),
    )


def _normalize_exact_batch(
    batch: ForwardBatch,
    *,
    context_blocks: int,
    block_size: int,
) -> ForwardBatch:
    """Give exact packed graphs their startup-fixed KV table geometry.

    Packed flow and mixed calls use request-variable KV prefix lengths, but the
    partition input buffer already owns a maximum-width, zero-scrubbed block
    table.  Capturing the active request-width view makes otherwise identical
    startup and serving calls different graph shapes.  Widening that view here
    keeps the physical kernel launch fixed while ``seqused_k`` and the other
    staged tensors carry the live lengths on every replay.
    """

    if batch.forward_mode is not AttentionMode.PACKED or context_blocks <= 0:
        return batch
    if batch.block_table is None:
        raise _GraphMiss("packed attention has no block table")
    if int(batch.block_table.shape[1]) > context_blocks:
        raise _GraphMiss("packed attention exceeds the configured context width")
    return replace(
        batch,
        block_table=_fixed_view(
            batch.block_table,
            (int(batch.block_table.shape[0]), int(context_blocks)),
        ),
        max_seqlen_k=int(context_blocks) * int(block_size),
    )


def _tensor_signature(value: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(int(extent) for extent in value.shape),
        str(value.dtype),
        value.device.type,
    )


def _graph_batch(
    batch: ForwardBatch,
    binding: int,
    *,
    own_inputs: bool,
) -> ForwardBatch:
    graph_batch = _clone_value(batch) if own_inputs else batch
    if not isinstance(graph_batch, ForwardBatch):
        raise TypeError("graph input cloning did not preserve ForwardBatch")
    if graph_batch.forward_mode is AttentionMode.REQUEST_INDEXED_DECODE:
        raise _GraphMiss("request-indexed decode metadata was not staged")
    return replace(
        graph_batch,
        binding=int(binding),
        cuda_graph_capture=True,
    )


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.preserve_format)
    if isinstance(value, AttentionSelection):
        return value
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{field.name: _clone_value(getattr(value, field.name)) for field in fields(value)},
        )
    return value


_LEAF_FIELD_NAMES: dict[type, tuple[str, ...] | None] = {}


def _leaf_field_names(value: Any) -> tuple[str, ...] | None:
    kind = type(value)
    names = _LEAF_FIELD_NAMES.get(kind, ())
    if names == ():
        names = (
            tuple(field.name for field in fields(value))
            if is_dataclass(value) and not isinstance(value, type)
            else None
        )
        _LEAF_FIELD_NAMES[kind] = names
    return names


def _tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, AttentionSelection):
        return
    if isinstance(value, tuple):
        for item in value:
            yield from _tensor_leaves(item)
        return
    names = _leaf_field_names(value)
    if names is not None:
        for name in names:
            yield from _tensor_leaves(getattr(value, name))


def _copy_into_leaves(
    target_tensors: tuple[torch.Tensor, ...],
    source: object,
    structure: str,
) -> None:
    index = 0
    limit = len(target_tensors)
    for value in _tensor_leaves(source):
        if index >= limit:
            raise _GraphMiss(f"{structure} tensor structure changed")
        destination = target_tensors[index]
        index += 1
        if (
            destination.shape != value.shape
            or destination.dtype != value.dtype
            or destination.device != value.device
        ):
            raise _GraphMiss(f"{structure} tensor geometry changed")
        destination.copy_(value, non_blocking=True)
    if index != limit:
        raise _GraphMiss(f"{structure} tensor structure changed")


def _graph_provider(selection: AttentionSelection, mode: AttentionMode):
    for provider in selection.providers:
        capabilities = provider.capabilities()
        available = capabilities.available
        if mode is AttentionMode.PACKED:
            capable = capabilities.segmented_attention
            safe = capabilities.segmented_attention_cuda_graph
        elif mode is AttentionMode.PAGED_VARLEN:
            capable = capabilities.varlen_attention and capabilities.varlen_paged_kv
            safe = capabilities.paged_varlen_cuda_graph or (
                callable(getattr(provider, "bind_paged_prefill_graph_wrapper", None))
                and callable(getattr(provider, "prepare_paged_prefill_cuda_graph", None))
            )
        elif mode is AttentionMode.PAGED_DECODE:
            capable = capabilities.paged_kv
            safe = capabilities.paged_kv
        else:
            capable = bool(capabilities.dense_ranks)
            safe = True
        if not available or not capable:
            continue
        if not safe:
            continue
        return provider
    raise _GraphMiss("no provisioned attention provider is graph-safe")


_PLAN_LENGTH_BOUNDS = frozenset({"max_seqlen_q", "max_seqlen_k"})


def _live_attention(static: object, live: object) -> object:
    if not is_dataclass(static) or not is_dataclass(live):
        raise _GraphMiss("attention plan is not immutable data")
    updates: dict[str, object] = {}
    for field in fields(static):
        static_value = getattr(static, field.name)
        if (
            isinstance(static_value, torch.Tensor)
            or field.name == "binding"
            or field.name in _PLAN_LENGTH_BOUNDS
        ):
            updates[field.name] = static_value
        else:
            updates[field.name] = getattr(live, field.name)
    return replace(cast(Any, live), **updates)


def _fixed_view(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if tensor.ndim != len(shape) or any(value < 0 for value in shape):
        raise _GraphMiss("fixed graph view rank changed")
    strides = tuple(int(value) for value in tensor.stride())
    if any(value < 0 for value in strides):
        raise _GraphMiss("fixed graph view has a negative stride")
    maximum = int(tensor.storage_offset())
    for extent, stride in zip(shape, strides, strict=True):
        if extent:
            maximum += (int(extent) - 1) * stride
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    if maximum >= storage_elements:
        raise _GraphMiss("graph bucket exceeds its fixed input storage")
    return tensor.as_strided(shape, strides, storage_offset=int(tensor.storage_offset()))


def _expand_token_axis(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
    if tensor.ndim == 1:
        return _fixed_view(tensor, (tokens,))
    if tensor.ndim == 2:
        return _fixed_view(tensor, (int(tensor.shape[0]), tokens))
    raise _GraphMiss("graph token positions have an invalid rank")


def _cuda_batch(batch: ForwardBatch) -> bool:
    tensors = tuple(_tensor_leaves(batch))
    return bool(
        tensors
        and torch.cuda.is_available()
        and all(value.device.type == "cuda" for value in tensors)
        and len({value.device for value in tensors}) == 1
    )


def _batch_device(batch: ForwardBatch) -> torch.device:
    for tensor in _tensor_leaves(batch):
        return tensor.device
    raise _GraphMiss("forward batch carries no device tensors")


def _attention_tensor_leaves(batch: ForwardBatch) -> Iterator[torch.Tensor]:
    for name in (
        "req_pool_indices",
        "seq_lens",
        "query_lens",
        "out_cache_loc",
        "block_table",
        "kv_lens",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "output_indices",
        "attention_indexes",
        "visible_end",
    ):
        value = getattr(batch, name)
        if isinstance(value, torch.Tensor):
            yield value


def _quantized_kv(runner: CudaGraphRunner) -> bool:
    return bool(runner.cache_pool.is_quantized)


def _private_pool_bytes(device: torch.device | None) -> int:
    if device is None or not torch.cuda.is_available():
        return 0
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = 0
    for segment in torch.cuda.memory_snapshot():
        if segment.get("device") != index:
            continue
        pool_id = segment.get("segment_pool_id")
        if isinstance(pool_id, tuple) and any(pool_id):
            total += int(segment.get("total_size", 0))
    return total


def _trim_output(output: ForwardOutput, rows: int) -> ForwardOutput:
    return ForwardOutput(tuple(output.values[:rows]))


def _greedy_decode(
    batch: ForwardBatch,
    output: ForwardOutput,
    predicate_state: torch.Tensor | None,
) -> GraphGreedyOutput | None:
    return _greedy_decode_values(
        batch,
        output,
        predicate_state,
        batch.decode_force_finish,
        clear_force_finish=True,
    )


def _greedy_decode_values(
    batch: ForwardBatch,
    output: ForwardOutput,
    predicate_state: torch.Tensor | None,
    force_finish: torch.Tensor | None,
    *,
    clear_force_finish: bool,
) -> GraphGreedyOutput | None:
    if (
        batch.forward_mode is not AttentionMode.PAGED_DECODE
        or predicate_state is None
        or force_finish is None
        or len(output.values) != batch.row_count
    ):
        return None
    rows = tuple(value.reshape(-1) for value in output.values)
    logits = packed_tensor_views(rows)
    if logits is None:
        raise _GraphMiss("decode logits are not one contiguous graph output")
    logits = logits.reshape(batch.row_count, -1)
    max_values, tokens = torch.max(logits, dim=-1)
    valid = torch.isfinite(max_values)
    active = predicate_state.index_select(0, batch.request_pool_indices.reshape(-1))
    finish = force_finish.reshape(-1) & valid & active
    continuation = valid & active & ~finish
    tags = torch.where(continuation, TOKEN_CONTINUATION_BIT, 0)
    tagged_tokens = tokens.bitwise_or(tags)
    completion = torch.cat(
        (
            valid,
            active,
            tokens,
            torch.zeros_like(tokens),
        )
    )
    if clear_force_finish:
        force_finish.zero_()
    return GraphGreedyOutput(
        request_pool_indices=batch.request_pool_indices,
        tokens=tokens,
        valid=valid,
        active=active,
        finish=finish,
        continuation=continuation,
        tagged_tokens=tagged_tokens,
        completion=completion,
    )


def _trim_greedy(
    output: GraphGreedyOutput | None,
    rows: int,
) -> GraphGreedyOutput | None:
    if output is None:
        return None
    total = int(output.tokens.numel())
    if rows < 0 or rows > total or int(output.completion.numel()) != 4 * total:
        raise GraphExecutionError("CUDA graph greedy output has invalid row geometry")
    completion = torch.cat(
        tuple(output.completion[index * total : index * total + rows] for index in range(4))
    )
    return GraphGreedyOutput(
        request_pool_indices=output.request_pool_indices[:rows],
        tokens=output.tokens[:rows],
        valid=output.valid[:rows],
        active=output.active[:rows],
        finish=output.finish[:rows],
        continuation=output.continuation[:rows],
        tagged_tokens=output.tagged_tokens[:rows],
        completion=completion,
    )


def _clone_output(output: ForwardOutput) -> ForwardOutput:
    return ForwardOutput(
        tuple(value.detach().clone(memory_format=torch.preserve_format) for value in output.values)
    )


def _equivalence_check(reference: ForwardOutput, candidate: ForwardOutput) -> torch.Tensor:
    if len(reference.values) != len(candidate.values):
        raise GraphExecutionError("CUDA graph output count differs from eager execution")
    checks: list[torch.Tensor] = []
    for expected, actual in zip(reference.values, candidate.values, strict=True):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise GraphExecutionError("CUDA graph output geometry differs from eager execution")
        if not (expected.is_floating_point() or expected.is_complex()):
            checks.append(torch.eq(expected, actual).all())
            continue
        tolerance = 0.01 if expected.element_size() <= 2 else 1e-5
        checks.append(
            torch.isclose(
                actual,
                expected,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=False,
            ).all()
        )
    if not checks:
        device = reference.values[0].device if reference.values else torch.device("cpu")
        return torch.ones((), dtype=torch.bool, device=device)
    return torch.stack(tuple(checks)).all()


def _release_call(method: Callable[[int], object], binding: int) -> Callable[[], None]:
    def release() -> None:
        method(binding)

    return release


def _release_state(state: _GraphState) -> None:
    for release in reversed(state.releases):
        try:
            release()
        except Exception:
            logger.warning("attention graph binding release failed", exc_info=True)
    reset = getattr(state.graph, "reset", None)
    if callable(reset):
        reset()


__all__ = ["CudaGraphRunner", "GraphExecutionError", "GraphGreedyOutput", "GraphRun"]
