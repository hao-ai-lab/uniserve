"""System-owned bucketed CUDA graph capture and replay."""

from __future__ import annotations

import itertools
import logging
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from threading import RLock
from typing import Any, cast

import torch

from uniserve_worker.forward import (
    AttentionSelection,
    DecodeOutput,
    EmptyKvView,
    EncodeOutput,
    FlowOutput,
    ForwardBatch,
    ForwardContext,
    ForwardOutput,
    ForwardRowOutput,
    GraphBinding,
    NoAttention,
    PackedAttentionPlan,
    PagedDecodePlan,
    PagedVarlenPlan,
    TokenHidden,
    TokenIds,
    TokenLogits,
    TokenOutput,
    TokenRow,
    TokenSelection,
    packed_tensor_views,
    packed_token_ids,
    packed_token_positions,
)
from uniserve_worker.foundation.sizing import bucketed_length
from uniserve_worker.models.runtime import CacheGeometry

__all__ = ["GraphExecutionError", "GraphRun", "GraphStore"]

logger = logging.getLogger(__name__)


class GraphExecutionError(RuntimeError):
    """A physical CUDA graph could not be captured or replayed safely."""


class _GraphMiss(RuntimeError):
    pass


@dataclass(slots=True)
class _GraphState:
    graph: Any
    batch: ForwardBatch
    output: ForwardOutput
    releases: tuple[Callable[[], None], ...]
    padding: _DecodePadding | None = None
    prefill_padding: _PrefillPadding | None = None


@dataclass(frozen=True, slots=True)
class _DecodePadding:
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_row: torch.Tensor
    cache_seqlens: torch.Tensor
    kv_seqlens: torch.Tensor
    query_lens: torch.Tensor
    page_ids: torch.Tensor
    page_offsets: torch.Tensor


@dataclass(frozen=True, slots=True)
class _PrefillPadding:
    block_row: torch.Tensor


@dataclass(frozen=True, slots=True)
class GraphRun:
    """One physical execution and its graph-padding geometry."""

    output: ForwardOutput
    path: str
    row_count: int
    padded_row_count: int


@dataclass(frozen=True, slots=True)
class _DecodeGeometry:
    bucket: int
    width: int
    padding: int
    reserved: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PrefillGeometry:
    token_bucket: int
    row_bucket: int
    width: int
    padding: int
    reserved: tuple[int, ...]
    max_query_len: int
    max_key_len: int


class GraphStore:
    """Own captured executables for one resolved model and physical KV store.

    Captures are keyed by the immutable runner key plus a request-identity-free
    structural signature. Every replay copies live tensors into capture-owned
    buffers, refreshes graph-scoped attention plans, and returns fresh output
    tensors so callers can retain results across later replays.

    Retained executables hold device memory for their whole lifetime, so the
    store keeps its own residency inside ``memory_budget_bytes`` and releases
    least-recently-replayed captures to stay there. Workloads whose batch
    geometry keeps changing therefore run the surplus shapes eagerly instead of
    accumulating executables until the device is exhausted.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        prefill_enabled: bool,
        cache: CacheGeometry,
        block_size: int,
        weight_digest: str,
        memory_budget_bytes: int,
        decode_batch_sizes: tuple[int, ...] = (),
        decode_context_blocks: int = 0,
        prefill_token_sizes: tuple[int, ...] = (),
        prefill_row_bucket: int = 8,
    ) -> None:
        if not weight_digest:
            raise ValueError("graph store requires a weight identity")
        if int(block_size) < 1:
            raise ValueError("graph store block size must be positive")
        if int(memory_budget_bytes) < 0:
            raise ValueError("graph store memory budget must not be negative")
        self.enabled = bool(enabled)
        self.prefill_enabled = bool(prefill_enabled)
        self.cache = cache
        self.block_size = int(block_size)
        self.weight_digest = str(weight_digest)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.decode_batch_sizes = tuple(
            sorted({int(value) for value in decode_batch_sizes if int(value) > 0})
        )
        self.decode_context_blocks = max(0, int(decode_context_blocks))
        self.prefill_token_sizes = tuple(
            sorted({int(value) for value in prefill_token_sizes if int(value) > 0})
        )
        self.prefill_row_bucket = max(1, int(prefill_row_bucket))
        self.captures = 0
        self.evictions = 0
        self._device: torch.device | None = None
        self._states: OrderedDict[tuple[Hashable, tuple[object, ...]], _GraphState] = OrderedDict()
        self._warmed: set[tuple[Hashable, tuple[object, ...]]] = set()
        self._disabled: set[tuple[Hashable, tuple[object, ...]]] = set()
        self._bindings = itertools.count(1)
        self._pool_handle: Any = None
        self._lock = RLock()

    @property
    def resident_bytes(self) -> int:
        """Device memory the allocator currently holds in graph private pools."""

        return _private_pool_bytes(self._device)

    def execute(
        self,
        key: Hashable,
        batch: ForwardBatch,
        forward: Callable[[ForwardBatch], ForwardOutput],
        *,
        eligible: bool,
    ) -> GraphRun:
        """Capture, replay, or run one batch in its stable execution bucket."""

        if not eligible or not self.enabled or not self._cuda_batch(batch):
            return _run(forward(batch), "eager", batch, len(batch.rows))
        if isinstance(batch.context.attention, PagedVarlenPlan) and not self.prefill_enabled:
            return _run(forward(batch), "eager", batch, len(batch.rows))
        if isinstance(batch.context.attention, PagedVarlenPlan) and any(
            isinstance(row, TokenRow) and row.selection is not TokenSelection.LAST_LOGITS
            for row in batch.rows
        ):
            return _run(forward(batch), "eager", batch, len(batch.rows))
        if isinstance(batch.context.attention, PackedAttentionPlan) and _quantized_kv(batch):
            return _run(forward(batch), "graph_fallback", batch, len(batch.rows))
        decode_geometry = _decode_geometry(
            batch,
            self.decode_batch_sizes,
            self.block_size,
            self.decode_context_blocks,
        )
        prefill_geometry = (
            None
            if decode_geometry is not None
            else _prefill_geometry(
                batch,
                self.prefill_token_sizes,
                self.block_size,
                self.prefill_row_bucket,
                self.decode_context_blocks,
            )
        )
        execution_batch: ForwardBatch | None = None

        def materialize() -> ForwardBatch:
            nonlocal execution_batch
            if execution_batch is None:
                if decode_geometry is not None:
                    execution_batch = _pad_decode_batch(batch, decode_geometry)
                elif prefill_geometry is not None:
                    execution_batch = _pad_prefill_batch(batch, prefill_geometry)
                else:
                    execution_batch = batch
            return execution_batch

        if decode_geometry is not None:
            padded_row_count = decode_geometry.bucket
            signature = _decode_signature(batch, decode_geometry)
        elif prefill_geometry is not None:
            padded_row_count = prefill_geometry.row_bucket
            signature = _prefill_signature(batch, prefill_geometry)
        else:
            padded_row_count = len(batch.rows)
            signature = _batch_signature(batch)
        state_key = (key, signature)
        with self._lock:
            if state_key in self._disabled:
                return _run(
                    forward(materialize()),
                    "graph_fallback",
                    batch,
                    padded_row_count,
                )
            state = self._states.get(state_key)
            if state is None:
                if state_key not in self._warmed:
                    output = forward(materialize())
                    self._warmed.add(state_key)
                    return _run(output, "graph_fallback", batch, padded_row_count)
                if not self._room_for_capture():
                    # The retained set already fills its budget, so this shape
                    # serves eagerly from here on rather than re-measuring the
                    # allocator every time it reappears.
                    self._disabled.add(state_key)
                    return _run(
                        forward(materialize()),
                        "graph_fallback",
                        batch,
                        padded_row_count,
                    )
                try:
                    state = self._capture(materialize(), forward)
                except _GraphMiss:
                    self._warmed.discard(state_key)
                    self._disabled.add(state_key)
                    return _run(
                        forward(materialize()),
                        "graph_fallback",
                        batch,
                        padded_row_count,
                    )
                except Exception as error:
                    logger.warning(
                        "CUDA graph capture failed for an execution bucket", exc_info=error
                    )
                    self._warmed.discard(state_key)
                    self._disabled.add(state_key)
                    return _run(
                        forward(materialize()),
                        "graph_fallback",
                        batch,
                        padded_row_count,
                    )
                self._warmed.discard(state_key)
                self._states[state_key] = state
                self.captures += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "graph residency %d/%d MiB across %d executables after capture %d",
                        self.resident_bytes >> 20,
                        self.memory_budget_bytes >> 20,
                        len(self._states),
                        self.captures,
                    )
                # Do not return the capture-pass output. Tensors produced while
                # the stream is capturing can reflect capture-time pool
                # bootstrapping rather than the real result, so the first
                # multimodal decode step would otherwise emit a garbage token
                # even though every later replay of the same graph is correct.
                # Replay the freshly captured graph against the live batch to
                # produce a clean result, identical to every subsequent replay.
                self._prepare_replay(
                    state,
                    batch,
                    decode_geometry,
                    prefill_geometry,
                )
                state.graph.replay()
                return _run(
                    _fresh_output(state.output, batch),
                    "graph_capture",
                    batch,
                    padded_row_count,
                )
            try:
                self._prepare_replay(
                    state,
                    batch,
                    decode_geometry,
                    prefill_geometry,
                )
                state.graph.replay()
                self._states.move_to_end(state_key)
                return _run(
                    _fresh_output(state.output, batch),
                    "graph_replay",
                    batch,
                    padded_row_count,
                )
            except Exception as error:
                logger.warning("CUDA graph replay failed for an execution bucket", exc_info=error)
                self._states.pop(state_key, None)
                self._disabled.add(state_key)
                _release_state(state)
                return _run(
                    forward(materialize()),
                    "graph_fallback",
                    batch,
                    padded_row_count,
                )

    def _prepare_replay(
        self,
        state: _GraphState,
        batch: ForwardBatch,
        decode_geometry: _DecodeGeometry | None,
        prefill_geometry: _PrefillGeometry | None,
    ) -> None:
        if decode_geometry is not None:
            _copy_decode_tensors(state, batch, decode_geometry)
            live = _decode_live_batch(state.batch, batch, decode_geometry)
        elif prefill_geometry is not None:
            _copy_prefill_tensors(state, batch, prefill_geometry)
            live = _prefill_live_batch(state.batch, batch, prefill_geometry)
        else:
            _copy_batch_tensors(state.batch, batch)
            live = batch
        self._prepare_attention(state.batch, live, capture=False)

    def close(self) -> None:
        with self._lock:
            states = tuple(self._states.values())
            self._states.clear()
            self._warmed.clear()
            self._disabled.clear()
        for state in states:
            _release_state(state)

    def _room_for_capture(self) -> bool:
        """Free least-recently-replayed executables until one more pool fits."""

        resident = self.resident_bytes
        while resident > self.memory_budget_bytes and self._states:
            _key, state = self._states.popitem(last=False)
            _release_state(state)
            self.evictions += 1
            if self.evictions == 1:
                logger.info(
                    "graph residency reached its %d MiB budget; "
                    "releasing least-recently-replayed executables",
                    self.memory_budget_bytes >> 20,
                )
            released = self.resident_bytes
            if released >= resident:
                # The allocator kept the released pool, so freeing more
                # executables would surrender replay speed without recovering
                # device memory. Serve this shape eagerly instead.
                return False
            resident = released
        return resident <= self.memory_budget_bytes

    def _capture(
        self,
        batch: ForwardBatch,
        forward: Callable[[ForwardBatch], ForwardOutput],
    ) -> _GraphState:
        self._device = _batch_device(batch)
        static = _graph_batch(batch, next(self._bindings))
        releases = self._prepare_attention(static, batch, capture=True)
        graph = torch.cuda.CUDAGraph()
        try:
            if self._pool_handle is None:
                self._pool_handle = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self._pool_handle):
                output = forward(static)
            if not isinstance(output, ForwardOutput):
                raise TypeError("captured model forward did not return ForwardOutput")
            return _GraphState(
                graph=graph,
                batch=static,
                output=output,
                releases=releases,
                padding=_decode_padding(static, self.block_size),
                prefill_padding=_prefill_padding(static),
            )
        except Exception:
            for release in reversed(releases):
                release()
            reset = getattr(graph, "reset", None)
            if callable(reset):
                reset()
            raise

    def _prepare_attention(
        self,
        static_batch: ForwardBatch,
        live_batch: ForwardBatch,
        *,
        capture: bool,
    ) -> tuple[Callable[[], None], ...]:
        static = static_batch.context.attention
        live = live_batch.context.attention
        if type(static) is not type(live):
            raise _GraphMiss("attention plan variant changed")
        if isinstance(static, (NoAttention, PackedAttentionPlan)):
            return ()
        prepared = _live_attention(static, live)
        providers = static.backends.providers
        if len(providers) != 1:
            raise _GraphMiss("graph execution requires one resolved attention provider")
        backend = providers[0]
        releases: list[Callable[[], None]] = []
        key_cache, _value_cache = static_batch.context.kv.layer_kv(0)
        q_dtype = _batch_compute_dtype(static_batch)
        kv_dtype = key_cache.dtype
        if isinstance(static, PagedDecodePlan):
            prepare = getattr(backend, "prepare_paged_decode_cuda_graph", None)
            if callable(prepare):
                prepare(
                    static.binding,
                    prepared,
                    batch_size=int(static.block_table.shape[0]),
                    max_indices=max(1, int(static.block_table.numel())),
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
                        releases.append(
                            _binding_release(
                                cast(Callable[[GraphBinding], object], release),
                                static.binding,
                            )
                        )
            return tuple(releases)

        bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
        prepare = getattr(backend, "prepare_paged_prefill_cuda_graph", None)
        if callable(bind) and callable(prepare):
            if capture:
                bind(static.binding, static, device=static.block_table.device)
                release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
                if callable(release):
                    releases.append(
                        _binding_release(
                            cast(Callable[[GraphBinding], object], release),
                            static.binding,
                        )
                    )
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

    @staticmethod
    def _cuda_batch(batch: ForwardBatch) -> bool:
        tensors = tuple(_tensor_leaves(batch))
        return bool(
            tensors
            and torch.cuda.is_available()
            and all(value.device.type == "cuda" for value in tensors)
            and len({value.device for value in tensors}) == 1
        )


def _private_pool_bytes(device: torch.device | None) -> int:
    """Allocator segments held in CUDA graph private pools on one device.

    Captured executables own their pool for as long as they are retained, and
    that memory serves no other allocation, so it is what the store's residency
    budget governs.
    """

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


def _batch_device(batch: ForwardBatch) -> torch.device:
    for tensor in _tensor_leaves(batch):
        return tensor.device
    raise _GraphMiss("forward batch carries no device tensors")


def _graph_batch(batch: ForwardBatch, binding_identity: int) -> ForwardBatch:
    cloned = _clone_value(batch)
    if not isinstance(cloned, ForwardBatch):
        raise TypeError("graph input cloning did not preserve ForwardBatch")
    cloned = _pack_token_batch(cloned)
    attention = cloned.context.attention
    selection = _graph_selection(attention)
    graph_attention: NoAttention | PagedDecodePlan | PagedVarlenPlan | PackedAttentionPlan
    if isinstance(attention, NoAttention):
        graph_attention = replace(attention, backends=selection)
    else:
        graph_attention = cast(
            Any,
            _plan_with_bucketed_bounds(
                replace(
                    attention,
                    backends=selection,
                    binding=GraphBinding(int(binding_identity)),
                )
            ),
        )
    return replace(cloned, context=replace(cloned.context, attention=graph_attention))


def _graph_selection(plan: object) -> AttentionSelection:
    selection = getattr(plan, "backends", None)
    if not isinstance(selection, AttentionSelection):
        raise _GraphMiss("attention plan has no resolved backend selection")
    for provider in selection.providers:
        capabilities = provider.capabilities()
        if isinstance(plan, PackedAttentionPlan):
            safe = bool(getattr(capabilities, "visible_end_cuda_graph", False))
        elif isinstance(plan, PagedVarlenPlan):
            safe = bool(getattr(capabilities, "paged_varlen_cuda_graph", False)) or (
                callable(getattr(provider, "bind_paged_prefill_graph_wrapper", None))
                and callable(getattr(provider, "prepare_paged_prefill_cuda_graph", None))
            )
        elif isinstance(plan, PagedDecodePlan):
            safe = bool(getattr(capabilities, "paged_kv", False))
        else:
            safe = True
        if safe:
            return AttentionSelection(
                identity=f"{selection.identity}:graph:{provider.name}",
                providers=(provider,),
            )
    raise _GraphMiss("no provisioned attention backend is graph-safe for this plan")


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.preserve_format)
    if isinstance(value, AttentionSelection):
        return value
    if isinstance(value, GraphBinding):
        return value
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{field.name: _clone_value(getattr(value, field.name)) for field in fields(value)},
        )
    return value


def _tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (AttentionSelection, GraphBinding)):
        return
    if isinstance(value, tuple):
        for item in value:
            yield from _tensor_leaves(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _tensor_leaves(getattr(value, field.name))


def _copy_batch_tensors(target: ForwardBatch, source: ForwardBatch) -> None:
    if _copy_packed_token_rows(target, source):
        _copy_value_tensors(target.context, source.context)
        return
    _copy_value_tensors(target, source)


def _copy_value_tensors(target: object, source: object) -> None:
    target_tensors = tuple(_tensor_leaves(target))
    source_tensors = tuple(_tensor_leaves(source))
    if len(target_tensors) != len(source_tensors):
        raise _GraphMiss("forward tensor structure changed")
    for destination, value in zip(target_tensors, source_tensors, strict=True):
        if (
            destination.shape != value.shape
            or destination.dtype != value.dtype
            or destination.device != value.device
            or destination.stride() != value.stride()
        ):
            raise _GraphMiss("forward tensor geometry changed")
        destination.copy_(value, non_blocking=True)


def _copy_packed_token_rows(target: ForwardBatch, source: ForwardBatch) -> bool:
    if (
        any(not isinstance(row, TokenRow) for row in target.rows)
        or any(not isinstance(row, TokenRow) for row in source.rows)
        or len(target.rows) != len(source.rows)
    ):
        return False
    target_rows = cast(tuple[TokenRow, ...], target.rows)
    source_rows = cast(tuple[TokenRow, ...], source.rows)
    target_ids = packed_token_ids(target_rows)
    source_ids = packed_token_ids(source_rows)
    target_positions = packed_token_positions(target_rows)
    source_positions = packed_token_positions(source_rows)
    if any(value is None for value in (target_ids, source_ids, target_positions, source_positions)):
        return False
    cast(torch.Tensor, target_ids).copy_(cast(torch.Tensor, source_ids), non_blocking=True)
    cast(torch.Tensor, target_positions).copy_(
        cast(torch.Tensor, source_positions), non_blocking=True
    )
    return True


def _pack_token_batch(batch: ForwardBatch) -> ForwardBatch:
    if any(not isinstance(row, TokenRow) for row in batch.rows):
        return batch
    rows = cast(tuple[TokenRow, ...], batch.rows)
    if any(not isinstance(row.inputs, TokenIds) for row in rows):
        return batch
    inputs = torch.cat(tuple(cast(TokenIds, row.inputs).values.reshape(-1) for row in rows))
    positions = torch.cat(tuple(row.positions.reshape(-1) for row in rows))
    packed_rows: list[TokenRow] = []
    input_offset = 0
    position_offset = 0
    for row in rows:
        input_count = int(cast(TokenIds, row.inputs).values.numel())
        position_count = int(row.positions.numel())
        packed_rows.append(
            replace(
                row,
                inputs=TokenIds(inputs[input_offset : input_offset + input_count]),
                positions=positions[position_offset : position_offset + position_count],
            )
        )
        input_offset += input_count
        position_offset += position_count
    return replace(batch, rows=tuple(packed_rows))


# Sequence-length bounds an attention plan carries as host scalars. A capture
# bakes them into its kernel launch, so an executable is only reusable for
# lengths at or below the bound it was captured with. Capturing at a bucket
# ceiling makes one executable serve every length inside that bucket, which is
# what keeps a growing conversation from capturing a new graph per step.
_PLAN_LENGTH_BOUNDS = frozenset({"max_seqlen_q", "max_seqlen_k"})


def _bucketed_length(value: object) -> object:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 1:
        return value
    return bucketed_length(value)


def _plan_with_bucketed_bounds(plan: object) -> object:
    if not is_dataclass(plan) or isinstance(plan, type):
        return plan
    updates = {
        field.name: _bucketed_length(getattr(plan, field.name))
        for field in fields(plan)
        if field.name in _PLAN_LENGTH_BOUNDS
    }
    if not updates:
        return plan
    return replace(cast(Any, plan), **updates)


def _live_attention(static: object, live: object) -> object:
    if not is_dataclass(static) or not is_dataclass(live):
        raise _GraphMiss("attention plan is not immutable data")
    updates: dict[str, object] = {}
    for field in fields(static):
        static_value = getattr(static, field.name)
        if (
            isinstance(static_value, torch.Tensor)
            or field.name in {"binding", "backends"}
            or field.name in _PLAN_LENGTH_BOUNDS
        ):
            updates[field.name] = static_value
        else:
            updates[field.name] = getattr(live, field.name)
    return replace(cast(Any, live), **updates)


def _batch_compute_dtype(batch: ForwardBatch) -> torch.dtype:
    for tensor in _tensor_leaves(batch.rows):
        if tensor.is_floating_point():
            return tensor.dtype
    key, _value = batch.context.kv.layer_kv(0)
    return key.dtype


def _batch_signature(batch: ForwardBatch) -> tuple[object, ...]:
    return (
        str(batch.route),
        tuple(_row_signature(row) for row in batch.rows),
        _attention_signature(batch.context.attention),
        _view_signature(batch.context),
    )


def _row_signature(row: object) -> tuple[object, ...]:
    values: list[object] = [type(row).__qualname__]
    for field in fields(cast(Any, row)):
        if field.name in {"row_id", "output_slot"}:
            continue
        values.append((field.name, _semantic_signature(getattr(row, field.name))))
    return tuple(values)


def _attention_signature(plan: object) -> tuple[object, ...]:
    values: list[object] = [type(plan).__qualname__]
    for field in fields(cast(Any, plan)):
        value = getattr(plan, field.name)
        if field.name == "binding":
            continue
        if field.name == "backends":
            values.append((field.name, value.identity))
        elif field.name.endswith("_cpu"):
            values.append((field.name, len(value)))
        elif field.name in _PLAN_LENGTH_BOUNDS:
            values.append((field.name, _bucketed_length(value)))
        else:
            values.append((field.name, _semantic_signature(value)))
    return tuple(values)


def _view_signature(context: ForwardContext) -> tuple[object, ...]:
    kv = context.kv
    if isinstance(kv, EmptyKvView):
        storage: object = "empty"
    else:
        pool = getattr(kv, "_pool", None)
        if pool is None:
            cache = getattr(kv, "_cache", None)
            pool = getattr(cache, "pool", None)
        storage = (
            type(kv).__qualname__,
            id(pool),
            bool(getattr(pool, "is_quantized", False)),
        )
    return (
        storage,
        type(context.latent).__qualname__,
        type(context.mesh).__qualname__,
        type(context.output).__qualname__,
    )


def _quantized_kv(batch: ForwardBatch) -> bool:
    kv = batch.context.kv
    pool = getattr(kv, "_pool", None)
    if pool is None:
        cache = getattr(kv, "_cache", None)
        pool = getattr(cache, "pool", None)
    return bool(getattr(pool, "is_quantized", False))


def _run(
    output: ForwardOutput,
    path: str,
    live_batch: ForwardBatch,
    padded_row_count: int,
) -> GraphRun:
    if int(padded_row_count) != len(live_batch.rows):
        output = ForwardOutput(tuple(output.rows[: len(live_batch.rows)]))
    return GraphRun(
        output=output,
        path=path,
        row_count=len(live_batch.rows),
        padded_row_count=int(padded_row_count),
    )


def _decode_geometry(
    batch: ForwardBatch,
    batch_sizes: tuple[int, ...],
    block_size: int,
    context_blocks: int,
) -> _DecodeGeometry | None:
    """Resolve the stable bucket and page-table capacity for a decode batch."""

    attention = batch.context.attention
    if not isinstance(attention, PagedDecodePlan) or not batch_sizes:
        return None
    live_rows = len(batch.rows)
    bucket = next((value for value in batch_sizes if value >= live_rows), None)
    if bucket is None:
        return None
    if any(not isinstance(row, TokenRow) for row in batch.rows):
        return None
    rows = cast(tuple[TokenRow, ...], batch.rows)
    if any(not isinstance(row.inputs, TokenIds) for row in rows):
        return None
    token_inputs = tuple(cast(TokenIds, row.inputs) for row in rows)
    if (
        any(int(value.values.numel()) != 1 for value in token_inputs)
        or any(int(row.positions.numel()) != 1 for row in rows)
        or len({row.selection for row in rows}) != 1
    ):
        return None
    pool = getattr(batch.context.kv, "_pool", None)
    if pool is None:
        cache = getattr(batch.context.kv, "_cache", None)
        pool = getattr(cache, "pool", None)
    reserved = tuple(int(value) for value in getattr(pool, "reserved_block_ids", ()))
    padding = int(bucket) - live_rows
    live_width = int(attention.block_table.shape[1])
    width = max(live_width, int(context_blocks), len(reserved))
    if (
        int(getattr(pool, "block_size", 0)) != int(block_size)
        or (padding > 0 and not reserved)
        or padding > len(reserved) * int(block_size)
        or (context_blocks > 0 and live_width > int(context_blocks))
    ):
        return None
    return _DecodeGeometry(
        bucket=int(bucket),
        width=width,
        padding=padding,
        reserved=reserved,
    )


def _prefill_geometry(
    batch: ForwardBatch,
    token_sizes: tuple[int, ...],
    block_size: int,
    row_bucket: int,
    context_blocks: int,
) -> _PrefillGeometry | None:
    """Resolve one padded token bucket for a paged-varlen prefill batch."""

    attention = batch.context.attention
    if not isinstance(attention, PagedVarlenPlan) or not token_sizes:
        return None
    if any(not isinstance(row, TokenRow) for row in batch.rows):
        return None
    rows = cast(tuple[TokenRow, ...], batch.rows)
    if any(not isinstance(row.inputs, TokenIds) for row in rows):
        return None
    if any(row.selection is not TokenSelection.LAST_LOGITS for row in rows) or len(rows) > int(
        row_bucket
    ):
        return None
    query_lens = tuple(int(value) for value in attention.query_lens_cpu)
    if (
        len(query_lens) != len(rows)
        or any(length < 1 for length in query_lens)
        or any(
            int(cast(TokenIds, row.inputs).values.numel()) != length
            or int(row.positions.numel()) != length
            for row, length in zip(rows, query_lens, strict=True)
        )
    ):
        return None
    live_tokens = sum(query_lens)
    token_bucket = next((value for value in token_sizes if value >= live_tokens), None)
    if token_bucket is None:
        return None
    padding = int(token_bucket) - live_tokens
    if len(rows) >= int(row_bucket):
        return None
    pool = getattr(batch.context.kv, "_pool", None)
    if pool is None:
        cache = getattr(batch.context.kv, "_cache", None)
        pool = getattr(cache, "pool", None)
    reserved = tuple(int(value) for value in getattr(pool, "reserved_block_ids", ()))
    previous_bucket = max((value for value in token_sizes if value < token_bucket), default=0)
    maximum_padding = int(token_bucket) - int(previous_bucket)
    required_blocks = (maximum_padding + int(block_size) - 1) // int(block_size)
    live_width = int(attention.block_table.shape[1])
    if (
        int(getattr(pool, "block_size", 0)) != int(block_size)
        or required_blocks > len(reserved)
        or (int(context_blocks) > 0 and live_width > int(context_blocks))
        or not callable(getattr(batch.context.kv, "with_synthetic_row", None))
    ):
        return None
    width = max(live_width, required_blocks, int(context_blocks))
    return _PrefillGeometry(
        token_bucket=int(token_bucket),
        row_bucket=int(row_bucket),
        width=width,
        padding=padding,
        reserved=reserved[:required_blocks],
        max_query_len=bucketed_length(int(token_bucket)),
        max_key_len=width * int(block_size),
    )


def _decode_signature(
    batch: ForwardBatch,
    geometry: _DecodeGeometry,
) -> tuple[object, ...]:
    attention = cast(PagedDecodePlan, batch.context.attention)
    rows = cast(tuple[TokenRow, ...], batch.rows)
    first = rows[0]
    token_ids = cast(TokenIds, first.inputs).values
    return (
        "paged_decode_bucket",
        str(batch.route),
        geometry.bucket,
        geometry.width,
        first.selection.value,
        str(token_ids.dtype),
        str(token_ids.device),
        str(first.positions.dtype),
        str(first.positions.device),
        str(attention.block_table.dtype),
        attention.backends.identity,
        bool(attention.causal),
        _view_signature(batch.context),
    )


def _prefill_signature(
    batch: ForwardBatch,
    geometry: _PrefillGeometry,
) -> tuple[object, ...]:
    attention = cast(PagedVarlenPlan, batch.context.attention)
    rows = cast(tuple[TokenRow, ...], batch.rows)
    first = rows[0]
    token_ids = cast(TokenIds, first.inputs).values
    return (
        "paged_prefill_bucket",
        str(batch.route),
        geometry.row_bucket,
        geometry.token_bucket,
        geometry.width,
        geometry.max_query_len,
        geometry.max_key_len,
        first.selection.value,
        str(token_ids.dtype),
        str(token_ids.device),
        str(first.positions.dtype),
        str(first.positions.device),
        str(attention.block_table.dtype),
        attention.backends.identity,
        bool(attention.causal),
        _view_signature(batch.context),
    )


def _pad_decode_batch(
    batch: ForwardBatch,
    geometry: _DecodeGeometry | None,
) -> ForwardBatch:
    """Materialize a decode bucket for eager warmup and physical capture."""

    if geometry is None:
        return batch
    attention = cast(PagedDecodePlan, batch.context.attention)
    rows = cast(tuple[TokenRow, ...], batch.rows)
    token_inputs = tuple(cast(TokenIds, row.inputs) for row in rows)
    live_rows = len(rows)
    bucket = geometry.bucket
    padding = geometry.padding
    width = geometry.width
    reserved = geometry.reserved
    block_size = int(batch.context.kv.block_size)
    live_width = int(attention.block_table.shape[1])

    first = rows[0]
    next_row_id = max(row.row_id for row in rows) + 1
    next_output_slot = max(row.output_slot for row in rows) + 1
    live_input_ids = packed_token_ids(rows)
    live_positions = packed_token_positions(rows)
    if live_input_ids is None:
        live_input_ids = torch.cat(tuple(value.values.reshape(-1) for value in token_inputs))
    if live_positions is None:
        live_positions = torch.cat(tuple(row.positions.reshape(-1) for row in rows))
    input_ids = torch.cat((live_input_ids, live_input_ids.new_zeros(padding)))
    positions = torch.cat(
        (
            live_positions,
            torch.arange(
                padding,
                dtype=live_positions.dtype,
                device=live_positions.device,
            ),
        )
    )
    live_packed_rows = tuple(
        replace(
            row,
            inputs=TokenIds(input_ids[index : index + 1]),
            positions=positions[index : index + 1],
        )
        for index, row in enumerate(rows)
    )
    dummy_rows = tuple(
        TokenRow(
            row_id=next_row_id + offset,
            inputs=TokenIds(input_ids[live_rows + offset : live_rows + offset + 1]),
            positions=positions[live_rows + offset : live_rows + offset + 1],
            output_slot=next_output_slot + offset,
            selection=first.selection,
        )
        for offset in range(padding)
    )

    block_table = attention.block_table.new_zeros((bucket, width))
    block_table[:live_rows, :live_width].copy_(attention.block_table)
    reserved_tensor = attention.block_table.new_tensor(reserved)
    if padding:
        block_table[live_rows:, : len(reserved)].copy_(
            reserved_tensor.unsqueeze(0).expand(padding, -1)
        )
    offsets = torch.arange(
        padding,
        dtype=attention.cache_seqlens.dtype,
        device=attention.cache_seqlens.device,
    )
    cache_seqlens = torch.cat((attention.cache_seqlens, offsets))
    kv_seqlens = torch.cat((attention.kv_seqlens, offsets + 1))
    query_lens = torch.cat((attention.query_lens, attention.query_lens.new_ones(padding)))
    decode_page_ids = torch.cat(
        (
            attention.decode_page_ids,
            reserved_tensor[(offsets // int(block_size)).to(dtype=torch.long)].to(
                dtype=attention.decode_page_ids.dtype
            ),
        )
    )
    decode_page_offsets = torch.cat(
        (
            attention.decode_page_offsets,
            (offsets % int(block_size)).to(dtype=attention.decode_page_offsets.dtype),
        )
    )
    padded_attention = replace(
        attention,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        kv_seqlens=kv_seqlens,
        query_lens=query_lens,
        cache_seqlens_cpu=(*attention.cache_seqlens_cpu, *range(padding)),
        kv_seqlens_cpu=(*attention.kv_seqlens_cpu, *(offset + 1 for offset in range(padding))),
        query_lens_cpu=(*attention.query_lens_cpu, *(1 for _ in range(padding))),
        decode_page_ids=decode_page_ids,
        decode_page_offsets=decode_page_offsets,
        max_context_len=max(int(attention.max_context_len), width * int(block_size)),
    )
    return replace(
        batch,
        rows=(*live_packed_rows, *dummy_rows),
        context=replace(batch.context, attention=padded_attention),
    )


def _pad_prefill_batch(
    batch: ForwardBatch,
    geometry: _PrefillGeometry,
) -> ForwardBatch:
    """Pad query tokens and rows into one stable paged-prefill bucket."""

    attention = cast(PagedVarlenPlan, batch.context.attention)
    rows = cast(tuple[TokenRow, ...], batch.rows)
    token_inputs = tuple(cast(TokenIds, row.inputs) for row in rows)
    live_rows = len(rows)
    dummy_rows = geometry.row_bucket - live_rows
    live_width = int(attention.block_table.shape[1])
    input_ids = packed_token_ids(rows)
    positions = packed_token_positions(rows)
    if input_ids is None:
        input_ids = torch.cat(tuple(value.values.reshape(-1) for value in token_inputs))
    if positions is None:
        positions = torch.cat(tuple(row.positions.reshape(-1) for row in rows))
    packed_ids = torch.cat((input_ids, input_ids.new_zeros(geometry.padding)))
    packed_positions = torch.cat(
        (
            positions,
            torch.arange(
                geometry.padding,
                dtype=positions.dtype,
                device=positions.device,
            ),
        )
    )
    live_packed_rows: list[TokenRow] = []
    offset = 0
    for row, token_input in zip(rows, token_inputs, strict=True):
        width = int(token_input.values.numel())
        live_packed_rows.append(
            replace(
                row,
                inputs=TokenIds(packed_ids[offset : offset + width]),
                positions=packed_positions[offset : offset + width],
            )
        )
        offset += width
    next_row_id = max(row.row_id for row in rows) + 1
    next_output_slot = max(row.output_slot for row in rows) + 1
    padded_rows: list[TokenRow] = []
    for index in range(dummy_rows):
        count = geometry.padding if index == 0 else 0
        padded_rows.append(
            TokenRow(
                row_id=next_row_id + index,
                inputs=TokenIds(packed_ids[offset : offset + count]),
                positions=packed_positions[offset : offset + count],
                output_slot=next_output_slot + index,
                selection=rows[0].selection,
            )
        )
        offset += count

    block_table = attention.block_table.new_zeros((geometry.row_bucket, geometry.width))
    block_table[:live_rows, :live_width].copy_(attention.block_table)
    if geometry.reserved:
        block_table[live_rows, : len(geometry.reserved)].copy_(
            attention.block_table.new_tensor(geometry.reserved)
        )
    cache_seqlens = torch.cat(
        (attention.cache_seqlens, attention.cache_seqlens.new_zeros(dummy_rows))
    )
    dummy_query_lens = (geometry.padding, *(0 for _ in range(dummy_rows - 1)))
    query_lens = torch.cat(
        (
            attention.query_lens,
            attention.query_lens.new_tensor(dummy_query_lens),
        )
    )
    kv_seqlens = torch.cat(
        (
            attention.kv_seqlens,
            attention.kv_seqlens.new_tensor(dummy_query_lens),
        )
    )
    cu_seqlens_q = torch.cat(
        (
            attention.cu_seqlens_q,
            attention.cu_seqlens_q.new_full((dummy_rows,), geometry.token_bucket),
        )
    )
    cu_seqlens_k = torch.cat(
        (
            attention.cu_seqlens_k,
            attention.cu_seqlens_k.new_full(
                (dummy_rows,),
                sum(int(value) for value in attention.kv_seqlens_cpu) + geometry.padding,
            ),
        )
    )
    dummy_output_indices = attention.output_indices.new_zeros(dummy_rows)
    if geometry.padding:
        dummy_output_indices[0] = geometry.token_bucket - 1
    output_indices = torch.cat((attention.output_indices, dummy_output_indices))
    padded_attention = replace(
        attention,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        query_lens=query_lens,
        kv_seqlens=kv_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        output_indices=output_indices,
        cache_seqlens_cpu=(*attention.cache_seqlens_cpu, *(0 for _ in range(dummy_rows))),
        query_lens_cpu=(*attention.query_lens_cpu, *dummy_query_lens),
        kv_seqlens_cpu=(*attention.kv_seqlens_cpu, *dummy_query_lens),
        max_seqlen_q=geometry.max_query_len,
        max_seqlen_k=geometry.max_key_len,
        max_context_len=geometry.max_key_len,
    )
    padded_kv = cast(Any, batch.context.kv)
    for index in range(dummy_rows):
        padded_kv = padded_kv.with_synthetic_row(
            geometry.reserved if index == 0 else (),
            base_len=0,
            query_len=geometry.padding if index == 0 else 0,
        )
    return replace(
        batch,
        rows=(*live_packed_rows, *padded_rows),
        context=replace(
            batch.context,
            kv=padded_kv,
            attention=padded_attention,
        ),
    )


def _copy_decode_tensors(
    state: _GraphState,
    source: ForwardBatch,
    geometry: _DecodeGeometry,
) -> None:
    target = state.batch
    target_rows = cast(tuple[TokenRow, ...], target.rows)
    source_rows = cast(tuple[TokenRow, ...], source.rows)
    target_ids = packed_token_ids(target_rows)
    source_ids = packed_token_ids(source_rows)
    target_positions = packed_token_positions(target_rows)
    source_positions = packed_token_positions(source_rows)
    if any(value is None for value in (target_ids, source_ids, target_positions, source_positions)):
        raise _GraphMiss("decode rows are not backed by packed tensors")
    live_rows = len(source_rows)
    cast(torch.Tensor, target_ids)[:live_rows].copy_(
        cast(torch.Tensor, source_ids),
        non_blocking=True,
    )
    cast(torch.Tensor, target_positions)[:live_rows].copy_(
        cast(torch.Tensor, source_positions),
        non_blocking=True,
    )
    padding = state.padding
    if geometry.padding:
        if padding is None:
            raise _GraphMiss("decode graph has no padding template")
        tail = slice(live_rows, geometry.bucket)
        cast(torch.Tensor, target_ids)[tail].copy_(
            padding.input_ids[: geometry.padding],
            non_blocking=True,
        )
        cast(torch.Tensor, target_positions)[tail].copy_(
            padding.positions[: geometry.padding],
            non_blocking=True,
        )
    target_attention = target.context.attention
    source_attention = source.context.attention
    if (
        not isinstance(target_attention, PagedDecodePlan)
        or not isinstance(source_attention, PagedDecodePlan)
        or tuple(target_attention.block_table.shape) != (geometry.bucket, geometry.width)
    ):
        raise _GraphMiss("decode attention bucket geometry changed")
    live_width = int(source_attention.block_table.shape[1])
    target_attention.block_table[:live_rows, :live_width].copy_(
        source_attention.block_table,
        non_blocking=True,
    )
    for destination, value in (
        (target_attention.cache_seqlens, source_attention.cache_seqlens),
        (target_attention.kv_seqlens, source_attention.kv_seqlens),
        (target_attention.query_lens, source_attention.query_lens),
        (target_attention.decode_page_ids, source_attention.decode_page_ids),
        (target_attention.decode_page_offsets, source_attention.decode_page_offsets),
    ):
        destination[:live_rows].copy_(value, non_blocking=True)
    if geometry.padding:
        assert padding is not None
        tail = slice(live_rows, geometry.bucket)
        target_attention.block_table[tail].copy_(
            padding.block_row.expand(geometry.padding, -1),
            non_blocking=True,
        )
        for destination, value in (
            (target_attention.cache_seqlens, padding.cache_seqlens),
            (target_attention.kv_seqlens, padding.kv_seqlens),
            (target_attention.query_lens, padding.query_lens),
            (target_attention.decode_page_ids, padding.page_ids),
            (target_attention.decode_page_offsets, padding.page_offsets),
        ):
            destination[tail].copy_(value[: geometry.padding], non_blocking=True)


def _copy_prefill_tensors(
    state: _GraphState,
    source: ForwardBatch,
    geometry: _PrefillGeometry,
) -> None:
    target = state.batch
    target_rows = cast(tuple[TokenRow, ...], target.rows)
    source_rows = cast(tuple[TokenRow, ...], source.rows)
    target_ids = packed_token_ids(target_rows)
    source_ids = packed_token_ids(source_rows)
    target_positions = packed_token_positions(target_rows)
    source_positions = packed_token_positions(source_rows)
    if any(value is None for value in (target_ids, source_ids, target_positions, source_positions)):
        raise _GraphMiss("prefill rows are not backed by packed tensors")
    live_tokens = int(cast(torch.Tensor, source_ids).numel())
    if live_tokens + geometry.padding != geometry.token_bucket:
        raise _GraphMiss("prefill token bucket changed")
    cast(torch.Tensor, target_ids)[:live_tokens].copy_(
        cast(torch.Tensor, source_ids),
        non_blocking=True,
    )
    cast(torch.Tensor, target_positions)[:live_tokens].copy_(
        cast(torch.Tensor, source_positions),
        non_blocking=True,
    )
    if live_tokens < geometry.token_bucket:
        cast(torch.Tensor, target_ids)[live_tokens:].zero_()
        cast(torch.Tensor, target_positions)[live_tokens:].zero_()

    target_attention = target.context.attention
    source_attention = source.context.attention
    live_rows = len(source_rows)
    dummy_rows = geometry.row_bucket - live_rows
    if (
        not isinstance(target_attention, PagedVarlenPlan)
        or not isinstance(source_attention, PagedVarlenPlan)
        or len(target_rows) != geometry.row_bucket
        or tuple(target_attention.block_table.shape) != (geometry.row_bucket, geometry.width)
    ):
        raise _GraphMiss("prefill attention bucket geometry changed")
    source_width = int(source_attention.block_table.shape[1])
    if source_width > geometry.width:
        raise _GraphMiss("prefill page-table width exceeds its bucket")
    target_attention.block_table.zero_()
    target_attention.block_table[:live_rows, :source_width].copy_(
        source_attention.block_table,
        non_blocking=True,
    )
    padding = state.prefill_padding
    if padding is None:
        raise _GraphMiss("prefill graph has no padding template")
    target_attention.block_table[live_rows].copy_(
        padding.block_row,
        non_blocking=True,
    )
    target_attention.cache_seqlens.zero_()
    target_attention.query_lens.zero_()
    target_attention.kv_seqlens.zero_()
    for destination, value in (
        (target_attention.cache_seqlens, source_attention.cache_seqlens),
        (target_attention.query_lens, source_attention.query_lens),
        (target_attention.kv_seqlens, source_attention.kv_seqlens),
    ):
        destination[:live_rows].copy_(value, non_blocking=True)
    if geometry.padding:
        target_attention.query_lens[live_rows] = geometry.padding
        target_attention.kv_seqlens[live_rows] = geometry.padding
    target_attention.cu_seqlens_q[: live_rows + 1].copy_(
        source_attention.cu_seqlens_q,
        non_blocking=True,
    )
    target_attention.cu_seqlens_q[live_rows + 1 :].fill_(geometry.token_bucket)
    target_attention.cu_seqlens_k[: live_rows + 1].copy_(
        source_attention.cu_seqlens_k,
        non_blocking=True,
    )
    padded_kv_tokens = sum(int(value) for value in source_attention.kv_seqlens_cpu)
    padded_kv_tokens += geometry.padding
    target_attention.cu_seqlens_k[live_rows + 1 :].fill_(padded_kv_tokens)
    target_attention.output_indices.zero_()
    target_attention.output_indices[:live_rows].copy_(
        source_attention.output_indices,
        non_blocking=True,
    )
    if geometry.padding:
        target_attention.output_indices[live_rows] = geometry.token_bucket - 1
    if dummy_rows < 0:
        raise _GraphMiss("prefill row bucket changed")


def _prefill_live_batch(
    target: ForwardBatch,
    source: ForwardBatch,
    geometry: _PrefillGeometry,
) -> ForwardBatch:
    target_attention = cast(PagedVarlenPlan, target.context.attention)
    source_attention = cast(PagedVarlenPlan, source.context.attention)
    dummy_rows = geometry.row_bucket - len(source.rows)
    dummy_query_lens = (geometry.padding, *(0 for _ in range(dummy_rows - 1)))
    attention = replace(
        source_attention,
        cache_seqlens_cpu=(
            *source_attention.cache_seqlens_cpu,
            *(0 for _ in range(dummy_rows)),
        ),
        query_lens_cpu=(*source_attention.query_lens_cpu, *dummy_query_lens),
        kv_seqlens_cpu=(*source_attention.kv_seqlens_cpu, *dummy_query_lens),
        max_seqlen_q=target_attention.max_seqlen_q,
        max_seqlen_k=target_attention.max_seqlen_k,
        max_context_len=target_attention.max_context_len,
    )
    return replace(
        source,
        context=replace(source.context, attention=attention),
    )


def _decode_padding(
    batch: ForwardBatch,
    block_size: int,
) -> _DecodePadding | None:
    attention = batch.context.attention
    if not isinstance(attention, PagedDecodePlan) or any(
        not isinstance(row, TokenRow) for row in batch.rows
    ):
        return None
    rows = cast(tuple[TokenRow, ...], batch.rows)
    input_ids = packed_token_ids(rows)
    positions = packed_token_positions(rows)
    if input_ids is None or positions is None:
        return None
    pool = getattr(batch.context.kv, "_pool", None)
    if pool is None:
        cache = getattr(batch.context.kv, "_cache", None)
        pool = getattr(cache, "pool", None)
    reserved = tuple(int(value) for value in getattr(pool, "reserved_block_ids", ()))
    capacity = len(rows)
    offsets = torch.arange(
        capacity,
        dtype=attention.cache_seqlens.dtype,
        device=attention.cache_seqlens.device,
    )
    reserved_tensor = attention.block_table.new_tensor(reserved)
    block_row = attention.block_table.new_zeros((int(attention.block_table.shape[1]),))
    block_row[: len(reserved)].copy_(reserved_tensor)
    return _DecodePadding(
        input_ids=input_ids.new_zeros(capacity),
        positions=torch.arange(
            capacity,
            dtype=positions.dtype,
            device=positions.device,
        ),
        block_row=block_row,
        cache_seqlens=offsets,
        kv_seqlens=offsets + 1,
        query_lens=attention.query_lens.new_ones(capacity),
        page_ids=reserved_tensor[(offsets // int(block_size)).to(dtype=torch.long)].to(
            dtype=attention.decode_page_ids.dtype
        ),
        page_offsets=(offsets % int(block_size)).to(dtype=attention.decode_page_offsets.dtype),
    )


def _prefill_padding(batch: ForwardBatch) -> _PrefillPadding | None:
    attention = batch.context.attention
    if not isinstance(attention, PagedVarlenPlan):
        return None
    pool = getattr(batch.context.kv, "_pool", None)
    if pool is None:
        cache = getattr(batch.context.kv, "_cache", None)
        pool = getattr(cache, "pool", None)
    reserved = tuple(int(value) for value in getattr(pool, "reserved_block_ids", ()))
    if not reserved:
        return None
    block_row = attention.block_table.new_zeros((int(attention.block_table.shape[1]),))
    block_row[: len(reserved)].copy_(attention.block_table.new_tensor(reserved))
    return _PrefillPadding(block_row=block_row)


def _decode_live_batch(
    target: ForwardBatch,
    source: ForwardBatch,
    geometry: _DecodeGeometry,
) -> ForwardBatch:
    target_attention = cast(PagedDecodePlan, target.context.attention)
    source_attention = cast(PagedDecodePlan, source.context.attention)
    padding = geometry.padding
    attention = replace(
        source_attention,
        cache_seqlens_cpu=(*source_attention.cache_seqlens_cpu, *range(padding)),
        kv_seqlens_cpu=(
            *source_attention.kv_seqlens_cpu,
            *(offset + 1 for offset in range(padding)),
        ),
        query_lens_cpu=(*source_attention.query_lens_cpu, *(1 for _ in range(padding))),
        max_context_len=target_attention.max_context_len,
    )
    return replace(
        source,
        context=replace(source.context, attention=attention),
    )


def _semantic_signature(value: Any) -> object:
    if isinstance(value, torch.Tensor):
        return (
            tuple(int(item) for item in value.shape),
            tuple(int(item) for item in value.stride()),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return tuple(_semantic_signature(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (field.name, _semantic_signature(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return type(value).__qualname__


def _fresh_output(output: ForwardOutput, batch: ForwardBatch) -> ForwardOutput:
    captured_rows = output.rows[: len(batch.rows)]
    packed = _fresh_packed_token_outputs(captured_rows, batch)
    if packed is not None:
        return packed
    rows: list[ForwardRowOutput] = []
    for captured, current in zip(captured_rows, batch.rows, strict=True):
        if isinstance(captured, TokenOutput):
            value = captured.value
            fresh_value: TokenLogits | TokenHidden
            if isinstance(value, TokenLogits):
                fresh_value = TokenLogits(value.value.clone())
            elif isinstance(value, TokenHidden):
                fresh_value = TokenHidden(value.value.clone())
            else:
                raise GraphExecutionError("captured token output has an invalid value variant")
            rows.append(
                TokenOutput(
                    row_id=current.row_id,
                    output_slot=current.output_slot,
                    value=fresh_value,
                )
            )
        elif isinstance(captured, FlowOutput):
            rows.append(
                FlowOutput(current.row_id, current.output_slot, captured.prediction.clone())
            )
        elif isinstance(captured, EncodeOutput):
            rows.append(
                EncodeOutput(current.row_id, current.output_slot, captured.features.clone())
            )
        elif isinstance(captured, DecodeOutput):
            rows.append(DecodeOutput(current.row_id, current.output_slot, captured.tensor.clone()))
        else:
            raise GraphExecutionError("captured output has an invalid row variant")
    return ForwardOutput(tuple(rows))


def _fresh_packed_token_outputs(
    captured_rows: tuple[ForwardRowOutput, ...],
    batch: ForwardBatch,
) -> ForwardOutput | None:
    if (
        any(not isinstance(row, TokenOutput) for row in captured_rows)
        or any(not isinstance(row, TokenRow) for row in batch.rows)
        or len(captured_rows) != len(batch.rows)
    ):
        return None
    captured = cast(tuple[TokenOutput, ...], captured_rows)
    value_types = {type(row.value) for row in captured}
    if len(value_types) != 1:
        return None
    tensors = tuple(row.value.value for row in captured)
    shared = packed_tensor_views(tensors)
    if shared is None:
        return None
    fresh = shared.clone()
    rows: list[ForwardRowOutput] = []
    offset = 0
    for captured_row, current, tensor in zip(
        captured,
        batch.rows,
        tensors,
        strict=True,
    ):
        count = int(tensor.numel())
        value = fresh[offset : offset + count].view(tensor.shape)
        offset += count
        fresh_value: TokenLogits | TokenHidden
        if isinstance(captured_row.value, TokenLogits):
            fresh_value = TokenLogits(value)
        else:
            fresh_value = TokenHidden(value)
        rows.append(
            TokenOutput(
                row_id=current.row_id,
                output_slot=current.output_slot,
                value=fresh_value,
            )
        )
    return ForwardOutput(tuple(rows))


def _binding_release(
    method: Callable[[GraphBinding], object],
    binding: GraphBinding,
) -> Callable[[], None]:
    def release() -> None:
        method(binding)

    return release


def _release_state(state: _GraphState) -> None:
    reset = getattr(state.graph, "reset", None)
    try:
        if callable(reset):
            reset()
    finally:
        for release in reversed(state.releases):
            release()
