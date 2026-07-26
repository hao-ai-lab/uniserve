"""System-owned exact-shape CUDA graph capture and replay."""

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
    TokenLogits,
    TokenOutput,
)
from uniserve_worker.spec import CacheSpec

__all__ = ["GraphExecutionError", "GraphStore"]

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
        cache: CacheSpec,
        block_size: int,
        spec_digest: str,
        memory_budget_bytes: int,
    ) -> None:
        if not spec_digest:
            raise ValueError("graph store requires a resolved spec digest")
        if int(block_size) < 1:
            raise ValueError("graph store block size must be positive")
        if int(memory_budget_bytes) < 0:
            raise ValueError("graph store memory budget must not be negative")
        self.enabled = bool(enabled)
        self.prefill_enabled = bool(prefill_enabled)
        self.cache = cache
        self.block_size = int(block_size)
        self.spec_digest = str(spec_digest)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.captures = 0
        self.evictions = 0
        self._device: torch.device | None = None
        self._states: OrderedDict[tuple[Hashable, tuple[object, ...]], _GraphState] = OrderedDict()
        self._warmed: set[tuple[Hashable, tuple[object, ...]]] = set()
        self._disabled: set[tuple[Hashable, tuple[object, ...]]] = set()
        self._bindings = itertools.count(1)
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
    ) -> tuple[ForwardOutput, str]:
        """Capture, replay, or run the identical mixed batch eagerly."""

        if not eligible or not self.enabled or not self._cuda_batch(batch):
            return forward(batch), "eager"
        if isinstance(batch.context.attention, PagedVarlenPlan) and not self.prefill_enabled:
            return forward(batch), "eager"
        if isinstance(batch.context.attention, PackedAttentionPlan) and _quantized_kv(batch):
            return forward(batch), "graph_fallback"
        signature = _batch_signature(batch)
        state_key = (key, signature)
        with self._lock:
            if state_key in self._disabled:
                return forward(batch), "graph_fallback"
            state = self._states.get(state_key)
            if state is None:
                if state_key not in self._warmed:
                    output = forward(batch)
                    self._warmed.add(state_key)
                    return output, "graph_fallback"
                if not self._room_for_capture():
                    # The retained set already fills its budget, so this shape
                    # serves eagerly from here on rather than re-measuring the
                    # allocator every time it reappears.
                    self._disabled.add(state_key)
                    return forward(batch), "graph_fallback"
                try:
                    state = self._capture(batch, forward)
                except _GraphMiss:
                    self._warmed.discard(state_key)
                    self._disabled.add(state_key)
                    return forward(batch), "graph_fallback"
                except Exception as error:
                    logger.warning("CUDA graph capture failed for an exact shape", exc_info=error)
                    self._warmed.discard(state_key)
                    self._disabled.add(state_key)
                    return forward(batch), "graph_fallback"
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
                _copy_batch_tensors(state.batch, batch)
                self._prepare_attention(state.batch, batch, capture=False)
                state.graph.replay()
                return _fresh_output(state.output, batch), "graph_capture"
            try:
                _copy_batch_tensors(state.batch, batch)
                self._prepare_attention(state.batch, batch, capture=False)
                state.graph.replay()
                self._states.move_to_end(state_key)
                return _fresh_output(state.output, batch), "graph_replay"
            except Exception as error:
                logger.warning("CUDA graph replay failed for an exact shape", exc_info=error)
                self._states.pop(state_key, None)
                self._disabled.add(state_key)
                _release_state(state)
                return forward(batch), "graph_fallback"

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
            with torch.cuda.graph(graph):
                output = forward(static)
            if not isinstance(output, ForwardOutput):
                raise TypeError("captured model forward did not return ForwardOutput")
            return _GraphState(graph=graph, batch=static, output=output, releases=releases)
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
        releases: list[Callable[[], None]] = []
        key_cache, _value_cache = static_batch.context.kv.layer_kv(0)
        q_dtype = _batch_compute_dtype(static_batch)
        kv_dtype = key_cache.dtype
        if isinstance(static, PagedDecodePlan):
            for backend in providers:
                prepare = getattr(backend, "prepare_paged_decode_cuda_graph", None)
                if not callable(prepare):
                    continue
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

        for backend in providers:
            bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
            prepare = getattr(backend, "prepare_paged_prefill_cuda_graph", None)
            if not callable(bind) or not callable(prepare):
                continue
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
    providers = []
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
            providers.append(provider)
    if not providers:
        raise _GraphMiss("no provisioned attention backend is graph-safe for this plan")
    return AttentionSelection(
        identity=f"{selection.identity}:graph",
        providers=tuple(providers),
    )


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


# Sequence-length bounds an attention plan carries as host scalars. A capture
# bakes them into its kernel launch, so an executable is only reusable for
# lengths at or below the bound it was captured with. Capturing at a bucket
# ceiling makes one executable serve every length inside that bucket, which is
# what keeps a growing conversation from capturing a new graph per step.
_PLAN_LENGTH_BOUNDS = frozenset({"max_seqlen_q", "max_seqlen_k"})


def _bucketed_length(value: object) -> object:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 1:
        return value
    return 1 << (int(value) - 1).bit_length()


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
    rows: list[ForwardRowOutput] = []
    for captured, current in zip(output.rows, batch.rows, strict=True):
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
