"""Consolidated CUDA graph runtime for model-backed execution.

One module owns every graph concern (``specs/unified_forward_execution.md``,
forward-subpackage disposition): the capacity-only bucket runtime
(:class:`CudaGraphRuntime`), stable graph input buffers and shape keys, graph
statistics, the model-neutral text decode/prefill capture-and-replay runners,
and the single graph-program replay dispatcher (:class:`CudaGraphForwardRunner`)
the engine drives. Family-specific graph runners (packed mixed, denoise step,
interleaved text) live with their families in ``models/`` and build on the
shared ``_GraphRunnerBase`` primitives exported here.
"""

from __future__ import annotations

import gc
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, replace
from enum import Enum, StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from uniserve_worker.backends.paged_kv_math import decode_write_locations
from uniserve_worker.contracts.batches import UniForwardBatch
from uniserve_worker.contracts.forward_batch import (
    ForwardBatch,
    ForwardGraphExecutionInfo,
    ForwardPlan,
    ForwardResult,
)
from uniserve_worker.contracts.forward_context import (
    ForwardContext,
    GraphBinding,
    PagedDecodePlan,
    PagedVarlenPlan,
    get_forward_context,
    use_forward_context,
)
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.foundation.errors import classify, invalid_descriptor
from uniserve_worker.foundation.runtime_config import (
    DEFAULT_DECODE_GRAPH_BATCH_SIZES,
    DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
    get_execution_config,
)
from uniserve_worker.foundation.sizing import ceil_div
from uniserve_worker.runtime.host_staging import cpu_int_staging_buffer, fill_cpu_ints, is_pinned
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import BatchedPagedRequestCache
from uniserve_worker.runtime.tensor_views import adjacent_one_token_view

from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import GraphCapacity, SegmentTableArrays

if TYPE_CHECKING:
    import torch


"""Capacity-only bucket runtime (Stage 7): graph identity is capacity identity.

A finite ordered set of :class:`~uniserve_worker.contracts.segment_table.GraphCapacity`
buckets is allocated and captured once at startup; production execution can only
refresh and replay an existing bucket. No graph key contains operation, phase,
route, request, overlay, or composition values, and nothing may capture, compile,
or allocate persistent storage after readiness. Each bucket owns pointer-stable
device columns for the closed ``SegmentTable`` schema, the packed
``ResidencyBatch`` mapping, and a fixed result slot per row.
"""




__all__ = [
    "BucketTensors",
    "CudaGraphError",
    "CudaGraphRuntime",
    "GraphReplayView",
]


class CudaGraphError(RuntimeError):
    """A graph-runtime law was violated (capacity miss, pointer drift, ...)."""


@dataclass(frozen=True)
class BucketTensors:
    """Pointer-stable device columns owned by one captured bucket."""

    # SegmentTable columns (int32, capacity `segments`).
    segment_active: torch.Tensor
    row_id: torch.Tensor
    operation_tag: torch.Tensor
    route_id: torch.Tensor
    local_segment_id: torch.Tensor
    token_begin: torch.Tensor
    token_count: torch.Tensor
    query_begin: torch.Tensor
    query_count: torch.Tensor
    context_length: torch.Tensor
    position_begin: torch.Tensor
    position_count: torch.Tensor
    branch_id: torch.Tensor
    branch_count: torch.Tensor
    attention_pattern: torch.Tensor
    attention_region_id: torch.Tensor
    kv_group: torch.Tensor
    kv_read_index: torch.Tensor
    kv_write_index: torch.Tensor
    cache_effect: torch.Tensor
    cache_write_count: torch.Tensor
    input_product_index: torch.Tensor
    output_product_index: torch.Tensor
    overlay_slot: torch.Tensor
    candidate_begin: torch.Tensor
    candidate_count: torch.Tensor
    result_slot: torch.Tensor
    # ResidencyBatch columns.
    binding_active: torch.Tensor
    binding_domain_id: torch.Tensor
    binding_committed_rows: torch.Tensor
    binding_provisional_rows: torch.Tensor
    binding_page_indptr: torch.Tensor
    page_ids: torch.Tensor
    write_page_ids: torch.Tensor
    write_page_offsets: torch.Tensor
    write_active: torch.Tensor
    # Fixed result slots (int64, capacity `rows`).
    row_results: torch.Tensor


@dataclass(frozen=True)
class GraphReplayView:
    """Graph-owned result view; copy out before the next replay."""

    bucket_index: int
    row_results: torch.Tensor


class _Bucket:
    def __init__(self, capacity: GraphCapacity, device: torch.device) -> None:
        self.capacity = capacity
        segment_columns = {
            name: torch.zeros(capacity.segments, dtype=torch.int32, device=device)
            for name in SegmentTableArrays.__dataclass_fields__
        }
        residency = capacity.residency
        residency_columns = {
            "binding_active": torch.zeros(
                residency.bindings + 1, dtype=torch.uint8, device=device
            ),
            "binding_domain_id": torch.zeros(
                residency.bindings + 1, dtype=torch.int32, device=device
            ),
            "binding_committed_rows": torch.zeros(
                residency.bindings + 1, dtype=torch.int32, device=device
            ),
            "binding_provisional_rows": torch.zeros(
                residency.bindings + 1, dtype=torch.int32, device=device
            ),
            "binding_page_indptr": torch.zeros(
                residency.bindings + 2, dtype=torch.int32, device=device
            ),
            "page_ids": torch.zeros(
                residency.page_references, dtype=torch.int32, device=device
            ),
            "write_page_ids": torch.zeros(
                residency.tokens, dtype=torch.int32, device=device
            ),
            "write_page_offsets": torch.zeros(
                residency.tokens, dtype=torch.int32, device=device
            ),
            "write_active": torch.zeros(
                residency.tokens, dtype=torch.uint8, device=device
            ),
        }
        self.tensors = BucketTensors(
            **segment_columns,
            **residency_columns,
            row_results=torch.zeros(
                max(capacity.rows, 1), dtype=torch.int64, device=device
            ),
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.replay_count = 0
        self._pointers = tuple(
            getattr(self.tensors, field.name).data_ptr()
            for field in fields(BucketTensors)
        )

    def assert_pointer_stability(self) -> None:
        current = tuple(
            getattr(self.tensors, field.name).data_ptr()
            for field in fields(BucketTensors)
        )
        if current != self._pointers:
            raise CudaGraphError("bucket tensor pointers drifted after capture")

    def refresh(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
    ) -> None:
        # The host arrays carry zero-sentinel tails at full capacity, so one
        # whole-column copy is both the active refresh and the neutral fill.
        for name in SegmentTableArrays.__dataclass_fields__:
            column = getattr(self.tensors, name)
            column.copy_(
                torch.as_tensor(
                    list(getattr(segments, name)), dtype=column.dtype
                )
            )
        for name in (
            "binding_active",
            "binding_domain_id",
            "binding_committed_rows",
            "binding_provisional_rows",
            "binding_page_indptr",
            "page_ids",
            "write_page_ids",
            "write_page_offsets",
            "write_active",
        ):
            column = getattr(self.tensors, name)
            column.copy_(
                torch.as_tensor(
                    list(getattr(residency, name)), dtype=column.dtype
                )
            )


class CudaGraphRuntime:
    """Finite ordered capacities, captured once, replayed forever."""

    def __init__(
        self,
        capacities: tuple[GraphCapacity, ...],
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        if not capacities:
            raise CudaGraphError("a graph runtime needs at least one capacity")
        self.device = torch.device(device)
        self._buckets = [
            _Bucket(capacity, self.device) for capacity in capacities
        ]
        self._ready = False
        self.capture_count = 0

    def capture_all(
        self,
        adapter_fn: Callable[[BucketTensors], None],
    ) -> None:
        """Warm and capture every configured bucket exactly once."""

        if self._ready:
            raise CudaGraphError("capture after readiness is prohibited")
        for bucket in self._buckets:
            # Warmup on a side stream, then capture one adapter invocation.
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                adapter_fn(bucket.tensors)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                adapter_fn(bucket.tensors)
            bucket.graph = graph
            bucket.assert_pointer_stability()
            self.capture_count += 1
        self._ready = True

    def execute(
        self,
        demand: GraphCapacity,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
    ) -> GraphReplayView:
        """Select the smallest dominating bucket, refresh, replay once."""

        if not self._ready:
            raise CudaGraphError("the runtime is not ready")
        bucket_index = self._select(demand)
        bucket = self._buckets[bucket_index]
        bucket.refresh(segments, residency)
        allocated_before = torch.cuda.memory_allocated(self.device)
        assert bucket.graph is not None
        bucket.graph.replay()
        torch.cuda.synchronize(self.device)
        if torch.cuda.memory_allocated(self.device) != allocated_before:
            raise CudaGraphError("replay changed persistent device allocation")
        bucket.assert_pointer_stability()
        bucket.replay_count += 1
        return GraphReplayView(
            bucket_index=bucket_index,
            row_results=bucket.tensors.row_results,
        )

    def replay_counts(self) -> tuple[int, ...]:
        return tuple(bucket.replay_count for bucket in self._buckets)

    def _select(self, demand: GraphCapacity) -> int:
        candidates = [
            index
            for index, bucket in enumerate(self._buckets)
            if bucket.capacity.dominates(demand)
        ]
        if not candidates:
            raise CudaGraphError(
                "no configured graph capacity dominates the demand"
            )
        return min(
            candidates,
            key=lambda index: (
                self._buckets[index].capacity.tokens,
                self._buckets[index].capacity.segments,
                self._buckets[index].capacity.rows,
            ),
        )


# ---------------------
# Shared graph-runner base, capture locks, and bucket configuration
# ---------------------

# Module-level buffer pool for ``_share_decode_graph_input_buffer`` (contract tests).
_DECODE_GRAPH_INPUT_BUFFER_POOL: dict[tuple[str, str, str], torch.Tensor] = {}
_DECODE_GRAPH_INPUT_BUFFER_POOL_LOCK = threading.Lock()
GraphOutput = TypeVar("GraphOutput")

# Warmup iterations run before each CUDA-graph capture to settle allocator and
# autotune state so the captured graph is stable. Two passes is the minimum that
# reliably clears first-call side effects.
_CAPTURE_WARMUP_ITERS = 2

# Default metric prefix; runners may override. Every production text runner
# passes ``text_`` explicitly; the default matches so component metrics land in
# one namespace regardless of construction site.
_DEFAULT_METRIC_PREFIX = "text_"


def _record_function_scope(name: str):
    record = getattr(torch.profiler, "record_function", None)
    return record(name) if callable(record) else nullcontext()

def maybe_weak_ref_cuda_graph_tensor(tensor: Any) -> Any:
    if not isinstance(tensor, torch.Tensor):
        return tensor
    weak_ref_tensor = _weak_ref_tensor_func()
    if weak_ref_tensor is None:
        return tensor
    try:
        return weak_ref_tensor(tensor)
    except Exception:
        return tensor


@lru_cache(maxsize=1)
def _weak_ref_tensor_func() -> Any:
    from uniserve_worker.ops.providers import weak_ref_tensor_provider

    return weak_ref_tensor_provider()


class GraphEvent(str, Enum):
    """CUDA-graph stat events. Values are the original dispatch strings."""

    CAPTURE_REPLAY = "capture_replay"
    REPLAY = "replay"
    MISS = "miss"
    FALLBACK = "fallback"

    def __str__(self) -> str:
        return self.value


def record_graph_stats(
    ctx: Any,
    event: GraphEvent | str,
    *,
    mode: ForwardMode,
    unpadded_tokens: int,
    padded_tokens: int,
) -> None:
    """Apply a CUDA-graph stat event to ``ctx.stats`` for decode and prefill alike.

    ``unpadded_tokens``/``padded_tokens`` are the raw vs bucket-padded token
    counts the event contributes; ``mode`` is the runtime graph mode recorded on
    a capture/replay. ``miss`` and ``fallback`` only bump their counter.
    """

    stats = getattr(ctx, "stats", None)
    if stats is None:
        return
    event = GraphEvent(event)
    if event is GraphEvent.MISS:
        stats.cuda_graph_misses += 1
        return
    if event is GraphEvent.FALLBACK:
        stats.cuda_graph_fallbacks += 1
        return
    if event is GraphEvent.CAPTURE_REPLAY:
        stats.cuda_graph_captures += 1
        stats.cuda_graph_replays += 1
    else:  # GraphEvent.REPLAY
        stats.cuda_graph_replays += 1
    unpadded_tokens = int(unpadded_tokens)
    padded_tokens = int(padded_tokens)
    stats.cuda_graph_unpadded_tokens += unpadded_tokens
    stats.cuda_graph_padded_tokens += max(0, padded_tokens - unpadded_tokens)
    stats.record_runtime_graph_mode(mode.value)

_DEFAULT_DECODE_GRAPH_BATCH_SIZES = DEFAULT_DECODE_GRAPH_BATCH_SIZES
_DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS = DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS


def _parse_positive_int_csv(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive ints into a sorted unique tuple."""

    sizes = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            sizes.append(value)
    return tuple(sorted(set(sizes)))


def _normalize_token_buckets(raw: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted({int(size) for size in raw if int(size) > 0}))


def _share_input_buffer(
    pool: dict[tuple[str, str, str], torch.Tensor],
    name: str,
    tensor: torch.Tensor,
    *,
    strict: bool,
) -> torch.Tensor:
    """Pool input buffers by (name, dtype, device), slicing into the largest.

    Production runners usually capture buckets largest-first, but interleaved
    decode captures lazily from live request batches. If a larger bucket arrives
    after a smaller one, install the larger tensor for future captures; already
    captured graph states keep their own tensor references, so their captured
    input addresses remain valid.
    """

    key = (str(name), str(tensor.dtype), str(tensor.device))
    existing = pool.get(key)
    if existing is not None:
        if int(existing.numel()) >= int(tensor.numel()):
            return existing.as_strided(tuple(tensor.shape), tuple(tensor.stride()))
    pool[key] = tensor
    return tensor


def _share_decode_graph_input_buffer(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Module-level buffer pool delegate (non-strict); runners use instance pools."""

    with _DECODE_GRAPH_INPUT_BUFFER_POOL_LOCK:
        return _share_input_buffer(_DECODE_GRAPH_INPUT_BUFFER_POOL, name, tensor, strict=False)


def _reset_for_testing() -> None:
    """Test-only; not part of the public API.

    Clear the module-level decode-graph input buffer pool so tests get
    deterministic isolation. The pool is repopulated on demand by
    ``_share_decode_graph_input_buffer``.
    """

    with _DECODE_GRAPH_INPUT_BUFFER_POOL_LOCK:
        _DECODE_GRAPH_INPUT_BUFFER_POOL.clear()


class _GraphRunnerBase:
    """Shared enable/warmup/capture-pool scaffolding for the graph runners."""

    name: str
    default_enabled: bool
    default_warmup: bool
    metric_prefix: str
    logger: Any
    states: dict[Any, Any]
    disabled: set[Any]
    _capture_pool: Any
    _graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor]

    def enabled(self) -> bool:
        return self.default_enabled

    def warmup_enabled(self) -> bool:
        return self.default_warmup

    def capture_pool(self) -> Any:
        if self._capture_pool is not None:
            return self._capture_pool
        graph_pool_handle = getattr(torch.cuda, "graph_pool_handle", None)
        if callable(graph_pool_handle):
            self._capture_pool = graph_pool_handle()
        return self._capture_pool

    def share_graph_input_buffer(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Reuse an already-allocated input buffer for a smaller bucket.

        Warmed-up runners capture largest-first (see ``warmup_capture_*``), so
        smaller buckets usually slice into the first allocation via
        ``as_strided``. Lazy runners may discover larger exact batches later; in
        that case the pool grows for subsequent captures without mutating
        already-captured graph states.
        """

        return _share_input_buffer(self._graph_input_buffer_pool, name, tensor, strict=True)

    @staticmethod
    def _destroy_graph_state(state: Any) -> None:
        """Destroy one graph executable and its graph-scoped backend binding."""

        graph = getattr(state, "graph", None)
        release_backend = getattr(state, "release_backend", None)
        if hasattr(state, "release_backend"):
            state.release_backend = None
        try:
            reset = getattr(graph, "reset", None)
            if callable(reset):
                reset()
        finally:
            if callable(release_backend):
                release_backend()

    def _retire_graph_states(
        self,
        keys: tuple[Any, ...],
        *,
        device: torch.device | str,
        reclaim: bool,
    ) -> None:
        """Synchronously retire graph states before another private-pool capture."""

        states = [self.states.pop(key) for key in keys if key in self.states]
        if not states:
            return
        torch.cuda.synchronize(device)
        try:
            for state in states:
                self._destroy_graph_state(state)
        finally:
            states.clear()
            if reclaim:
                gc.collect()
                torch.cuda.empty_cache()

    def _capture_or_replay(
        self,
        *,
        key: Any,
        ctx: Any,
        capture: Callable[[], Any],
        copy_inputs: Callable[[Any], None],
        replay: Callable[[Any], GraphOutput],
        record: Callable[[GraphEvent], None],
        disable: Callable[[BaseException], None],
        capture_metric: str,
        input_copy_metric: str,
        replay_metric: str,
        after_copy: Callable[[Any], None] | None = None,
        after_copy_metric: str | None = None,
    ) -> GraphOutput | None:
        """Template method for capture-or-replay lifecycle.

        Subclasses provide bucket selection, state construction, input copying,
        optional backend preparation, and stats mapping. The error handling,
        capture timing, replay timing, state storage, and fatal-CUDA propagation
        live here once for decode and prefill runners.
        """

        state = self.states.get(key)
        try:
            event = GraphEvent.REPLAY
            if state is None:
                start = ctx.component_timer_start()
                with _record_function_scope(f"uniserve.cuda_graph.{capture_metric}"):
                    state = capture()
                ctx.record_component_elapsed(capture_metric, start)
                self.states[key] = state
                event = GraphEvent.CAPTURE_REPLAY
            start = ctx.component_timer_start()
            with _record_function_scope(f"uniserve.cuda_graph.{input_copy_metric}"):
                copy_inputs(state)
            ctx.record_component_elapsed(input_copy_metric, start)
            if after_copy is not None:
                start = ctx.component_timer_start()
                metric = after_copy_metric or "prepare"
                with _record_function_scope(f"uniserve.cuda_graph.{metric}"):
                    after_copy(state)
                if after_copy_metric is not None:
                    ctx.record_component_elapsed(after_copy_metric, start)
            start = ctx.component_timer_start()
            with _record_function_scope(f"uniserve.cuda_graph.{replay_metric}"):
                logits = replay(state)
            ctx.record_component_elapsed(replay_metric, start)
            record(event)
            return logits
        except Exception as exc:
            if classify(exc).fatal:
                raise
            disable(exc)
            record(GraphEvent.FALLBACK)
            return None

    def _capture_graph_state(
        self,
        *,
        device: torch.device | str,
        state: Any,
        run: Callable[[], GraphOutput],
        copy_inputs: Callable[[Any], None],
        before_run: Callable[[Any], None] | None = None,
    ) -> Any:
        """Template method for CUDA graph capture after state allocation."""

        device = torch.device(device)
        current_stream = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(_CAPTURE_WARMUP_ITERS):
                copy_inputs(state)
                if before_run is not None:
                    before_run(state)
                state.logits = run()
        current_stream.wait_stream(warmup_stream)
        copy_inputs(state)
        if before_run is not None:
            before_run(state)
        graph_kwargs = {}
        capture_pool = self.capture_pool()
        if capture_pool is not None:
            graph_kwargs["pool"] = capture_pool
        with torch.cuda.graph(state.graph, **graph_kwargs):
            state.logits = run()
        state.logits = maybe_weak_ref_cuda_graph_tensor(state.logits)
        return state

    def _warmup_capture_buckets(
        self,
        *,
        device: torch.device | str,
        ctx: Any,
        buckets: Callable[[], Any],
        key_for: Callable[[Any], Any],
        should_skip: Callable[[Any], bool],
        capture_bucket: Callable[[Any, Any], Any],
        copy_inputs: Callable[[Any, Any], None],
        replay: Callable[[Any], None],
        disable: Callable[[Any, BaseException], None],
    ) -> None:
        """Template method for warmup capture/replay bucket loops."""

        if not self.enabled() or not self.warmup_enabled():
            return
        with torch.inference_mode():
            for bucket in buckets():
                key = key_for(bucket)
                if key in self.states or key in self.disabled or should_skip(bucket):
                    continue
                try:
                    state = capture_bucket(bucket, ctx)
                    self.states[key] = state
                    copy_inputs(bucket, state)
                    replay(state)
                except Exception as exc:
                    disable(bucket, exc)
        if self.states:
            torch.cuda.synchronize(device)


# ---------------------
# Stable graph slot buffers
# ---------------------

class SlotAxis(StrEnum):
    TOKENS = "tokens"
    ROWS = "rows"
    SEGMENTS = "segments"
    BRANCHES = "branches"
    SCALAR = "scalar"
    OPAQUE = "opaque"


class PaddingPolicy(StrEnum):
    KEEP = "keep"
    ZERO = "zero"
    SENTINEL = "sentinel"
    COPY_HEAD = "copy_head"
    FILL_ONCE = "fill_once"
    CUSTOM = "custom"


@dataclass
class ForwardGraphSlot:
    name: str
    axis: SlotAxis
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    padding: PaddingPolicy = PaddingPolicy.KEEP
    sentinel: int | float = 0
    refresh: Callable[[torch.Tensor, torch.Tensor, int], None] | None = None
    pin_cpu: bool = False
    tensor: torch.Tensor | None = None
    cpu_staging: torch.Tensor | None = None

    def allocate(self) -> None:
        if self.tensor is None:
            self.tensor = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        if self.pin_cpu:
            try:
                self.cpu_staging = torch.empty(self.shape, dtype=self.dtype, device="cpu", pin_memory=True)
            except RuntimeError:
                self.cpu_staging = torch.empty(self.shape, dtype=self.dtype, device="cpu")


class ForwardGraphBufferRegistry:
    def __init__(self) -> None:
        self._slots: dict[str, ForwardGraphSlot] = {}
        self.copy_bytes: dict[str, int] = {}

    def register_slot(
        self,
        name: str,
        *,
        axis: SlotAxis | str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str,
        padding: PaddingPolicy | str = PaddingPolicy.KEEP,
        sentinel: int | float = 0,
        refresh: Callable[[torch.Tensor, torch.Tensor, int], None] | None = None,
        pin_cpu: bool = False,
    ) -> ForwardGraphSlot:
        if name in self._slots:
            raise invalid_descriptor(f"forward graph slot {name!r} already exists")
        slot = ForwardGraphSlot(
            name=name,
            axis=SlotAxis(axis),
            shape=tuple(int(v) for v in shape),
            dtype=dtype,
            device=torch.device(device),
            padding=PaddingPolicy(padding),
            sentinel=sentinel,
            refresh=refresh,
            pin_cpu=pin_cpu,
        )
        slot.allocate()
        self._slots[name] = slot
        return slot

    def slot(self, name: str) -> ForwardGraphSlot:
        try:
            return self._slots[name]
        except KeyError as exc:
            raise invalid_descriptor(f"unknown forward graph slot {name!r}") from exc

    def tensor(self, name: str) -> torch.Tensor:
        tensor = self.slot(name).tensor
        if tensor is None:
            raise invalid_descriptor(f"forward graph slot {name!r} is not allocated")
        return tensor

    def refresh_slot(self, name: str, value: torch.Tensor, *, raw_length: int | None = None) -> torch.Tensor:
        slot = self.slot(name)
        target = self.tensor(name)
        if value.dtype != slot.dtype:
            value = value.to(dtype=slot.dtype)
        if value.device != target.device:
            value = value.to(device=target.device, non_blocking=True)
        raw = int(raw_length if raw_length is not None else _head_length(value, target))
        if raw < 0 or raw > target.shape[0]:
            raise invalid_descriptor("forward graph slot refresh length exceeds slot capacity")
        if slot.refresh is not None:
            slot.refresh(target, value, raw)
        else:
            _copy_head(target, value, raw)
            _pad_tail(slot, target, raw)
        self.copy_bytes[name] = self.copy_bytes.get(name, 0) + int(value.element_size() * value.numel())
        return target

    def validate_geometry(self, name: str, value: torch.Tensor) -> None:
        slot = self.slot(name)
        if value.ndim != len(slot.shape):
            raise invalid_descriptor("forward graph slot rank mismatch")
        for actual, expected in zip(value.shape[1:], slot.shape[1:]):
            if int(actual) != int(expected):
                raise invalid_descriptor("forward graph slot static geometry mismatch")

    def batch_view(self) -> dict[str, torch.Tensor]:
        return {name: self.tensor(name) for name in self._slots}


def _head_length(value: torch.Tensor, target: torch.Tensor) -> int:
    if value.ndim == 0:
        return 1
    if target.ndim == 0:
        return 1
    return int(value.shape[0])


def _copy_head(target: torch.Tensor, value: torch.Tensor, raw: int) -> None:
    if target.ndim == 0:
        target.copy_(value.reshape(()))
        return
    if raw == 0:
        return
    target[:raw].copy_(value[:raw])


def _pad_tail(slot: ForwardGraphSlot, target: torch.Tensor, raw: int) -> None:
    if target.ndim == 0 or raw >= target.shape[0]:
        return
    tail = target[raw:]
    if slot.padding is PaddingPolicy.KEEP:
        return
    if slot.padding is PaddingPolicy.ZERO:
        tail.zero_()
        return
    if slot.padding is PaddingPolicy.SENTINEL:
        tail.fill_(slot.sentinel)
        return
    if slot.padding is PaddingPolicy.COPY_HEAD:
        if raw > 0:
            tail.copy_(target[raw - 1].expand_as(tail))
        return
    if slot.padding is PaddingPolicy.FILL_ONCE:
        return
    if slot.padding is PaddingPolicy.CUSTOM:
        if slot.refresh is None:
            raise invalid_descriptor("custom forward graph slot padding requires a refresh hook")
        return


# ---------------------
# Graph shape keys
# ---------------------

@dataclass(frozen=True)
class ForwardGraphShapeKey:
    program: str
    mode: ForwardMode
    op_modes: tuple[ForwardMode, ...]
    token_bucket: int
    row_bucket: int
    segment_geometry: tuple[tuple[int, str, str, int], ...]
    kv_geometry: tuple[Any, ...]
    tensor_geometry: tuple[Any, ...]
    backend: str | None = None
    descriptor_variant: str | None = None


def graph_shape_key(
    *,
    program: str,
    batch: ForwardBatch,
    plan: ForwardPlan,
    backend: str | None = None,
    descriptor_variant: str | None = None,
) -> ForwardGraphShapeKey:
    token_bucket = int(batch.padded_num_tokens or plan.shape.padded_token_count)
    row_bucket = max(int(plan.shape.padded_row_count), len(batch.req_ids))
    segment_geometry = tuple(
        (
            int(segment.q_len),
            str(segment.visible_policy.value),
            str(segment.segment_class.value),
            int(segment.branch_id),
        )
        for segment in plan.segments
    )
    max_block_width = 0
    block_table = getattr(batch, "block_table", None)
    if block_table is not None and getattr(block_table, "ndim", 0) >= 2:
        max_block_width = int(block_table.shape[-1])
    kv_geometry = (
        max_block_width,
        tuple(str(segment.kv_write_policy.value) for segment in plan.segments),
        int(plan.shape.branch_count),
    )
    tensor_geometry = (
        str(getattr(getattr(batch, "input_ids", None), "dtype", "")),
        str(getattr(getattr(batch, "input_ids", None), "device", getattr(batch, "device", ""))),
        tuple(getattr(getattr(batch, "input_ids", None), "shape", ())),
        tuple(getattr(getattr(batch, "is_gen", None), "shape", ())),
    )
    return ForwardGraphShapeKey(
        program=program,
        mode=plan.forward_mode,
        op_modes=plan.op_modes,
        token_bucket=token_bucket,
        row_bucket=row_bucket,
        segment_geometry=segment_geometry,
        kv_geometry=kv_geometry,
        tensor_geometry=tensor_geometry,
        backend=backend,
        descriptor_variant=descriptor_variant,
    )


# ---------------------
# Decode-graph padding
# ---------------------

def decode_graph_padding_block_ids(pool: Any) -> tuple[int, ...]:
    block_ids = tuple(int(block_id) for block_id in getattr(pool, "reserved_block_ids", ()))
    if len(set(block_ids)) != len(block_ids):
        raise invalid_descriptor("decode graph padding block ids must be unique")
    num_blocks = int(getattr(pool, "num_blocks", 0) or 0)
    if any(block_id < 0 or block_id >= num_blocks for block_id in block_ids):
        raise invalid_descriptor("decode graph padding block id is out of range")
    return block_ids


# ---------------------
# Graph statistics
# ---------------------

@dataclass
class ForwardGraphStats:
    captures: int = 0
    replays: int = 0
    misses: int = 0
    fallbacks: int = 0
    capture_failures: int = 0
    replay_failures: int = 0
    shape_counts: dict[str, int] = field(default_factory=dict)
    mode_counts: dict[str, int] = field(default_factory=dict)
    padded_tokens: int = 0
    unpadded_tokens: int = 0

    def record_capture(
        self,
        key: Any,
        *,
        unpadded_tokens: int = 0,
        forward_stats: Any | None = None,
    ) -> None:
        self.captures += 1
        self._record_key(key, unpadded_tokens=unpadded_tokens)
        _record_forward_stats_key(
            forward_stats,
            key,
            unpadded_tokens=unpadded_tokens,
            event="capture",
        )

    def record_replay(
        self,
        key: Any,
        *,
        unpadded_tokens: int = 0,
        forward_stats: Any | None = None,
    ) -> None:
        self.replays += 1
        self._record_key(key, unpadded_tokens=unpadded_tokens)
        _record_forward_stats_key(
            forward_stats,
            key,
            unpadded_tokens=unpadded_tokens,
            event="replay",
        )

    def record_miss(self, mode: str, *, forward_stats: Any | None = None) -> None:
        self.misses += 1
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1
        if forward_stats is not None:
            forward_stats.cuda_graph_misses += 1
            forward_stats.record_runtime_graph_mode(mode)

    def _record_key(self, key: Any, *, unpadded_tokens: int) -> None:
        text = repr(key)
        self.shape_counts[text] = self.shape_counts.get(text, 0) + 1
        mode = getattr(getattr(key, "mode", None), "value", str(getattr(key, "mode", "")))
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1
        self.padded_tokens += int(getattr(key, "token_bucket", 0) or 0)
        self.unpadded_tokens += max(0, int(unpadded_tokens))

    def to_wire(self) -> dict[str, Any]:
        return {
            "forward_graph_captures": self.captures,
            "forward_graph_replays": self.replays,
            "forward_graph_misses": self.misses,
            "forward_graph_fallbacks": self.fallbacks,
            "forward_graph_capture_failures": self.capture_failures,
            "forward_graph_replay_failures": self.replay_failures,
            "forward_graph_runtime_mode_counts": dict(self.mode_counts),
            "forward_graph_shape_counts": dict(self.shape_counts),
            "forward_graph_padded_tokens": self.padded_tokens,
            "forward_graph_unpadded_tokens": self.unpadded_tokens,
        }


def _record_forward_stats_key(
    forward_stats: Any | None,
    key: Any,
    *,
    unpadded_tokens: int,
    event: str,
) -> None:
    if forward_stats is None:
        return
    if event == "capture":
        forward_stats.cuda_graph_captures += 1
    elif event == "replay":
        forward_stats.cuda_graph_replays += 1
    mode = getattr(getattr(key, "mode", None), "value", str(getattr(key, "mode", "")))
    forward_stats.record_runtime_graph_mode(mode)
    shape_text = repr(key)
    shape_counts = forward_stats.forward_graph_shape_counts
    shape_counts[shape_text] = int(shape_counts.get(shape_text, 0)) + 1
    forward_stats.cuda_graph_padded_tokens += int(getattr(key, "token_bucket", 0) or 0)
    forward_stats.cuda_graph_unpadded_tokens += max(0, int(unpadded_tokens))


# ---------------------
# Graph programs and captured-graph dispatch
# ---------------------

@dataclass(frozen=True)
class GraphEligibility:
    eligible: bool
    reason: str = ""


@dataclass
class CapturedForwardGraph:
    key: ForwardGraphShapeKey
    program: "ForwardGraphProgram"
    payload: Any = None
    replay_fn: Callable[[ForwardBatch], ForwardResult] | None = None


class ForwardGraphProgram(ABC):
    program_id: str = "program"

    @abstractmethod
    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility: ...

    def shape_key(self, batch: ForwardBatch, plan: ForwardPlan) -> ForwardGraphShapeKey:
        return graph_shape_key(program=self.program_id, batch=batch, plan=plan)

    def capture(
        self,
        key: ForwardGraphShapeKey,
        batch: ForwardBatch,
        plan: ForwardPlan,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
    ) -> CapturedForwardGraph | None:
        del plan
        return CapturedForwardGraph(
            key=key,
            program=self,
            payload=forward_fn(batch),
            replay_fn=forward_fn,
        )

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult | None:
        del plan
        if graph.replay_fn is not None:
            return graph.replay_fn(batch)
        if not isinstance(graph.payload, ForwardResult):
            raise TypeError("captured forward graph payload must be a ForwardResult")
        return graph.payload


class _TextGraphProgramMixin(ForwardGraphProgram):
    def __init__(
        self,
        *,
        text_driver: Any | None = None,
        model: Any | None = None,
        request_states: Any | None = None,
        text_graph_runner: Any | None = None,
    ) -> None:
        self.text_driver = text_driver
        self.model = model
        self.request_states = request_states
        self.text_graph_runner = text_graph_runner

    @property
    def _bound_text_graph(self) -> bool:
        return (
            self.text_driver is not None
            and self.model is not None
            and self.request_states is not None
            and self.text_graph_runner is not None
        )

    def _text_graph_eligible(self, plan: ForwardPlan) -> GraphEligibility:
        if not self._bound_text_graph:
            return GraphEligibility(True)
        for row in plan.rows:
            op = row.op
            if op.get("spec_token_ids"):
                return GraphEligibility(False, "speculative text rows are not graphable")
            if row.mode is ForwardMode.DECODE and int(op.get("decode_token_count") or 1) > 1:
                forward_graph_result = getattr(self.text_driver, "forward_graph_result", None)
                if not callable(forward_graph_result):
                    return GraphEligibility(False, "decode burst rows are not graphable")
        return GraphEligibility(True)

    def _run_text_graph(self, plan: ForwardPlan) -> ForwardResult | None:
        if not self._bound_text_graph:
            return None
        forward_graph_result = getattr(self.text_driver, "forward_graph_result", None)
        if callable(forward_graph_result):
            dispatch_batch = plan.runtime_handles.get("dispatch_batch")
            if not isinstance(dispatch_batch, UniForwardBatch):
                dispatch_batch = UniForwardBatch.from_ops(plan.ops)
            result = forward_graph_result(
                dispatch_batch,
                self.request_states,
                self.model,
                graph_runner=self.text_graph_runner,
                defer_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results", False)),
                defer_sampling=bool(plan.runtime_handles.get("defer_sampling", False)),
            )
            if result is not None and not isinstance(result, ForwardResult):
                raise invalid_descriptor("text graph result must be a ForwardResult")
            return result
        forward_logits_graph = getattr(self.text_driver, "forward_logits_graph", None)
        if not callable(forward_logits_graph):
            return None
        dispatch_batch = plan.runtime_handles.get("dispatch_batch")
        if not isinstance(dispatch_batch, UniForwardBatch):
            dispatch_batch = UniForwardBatch.from_ops(plan.ops)
        text_result = forward_logits_graph(
            dispatch_batch,
            self.request_states,
            self.model,
            graph_runner=self.text_graph_runner,
            defer_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results", False)),
            defer_sampling=bool(plan.runtime_handles.get("defer_sampling", False)),
        )
        if text_result is None:
            return None
        expected_req_ids = tuple(int(row.req_id) for row in plan.rows)
        req_ids = tuple(int(req_id) for req_id in getattr(text_result, "req_ids", ()))
        if req_ids != expected_req_ids:
            raise invalid_descriptor("text graph result req_ids must align with forward rows")
        return ForwardResult(
            text_logits=text_result.logits,
            text_cuda_ready_start_event=getattr(text_result, "cuda_ready_start_event", None),
        )

    def capture(
        self,
        key: ForwardGraphShapeKey,
        batch: ForwardBatch,
        plan: ForwardPlan,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
    ) -> CapturedForwardGraph | None:
        if not self._bound_text_graph:
            return super().capture(key, batch, plan, forward_fn)
        result = self._run_text_graph(plan)
        if result is None:
            return None
        return CapturedForwardGraph(key=key, program=self, payload=result)

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult | None:
        if not self._bound_text_graph:
            return super().replay(graph, batch, plan)
        return self._run_text_graph(plan)


class DecodeGraphProgram(_TextGraphProgramMixin):
    program_id = "decode"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count == plan.shape.row_count and all(
            segment.segment_class.value == "decode" for segment in plan.segments
        ):
            return self._text_graph_eligible(plan)
        return GraphEligibility(False, "not a pure decode shape")


class PrefillGraphProgram(_TextGraphProgramMixin):
    program_id = "prefill"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count == plan.shape.row_count and plan.shape.token_count > 0:
            return self._text_graph_eligible(plan)
        return GraphEligibility(False, "not a text prefill shape")


class ModelOwnedTextGraphProgram(_TextGraphProgramMixin):
    program_id = "model_owned_text"

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count != plan.shape.row_count:
            return GraphEligibility(False, "not a pure model-owned text shape")
        if not self._bound_model_owned_text_graph:
            return GraphEligibility(False, "model-owned text graph program is not bound")
        return self._text_graph_eligible(plan)

    @property
    def _bound_model_owned_text_graph(self) -> bool:
        return (
            self.text_driver is not None
            and self.model is not None
            and self.request_states is not None
            and callable(getattr(self.model, "try_run_text_graph_logits_batch", None))
        )

    @property
    def _bound_text_graph(self) -> bool:
        return self._bound_model_owned_text_graph


class PackedVisibleGraphProgram(ForwardGraphProgram):
    program_id = "packed_visible"

    def __init__(
        self,
        *,
        owner: Any | None = None,
        request_states: Any | None = None,
        image_decode_driver: Any | None = None,
    ) -> None:
        self.owner = owner
        self.request_states = request_states
        self.image_decode_driver = image_decode_driver

    @property
    def _bound_packed_visible(self) -> bool:
        return (
            self.owner is not None
            and self.request_states is not None
            and callable(getattr(self.owner, "prepare_denoise", None))
            and callable(getattr(self.owner, "packed_decoder_forward", None))
            and callable(getattr(self.owner, "packed_text_embeddings", None))
            and callable(getattr(self.owner, "packed_text_logits", None))
            and callable(getattr(self.owner, "packed_graph_attention", None))
        )

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.text_row_count and (
            plan.shape.denoise_row_count or plan.shape.commit_row_count
        ):
            if not self._bound_packed_visible:
                if self.owner is None and self.request_states is None:
                    return GraphEligibility(True)
                return GraphEligibility(False, "packed visible graph program is not bound")
            if plan.shape.denoise_row_count and not callable(
                getattr(self.owner, "packed_hidden_to_velocity", None)
            ):
                return GraphEligibility(False, "packed denoise projection is not bound")
            if self._has_burst_rows(plan) and not callable(
                getattr(self.owner, "_run_forward_adapter", None)
            ):
                return GraphEligibility(False, "packed burst graph adapter is not bound")
            if plan.shape.commit_row_count and self.image_decode_driver is None:
                return GraphEligibility(False, "packed visible commit publication is not bound")
            return GraphEligibility(True)
        return GraphEligibility(False, "not a packed visible generation shape")

    def capture(
        self,
        key: ForwardGraphShapeKey,
        batch: ForwardBatch,
        plan: ForwardPlan,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
    ) -> CapturedForwardGraph | None:
        del batch, forward_fn
        result = self._run_packed_visible_graph(plan)
        if result is None:
            return None
        return CapturedForwardGraph(key=key, program=self, payload=result)

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult | None:
        del graph, batch
        return self._run_packed_visible_graph(plan)

    def _run_packed_visible_graph(self, plan: ForwardPlan) -> ForwardResult | None:
        owner = self.owner
        request_states = self.request_states
        if not self._bound_packed_visible or owner is None or request_states is None:
            return None
        dispatch_batch = plan.runtime_handles.get("dispatch_batch")
        if not isinstance(dispatch_batch, UniForwardBatch):
            dispatch_batch = UniForwardBatch.from_ops(plan.ops)
        if self._has_burst_rows(plan):
            run = getattr(owner, "_run_forward_adapter", None)
            if not callable(run):
                return None
            outputs = run(
                dispatch_batch,
                request_states=request_states,
                group=list(enumerate(dispatch_batch.ops)),
                defer_text_cpu_results=bool(
                    plan.runtime_handles.get("defer_text_cpu_results", False)
                ),
            )
            if isinstance(outputs, ForwardResult):
                return outputs
            if not isinstance(outputs, Sequence) or isinstance(
                outputs, (str, bytes, bytearray)
            ):
                raise invalid_descriptor(
                    "packed burst graph adapter must return one result per forward row"
                )
            if len(outputs) != len(plan.rows):
                raise invalid_descriptor(
                    "packed burst graph adapter returned the wrong number of results"
                )
            return ForwardResult(runtime_outputs=tuple(outputs))
        denoise_steps = []
        for row in plan.rows:
            if row.mode is not ForwardMode.DENOISE:
                continue
            state = request_states.get(int(row.req_id))
            step = owner.prepare_denoise(state, dict(row.op))
            denoise_steps.append((int(row.row_index), step))
            extra = getattr(step, "extra", None)
            img = extra.get("img") if isinstance(extra, dict) else None
            residual_state = getattr(img, "residual_cache", None)
            if residual_state is not None:
                residual_state.invalidate()
        run_packed_visible = getattr(owner, "run_packed_visible_forward_result", None)
        if not callable(run_packed_visible):
            return None
        result = run_packed_visible(
            dispatch_batch,
            request_states,
            denoise_steps,
            defer_text_cpu_results=bool(plan.runtime_handles.get("defer_text_cpu_results", False)),
            allow_graph=True,
            require_graph=True,
        )
        if result is None or not plan.shape.commit_row_count:
            return result
        image_decode_driver = self.image_decode_driver
        if image_decode_driver is None:
            raise invalid_descriptor("packed visible commit publication is not bound")
        commit_rows = tuple(row for row in plan.rows if row.mode is ForwardMode.COMMIT)
        commit_result = image_decode_driver.forward_result(
            tuple(
                (int(row.req_id), request_states.get(int(row.req_id)), row.op)
                for row in commit_rows
            ),
            owner,
            row_indices=tuple(int(row.row_index) for row in commit_rows),
        )
        if not isinstance(commit_result, ForwardResult) or commit_result.commit_outputs is None:
            raise invalid_descriptor("packed visible commit publication returned no commit outputs")
        commit_outputs = dict(result.commit_outputs or {})
        commit_outputs.update(commit_result.commit_outputs)
        result.commit_outputs = commit_outputs
        return result

    @staticmethod
    def _has_burst_rows(plan: ForwardPlan) -> bool:
        return any(
            int(op.get("decode_token_count") or 1) > 1
            or int(op.get("denoise_step_count") or 1) > 1
            for op in plan.ops
        )


class DenoiseStepGraphProgram(ForwardGraphProgram):
    program_id = "denoise_step"

    def __init__(
        self,
        *,
        denoise_driver: Any | None = None,
        model: Any | None = None,
        request_states: Any | None = None,
    ) -> None:
        self.denoise_driver = denoise_driver
        self.model = model
        self.request_states = request_states

    @property
    def _bound_denoise_graph(self) -> bool:
        return (
            self.denoise_driver is not None
            and self.model is not None
            and self.request_states is not None
            and callable(getattr(self.denoise_driver, "forward_result", None))
        )

    def can_run(self, batch: ForwardBatch, plan: ForwardPlan) -> GraphEligibility:
        del batch
        if plan.shape.denoise_row_count == plan.shape.row_count:
            if not self._bound_denoise_graph:
                if (
                    self.denoise_driver is None
                    and self.model is None
                    and self.request_states is None
                ):
                    return GraphEligibility(True)
                return GraphEligibility(False, "denoise graph program is not bound")
            for op in plan.ops:
                if int(op.get("denoise_step_count") or 1) > 1:
                    return GraphEligibility(False, "denoise burst rows are not graphable")
            return GraphEligibility(True)
        return GraphEligibility(False, "not a pure denoise shape")

    def capture(
        self,
        key: ForwardGraphShapeKey,
        batch: ForwardBatch,
        plan: ForwardPlan,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
    ) -> CapturedForwardGraph | None:
        del batch, forward_fn
        result = self._run_denoise_graph(plan)
        if result is None:
            return None
        return CapturedForwardGraph(key=key, program=self, payload=result)

    def replay(
        self,
        graph: CapturedForwardGraph,
        batch: ForwardBatch,
        plan: ForwardPlan,
    ) -> ForwardResult | None:
        del graph, batch
        return self._run_denoise_graph(plan)

    def _run_denoise_graph(self, plan: ForwardPlan) -> ForwardResult | None:
        denoise_driver = self.denoise_driver
        model = self.model
        request_states = self.request_states
        if (
            not self._bound_denoise_graph
            or denoise_driver is None
            or model is None
            or request_states is None
        ):
            return None
        forward_result = denoise_driver.forward_result
        kwargs: dict[str, Any] = {"row_indices": tuple(int(row.row_index) for row in plan.rows)}
        if _accepts_keyword(forward_result, "graph_mode"):
            kwargs["graph_mode"] = "require"
        items = [
            (int(row.req_id), request_states.get(int(row.req_id)), row.op)
            for row in plan.rows
        ]
        return forward_result(items, model, **kwargs)


def _accepts_keyword(hook: Any, name: str) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


# ---------------------
# Graph-program replay runner
# ---------------------

logger = logging.getLogger(__name__)


class CudaGraphForwardRunner:
    def __init__(
        self,
        programs: tuple[ForwardGraphProgram, ...] = (),
        *,
        stats: ForwardGraphStats | None = None,
    ) -> None:
        self.programs = list(programs)
        self.stats = stats or ForwardGraphStats()
        self._graphs: dict[object, CapturedForwardGraph] = {}

    def register(self, program: ForwardGraphProgram) -> None:
        self.programs.append(program)

    def run(
        self,
        batch: ForwardBatch,
        plan: ForwardPlan,
        *,
        forward_fn: Callable[[ForwardBatch], ForwardResult],
        allow_capture: bool = True,
    ) -> ForwardResult | None:
        forward_stats = get_forward_context().stats
        ineligible: list[str] = []
        strict = bool(getattr(getattr(plan, "graph_policy", None), "strict", False))
        for program in self.programs:
            eligibility = program.can_run(batch, plan)
            if not eligibility.eligible:
                ineligible.append(f"{program.program_id}:{eligibility.reason or 'ineligible'}")
                continue
            key = program.shape_key(batch, plan)
            graph = self._graphs.get(key)
            if graph is not None:
                result = program.replay(graph, batch, plan)
                if result is None:
                    if strict:
                        logger.warning(
                            "strict forward graph replay miss: program=%s mode=%s rows=%d tokens=%d",
                            program.program_id,
                            plan.forward_mode.value,
                            plan.shape.row_count,
                            plan.shape.token_count,
                        )
                    self.stats.record_miss(plan.forward_mode.value, forward_stats=forward_stats)
                    return None
                result.graph = ForwardGraphExecutionInfo(
                    program=program.program_id,
                    shape_key=key,
                    replayed=True,
                )
                self.stats.record_replay(
                    key,
                    unpadded_tokens=plan.shape.token_count,
                    forward_stats=forward_stats,
                )
                return result
            if not allow_capture:
                if strict:
                    logger.warning(
                        "strict forward graph capture disabled: program=%s mode=%s rows=%d tokens=%d",
                        program.program_id,
                        plan.forward_mode.value,
                        plan.shape.row_count,
                        plan.shape.token_count,
                    )
                self.stats.record_miss(plan.forward_mode.value, forward_stats=forward_stats)
                return None
            graph = program.capture(key, batch, plan, forward_fn)
            if graph is None:
                if strict:
                    logger.warning(
                        "strict forward graph capture miss: program=%s mode=%s rows=%d tokens=%d",
                        program.program_id,
                        plan.forward_mode.value,
                        plan.shape.row_count,
                        plan.shape.token_count,
                    )
                self.stats.record_miss(plan.forward_mode.value, forward_stats=forward_stats)
                return None
            self._graphs[key] = graph
            result = graph.payload
            if not isinstance(result, ForwardResult):
                result = forward_fn(batch)
                graph.payload = result
            result.graph = ForwardGraphExecutionInfo(
                program=program.program_id,
                shape_key=key,
                captured=True,
            )
            self.stats.record_capture(
                key,
                unpadded_tokens=plan.shape.token_count,
                forward_stats=forward_stats,
            )
            return result
        if strict:
            logger.warning(
                "strict forward graph miss: no eligible program mode=%s rows=%d tokens=%d reasons=%s",
                plan.forward_mode.value,
                plan.shape.row_count,
                plan.shape.token_count,
                ";".join(ineligible),
            )
        self.stats.record_miss(plan.forward_mode.value, forward_stats=forward_stats)
        return None


# ---------------------
# Text decode graph runner
# ---------------------

@dataclass
class TextDecodeGraphState:
    """Static buffers bound into a captured paged-text decode graph."""

    batch_size: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    sequence_lens: torch.Tensor
    cache_seqlens: torch.Tensor
    graph: torch.cuda.CUDAGraph
    cache: BatchedPagedRequestCache
    plan: PagedDecodePlan
    graph_binding: GraphBinding
    logits: torch.Tensor | None = None
    long_inputs: torch.Tensor | None = None
    block_table_rows: tuple[tuple[int, ...], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TextDecodeGraphHostInputs:
    """Host-resident dynamic inputs for a one-token decode graph replay."""

    input_ids: Sequence[int]
    positions: Sequence[int]
    block_ids_by_row: Sequence[Sequence[int]]
    cache_seqlens_cpu: Sequence[int]
    kv_seqlens_cpu: Sequence[int]
    token_replacements: Sequence[tuple[int, torch.Tensor]] = field(default_factory=tuple)
    max_context_len: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.input_ids)


class DecodeCudaGraphRunner(_GraphRunnerBase):
    """Own per-bucket decode CUDA graph state and its capture/replay lifecycle."""

    def __init__(
        self,
        *,
        name: str,
        default_enabled: bool = True,
        default_warmup: bool = True,
        default_warmup_batch_sizes: tuple[int, ...] = _DEFAULT_DECODE_GRAPH_BATCH_SIZES,
        metric_prefix: str = _DEFAULT_METRIC_PREFIX,
        logger: Any = None,
    ) -> None:
        self.name = str(name)
        self.default_enabled = bool(default_enabled)
        self.default_warmup = bool(default_warmup)
        self.default_warmup_batch_sizes = tuple(
            sorted({int(size) for size in default_warmup_batch_sizes if int(size) > 0})
        )
        self.metric_prefix = str(metric_prefix)
        self.logger = logger
        self.states: dict[int, TextDecodeGraphState] = {}
        self.disabled: set[int] = set()
        self._capture_pool: Any = None
        self._graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor] = {}

    def warmup_batch_sizes(self) -> tuple[int, ...]:
        return self.default_warmup_batch_sizes

    def warmup_capture_batch_sizes(self) -> tuple[int, ...]:
        return tuple(reversed(self.warmup_batch_sizes()))

    def bucket_batch_size(self, batch_size: int) -> int:
        batch_size = int(batch_size)
        candidates = [
            int(size)
            for size in self.warmup_batch_sizes()
            if int(size) >= batch_size and int(size) not in self.disabled
        ]
        if candidates:
            return min(candidates)
        return batch_size

    def resolve_bucket(self, batch_size: int) -> int:
        """Authoritative decode bucket: prefer an already-captured bucket."""

        batch_size = int(batch_size)
        candidates = [
            int(size)
            for size in self.states
            if int(size) >= batch_size and int(size) not in self.disabled
        ]
        if candidates:
            return min(candidates)
        return self.bucket_batch_size(batch_size)

    def can_use(self, batch_size: int) -> bool:
        return self.enabled() and int(batch_size) > 0 and int(batch_size) not in self.disabled

    def disable(self, batch_size: int, exc: BaseException, *, phase: str) -> None:
        batch_size = int(batch_size)
        self.disabled.add(batch_size)
        self.states.pop(batch_size, None)
        if self.logger is not None:
            self.logger.warning(
                "%s %s decode CUDA graph for batch size %s: %s",
                phase,
                self.name,
                batch_size,
                exc,
            )

    @staticmethod
    def record_stats(
        ctx: Any,
        event: str,
        *,
        batch_size: int,
        graph_batch_size: int | None = None,
    ) -> None:
        batch_size = int(batch_size)
        graph_batch_size = int(graph_batch_size or batch_size)
        record_graph_stats(
            ctx,
            event,
            mode=ForwardMode.DECODE,
            unpadded_tokens=batch_size,
            padded_tokens=graph_batch_size,
        )

    def make_state(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        batch_size: int,
        device: torch.device | str,
        max_context_len: int = 0,
    ) -> TextDecodeGraphState:
        return make_text_decode_graph_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=batch_size,
            device=device,
            buffer_pool=self,
            max_context_len=max_context_len,
        )

    def capture(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_plan: PagedDecodePlan,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
    ) -> TextDecodeGraphState:
        """Capture a decode graph bucket bound to the model's forward closure."""

        device = input_ids.device
        state = self.make_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=batch_size,
            device=device,
            max_context_len=int(getattr(attention_plan, "max_context_len", 0) or 0),
        )
        copy_text_decode_graph_inputs(
            state,
            input_ids=input_ids,
            positions=positions,
            attention_plan=attention_plan,
        )
        graph_ctx = replace(
            ctx,
            attention_plan=state.plan,
            graph_binding=state.graph_binding,
            stats=None,
        )

        def run() -> torch.Tensor:
            # copy_inputs rebuilds state.plan before each warmup and capture run;
            # publish that snapshot so capture records the same plan the backend
            # prepare step plans from.
            with use_forward_context(replace(graph_ctx, attention_plan=state.plan)):
                return forward_fn(state)

        def copy_inputs(capture_state: TextDecodeGraphState) -> None:
            copy_text_decode_graph_inputs(
                capture_state,
                input_ids=input_ids,
                positions=positions,
                attention_plan=attention_plan,
            )

        def prepare(capture_state: TextDecodeGraphState) -> None:
            if prepare_backend is not None:
                prepare_backend(capture_state, graph_ctx)

        return self._capture_graph_state(
            device=device,
            state=state,
            run=run,
            copy_inputs=copy_inputs,
            before_run=prepare,
        )

    def capture_host_inputs(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        batch_size: int,
        device: torch.device | str,
        host_inputs: TextDecodeGraphHostInputs,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
        staging_slot: Any | None = None,
    ) -> TextDecodeGraphState:
        """Capture a decode graph bucket using host-staged dynamic inputs."""

        state = self.make_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=batch_size,
            device=device,
            max_context_len=int(host_inputs.max_context_len),
        )
        copy_text_decode_graph_host_inputs(state, host_inputs, staging_slot=staging_slot)
        graph_ctx = replace(
            ctx,
            attention_plan=state.plan,
            graph_binding=state.graph_binding,
            stats=None,
        )

        def run() -> torch.Tensor:
            # copy_inputs rebuilds state.plan before each warmup and capture run;
            # publish that snapshot so capture records the same plan the backend
            # prepare step plans from.
            with use_forward_context(replace(graph_ctx, attention_plan=state.plan)):
                return forward_fn(state)

        def copy_inputs(capture_state: TextDecodeGraphState) -> None:
            copy_text_decode_graph_host_inputs(
                capture_state, host_inputs, staging_slot=staging_slot
            )

        def prepare(capture_state: TextDecodeGraphState) -> None:
            if prepare_backend is not None:
                prepare_backend(capture_state, graph_ctx)

        return self._capture_graph_state(
            device=device,
            state=state,
            run=run,
            copy_inputs=copy_inputs,
            before_run=prepare,
        )

    def _record_decode_graph_miss(
        self,
        *,
        ctx: Any,
        batch_size: int,
        graph_batch_size: int,
    ) -> None:
        self.record_stats(
            ctx,
            GraphEvent.MISS,
            batch_size=batch_size,
            graph_batch_size=graph_batch_size,
        )

    def _capture_decode_graph(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        graph_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_plan: PagedDecodePlan,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None,
    ) -> TextDecodeGraphState:
        return self.capture(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            batch_size=graph_batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_plan=attention_plan,
            ctx=ctx,
            forward_fn=forward_fn,
            prepare_backend=prepare_backend,
        )

    def _prepare_decode_graph(
        self,
        state: TextDecodeGraphState,
        ctx: Any,
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None,
    ) -> None:
        if prepare_backend is not None:
            prepare_backend(state, ctx)

    def _record_decode_graph_event(
        self,
        *,
        ctx: Any,
        event: GraphEvent,
        batch_size: int,
        graph_batch_size: int,
    ) -> None:
        self.record_stats(
            ctx,
            event,
            batch_size=batch_size,
            graph_batch_size=graph_batch_size,
        )

    def maybe_run(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_plan: PagedDecodePlan,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
    ) -> torch.Tensor | None:
        """Capture-or-replay the decode graph for ``batch_size``; ``None`` on miss."""

        graph_batch_size = self.resolve_bucket(batch_size)
        if not self.can_use(graph_batch_size):
            self._record_decode_graph_miss(
                ctx=ctx,
                batch_size=batch_size,
                graph_batch_size=graph_batch_size,
            )
            return None

        return self._capture_or_replay(
            key=graph_batch_size,
            ctx=ctx,
            capture=lambda: self._capture_decode_graph(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                graph_batch_size=graph_batch_size,
                input_ids=input_ids,
                positions=positions,
                attention_plan=attention_plan,
                ctx=ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
            ),
            copy_inputs=lambda state: copy_text_decode_graph_inputs(
                state,
                input_ids=input_ids,
                positions=positions,
                attention_plan=attention_plan,
            ),
            replay=lambda state: _replay_decode_graph(state, batch_size),
            record=lambda event: self._record_decode_graph_event(
                ctx=ctx,
                event=event,
                batch_size=batch_size,
                graph_batch_size=graph_batch_size,
            ),
            disable=lambda exc: self.disable(graph_batch_size, exc, phase="disabling"),
            capture_metric=f"{self.metric_prefix}decode_graph_capture",
            input_copy_metric=f"{self.metric_prefix}decode_graph_input_copy",
            replay_metric=f"{self.metric_prefix}decode_graph_replay_launch",
            after_copy=lambda state: self._prepare_decode_graph(state, ctx, prepare_backend),
            after_copy_metric=f"{self.metric_prefix}decode_graph_attention_prepare",
        )

    def maybe_run_host_inputs(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        device: torch.device | str,
        host_inputs: TextDecodeGraphHostInputs,
        ctx: Any,
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
        staging_slot: Any | None = None,
    ) -> torch.Tensor | None:
        """Capture-or-replay the decode graph from host-staged inputs."""

        batch_size = int(host_inputs.batch_size)
        graph_batch_size = self.resolve_bucket(batch_size)
        if not self.can_use(graph_batch_size):
            self._record_decode_graph_miss(
                ctx=ctx,
                batch_size=batch_size,
                graph_batch_size=graph_batch_size,
            )
            return None

        return self._capture_or_replay(
            key=graph_batch_size,
            ctx=ctx,
            capture=lambda: self.capture_host_inputs(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                batch_size=graph_batch_size,
                device=device,
                host_inputs=host_inputs,
                ctx=ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
                staging_slot=staging_slot,
            ),
            copy_inputs=lambda state: copy_text_decode_graph_host_inputs(
                state,
                host_inputs,
                staging_slot=staging_slot,
            ),
            replay=lambda state: _replay_decode_graph(state, batch_size),
            record=lambda event: self._record_decode_graph_event(
                ctx=ctx,
                event=event,
                batch_size=batch_size,
                graph_batch_size=graph_batch_size,
            ),
            disable=lambda exc: self.disable(graph_batch_size, exc, phase="disabling"),
            capture_metric=f"{self.metric_prefix}decode_graph_capture",
            input_copy_metric=f"{self.metric_prefix}decode_graph_input_copy",
            replay_metric=f"{self.metric_prefix}decode_graph_replay_launch",
            after_copy=lambda state: self._prepare_decode_graph(state, ctx, prepare_backend),
            after_copy_metric=f"{self.metric_prefix}decode_graph_attention_prepare",
        )

    def warmup(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        device: torch.device,
        max_context_len: int = 0,
        attention_preference: str | None = "auto",
        forward_fn: Callable[[TextDecodeGraphState], torch.Tensor],
        prepare_backend: Callable[[TextDecodeGraphState, Any], None] | None = None,
    ) -> None:
        """Pre-capture decode graph buckets ahead of serving."""

        ctx = ForwardContext(attention_preference=attention_preference or "auto")

        def should_skip(batch_size: int) -> bool:
            return int(batch_size) <= 0 or int(batch_size) > int(num_blocks)

        def capture_bucket(batch_size: int, capture_ctx: ForwardContext) -> TextDecodeGraphState:
            batch_size = int(batch_size)
            input_ids, positions = _synthetic_decode_inputs(batch_size, device)
            plan = _synthetic_decode_plan(
                _synthetic_decode_cache(kv_pool, batch_size),
                batch_size,
                device,
                max_context_len=max_context_len,
            )
            return self.capture(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                batch_size=batch_size,
                input_ids=input_ids,
                positions=positions,
                attention_plan=plan,
                ctx=capture_ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
            )

        def copy_inputs(batch_size: int, state: TextDecodeGraphState) -> None:
            batch_size = int(batch_size)
            input_ids, positions = _synthetic_decode_inputs(batch_size, device)
            plan = _synthetic_decode_plan(
                state.cache,
                batch_size,
                device,
                max_context_len=max_context_len,
            )
            copy_text_decode_graph_inputs(
                state,
                input_ids=input_ids,
                positions=positions,
                attention_plan=plan,
            )

        self._warmup_capture_buckets(
            device=device,
            ctx=ctx,
            buckets=self.warmup_capture_batch_sizes,
            key_for=int,
            should_skip=should_skip,
            capture_bucket=capture_bucket,
            copy_inputs=copy_inputs,
            replay=lambda state: state.graph.replay(),
            disable=lambda batch_size, exc: self.disable(
                int(batch_size), exc, phase="skipping warmup for"
            ),
        )


def _synthetic_decode_inputs(
    batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (int(batch_size), 1)
    return (
        torch.zeros(shape, dtype=torch.long, device=device),
        torch.zeros(shape, dtype=torch.long, device=device),
    )


def _synthetic_decode_cache(kv_pool: PagedKVPool, batch_size: int) -> BatchedPagedRequestCache:
    batch_size = int(batch_size)
    return BatchedPagedRequestCache(
        kv_pool,
        [[row] for row in range(batch_size)],
        [0 for _ in range(batch_size)],
    )


def _replay_decode_graph(state: TextDecodeGraphState, batch_size: int) -> torch.Tensor:
    state.graph.replay()
    if state.logits is None:
        raise RuntimeError("captured decode graph has no logits buffer")
    return state.logits[:batch_size]


def _synthetic_decode_plan(
    cache: BatchedPagedRequestCache,
    batch_size: int,
    device: torch.device,
    *,
    max_context_len: int = 0,
) -> PagedDecodePlan:
    batch_size = int(batch_size)
    block_table = cache.block_table(device=device)
    cache_seqlens = cache.cache_seqlens(device=device)
    decode_page_ids, decode_page_offsets = decode_write_locations(
        block_table,
        cache_seqlens,
        cache.pool.block_size,
    )
    return PagedDecodePlan(
        residency_cache=cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=tuple(0 for _ in range(batch_size)),
        kv_seqlens=cache_seqlens + 1,
        query_lens=torch.ones(batch_size, dtype=torch.int32, device=device),
        query_lens_cpu=tuple(1 for _ in range(batch_size)),
        kv_seqlens_cpu=tuple(1 for _ in range(batch_size)),
        decode_page_ids=decode_page_ids,
        decode_page_offsets=decode_page_offsets,
        max_context_len=int(max_context_len),
    )


def make_text_decode_graph_state(
    *,
    kv_pool: PagedKVPool,
    num_blocks: int,
    batch_size: int,
    device: torch.device | str,
    buffer_pool: _GraphRunnerBase | None = None,
    max_context_len: int = 0,
) -> TextDecodeGraphState:
    """Allocate fixed buffers for a paged one-token decode graph bucket."""

    batch_size = int(batch_size)
    device = torch.device(device)
    max_blocks_per_seq = max(1, int(num_blocks))
    context_len = max(0, int(max_context_len))
    if context_len > 0:
        max_blocks_per_seq = max(
            1, min(max_blocks_per_seq, ceil_div(context_len, kv_pool.block_size))
        )
    if buffer_pool is not None:
        share = buffer_pool.share_graph_input_buffer
    else:
        share = _share_decode_graph_input_buffer
    graph_cache = BatchedPagedRequestCache(
        kv_pool,
        [[] for _ in range(batch_size)],
        [0 for _ in range(batch_size)],
    )
    long_inputs = share(
        "text_decode.long_inputs",
        torch.empty(4 * batch_size, dtype=torch.long, device=device),
    )
    block_table = share(
        "text_decode.block_table",
        torch.empty((batch_size, max_blocks_per_seq), dtype=torch.int32, device=device),
    )
    sequence_lens = share(
        "text_decode.sequence_lens",
        torch.empty(2 * batch_size, dtype=torch.int32, device=device),
    )
    cache_seqlens = sequence_lens[:batch_size]
    kv_seqlens = sequence_lens[batch_size:]
    state = TextDecodeGraphState(
        batch_size=batch_size,
        input_ids=long_inputs[:batch_size].view(batch_size, 1),
        positions=long_inputs[batch_size : 2 * batch_size].view(batch_size, 1),
        block_table=block_table,
        sequence_lens=sequence_lens,
        cache_seqlens=cache_seqlens,
        graph=torch.cuda.CUDAGraph(),
        cache=graph_cache,
        plan=PagedDecodePlan.for_decode_graph(
            residency_cache=graph_cache,
            batch_size=batch_size,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            kv_seqlens=kv_seqlens,
            query_lens=share(
                "text_decode.query_lens",
                torch.ones(batch_size, dtype=torch.int32, device=device),
            ),
            decode_page_ids=long_inputs[2 * batch_size : 3 * batch_size],
            decode_page_offsets=long_inputs[3 * batch_size : 4 * batch_size],
            max_context_len=max_context_len,
        ),
        graph_binding=GraphBinding(),
        long_inputs=long_inputs,
    )
    return state


def copy_text_decode_graph_inputs(
    state: TextDecodeGraphState,
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    attention_plan: PagedDecodePlan,
) -> None:
    """Refresh dynamic tensors feeding a captured paged one-token decode graph."""

    actual_batch = int(input_ids.shape[0])
    if actual_batch <= 0 or actual_batch > state.batch_size or int(input_ids.shape[1]) != 1:
        raise invalid_descriptor("decode CUDA graph input shape mismatch")
    state.input_ids[:actual_batch].copy_(input_ids, non_blocking=True)
    state.positions[:actual_batch].copy_(positions, non_blocking=True)
    if actual_batch < state.batch_size:
        state.input_ids[actual_batch:].zero_()
        state.positions[actual_batch:].zero_()
    source_cache = attention_plan.residency_cache
    if isinstance(state.cache, BatchedPagedRequestCache) and isinstance(
        source_cache, BatchedPagedRequestCache
    ):
        if (
            len(source_cache.block_ids_by_row) != actual_batch
            or len(source_cache.base_lens) != actual_batch
        ):
            raise invalid_descriptor("decode CUDA graph cache row batch mismatch")
        graph_block_rows = [list(row) for row in source_cache.block_ids_by_row]
        graph_base_lens = [int(length) for length in source_cache.base_lens]
        if actual_batch < state.batch_size:
            graph_block_rows.extend([] for _ in range(state.batch_size - actual_batch))
            graph_base_lens.extend(0 for _ in range(state.batch_size - actual_batch))
        state.cache.reset_rows(graph_block_rows, graph_base_lens)
    block_table = attention_plan.block_table
    if block_table is None:
        raise invalid_descriptor("decode CUDA graph block table is missing")
    if block_table.shape[0] != actual_batch:
        raise invalid_descriptor("decode CUDA graph block-table batch mismatch")
    if block_table.shape[1] > state.block_table.shape[1]:
        raise invalid_descriptor("decode CUDA graph block-table width exceeded")
    block_rows_key: tuple[tuple[int, ...], ...] | None = None
    if isinstance(source_cache, BatchedPagedRequestCache):
        block_rows_key = _block_rows_key(source_cache.block_ids_by_row[:actual_batch])
    if not _block_table_rows_match(state, block_rows_key):
        state.block_table[:actual_batch, : block_table.shape[1]].copy_(
            block_table.to(dtype=torch.int32),
            non_blocking=True,
        )
        if block_table.shape[1] < state.block_table.shape[1]:
            state.block_table[:actual_batch, block_table.shape[1] :].zero_()
        if actual_batch < state.batch_size:
            state.block_table[actual_batch:].zero_()
        state.block_table_rows = block_rows_key or ()
    cache_seqlens = attention_plan.cache_seqlens
    if cache_seqlens is None:
        raise invalid_descriptor("decode CUDA graph cache lengths are missing")
    if int(cache_seqlens.shape[0]) != actual_batch:
        raise invalid_descriptor("decode CUDA graph cache length batch mismatch")
    state.cache_seqlens[:actual_batch].copy_(cache_seqlens.to(dtype=torch.int32), non_blocking=True)
    graph_kv_seqlens = state.plan.kv_seqlens
    source_kv_seqlens = attention_plan.kv_seqlens
    if not isinstance(graph_kv_seqlens, torch.Tensor):
        raise invalid_descriptor("decode CUDA graph KV lengths are missing")
    if isinstance(source_kv_seqlens, torch.Tensor):
        if int(source_kv_seqlens.shape[0]) != actual_batch:
            raise invalid_descriptor("decode CUDA graph KV length batch mismatch")
        graph_kv_seqlens[:actual_batch].copy_(
            source_kv_seqlens.to(dtype=torch.int32),
            non_blocking=True,
        )
    else:
        graph_kv_seqlens[:actual_batch].copy_(state.cache_seqlens[:actual_batch], non_blocking=True)
        graph_kv_seqlens[:actual_batch].add_(1)
    if actual_batch < state.batch_size:
        state.cache_seqlens[actual_batch:].zero_()
        graph_kv_seqlens[actual_batch:].fill_(1)
    decode_page_ids = state.plan.decode_page_ids
    decode_page_offsets = state.plan.decode_page_offsets
    if (
        isinstance(decode_page_ids, torch.Tensor)
        and isinstance(decode_page_offsets, torch.Tensor)
        and isinstance(state.cache, BatchedPagedRequestCache)
    ):
        page_ids, offsets = decode_write_locations(
            state.block_table[:actual_batch],
            state.cache_seqlens[:actual_batch],
            state.cache.pool.block_size,
        )
        decode_page_ids[:actual_batch].copy_(page_ids, non_blocking=True)
        decode_page_offsets[:actual_batch].copy_(offsets, non_blocking=True)
        if actual_batch < state.batch_size:
            decode_page_ids[actual_batch:].zero_()
            decode_page_offsets[actual_batch:].zero_()
    cache_cpu = tuple(int(x) for x in attention_plan.cache_seqlens_cpu[:actual_batch])
    kv_cpu = tuple(int(x) for x in attention_plan.kv_seqlens_cpu[:actual_batch])
    if len(cache_cpu) != actual_batch:
        cache_cpu = ()
    if len(kv_cpu) != actual_batch:
        kv_cpu = ()
    if cache_cpu and actual_batch < state.batch_size:
        cache_cpu = cache_cpu + tuple(0 for _ in range(state.batch_size - actual_batch))
    if kv_cpu and actual_batch < state.batch_size:
        kv_cpu = kv_cpu + tuple(1 for _ in range(state.batch_size - actual_batch))
    # Publish a fresh frozen plan reusing the state's static device tensors with
    # the current per-replay CPU summaries. The graph reads the buffers (stable
    # addresses); graph-wrapper identity stays on the separate graph binding.
    state.plan = replace(
        state.plan,
        cache_seqlens_cpu=cache_cpu,
        kv_seqlens_cpu=kv_cpu,
        max_context_len=int(attention_plan.max_context_len or state.plan.max_context_len),
    )


def copy_text_decode_graph_host_inputs(
    state: TextDecodeGraphState,
    host_inputs: TextDecodeGraphHostInputs,
    *,
    staging_slot: Any | None = None,
) -> None:
    """Refresh decode graph inputs directly from host row descriptors."""

    rows = _normalize_text_decode_graph_host_inputs(state, host_inputs)
    actual_batch = len(rows["input_ids"])
    block_rows = rows["block_ids_by_row"]
    max_blocks = max(len(row) for row in block_rows)
    if max_blocks > int(state.block_table.shape[1]):
        raise invalid_descriptor("decode CUDA graph block-table width exceeded")

    cache_lens = rows["cache_seqlens_cpu"]
    kv_lens = rows["kv_seqlens_cpu"]
    decode_page_ids = state.plan.decode_page_ids
    decode_page_offsets = state.plan.decode_page_offsets
    page_ids: list[int] = []
    offsets: list[int] = []
    if isinstance(decode_page_ids, torch.Tensor) and isinstance(decode_page_offsets, torch.Tensor):
        page_ids, offsets = _host_decode_write_locations(
            block_rows,
            cache_lens,
            int(state.cache.pool.block_size),
        )

    dense_replacements = _dense_token_replacements(
        state,
        host_inputs.token_replacements,
        actual_batch=actual_batch,
    )
    copied_long = _copy_fused_long_host_inputs(
        state,
        input_ids=rows["input_ids"],
        positions=rows["positions"],
        page_ids=page_ids,
        page_offsets=offsets,
        include_input_ids=dense_replacements is None,
        actual_batch=actual_batch,
        slot=staging_slot,
    )
    if not copied_long:
        _copy_host_ints_to_device(
            rows["input_ids"],
            state.input_ids[:actual_batch],
            dtype=torch.long,
            slot=staging_slot,
            name="text_decode.input_ids",
            view_shape=(actual_batch, 1),
        )
        _copy_host_ints_to_device(
            rows["positions"],
            state.positions[:actual_batch],
            dtype=torch.long,
            slot=staging_slot,
            name="text_decode.positions",
            view_shape=(actual_batch, 1),
        )
        if isinstance(decode_page_ids, torch.Tensor) and isinstance(
            decode_page_offsets, torch.Tensor
        ):
            _copy_host_ints_to_device(
                page_ids,
                decode_page_ids[:actual_batch],
                dtype=torch.long,
                slot=staging_slot,
                name="text_decode.decode_page_ids",
            )
            _copy_host_ints_to_device(
                offsets,
                decode_page_offsets[:actual_batch],
                dtype=torch.long,
                slot=staging_slot,
                name="text_decode.decode_page_offsets",
            )
        if actual_batch < state.batch_size:
            state.input_ids[actual_batch:].zero_()
            state.positions[actual_batch:].zero_()
            if isinstance(decode_page_ids, torch.Tensor) and isinstance(
                decode_page_offsets, torch.Tensor
            ):
                decode_page_ids[actual_batch:].zero_()
                decode_page_offsets[actual_batch:].zero_()
    if dense_replacements is None:
        _copy_token_replacements(
            state,
            host_inputs.token_replacements,
            actual_batch=actual_batch,
        )
    else:
        _copy_dense_token_replacements(state, dense_replacements)

    block_rows_key = _block_rows_key(block_rows)
    if not _block_table_rows_match(state, block_rows_key):
        block_values: list[int] = []
        for row in block_rows:
            block_values.extend(row)
            block_values.extend(0 for _ in range(max_blocks - len(row)))
        _copy_host_ints_to_device(
            block_values,
            state.block_table[:actual_batch, :max_blocks],
            dtype=torch.int32,
            slot=staging_slot,
            name="text_decode.block_table",
            view_shape=(actual_batch, max_blocks),
        )
        if max_blocks < int(state.block_table.shape[1]):
            state.block_table[:actual_batch, max_blocks:].zero_()
        if actual_batch < state.batch_size:
            state.block_table[actual_batch:].zero_()
        state.block_table_rows = block_rows_key

    sequence_lens = list(cache_lens)
    sequence_lens.extend(0 for _ in range(state.batch_size - actual_batch))
    sequence_lens.extend(kv_lens)
    sequence_lens.extend(1 for _ in range(state.batch_size - actual_batch))
    _copy_host_ints_to_device(
        sequence_lens,
        state.sequence_lens,
        dtype=torch.int32,
        slot=staging_slot,
        name="text_decode.sequence_lens",
    )

    if isinstance(state.cache, BatchedPagedRequestCache):
        graph_block_rows = [list(row) for row in block_rows]
        graph_base_lens = [int(length) for length in cache_lens]
        if actual_batch < state.batch_size:
            graph_block_rows.extend([] for _ in range(state.batch_size - actual_batch))
            graph_base_lens.extend(0 for _ in range(state.batch_size - actual_batch))
        state.cache.reset_rows(graph_block_rows, graph_base_lens)

    cache_cpu = tuple(int(x) for x in rows["cache_seqlens_cpu"])
    kv_cpu = tuple(int(x) for x in kv_lens)
    if actual_batch < state.batch_size:
        cache_cpu = cache_cpu + tuple(0 for _ in range(state.batch_size - actual_batch))
        kv_cpu = kv_cpu + tuple(1 for _ in range(state.batch_size - actual_batch))
    state.plan = replace(
        state.plan,
        cache_seqlens_cpu=cache_cpu,
        kv_seqlens_cpu=kv_cpu,
        max_context_len=int(host_inputs.max_context_len or state.plan.max_context_len),
    )


def _normalize_text_decode_graph_host_inputs(
    state: TextDecodeGraphState,
    host_inputs: TextDecodeGraphHostInputs,
) -> dict[str, Any]:
    actual_batch = int(host_inputs.batch_size)
    if actual_batch <= 0 or actual_batch > int(state.batch_size):
        raise invalid_descriptor("decode CUDA graph input shape mismatch")
    if (
        len(host_inputs.positions) != actual_batch
        or len(host_inputs.block_ids_by_row) != actual_batch
        or len(host_inputs.cache_seqlens_cpu) != actual_batch
        or len(host_inputs.kv_seqlens_cpu) != actual_batch
    ):
        raise invalid_descriptor("decode CUDA graph host input row mismatch")
    block_rows = [[int(block_id) for block_id in row] for row in host_inputs.block_ids_by_row]
    if any(not row for row in block_rows):
        raise invalid_descriptor("decode CUDA graph host block rows must be non-empty")
    cache_lens = [int(length) for length in host_inputs.cache_seqlens_cpu]
    kv_lens = [int(length) for length in host_inputs.kv_seqlens_cpu]
    if any(length < 0 for length in cache_lens) or any(length <= 0 for length in kv_lens):
        raise invalid_descriptor("decode CUDA graph host sequence lengths are invalid")
    return {
        "input_ids": [int(token) for token in host_inputs.input_ids],
        "positions": [int(pos) for pos in host_inputs.positions],
        "block_ids_by_row": block_rows,
        "cache_seqlens_cpu": cache_lens,
        "kv_seqlens_cpu": kv_lens,
    }


def _copy_host_ints_to_device(
    values: Sequence[int],
    target: torch.Tensor,
    *,
    dtype: torch.dtype,
    slot: Any | None,
    name: str,
    view_shape: tuple[int, ...] | None = None,
) -> None:
    cpu = cpu_int_staging_buffer(
        len(values),
        dtype=dtype,
        pin=target.device.type == "cuda",
        slot=slot,
        name=name,
    )
    fill_cpu_ints(cpu, [int(value) for value in values])
    source = cpu if view_shape is None else cpu.view(*view_shape)
    target.copy_(source, non_blocking=target.device.type == "cuda" and is_pinned(cpu))


def _block_rows_key(rows: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(block_id) for block_id in row) for row in rows)


def _block_table_rows_match(
    state: TextDecodeGraphState,
    rows: tuple[tuple[int, ...], ...] | None,
) -> bool:
    return rows is not None and bool(rows) and state.block_table_rows == rows


def _copy_fused_long_host_inputs(
    state: TextDecodeGraphState,
    *,
    input_ids: Sequence[int],
    positions: Sequence[int],
    page_ids: Sequence[int],
    page_offsets: Sequence[int],
    include_input_ids: bool,
    actual_batch: int,
    slot: Any | None,
) -> bool:
    if not _long_inputs_match_state(state):
        return False
    actual_batch = int(actual_batch)
    batch = int(state.batch_size)
    if len(page_ids) > 0 and len(page_ids) != actual_batch:
        raise invalid_descriptor("decode CUDA graph page-id batch mismatch")
    if len(page_offsets) > 0 and len(page_offsets) != actual_batch:
        raise invalid_descriptor("decode CUDA graph page-offset batch mismatch")
    row_values: list[int] = []
    if include_input_ids:
        row_values.extend(int(value) for value in input_ids)
        row_values.extend(0 for _ in range(batch - actual_batch))
    row_values.extend(int(value) for value in positions)
    row_values.extend(0 for _ in range(batch - actual_batch))
    if len(page_ids) > 0:
        row_values.extend(int(value) for value in page_ids)
        row_values.extend(0 for _ in range(batch - actual_batch))
    if len(page_offsets) > 0:
        row_values.extend(int(value) for value in page_offsets)
        row_values.extend(0 for _ in range(batch - actual_batch))
    if not row_values:
        return True
    start = 0 if include_input_ids else batch
    if state.long_inputs is None:
        raise RuntimeError("captured decode graph has no host-input buffer")
    target = state.long_inputs[start : start + len(row_values)]
    _copy_host_ints_to_device(
        row_values,
        target,
        dtype=torch.long,
        slot=slot,
        name="text_decode.long_inputs",
    )
    return True


def _long_inputs_match_state(state: TextDecodeGraphState) -> bool:
    long_inputs = state.long_inputs
    if not isinstance(long_inputs, torch.Tensor):
        return False
    batch = int(state.batch_size)
    if int(long_inputs.numel()) < 4 * batch or long_inputs.dtype != torch.long:
        return False
    decode_page_ids = state.plan.decode_page_ids
    decode_page_offsets = state.plan.decode_page_offsets
    if not isinstance(decode_page_ids, torch.Tensor) or not isinstance(
        decode_page_offsets, torch.Tensor
    ):
        return False
    return (
        state.input_ids.data_ptr() == long_inputs[:batch].data_ptr()
        and state.positions.data_ptr() == long_inputs[batch : 2 * batch].data_ptr()
        and decode_page_ids.data_ptr() == long_inputs[2 * batch : 3 * batch].data_ptr()
        and decode_page_offsets.data_ptr() == long_inputs[3 * batch : 4 * batch].data_ptr()
    )


def _copy_token_replacements(
    state: TextDecodeGraphState,
    replacements: Sequence[tuple[int, torch.Tensor]],
    *,
    actual_batch: int,
) -> None:
    for row_idx, token in replacements:
        row = int(row_idx)
        if row < 0 or row >= actual_batch:
            raise invalid_descriptor("decode CUDA graph token replacement row is out of range")
        if token.dtype != torch.long or int(token.numel()) != 1:
            raise invalid_descriptor("decode CUDA graph token replacement must be one int64 token")
        if torch.device(token.device) != torch.device(state.input_ids.device):
            raise invalid_descriptor("decode CUDA graph token replacement device mismatch")
        state.input_ids[row, 0:1].copy_(token.reshape(1), non_blocking=True)


def _dense_token_replacements(
    state: TextDecodeGraphState,
    replacements: Sequence[tuple[int, torch.Tensor]],
    *,
    actual_batch: int,
) -> torch.Tensor | list[torch.Tensor] | None:
    if len(replacements) != int(actual_batch):
        return None
    tokens: list[torch.Tensor | None] = [None for _ in range(int(actual_batch))]
    for row_idx, token in replacements:
        row = int(row_idx)
        if row < 0 or row >= int(actual_batch):
            raise invalid_descriptor("decode CUDA graph token replacement row is out of range")
        if tokens[row] is not None:
            return None
        if token.dtype != torch.long or int(token.numel()) != 1:
            raise invalid_descriptor("decode CUDA graph token replacement must be one int64 token")
        if torch.device(token.device) != torch.device(state.input_ids.device):
            raise invalid_descriptor("decode CUDA graph token replacement device mismatch")
        tokens[row] = token.reshape(1)
    if any(token is None for token in tokens):
        return None
    dense = [token for token in tokens if token is not None]
    view = adjacent_one_token_view(dense)
    return view if view is not None else dense


def _copy_dense_token_replacements(
    state: TextDecodeGraphState,
    replacements: torch.Tensor | Sequence[torch.Tensor],
) -> None:
    if isinstance(replacements, torch.Tensor):
        source = replacements.reshape(-1)
        count = int(source.numel())
        if count <= 0:
            return
        state.input_ids[:count, 0].copy_(source, non_blocking=True)
        if count < int(state.batch_size):
            state.input_ids[count:].zero_()
        return
    if not replacements:
        return
    target = state.input_ids[: len(replacements), 0]
    if len(replacements) == 1:
        target.copy_(replacements[0], non_blocking=True)
        if len(replacements) < int(state.batch_size):
            state.input_ids[len(replacements) :].zero_()
        return
    torch.cat(tuple(replacements), out=target)
    if len(replacements) < int(state.batch_size):
        state.input_ids[len(replacements) :].zero_()


def _host_decode_write_locations(
    block_ids_by_row: Sequence[Sequence[int]],
    cache_lens: Sequence[int],
    page_size: int,
) -> tuple[list[int], list[int]]:
    page_size = max(1, int(page_size))
    page_ids: list[int] = []
    offsets: list[int] = []
    for row, length in zip(block_ids_by_row, cache_lens, strict=True):
        page_slot = int(length) // page_size
        if page_slot < 0 or page_slot >= len(row):
            raise invalid_descriptor("decode CUDA graph host row lacks write page")
        page_ids.append(int(row[page_slot]))
        offsets.append(int(length) % page_size)
    return page_ids, offsets


def resolve_paged_decode_graph_backend(attention_preference: str | None) -> Any | None:
    """Return the attention backend that can host a *captured* paged-decode graph.

    A paged-decode graph is only correct when its backend refills the page-index /
    length plan buffers before every replay, or when the backend has no wrapper
    plan state to bake into the graph. FlashInfer's wrapper path uses
    ``prepare_paged_decode_cuda_graph``; direct paged-decode backends consume the
    live block-table and sequence-length tensors directly, so a no-op prepare is
    enough. Other backends stay eager rather than capture a stale paged plan.
    """

    from uniserve_worker.backends.attention import (
        get_attention_backend,
        has_attention_backend,
        normalize_attention_backend_name,
    )

    normalized = normalize_attention_backend_name(attention_preference)
    if normalized == "auto" and has_attention_backend("trtllm_mha"):
        backend = get_attention_backend("trtllm_mha")
        if bool(getattr(backend.capabilities(), "available", True)):
            return backend
    if normalized == "trtllm_mha" and has_attention_backend("trtllm_mha"):
        backend = get_attention_backend("trtllm_mha")
        if bool(getattr(backend.capabilities(), "available", True)):
            return backend
        return None
    if normalized not in ("auto", "flashinfer"):
        if normalized != "fa4_cute":
            return None
        if not has_attention_backend("fa4_cute"):
            return None
        backend = get_attention_backend("fa4_cute")
        caps = backend.capabilities()
        if not bool(getattr(caps, "available", True)) or not bool(getattr(caps, "paged_kv", False)):
            return None
        if not hasattr(backend, "forward_paged"):
            return None
        return backend
    if has_attention_backend("flashinfer"):
        backend = get_attention_backend("flashinfer")
        if hasattr(backend, "prepare_paged_decode_cuda_graph"):
            return backend
    if normalized != "auto" or not has_attention_backend("fa4_cute"):
        return None
    backend = get_attention_backend("fa4_cute")
    caps = backend.capabilities()
    if not bool(getattr(caps, "available", True)) or not bool(getattr(caps, "paged_kv", False)):
        return None
    if not hasattr(backend, "forward_paged"):
        return None
    return backend


def resolve_paged_decode_graph_prepare(
    *,
    owner: Any,
    kv_pool: PagedKVPool,
    num_blocks: int,
    attention_preference: str | None,
    before: Callable[[TextDecodeGraphState, Any], None] | None = None,
) -> Callable[[TextDecodeGraphState, Any], None] | None:
    """Build the per-replay decode-graph prepare hook, or ``None`` to stay eager.

    The single assembly point for paged-decode graph preparation shared by the
    thin :class:`~uniserve_worker.execution.forward.graph.text.TextGraphRunner`
    and the interleaved decode adapter: KV-side geometry comes off the shared
    pool, query-side geometry from the owner's
    ``text_decode_graph_query_geometry`` hook, and the backend must expose
    graph-aware planning (otherwise the capture-time plan would be baked in and
    the caller must stay eager). ``before`` runs first on every capture/replay
    for caller-specific static state (e.g. the interleaved indexes sidecar).
    """

    geometry_hook = getattr(owner, "text_decode_graph_query_geometry", None)
    if not callable(geometry_hook):
        return None
    backend = resolve_paged_decode_graph_backend(attention_preference)
    if backend is None:
        return None
    caps = backend.capabilities()
    multiple = int(getattr(caps, "paged_block_size_multiple", 1) or 1)
    if int(kv_pool.block_size) % max(1, multiple) != 0:
        return None
    num_q_heads, scale, q_dtype = geometry_hook()
    num_blocks = int(num_blocks)

    def prepare(state: TextDecodeGraphState, ctx: Any) -> None:
        if before is not None:
            before(state, ctx)
        if hasattr(backend, "prepare_paged_decode_cuda_graph"):
            prepare_paged_decode_graph_backend(
                state,
                backend=backend,
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(kv_pool.n_kv),
                head_dim=int(kv_pool.head_dim),
                page_size=int(kv_pool.block_size),
                q_dtype=q_dtype,
                kv_dtype=kv_pool.k.dtype,
                scale=scale,
                max_indices=num_blocks * int(state.batch_size),
            )

    return prepare


def prepare_paged_decode_graph_backend(
    state: TextDecodeGraphState,
    *,
    backend: Any,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    scale: float | None,
    max_indices: int,
) -> None:
    """Refill ``backend``'s graph-decode plan buffers for the pending replay.

    Called from the runner's ``prepare_backend`` hook (outside the captured
    region) after ``copy_text_decode_graph_inputs`` has refreshed ``state``'s
    static block table + cache lengths. It re-plans the graph decode wrapper from
    ``state.plan`` so the captured ``wrapper.run`` reads
    the current pages/lengths — the mechanism that makes one capture correct across
    growth and across requests.
    """

    backend.prepare_paged_decode_cuda_graph(
        state.graph_binding,
        state.plan,
        batch_size=int(state.batch_size),
        max_indices=int(max_indices),
        num_q_heads=int(num_q_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        scale=scale,
    )


# ---------------------
# Text prefill graph runner
# ---------------------

_DEFAULT_PREFILL_GRAPH_BATCH_SIZES = (1, 2, 4, 8)
_PREFILL_GRAPH_PADDING_BLOCK_ID = 0


def _reset_append_plan(state: "TextInitialPrefillGraphState") -> None:
    """Drop the cache's staged append plan so capture/replay re-derives it."""

    state.cache._append_plan = None


@dataclass
class TextInitialPrefillGraphState:
    """Static buffers bound into a text prefill graph bucket."""

    num_tokens: int
    max_kv_tokens: int
    batch_size: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    query_lens: torch.Tensor
    kv_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    last_token_indices: torch.Tensor
    graph: torch.cuda.CUDAGraph
    cache: BatchedPagedRequestCache
    plan: PagedVarlenPlan
    graph_binding: GraphBinding
    logits: torch.Tensor | None = None
    release_backend: Callable[[], None] | None = None


class PrefillCudaGraphRunner(_GraphRunnerBase):
    """Own per-token-bucket initial-prefill graph state and its lifecycle."""

    def __init__(
        self,
        *,
        name: str,
        default_enabled: bool = False,
        default_warmup: bool = False,
        default_warmup_token_buckets: tuple[int, ...] = _DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
        default_warmup_batch_sizes: tuple[int, ...] = _DEFAULT_PREFILL_GRAPH_BATCH_SIZES,
        token_bucket_parser: Callable[[str], tuple[int, ...]] | None = None,
        metric_prefix: str = _DEFAULT_METRIC_PREFIX,
        logger: Any = None,
    ) -> None:
        self.name = str(name)
        self.default_enabled = bool(default_enabled)
        self.default_warmup = bool(default_warmup)
        self.default_warmup_token_buckets = tuple(
            sorted({int(size) for size in default_warmup_token_buckets if int(size) > 0})
        )
        self.default_warmup_batch_sizes = tuple(
            sorted({int(size) for size in default_warmup_batch_sizes if int(size) > 0})
        )
        self.token_bucket_parser = token_bucket_parser
        self.metric_prefix = str(metric_prefix)
        self.logger = logger
        self.states: dict[tuple[int, int, int], TextInitialPrefillGraphState] = {}
        self.disabled: set[tuple[int, int, int]] = set()
        self._capture_pool: Any = None
        self._graph_input_buffer_pool: dict[tuple[str, str, str], torch.Tensor] = {}

    def warmup_token_buckets(self) -> tuple[int, ...]:
        return self.default_warmup_token_buckets

    def warmup_capture_token_buckets(self) -> tuple[int, ...]:
        return tuple(reversed(self.warmup_token_buckets()))

    def warmup_batch_sizes(self) -> tuple[int, ...]:
        return self.default_warmup_batch_sizes

    def warmup_capture_batch_sizes(self) -> tuple[int, ...]:
        return tuple(reversed(self.warmup_batch_sizes()))

    def state_key(
        self, num_tokens: int, batch_size: int, max_kv_tokens: int
    ) -> tuple[int, int, int]:
        return (int(num_tokens), int(batch_size), int(max_kv_tokens))

    def has_state(self, num_tokens: int, batch_size: int, max_kv_tokens: int) -> bool:
        return self.state_key(num_tokens, batch_size, max_kv_tokens) in self.states

    def bucket_num_tokens(self, num_tokens: int) -> int:
        num_tokens = int(num_tokens)
        candidates = [int(size) for size in self.warmup_token_buckets() if int(size) >= num_tokens]
        if candidates:
            return min(candidates)
        return num_tokens

    def bucket_batch_size(self, batch_size: int, *, num_tokens: int | None = None) -> int:
        batch_size = int(batch_size)
        prefer_reusable = num_tokens is not None
        candidates = [int(size) for size in self.warmup_batch_sizes() if int(size) >= batch_size]
        if candidates:
            return max(candidates) if prefer_reusable else min(candidates)
        return batch_size

    def padding_batch_size(self, batch_size: int, *, num_tokens: int) -> int:
        """Return a graph row bucket with one isolated token-padding row."""

        batch_size = int(batch_size)
        configured = self.warmup_batch_sizes()
        if not configured or batch_size > max(configured):
            return batch_size
        return self.bucket_batch_size(batch_size + 1, num_tokens=int(num_tokens))

    def resolve_batch_size(self, num_tokens: int, batch_size: int, max_kv_tokens: int) -> int:
        batch_size = int(batch_size)
        candidates = [
            int(state_batch_size)
            for state_tokens, state_batch_size, state_kv_tokens in self.states
            if int(state_tokens) == int(num_tokens)
            and int(state_batch_size) >= batch_size
            and int(state_kv_tokens) == int(max_kv_tokens)
            and (int(state_tokens), int(state_batch_size), int(state_kv_tokens))
            not in self.disabled
        ]
        if candidates:
            return min(candidates)
        return self.bucket_batch_size(batch_size, num_tokens=num_tokens)

    def bucket_kv_tokens(self, max_kv_tokens: int, *, max_context_len: int = 0) -> int:
        max_kv_tokens = int(max_kv_tokens)
        context_len = int(max_context_len)
        if context_len > 0 and max_kv_tokens <= context_len:
            return context_len
        return self.bucket_num_tokens(max_kv_tokens)

    def can_use(
        self, num_tokens: int, batch_size: int = 1, max_kv_tokens: int | None = None
    ) -> bool:
        max_kv_tokens = int(max_kv_tokens if max_kv_tokens is not None else num_tokens)
        key = self.state_key(num_tokens, batch_size, max_kv_tokens)
        return (
            self.enabled()
            and key[0] > 0
            and key[1] > 0
            and key[2] >= key[0]
            and key not in self.disabled
        )

    def disable(
        self,
        num_tokens: int,
        exc: BaseException,
        *,
        phase: str,
        batch_size: int = 1,
        max_kv_tokens: int | None = None,
    ) -> None:
        max_kv_tokens = int(max_kv_tokens if max_kv_tokens is not None else num_tokens)
        key = self.state_key(num_tokens, batch_size, max_kv_tokens)
        self.disabled.add(key)
        state = self.states.pop(key, None)
        if state is not None and state.release_backend is not None:
            state.release_backend()
        if self.logger is not None:
            self.logger.warning(
                "%s %s prefill CUDA graph for %s query tokens x %s rows x %s kv tokens: %s",
                phase,
                self.name,
                key[0],
                key[1],
                key[2],
                exc,
            )

    @staticmethod
    def record_stats(
        ctx: Any,
        event: str,
        *,
        mode: ForwardMode,
        raw_tokens: int,
        padded_tokens: int,
    ) -> None:
        record_graph_stats(
            ctx,
            event,
            mode=mode,
            unpadded_tokens=raw_tokens,
            padded_tokens=padded_tokens,
        )

    def capture(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        num_tokens: int,
        max_kv_tokens: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_plan: PagedVarlenPlan,
        last_token_indices: torch.Tensor,
        raw_num_tokens: int,
        ctx: Any,
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
        prepare_backend: Callable[[TextInitialPrefillGraphState, Any], None] | None = None,
    ) -> TextInitialPrefillGraphState:
        """Capture an initial-prefill token bucket bound to the model forward."""

        device = input_ids.device
        state = make_text_initial_prefill_graph_state(
            kv_pool=kv_pool,
            num_blocks=num_blocks,
            num_tokens=int(num_tokens),
            max_kv_tokens=int(max_kv_tokens),
            batch_size=int(batch_size),
            device=device,
            max_context_len=int(getattr(attention_plan, "max_context_len", 0) or 0),
        )
        copy_text_initial_prefill_graph_inputs(
            state,
            input_ids=input_ids,
            positions=positions,
            attention_plan=attention_plan,
            raw_num_tokens=raw_num_tokens,
            last_token_indices=last_token_indices,
        )
        graph_ctx = replace(
            ctx,
            attention_plan=state.plan,
            graph_binding=state.graph_binding,
            stats=None,
        )
        bind_paged_prefill_graph_wrapper(state, graph_ctx)

        def run() -> torch.Tensor:
            # copy_inputs rebuilds state.plan before each warmup and capture run;
            # publish that snapshot so capture records the same plan the backend
            # prepare step plans from.
            with use_forward_context(replace(graph_ctx, attention_plan=state.plan)):
                return forward_fn(state)

        def copy_inputs(capture_state: TextInitialPrefillGraphState) -> None:
            copy_text_initial_prefill_graph_inputs(
                capture_state,
                input_ids=input_ids,
                positions=positions,
                attention_plan=attention_plan,
                raw_num_tokens=raw_num_tokens,
                last_token_indices=last_token_indices,
            )

        def prepare(capture_state: TextInitialPrefillGraphState) -> None:
            _reset_append_plan(capture_state)
            if prepare_backend is not None:
                prepare_backend(capture_state, graph_ctx)

        try:
            captured = self._capture_graph_state(
                device=device,
                state=state,
                run=run,
                copy_inputs=copy_inputs,
                before_run=prepare,
            )
            assert_paged_prefill_graph_wrapper_planned(captured, graph_ctx)
            return captured
        except BaseException:
            release = state.release_backend
            state.release_backend = None
            if callable(release):
                release()
            raise

    def _prepare_prefill_graph(
        self,
        state: TextInitialPrefillGraphState,
        ctx: Any,
        prepare_backend: Callable[[TextInitialPrefillGraphState, Any], None] | None,
    ) -> None:
        _reset_append_plan(state)
        if prepare_backend is not None:
            prepare_backend(state, ctx)

    def maybe_run(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        num_tokens: int,
        max_kv_tokens: int,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_plan: PagedVarlenPlan,
        last_token_indices: torch.Tensor,
        raw_num_tokens: int,
        ctx: Any,
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
        prepare_backend: Callable[[TextInitialPrefillGraphState, Any], None] | None = None,
    ) -> torch.Tensor | None:
        """Capture-or-replay an initial-prefill bucket; ``None`` on miss/fallback."""

        graph_batch_size = self.resolve_batch_size(num_tokens, int(batch_size), max_kv_tokens)
        graph_key = self.state_key(num_tokens, graph_batch_size, max_kv_tokens)
        if graph_key not in self.states and self.warmup_enabled():
            self.record_stats(
                ctx,
                GraphEvent.MISS,
                mode=ForwardMode.EXTEND,
                raw_tokens=raw_num_tokens,
                padded_tokens=num_tokens,
            )
            return None

        def capture() -> TextInitialPrefillGraphState:
            return self.capture(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                num_tokens=num_tokens,
                max_kv_tokens=max_kv_tokens,
                batch_size=graph_batch_size,
                input_ids=input_ids,
                positions=positions,
                attention_plan=attention_plan,
                last_token_indices=last_token_indices,
                raw_num_tokens=raw_num_tokens,
                ctx=ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
            )

        def copy_inputs(state: TextInitialPrefillGraphState) -> None:
            copy_text_initial_prefill_graph_inputs(
                state,
                input_ids=input_ids,
                positions=positions,
                attention_plan=attention_plan,
                raw_num_tokens=raw_num_tokens,
                last_token_indices=last_token_indices,
            )

        def replay(state: TextInitialPrefillGraphState) -> torch.Tensor:
            state.graph.replay()
            if state.logits is None:
                raise RuntimeError("captured prefill graph has no logits buffer")
            return _slice_prefill_graph_logits(state.logits, batch_size)

        def record(event: GraphEvent) -> None:
            self.record_stats(
                ctx,
                event,
                mode=ForwardMode.EXTEND,
                raw_tokens=raw_num_tokens,
                padded_tokens=num_tokens,
            )

        return self._capture_or_replay(
            key=graph_key,
            ctx=ctx,
            capture=capture,
            copy_inputs=copy_inputs,
            replay=replay,
            record=record,
            disable=lambda exc: self.disable(
                num_tokens,
                exc,
                phase="disabling",
                batch_size=graph_batch_size,
                max_kv_tokens=max_kv_tokens,
            ),
            capture_metric=f"{self.metric_prefix}prefill_graph_capture",
            input_copy_metric=f"{self.metric_prefix}prefill_graph_input_copy",
            replay_metric=f"{self.metric_prefix}prefill_graph_replay_launch",
            after_copy=lambda state: self._prepare_prefill_graph(state, ctx, prepare_backend),
            after_copy_metric=f"{self.metric_prefix}prefill_graph_attention_prepare",
        )

    def warmup(
        self,
        *,
        kv_pool: PagedKVPool,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        max_context_len: int = 0,
        attention_preference: str | None = "auto",
        forward_fn: Callable[[TextInitialPrefillGraphState], torch.Tensor],
        prepare_backend: Callable[[TextInitialPrefillGraphState, Any], None] | None = None,
    ) -> None:
        """Pre-capture initial-prefill token/batch buckets ahead of serving."""

        max_tokens = int(num_blocks) * int(block_size)
        ctx = ForwardContext(attention_preference=attention_preference or "auto")

        def max_kv_bucket() -> int:
            context_len = int(max_context_len)
            if context_len > 0:
                return min(max_tokens, context_len)
            return 0

        def key_for(bucket: tuple[int, int]) -> tuple[int, int, int]:
            num_tokens, batch_size = bucket
            kv_tokens = max_kv_bucket() or int(num_tokens)
            return self.state_key(int(num_tokens), int(batch_size), int(kv_tokens))

        def should_skip(bucket: tuple[int, int]) -> bool:
            num_tokens, batch_size = bucket
            num_tokens = int(num_tokens)
            batch_size = int(batch_size)
            if num_tokens <= 0 or batch_size <= 0 or num_tokens > max_tokens:
                return True
            return ceil_div(num_tokens, block_size) > int(num_blocks)

        def capture_bucket(
            bucket: tuple[int, int],
            capture_ctx: ForwardContext,
        ) -> TextInitialPrefillGraphState:
            num_tokens, batch_size = bucket
            num_tokens = int(num_tokens)
            batch_size = int(batch_size)
            kv_tokens = max_kv_bucket() or num_tokens
            inputs = _synthetic_initial_prefill_inputs(
                kv_pool=kv_pool,
                num_tokens=num_tokens,
                max_kv_tokens=kv_tokens,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
                max_context_len=max_context_len,
            )
            return self.capture(
                kv_pool=kv_pool,
                num_blocks=num_blocks,
                num_tokens=num_tokens,
                max_kv_tokens=kv_tokens,
                batch_size=batch_size,
                input_ids=inputs.input_ids,
                positions=inputs.positions,
                attention_plan=inputs.plan,
                last_token_indices=inputs.last_token_indices,
                raw_num_tokens=num_tokens,
                ctx=capture_ctx,
                forward_fn=forward_fn,
                prepare_backend=prepare_backend,
            )

        def copy_inputs(bucket: tuple[int, int], state: TextInitialPrefillGraphState) -> None:
            num_tokens = int(bucket[0])
            inputs = _synthetic_initial_prefill_inputs(
                kv_pool=kv_pool,
                num_tokens=num_tokens,
                max_kv_tokens=state.max_kv_tokens,
                batch_size=state.batch_size,
                block_size=block_size,
                device=device,
                max_context_len=max_context_len,
            )
            copy_text_initial_prefill_graph_inputs(
                state,
                input_ids=inputs.input_ids,
                positions=inputs.positions,
                attention_plan=inputs.plan,
                raw_num_tokens=num_tokens,
                last_token_indices=inputs.last_token_indices,
            )

        def replay_state(state: TextInitialPrefillGraphState) -> None:
            self._prepare_prefill_graph(state, ctx, prepare_backend)
            state.graph.replay()

        self._warmup_capture_buckets(
            device=device,
            ctx=ctx,
            buckets=self.warmup_capture_buckets,
            key_for=key_for,
            should_skip=should_skip,
            capture_bucket=capture_bucket,
            copy_inputs=copy_inputs,
            replay=replay_state,
            disable=lambda bucket, exc: self.disable(
                int(bucket[0]),
                exc,
                phase="skipping warmup for",
                batch_size=int(bucket[1]),
                max_kv_tokens=max_kv_bucket() or int(bucket[0]),
            ),
        )

    def warmup_capture_buckets(self) -> tuple[tuple[int, int], ...]:
        configured = self.warmup_batch_sizes()
        if not configured:
            return ()
        max_batch_size = max(configured)
        return tuple(
            (
                int(num_tokens),
                max_batch_size,
            )
            for num_tokens in self.warmup_capture_token_buckets()
        )


@dataclass(frozen=True)
class _SyntheticInitialPrefillInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    plan: PagedVarlenPlan
    last_token_indices: torch.Tensor


def _synthetic_initial_prefill_inputs(
    *,
    kv_pool: PagedKVPool,
    num_tokens: int,
    max_kv_tokens: int | None = None,
    batch_size: int = 1,
    block_size: int,
    device: torch.device,
    max_context_len: int = 0,
) -> _SyntheticInitialPrefillInputs:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise invalid_descriptor("synthetic prefill graph batch size must be positive")
    query_lens = _synthetic_query_lens(int(num_tokens), batch_size)
    block_ids_by_row: list[list[int]] = []
    next_block = 0
    for query_len in query_lens:
        block_count = ceil_div(query_len, block_size)
        block_ids_by_row.append(list(range(next_block, next_block + block_count)))
        next_block += block_count
    cache = BatchedPagedRequestCache(kv_pool, block_ids_by_row, [0] * batch_size)
    positions = torch.cat(
        [torch.arange(length, dtype=torch.long, device=device) for length in query_lens],
        dim=0,
    )
    last_indices = []
    offset = 0
    for length in query_lens:
        last_indices.append(offset + int(length) - 1 if int(length) > 0 else 0)
        offset += int(length)
    return _SyntheticInitialPrefillInputs(
        input_ids=torch.zeros(num_tokens, dtype=torch.long, device=device),
        positions=positions,
        plan=_synthetic_initial_prefill_plan(
            cache,
            num_tokens,
            batch_size,
            device,
            max_context_len=max_context_len,
        ),
        last_token_indices=torch.tensor(last_indices, dtype=torch.long, device=device),
    )


def _synthetic_initial_prefill_plan(
    cache: BatchedPagedRequestCache,
    num_tokens: int,
    batch_size: int,
    device: torch.device,
    max_context_len: int = 0,
) -> PagedVarlenPlan:
    query_lens_cpu = _synthetic_query_lens(int(num_tokens), int(batch_size))
    query_lens = torch.tensor(query_lens_cpu, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(query_lens, dim=0).to(torch.int32),
        ]
    )
    return PagedVarlenPlan(
        residency_cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=tuple(0 for _ in range(int(batch_size))),
        query_lens=query_lens,
        query_lens_cpu=query_lens_cpu,
        kv_seqlens=query_lens,
        kv_seqlens_cpu=query_lens_cpu,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(query_lens_cpu, default=0),
        max_seqlen_k=max(query_lens_cpu, default=0),
        max_context_len=int(max_context_len),
        mode=ForwardMode.EXTEND,
    )


def _synthetic_query_lens(num_tokens: int, batch_size: int) -> tuple[int, ...]:
    num_tokens = int(num_tokens)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise invalid_descriptor("synthetic prefill graph batch size must be positive")
    if num_tokens <= 0:
        raise invalid_descriptor("synthetic prefill graph token count must be positive")
    return (num_tokens,) + tuple(0 for _ in range(batch_size - 1))


def make_text_initial_prefill_graph_state(
    *,
    kv_pool: PagedKVPool,
    num_blocks: int,
    num_tokens: int,
    max_kv_tokens: int | None = None,
    batch_size: int = 1,
    device: torch.device | str,
    max_context_len: int = 0,
) -> TextInitialPrefillGraphState:
    """Allocate fixed buffers for a text prefill graph bucket."""

    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise invalid_descriptor("prefill graph token bucket must be positive")
    max_kv_tokens = int(max_kv_tokens if max_kv_tokens is not None else num_tokens)
    if max_kv_tokens < num_tokens:
        raise invalid_descriptor("prefill graph KV bucket must cover the query bucket")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise invalid_descriptor("prefill graph batch size must be positive")
    device = torch.device(device)
    max_blocks_per_seq = max(
        1,
        min(int(num_blocks), ceil_div(max_kv_tokens, kv_pool.block_size)),
    )
    graph_cache = BatchedPagedRequestCache(
        kv_pool, [[] for _ in range(batch_size)], [0] * batch_size
    )
    query_lens_cpu = (num_tokens,) + tuple(0 for _ in range(batch_size - 1))
    input_ids = torch.empty(num_tokens, dtype=torch.long, device=device)
    positions = torch.empty(num_tokens, dtype=torch.long, device=device)
    block_table = torch.empty((batch_size, max_blocks_per_seq), dtype=torch.int32, device=device)
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    query_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    kv_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    last_token_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    query_lens[0] = num_tokens
    kv_seqlens[0] = num_tokens
    cu_seqlens_q[1:] = num_tokens
    cu_seqlens_k[1:] = num_tokens
    state = TextInitialPrefillGraphState(
        num_tokens=num_tokens,
        max_kv_tokens=max_kv_tokens,
        batch_size=batch_size,
        input_ids=input_ids,
        positions=positions,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        query_lens=query_lens,
        kv_seqlens=kv_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        last_token_indices=last_token_indices,
        graph=torch.cuda.CUDAGraph(),
        cache=graph_cache,
        plan=PagedVarlenPlan(
            residency_cache=graph_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cache_seqlens_cpu=tuple(0 for _ in range(batch_size)),
            query_lens=query_lens,
            query_lens_cpu=query_lens_cpu,
            kv_seqlens=kv_seqlens,
            kv_seqlens_cpu=query_lens_cpu,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=num_tokens,
            max_seqlen_k=max_kv_tokens,
            max_context_len=int(max_context_len),
            mode=ForwardMode.EXTEND,
        ),
        graph_binding=GraphBinding(),
    )
    return state


def copy_text_initial_prefill_graph_inputs(
    state: TextInitialPrefillGraphState,
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    attention_plan: PagedVarlenPlan,
    raw_num_tokens: int,
    last_token_indices: torch.Tensor | None = None,
) -> None:
    """Refresh dynamic tensors feeding a captured prefill graph."""

    raw_num_tokens = int(raw_num_tokens)
    if raw_num_tokens <= 0 or raw_num_tokens > state.num_tokens:
        raise invalid_descriptor("prefill graph raw token count mismatch")
    if int(input_ids.numel()) > state.num_tokens or int(positions.numel()) > state.num_tokens:
        raise invalid_descriptor("prefill graph input token bucket exceeded")
    flat_ids = input_ids.reshape(-1)
    flat_positions = positions.reshape(-1)
    state.input_ids[: flat_ids.numel()].copy_(flat_ids, non_blocking=True)
    state.positions[: flat_positions.numel()].copy_(flat_positions, non_blocking=True)
    if int(flat_ids.numel()) < state.num_tokens:
        state.input_ids[flat_ids.numel() :].zero_()
    if int(flat_positions.numel()) < state.num_tokens:
        state.positions[flat_positions.numel() :].zero_()
    block_table = attention_plan.block_table
    if block_table is None:
        raise invalid_descriptor("prefill CUDA graph block table is missing")
    real_rows = int(block_table.shape[0])
    if real_rows <= 0 or real_rows > state.batch_size:
        raise invalid_descriptor("prefill CUDA graph row count mismatch")
    if int(block_table.shape[1]) > int(state.block_table.shape[1]):
        raise invalid_descriptor("prefill CUDA graph block-table width exceeded")
    state.block_table[:real_rows, : block_table.shape[1]].copy_(
        block_table.to(dtype=torch.int32),
        non_blocking=True,
    )
    if int(block_table.shape[1]) < int(state.block_table.shape[1]):
        state.block_table[:real_rows, block_table.shape[1] :].zero_()
    if real_rows < state.batch_size:
        state.block_table[real_rows:].zero_()
    cache_seqlens = attention_plan.cache_seqlens
    if not isinstance(cache_seqlens, torch.Tensor):
        raise invalid_descriptor("prefill CUDA graph cache-seqlens tensor is missing")
    if int(cache_seqlens.numel()) != real_rows:
        raise invalid_descriptor("prefill CUDA graph cache-seqlens row count mismatch")
    state.cache_seqlens[:real_rows].copy_(cache_seqlens.to(dtype=torch.int32), non_blocking=True)
    if real_rows < state.batch_size:
        state.cache_seqlens[real_rows:].zero_()
    cache_lens_cpu = tuple(int(length) for length in attention_plan.cache_seqlens_cpu)
    if len(cache_lens_cpu) != real_rows:
        raise invalid_descriptor("prefill CUDA graph cache length count mismatch")
    graph_cache_lens_cpu = cache_lens_cpu + tuple(0 for _ in range(state.batch_size - real_rows))
    raw_lens = tuple(int(length) for length in attention_plan.query_lens_cpu)
    if len(raw_lens) != real_rows:
        raise invalid_descriptor("prefill CUDA graph query lengths mismatch")
    if sum(raw_lens) != raw_num_tokens:
        raise invalid_descriptor("prefill CUDA graph raw token count does not match query lengths")
    graph_lens = list(raw_lens) + [0 for _ in range(state.batch_size - real_rows)]
    source_cache = attention_plan.residency_cache
    block_ids_by_row = getattr(source_cache, "block_ids_by_row", None)
    padding_tokens = int(state.num_tokens) - raw_num_tokens
    padding_row = real_rows - 1
    if padding_tokens > 0:
        graph_lens[padding_row] += padding_tokens
    if block_ids_by_row is not None:
        graph_block_ids = [list(row) for row in block_ids_by_row] + [
            [] for _ in range(state.batch_size - real_rows)
        ]
        if padding_tokens > 0:
            required_blocks = ceil_div(
                int(graph_cache_lens_cpu[padding_row]) + int(graph_lens[padding_row]),
                int(state.cache.pool.block_size),
            )
            missing_blocks = required_blocks - len(graph_block_ids[padding_row])
            if missing_blocks > 0:
                graph_block_ids[padding_row].extend(
                    [_PREFILL_GRAPH_PADDING_BLOCK_ID] * missing_blocks
                )
            if len(graph_block_ids[padding_row]) > int(state.block_table.shape[1]):
                raise invalid_descriptor(
                    "prefill CUDA graph sink tail exceeds block-table capacity"
                )
        state.cache.reset_rows(graph_block_ids, graph_cache_lens_cpu)
    graph_lens_tuple = tuple(int(length) for length in graph_lens)
    kv_lens_tuple = tuple(
        int(base) + int(query)
        for base, query in zip(graph_cache_lens_cpu, graph_lens_tuple, strict=True)
    )
    if max(kv_lens_tuple, default=0) > int(state.max_kv_tokens):
        raise invalid_descriptor("prefill CUDA graph KV bucket exceeded")
    query_lens = attention_plan.query_lens
    if not isinstance(query_lens, torch.Tensor):
        raise invalid_descriptor("prefill CUDA graph query lens tensor is missing")
    if int(query_lens.numel()) != real_rows:
        raise invalid_descriptor("prefill CUDA graph query lens tensor count mismatch")
    state.query_lens[:real_rows].copy_(query_lens.to(dtype=torch.int32), non_blocking=True)
    if real_rows < state.batch_size:
        state.query_lens[real_rows:].zero_()
    if padding_tokens > 0:
        state.query_lens[padding_row] += padding_tokens
    state.kv_seqlens.copy_(state.cache_seqlens, non_blocking=True)
    state.kv_seqlens.add_(state.query_lens)
    state.cu_seqlens_q[:1].zero_()
    torch.cumsum(state.query_lens, dim=0, out=state.cu_seqlens_q[1:])
    state.cu_seqlens_k[:1].zero_()
    torch.cumsum(state.kv_seqlens, dim=0, out=state.cu_seqlens_k[1:])
    # Publish a fresh frozen plan reusing the state's static device tensors with
    # the current per-replay CPU summaries and varlen extents. The graph reads
    # the buffers (stable addresses); wrapper identity stays on the binding.
    state.plan = replace(
        state.plan,
        cache_seqlens_cpu=graph_cache_lens_cpu,
        query_lens_cpu=graph_lens_tuple,
        kv_seqlens_cpu=kv_lens_tuple,
        max_seqlen_q=max(graph_lens_tuple, default=state.num_tokens),
        max_seqlen_k=state.max_kv_tokens,
        max_context_len=int(attention_plan.max_context_len or state.plan.max_context_len),
    )
    if last_token_indices is None:
        if state.batch_size != 1:
            raise invalid_descriptor("multi-row prefill CUDA graph requires last-token indices")
        state.last_token_indices[:1].fill_(raw_num_tokens - 1)
    else:
        flat_indices = last_token_indices.reshape(-1)
        if int(flat_indices.numel()) != real_rows:
            raise invalid_descriptor("prefill CUDA graph last-token index count mismatch")
        state.last_token_indices[:real_rows].copy_(
            flat_indices.to(dtype=torch.long), non_blocking=True
        )
        if real_rows < state.batch_size:
            state.last_token_indices[real_rows:].zero_()


def resolve_paged_prefill_graph_prepare(
    *,
    owner: Any,
    kv_pool: PagedKVPool,
    attention_preference: str | None,
    before: Callable[[TextInitialPrefillGraphState, Any], None] | None = None,
) -> Callable[[TextInitialPrefillGraphState, Any], None] | None:
    """Build the per-replay prefill-graph prepare hook, or ``None`` to stay eager."""

    backend = _resolve_graph_prefill_backend(
        ForwardContext(attention_preference=attention_preference)
    )
    if backend is None:
        return None
    prepare = getattr(backend, "prepare_paged_prefill_cuda_graph", None)
    if callable(prepare):
        geometry_hook = getattr(owner, "text_decode_graph_query_geometry", None)
        if not callable(geometry_hook):
            return None
        num_q_heads, scale, q_dtype = geometry_hook()

        def prepare_with_backend(state: TextInitialPrefillGraphState, ctx: Any) -> None:
            if before is not None:
                before(state, ctx)
            bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
            release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
            if state.release_backend is None and callable(bind) and callable(release):
                bind(state.graph_binding, state.plan, device=state.input_ids.device)
                state.release_backend = lambda: release(state.graph_binding)
            prepare_paged_prefill_graph_backend(
                state,
                backend=backend,
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(kv_pool.n_kv),
                head_dim=int(kv_pool.head_dim),
                page_size=int(kv_pool.block_size),
                q_dtype=q_dtype,
                kv_dtype=kv_pool.k.dtype,
                causal=True,
                scale=scale,
            )

        return prepare_with_backend
    try:
        caps = backend.capabilities()
    except Exception:
        return None
    if not bool(getattr(caps, "paged_varlen_cuda_graph", False)):
        return None

    def prepare_direct_graph_backend(state: TextInitialPrefillGraphState, ctx: Any) -> None:
        if before is not None:
            before(state, ctx)

    return prepare_direct_graph_backend


def prepare_paged_prefill_graph_backend(
    state: TextInitialPrefillGraphState,
    *,
    backend: Any,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    causal: bool,
    scale: float | None,
) -> None:
    """Refresh ``backend``'s graph-prefill plan buffers for the pending replay."""

    backend.prepare_paged_prefill_cuda_graph(
        state.graph_binding,
        state.plan,
        num_q_heads=int(num_q_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        causal=bool(causal),
        scale=scale,
    )


def bind_paged_prefill_graph_wrapper(state: TextInitialPrefillGraphState, ctx: Any) -> None:
    """Bind a graph-scoped paged-prefill wrapper when the selected backend owns one."""

    backend = _resolve_graph_prefill_backend(ctx)
    bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
    release = getattr(backend, "release_paged_prefill_graph_wrapper", None)
    if not callable(bind) or not callable(release):
        return
    bind(state.graph_binding, state.plan, device=state.input_ids.device)
    state.release_backend = lambda: release(state.graph_binding)


def assert_paged_prefill_graph_wrapper_planned(
    state: TextInitialPrefillGraphState, ctx: Any
) -> None:
    if state.release_backend is None:
        return
    backend = _resolve_graph_prefill_backend(ctx)
    planned = getattr(backend, "paged_prefill_graph_wrapper_planned", None)
    if callable(planned) and not planned(state.graph_binding):
        raise invalid_descriptor(
            "captured prefill forward did not plan the graph-scoped prefill backend"
        )


def _resolve_graph_prefill_backend(ctx: Any) -> Any:
    backend = getattr(ctx, "attention_backend", None)
    if backend is not None:
        return backend
    try:
        from uniserve_worker.backends.attention import (
            get_attention_backend,
            has_attention_backend,
            normalize_attention_backend_name,
        )

        name = normalize_attention_backend_name(getattr(ctx, "attention_preference", None))
        if name == "auto" and has_attention_backend("trtllm_mha"):
            backend = get_attention_backend("trtllm_mha")
            if bool(getattr(backend.capabilities(), "available", True)):
                return backend
        if name == "trtllm_mha" and has_attention_backend("trtllm_mha"):
            backend = get_attention_backend("trtllm_mha")
            if bool(getattr(backend.capabilities(), "available", True)):
                return backend
            return None
        if name == "flashinfer" or (name == "auto" and has_attention_backend("flashinfer")):
            return get_attention_backend("flashinfer")
    except Exception:
        return None
    return None


def _slice_prefill_graph_logits(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    batch_size = int(batch_size)
    if batch_size <= 0 or int(logits.shape[0]) == batch_size:
        return logits
    return logits[:batch_size]


# ---------------------
# Combined text graph runner
# ---------------------

logger = logging.getLogger(__name__)



class TextGraphRunner:
    """Owns the decode + text-prefill CUDA graphs, keyed on the system pool."""

    def __init__(
        self,
        *,
        kv_pool: "PagedKVPool",
        num_blocks: int,
        block_size: int,
        device: "torch.device",
        attention_preference: str | None = None,
        max_context_len: int = 0,
    ) -> None:
        self.kv_pool = kv_pool
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.device = device
        self.max_context_len = max(0, int(max_context_len))
        # Startup (warmup) has no forward context to read the backend name from;
        # per-forward calls prefer the context's resolved name.
        self.attention_preference = attention_preference
        runtime = get_execution_config()
        self._decode = DecodeCudaGraphRunner(
            name="text",
            default_enabled=runtime.cuda_graph,
            default_warmup=runtime.cuda_graph_warmup,
            default_warmup_batch_sizes=runtime.cuda_graph_warmup_batches,
            metric_prefix="text_",
            logger=logger,
        )
        self._prefill = PrefillCudaGraphRunner(
            name="text",
            default_enabled=runtime.prefill_cuda_graph,
            default_warmup=runtime.prefill_cuda_graph_warmup,
            default_warmup_token_buckets=runtime.prefill_cuda_graph_warmup_tokens,
            metric_prefix="text_",
            logger=logger,
        )

    # ---- capture-or-replay --------------------------------------------------

    def maybe_run(
        self,
        model: Any,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        fb: ForwardBatch,
        ctx: Any,
    ) -> "torch.Tensor | None":
        """Replay (or capture) the graph for this forward; ``None`` to fall back."""

        plan = fb.attn_plan
        if not isinstance(getattr(plan, "residency_cache", None), BatchedPagedRequestCache):
            return None
        if getattr(input_ids, "device", None) is None or input_ids.device.type != "cuda":
            return None
        if fb.forward_mode == ForwardMode.DECODE:
            return self._maybe_decode(model, input_ids, positions, fb, plan, ctx)
        if fb.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            # A mixed extend+decode group is shape-identical to a cached-prefix
            # extend group (flat varlen rows, per-row context lengths, per-row
            # last-token sampling), so it replays the same prefill buckets.
            return self._maybe_prefill(model, input_ids, positions, fb, plan, ctx)
        return None

    def _maybe_decode(self, model, input_ids, positions, fb, plan, ctx):
        if not self._decode.enabled():
            return None
        if input_ids.ndim != 2 or int(input_ids.shape[1]) != 1:
            return None
        batch_size = int(input_ids.shape[0])
        prepare_backend = self._decode_prepare_backend(
            model,
            attention_preference=getattr(ctx, "attention_preference", None)
            or self.attention_preference,
        )
        if prepare_backend is None:
            # A captured paged-decode graph is only correct when the backend can
            # refill its plan buffers before every replay; without that, the
            # capture-time plan is baked in and later steps silently read stale
            # pages. Stay eager rather than capture a wrong graph.
            return None
        return self._decode.maybe_run(
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_plan=plan,
            ctx=ctx,
            forward_fn=lambda state: self._decode_forward(model, state),
            prepare_backend=prepare_backend,
        )

    def _decode_prepare_backend(
        self,
        model: Any,
        *,
        attention_preference: str | None,
    ) -> Any | None:
        return resolve_paged_decode_graph_prepare(
            owner=model,
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            attention_preference=attention_preference,
        )

    def _maybe_prefill(self, model, input_ids, positions, fb, plan, ctx):
        if not self._prefill.enabled():
            return None
        batch_size = int(fb.batch_size)
        if batch_size <= 0 or fb.last_token_indices is None:
            return None
        if any(fb.spec_token_ids):
            return None
        raw_tokens = int(fb.num_token_non_padded)
        padded_tokens = int(input_ids.numel())
        if raw_tokens <= 0 or padded_tokens < raw_tokens:
            return None
        max_kv_tokens = self._prefill.bucket_kv_tokens(
            _padded_prefill_max_kv_tokens(
                plan,
                padded_tokens=padded_tokens,
                raw_tokens=raw_tokens,
                batch_size=batch_size,
            ),
            max_context_len=self.max_context_len,
        )
        if not self._prefill.can_use(
            padded_tokens,
            batch_size=batch_size,
            max_kv_tokens=max_kv_tokens,
        ):
            return None
        prepare_backend = self._prefill_prepare_backend(
            model,
            attention_preference=getattr(ctx, "attention_preference", None)
            or getattr(self, "attention_preference", None),
        )
        if prepare_backend is None:
            return None
        return self._prefill.maybe_run(
            kv_pool=self.kv_pool,
            num_blocks=self.num_blocks,
            num_tokens=padded_tokens,
            max_kv_tokens=max_kv_tokens,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            attention_plan=plan,
            last_token_indices=fb.last_token_indices,
            raw_num_tokens=raw_tokens,
            ctx=ctx,
            forward_fn=lambda state: self._prefill_forward(model, state),
            prepare_backend=prepare_backend,
        )

    def _prefill_prepare_backend(
        self,
        model: Any,
        *,
        attention_preference: str | None,
    ) -> Any | None:
        return resolve_paged_prefill_graph_prepare(
            owner=model,
            kv_pool=self.kv_pool,
            attention_preference=attention_preference,
        )

    # ---- the graph-unaware model forward ------------------------------------

    def _decode_forward(self, model: Any, state: TextDecodeGraphState) -> "torch.Tensor":
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            req_ids=tuple(range(int(state.batch_size))),
            input_ids=state.input_ids,
            positions=state.positions,
            attn_plan=state.plan,
        )
        return model.forward(state.input_ids, state.positions, fb)

    def _prefill_forward(self, model: Any, state: TextInitialPrefillGraphState) -> "torch.Tensor":
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            req_ids=tuple(range(int(state.batch_size))),
            input_ids=state.input_ids,
            positions=state.positions,
            last_token_indices=state.last_token_indices,
            attn_plan=state.plan,
        )
        return model.forward(state.input_ids, state.positions, fb)

    # ---- prefill bucket padding + warmup ------------------------------------

    def reorder_mixed_for_padding(self, text: Any) -> Any:
        """Keep row order stable; graph token padding has an isolated row."""

        return text

    def padded_num_tokens(self, text: Any, *, attention_preference: str | None) -> int | None:
        """Pad an extend group up to a captured prefill bucket, else ``None``."""

        del attention_preference
        if not self._prefill.enabled():
            return None
        if text.mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
            return None
        if any(text.spec_token_ids):
            return None
        lengths = [len(tokens) for tokens in text.token_ids]
        if not lengths or any(length <= 0 for length in lengths):
            return None
        raw_tokens = sum(int(length) for length in lengths)
        bucket = self._prefill.bucket_num_tokens(raw_tokens)
        if bucket <= raw_tokens:
            return None
        graph_batch_size = self._prefill.padding_batch_size(
            len(lengths),
            num_tokens=int(bucket),
        )
        if graph_batch_size <= len(lengths):
            return None
        return bucket

    def warmup(self, model: Any) -> None:
        if self.device.type != "cuda":
            return
        if self._decode.enabled() and self._decode.warmup_enabled():
            prepare_backend = self._decode_prepare_backend(
                model, attention_preference=self.attention_preference
            )
            if prepare_backend is None:
                logger.info(
                    "skipping decode graph warmup: no graph-capable paged-decode "
                    "backend or model geometry hook; decode stays eager"
                )
            else:
                self._decode.warmup(
                    kv_pool=self.kv_pool,
                    num_blocks=self.num_blocks,
                    device=self.device,
                    max_context_len=self.max_context_len,
                    attention_preference=self.attention_preference,
                    forward_fn=lambda state: self._decode_forward(model, state),
                    prepare_backend=prepare_backend,
                )
        if self._prefill.enabled() and self._prefill.warmup_enabled():
            prepare_backend = self._prefill_prepare_backend(
                model, attention_preference=self.attention_preference
            )
            if prepare_backend is None:
                logger.info(
                    "skipping prefill graph warmup: no graph-capable paged-prefill "
                    "backend or model geometry hook; prefill stays eager"
                )
            else:
                self._prefill.warmup(
                    kv_pool=self.kv_pool,
                    num_blocks=self.num_blocks,
                    block_size=self.block_size,
                    device=self.device,
                    max_context_len=self.max_context_len,
                    attention_preference=self.attention_preference,
                    forward_fn=lambda state: self._prefill_forward(model, state),
                    prepare_backend=prepare_backend,
                )


def _padded_prefill_max_kv_tokens(
    plan: Any,
    *,
    padded_tokens: int,
    raw_tokens: int,
    batch_size: int,
) -> int:
    pad = max(0, int(padded_tokens) - int(raw_tokens))
    fallback = max(int(getattr(plan, "max_seqlen_k", 0) or 0), pad)
    cache_lens = tuple(int(length) for length in getattr(plan, "cache_seqlens_cpu", ()) or ())
    query_lens = tuple(int(length) for length in getattr(plan, "query_lens_cpu", ()) or ())
    if len(cache_lens) != int(batch_size) or len(query_lens) != int(batch_size) or not query_lens:
        return max(1, fallback)
    padded_max = max(
        (int(base) + int(query) for base, query in zip(cache_lens, query_lens, strict=True)),
        default=fallback,
    )
    return max(1, padded_max, pad)

def explicit_attention_backend_name(name: str | None) -> str | None:
    """Resolve an explicit attention-backend request to its registry name.

    Auto/boolean sentinels mean "no explicit choice"; ``eager`` maps to the
    dense SDPA backend. Family graph runners consult this instead of touching
    the backend registry themselves.
    """
    from uniserve_worker.backends.attention.registry import normalize_attention_backend_name

    normalized = normalize_attention_backend_name(name)
    if normalized in {"", "auto", "0", "false", "off", "1", "true", "on"}:
        return None
    if normalized == "eager":
        return "torch_sdpa"
    return normalized
