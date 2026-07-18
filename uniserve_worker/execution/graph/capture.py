"""Shared CUDA graph capture lifecycle."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import nullcontext
from enum import Enum
from functools import lru_cache
from typing import Any, TypeVar

import torch

from uniserve_worker.foundation.errors import classify

# ---------------------
# Shared graph-runner base, capture locks, and bucket configuration
# ---------------------

# Module-level buffer pool for ``_share_step_input`` (contract tests).
_STEP_INPUT_POOL: dict[tuple[str, str, str], torch.Tensor] = {}
_STEP_INPUT_POOL_LOCK = threading.Lock()
Output = TypeVar("Output")

# Warmup iterations run before each CUDA-graph capture to settle allocator and
# autotune state so the captured graph is stable. Two passes is the minimum that
# reliably clears first-call side effects.
_CAPTURE_WARMUP_ITERS = 2
# Reserve 5% of device memory for capture-time private-pool growth, bounded so
# the policy remains meaningful on both small and large accelerators. Avoid an
# allocator flush unless it can release a material amount of cached storage.
_CAPTURE_HEADROOM_MIN_BYTES = 512 * 1024**2
_CAPTURE_HEADROOM_MAX_BYTES = 8 * 1024**3
_CAPTURE_RECLAIM_MIN_BYTES = 256 * 1024**2

# Neutral default used by model-owned specializations that do not select a
# narrower metric namespace.
_DEFAULT_METRIC_PREFIX = "graph_"


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


def _reclaim_for_capture(device: torch.device | str) -> bool:
    """Release unused allocator blocks when a new graph lacks safe headroom."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return False
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
        allocated_bytes = torch.cuda.memory_allocated(resolved)
        reserved_bytes = torch.cuda.memory_reserved(resolved)
    except Exception:
        return False
    headroom = min(
        _CAPTURE_HEADROOM_MAX_BYTES,
        max(_CAPTURE_HEADROOM_MIN_BYTES, int(total_bytes) // 20),
    )
    reclaimable = max(0, int(reserved_bytes) - int(allocated_bytes))
    if int(free_bytes) >= headroom or reclaimable < _CAPTURE_RECLAIM_MIN_BYTES:
        return False
    torch.cuda.synchronize(resolved)
    torch.cuda.empty_cache()
    return True


@lru_cache(maxsize=1)
def _weak_ref_tensor_func() -> Any:
    from uniserve_worker.ops.providers import weak_ref_tensor_provider

    return weak_ref_tensor_provider()


class Event(str, Enum):
    """Events emitted by a physical graph runner."""

    CAPTURE_REPLAY = "capture_replay"
    REPLAY = "replay"
    MISS = "miss"
    FALLBACK = "fallback"

    def __str__(self) -> str:
        return self.value


def record(
    ctx: Any,
    event: Event | str,
    *,
    unpadded_tokens: int,
    padded_tokens: int,
) -> None:
    """Apply a graph event to the active forward statistics collector.

    ``unpadded_tokens``/``padded_tokens`` are the raw vs bucket-padded token
    counts the event contributes. ``miss`` and ``fallback`` only bump their
    counter.
    """

    stats = getattr(ctx, "stats", None)
    if stats is None:
        return
    event = Event(event)
    if event is Event.MISS:
        stats.cuda_graph_misses += 1
        return
    if event is Event.FALLBACK:
        stats.cuda_graph_fallbacks += 1
        return
    if event is Event.CAPTURE_REPLAY:
        stats.cuda_graph_captures += 1
        stats.cuda_graph_replays += 1
    else:  # Event.REPLAY
        stats.cuda_graph_replays += 1
    unpadded_tokens = int(unpadded_tokens)
    padded_tokens = int(padded_tokens)
    stats.cuda_graph_unpadded_tokens += unpadded_tokens
    stats.cuda_graph_padded_tokens += max(0, padded_tokens - unpadded_tokens)


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


def _share_step_input(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Module-level buffer pool delegate (non-strict); runners use instance pools."""

    with _STEP_INPUT_POOL_LOCK:
        return _share_input_buffer(_STEP_INPUT_POOL, name, tensor, strict=False)


def _reset_for_testing() -> None:
    """Test-only; not part of the public API.

    Clear the module-level decode-graph input buffer pool so tests get
    deterministic isolation. The pool is repopulated on demand by
    ``_share_step_input``.
    """

    with _STEP_INPUT_POOL_LOCK:
        _STEP_INPUT_POOL.clear()


class Runner:
    """Own the shared lifecycle for physical graph implementations."""

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
    ) -> None:
        """Synchronously retire graph states and reclaim their unoccupied pools.

        Synchronization makes every retired executable quiescent. Dropping the
        final state reference before ``empty_cache`` lets the allocator release
        only storage that became unoccupied; allocations owned by live graph
        executables remain occupied and keep their captured addresses.
        """

        states = [self.states.pop(key) for key in keys if key in self.states]
        if not states:
            return
        torch.cuda.synchronize(device)
        state = None
        states.reverse()
        try:
            while states:
                state = states.pop()
                self._destroy_graph_state(state)
                state = None
        finally:
            state = None
            states.clear()
            torch.cuda.empty_cache()

    def _capture_or_replay(
        self,
        *,
        key: Any,
        device: torch.device | str,
        ctx: Any,
        capture: Callable[[], Any],
        copy_inputs: Callable[[Any], None],
        replay: Callable[[Any], Output],
        record: Callable[[Event], None],
        disable: Callable[[BaseException], None],
        capture_metric: str,
        input_copy_metric: str,
        replay_metric: str,
        after_copy: Callable[[Any], None] | None = None,
        after_copy_metric: str | None = None,
    ) -> Output | None:
        """Template method for capture-or-replay lifecycle.

        Implementations provide bucket selection, state construction, input
        copying, backend preparation, and event mapping. This method owns state
        storage, timing, failure classification, and capture permission.
        """

        state = self.states.get(key)
        if state is None and not bool(getattr(ctx, "allow_capture", True)):
            record(Event.MISS)
            return None
        try:
            event = Event.REPLAY
            if state is None:
                _reclaim_for_capture(device)
                start = ctx.component_timer_start()
                with _record_function_scope(f"uniserve.cuda_graph.{capture_metric}"):
                    state = capture()
                ctx.record_component_elapsed(capture_metric, start)
                self.states[key] = state
                event = Event.CAPTURE_REPLAY
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
            record(Event.FALLBACK)
            return None

    def _capture_graph_state(
        self,
        *,
        device: torch.device | str,
        state: Any,
        run: Callable[[], Output],
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
        # The standard context synchronizes and releases unoccupied allocator
        # cache after warmup, immediately before graph-pool allocation. Live
        # graph allocations remain occupied and keep their captured addresses.
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
