"""Shared CUDA graph plumbing for text decode and initial prefill."""
from __future__ import annotations

import threading
from enum import Enum
from functools import lru_cache
from typing import Any, Callable

import torch

from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import classify
from ..foundation.runtime_config import (
    DEFAULT_DECODE_GRAPH_BATCH_SIZES,
    DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
)

# Module-level buffer pool for ``_share_decode_graph_input_buffer`` (contract tests).
_DECODE_GRAPH_INPUT_BUFFER_POOL: dict[tuple[str, str, str], torch.Tensor] = {}
_DECODE_GRAPH_INPUT_BUFFER_POOL_LOCK = threading.Lock()

# Warmup iterations run before each CUDA-graph capture to settle allocator and
# autotune state so the captured graph is stable. Two passes is the minimum that
# reliably clears first-call side effects.
_CAPTURE_WARMUP_ITERS = 2

# Default metric prefix; model runners may override (e.g. ``qwen3_``).
_DEFAULT_METRIC_PREFIX = "qwen3_"

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
    from ..ops.providers import weak_ref_tensor_provider

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

    Production runners use ``strict=True`` (buckets captured largest-first).
    The module-level shim uses ``strict=False``.
    """

    key = (str(name), str(tensor.dtype), str(tensor.device))
    existing = pool.get(key)
    if existing is not None:
        if strict:
            assert int(existing.numel()) >= int(tensor.numel()), (
                "graph input buffer reused for a larger bucket; "
                "capture buckets must run largest-first"
            )
            return existing.as_strided(tuple(tensor.shape), tuple(tensor.stride()))
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

        Buckets are captured largest-first (see ``warmup_capture_*``), so the
        first allocation under a given name is the largest and later buckets
        slice into it via ``as_strided``.  Assert the existing buffer is large
        enough rather than silently overwriting a captured graph's buffer.
        """

        return _share_input_buffer(self._graph_input_buffer_pool, name, tensor, strict=True)

    def _capture_or_replay(
        self,
        *,
        key: Any,
        ctx: Any,
        capture: Callable[[], Any],
        copy_inputs: Callable[[Any], None],
        replay: Callable[[Any], torch.Tensor],
        record: Callable[[GraphEvent], None],
        disable: Callable[[BaseException], None],
        capture_metric: str,
        input_copy_metric: str,
        replay_metric: str,
        after_copy: Callable[[Any], None] | None = None,
        after_copy_metric: str | None = None,
    ) -> torch.Tensor | None:
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
                state = capture()
                ctx.record_component_elapsed(capture_metric, start)
                self.states[key] = state
                event = GraphEvent.CAPTURE_REPLAY
            start = ctx.component_timer_start()
            copy_inputs(state)
            ctx.record_component_elapsed(input_copy_metric, start)
            if after_copy is not None:
                start = ctx.component_timer_start()
                after_copy(state)
                if after_copy_metric is not None:
                    ctx.record_component_elapsed(after_copy_metric, start)
            start = ctx.component_timer_start()
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
        run: Callable[[], torch.Tensor],
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
