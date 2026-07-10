"""Worker-side execution and forward-stats metrics.

Collects per-op-kind latency, control acks, typed-error counts, and recv/send
bracketing. ``snapshot()`` returns a plain dict; ``worker_exec_us`` is also
stamped onto execute results for host-side attribution.

Timing uses a caller-supplied clock (default ``time.perf_counter_ns``) so the
module stays free of wall-clock calls in restricted contexts.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, cast

__all__ = [
    'MetricsService',
]

logger = logging.getLogger(__name__)

# Emit the malformed-stats warning at most once per process (hot-path coercion).
_coerce_warned = False
_coerce_warned_lock = threading.Lock()

# Log-spaced execute-latency histogram upper bounds (microseconds). The implicit
# final bucket (+inf) captures anything above the last bound.
_EXEC_LATENCY_BUCKETS_US: tuple[int, ...] = (
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
)


def _coerce_int(value: object) -> int:
    """Best-effort int coercion that never raises inside the metrics hot path.

    Malformed stats payloads (unexpected types, NaN/inf, un-parseable strings)
    are dropped to 0 rather than propagating an exception out of a recording
    call that runs on the execute path.
    """
    if value is None:
        return 0
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        _warn_bad_stats_once(type(value).__name__)
        return 0


def _warn_bad_stats_once(type_name: str) -> None:
    global _coerce_warned
    with _coerce_warned_lock:
        if _coerce_warned:
            return
        _coerce_warned = True
    logger.warning(
        "metrics dropped a non-coercible stats value to 0 (type=%s); "
        "further occurrences are suppressed",
        type_name,
    )


def _percentile(buckets: list[int], bounds: tuple[int, ...], q: float) -> int:
    """Estimate the q-th percentile (0..1) from cumulative histogram buckets.

    Returns the bucket upper bound at or above the requested rank; values that
    fall in the open-ended final bucket report that bucket's lower bound. With
    no samples the result is 0.
    """
    total = sum(buckets)
    if total <= 0:
        return 0
    rank = q * total
    cumulative = 0
    for i, count in enumerate(buckets):
        cumulative += count
        if cumulative >= rank:
            if i < len(bounds):
                return int(bounds[i])
            return int(bounds[-1]) if bounds else 0
    return int(bounds[-1]) if bounds else 0


@dataclass(frozen=True)
class _MetricSpec:
    """Declarative descriptor for one forward-stats accumulator.

    Drives attribute initialization, ``record_forward_stats`` accumulation, and
    the ``forward`` sub-dict of ``snapshot()``. Adding a forward metric is one
    table row.

    - ``stats_key``: key read from the incoming forward-stats payload.
    - ``attr``: ``MetricsService`` attribute that stores the running total.
    - ``snap_key``: key emitted inside the ``snapshot()`` ``forward`` map
      (always equal to ``stats_key``).
    - ``is_map``: ``True`` for counter sub-maps merged with ``_merge_counter``;
      ``False`` for scalar counters folded with ``_coerce_int``.
    """

    stats_key: str
    attr: str
    is_map: bool

    @property
    def snap_key(self) -> str:
        return self.stats_key


_FORWARD_METRICS: tuple[_MetricSpec, ...] = (
    _MetricSpec("mode_counts", "forward_mode_counts", is_map=True),
    _MetricSpec("mode_tokens", "forward_mode_tokens", is_map=True),
    _MetricSpec("mode_us", "forward_mode_us", is_map=True),
    _MetricSpec("component_us", "forward_component_us", is_map=True),
    _MetricSpec("attention_launches", "attention_launches", is_map=False),
    _MetricSpec("attention_us", "attention_us", is_map=False),
    _MetricSpec("attention_backend_counts", "attention_backend_counts", is_map=True),
    _MetricSpec("operator_launches", "operator_launches", is_map=False),
    _MetricSpec("operator_us", "operator_us", is_map=False),
    _MetricSpec("operator_counts", "operator_counts", is_map=True),
    _MetricSpec("cuda_graph_captures", "cuda_graph_captures", is_map=False),
    _MetricSpec("cuda_graph_replays", "cuda_graph_replays", is_map=False),
    _MetricSpec("cuda_graph_misses", "cuda_graph_misses", is_map=False),
    _MetricSpec("cuda_graph_fallbacks", "cuda_graph_fallbacks", is_map=False),
    _MetricSpec("cuda_graph_unpadded_tokens", "cuda_graph_unpadded_tokens", is_map=False),
    _MetricSpec("cuda_graph_padded_tokens", "cuda_graph_padded_tokens", is_map=False),
    _MetricSpec(
        "cuda_graph_runtime_mode_counts", "cuda_graph_runtime_mode_counts", is_map=True
    ),
    _MetricSpec("forward_graph_captures", "forward_graph_captures", is_map=False),
    _MetricSpec("forward_graph_replays", "forward_graph_replays", is_map=False),
    _MetricSpec("forward_graph_misses", "forward_graph_misses", is_map=False),
    _MetricSpec("forward_graph_fallbacks", "forward_graph_fallbacks", is_map=False),
    _MetricSpec("forward_graph_capture_failures", "forward_graph_capture_failures", is_map=False),
    _MetricSpec("forward_graph_replay_failures", "forward_graph_replay_failures", is_map=False),
    _MetricSpec("forward_eager_fallbacks", "forward_eager_fallbacks", is_map=False),
    _MetricSpec("forward_eager_tokens", "forward_eager_tokens", is_map=False),
    _MetricSpec("forward_eager_rows", "forward_eager_rows", is_map=False),
    _MetricSpec(
        "forward_graph_runtime_mode_counts", "forward_graph_runtime_mode_counts", is_map=True
    ),
    _MetricSpec("forward_graph_shape_counts", "forward_graph_shape_counts", is_map=True),
    _MetricSpec("forward_graph_unpadded_tokens", "forward_graph_unpadded_tokens", is_map=False),
    _MetricSpec("forward_graph_padded_tokens", "forward_graph_padded_tokens", is_map=False),
    _MetricSpec("text_decode_token_relay_hits", "text_decode_token_relay_hits", is_map=False),
    _MetricSpec("text_decode_token_relay_misses", "text_decode_token_relay_misses", is_map=False),
    _MetricSpec("text_decode_position_relay_hits", "text_decode_position_relay_hits", is_map=False),
    _MetricSpec(
        "text_decode_position_relay_misses", "text_decode_position_relay_misses", is_map=False
    ),
    _MetricSpec("flashinfer_decode_plan_calls", "flashinfer_decode_plan_calls", is_map=False),
    _MetricSpec("flashinfer_decode_plan_reuses", "flashinfer_decode_plan_reuses", is_map=False),
    _MetricSpec("flashinfer_decode_plan_rows", "flashinfer_decode_plan_rows", is_map=False),
    _MetricSpec("flashinfer_decode_plan_indices", "flashinfer_decode_plan_indices", is_map=False),
    _MetricSpec(
        "flashinfer_decode_graph_plan_calls", "flashinfer_decode_graph_plan_calls", is_map=False
    ),
    _MetricSpec(
        "flashinfer_decode_graph_plan_reuses", "flashinfer_decode_graph_plan_reuses", is_map=False
    ),
    _MetricSpec("spec_verify_rows", "spec_verify_rows", is_map=False),
    _MetricSpec("spec_verify_draft_tokens", "spec_verify_draft_tokens", is_map=False),
    _MetricSpec("spec_verify_accepted_tokens", "spec_verify_accepted_tokens", is_map=False),
    _MetricSpec("spec_verify_rejected_tokens", "spec_verify_rejected_tokens", is_map=False),
    _MetricSpec("spec_verify_committed_tokens", "spec_verify_committed_tokens", is_map=False),
    _MetricSpec("spec_verify_path_counts", "spec_verify_path_counts", is_map=True),
)

# CUDA-graph counters duplicated at snapshot top level as well as in forward{}.
_TOP_LEVEL_CUDA_GRAPH_SCALARS: tuple[str, ...] = (
    "cuda_graph_captures",
    "cuda_graph_replays",
    "cuda_graph_misses",
    "cuda_graph_fallbacks",
    "cuda_graph_unpadded_tokens",
    "cuda_graph_padded_tokens",
)


class MetricsService:
    """Accumulates execute, control, error, and forward-stats counters."""

    def __init__(self, clock_ns: Callable[[], int] | None = None):
        self._clock = clock_ns or time.perf_counter_ns
        self.executes = 0
        self.ops_total = 0
        self.exec_ns_total = 0
        self.last_exec_ns = 0
        # Serve-loop pipeline timing: wall time the worker spends in each pipeline
        # stage. ``idle`` is time blocked on a recv with nothing in flight;
        # ``dispatch`` is request decode + GPU launch; ``finalize`` is the deferred
        # D2H materialize; ``encode_send`` is the response encode + ring write.
        self.pipeline_ns: dict[str, int] = defaultdict(int)
        self.pipeline_counts: dict[str, int] = defaultdict(int)
        # Whole-batch execute latency histogram (microseconds) for p50/p90/p99.
        self.exec_latency_bounds_us = _EXEC_LATENCY_BUCKETS_US
        self.exec_latency_buckets = [0] * (len(_EXEC_LATENCY_BUCKETS_US) + 1)
        self.op_kind_counts: dict[str, int] = defaultdict(int)
        self.op_kind_ns: dict[str, int] = defaultdict(int)
        self.control_ok: dict[str, int] = defaultdict(int)
        self.control_err: dict[str, int] = defaultdict(int)
        self.error_counts: dict[str, int] = defaultdict(int)
        for spec in _FORWARD_METRICS:
            setattr(self, spec.attr, defaultdict(int) if spec.is_map else 0)

    def now_ns(self) -> int:
        return self._clock()

    def record_execute(self, dur_ns: int, op_kinds: list[str]) -> None:
        self.executes += 1
        self.ops_total += len(op_kinds)
        self.exec_ns_total += dur_ns
        self.last_exec_ns = dur_ns
        self._observe_exec_latency(dur_ns)
        # Attribute batch wall time evenly across ops (coarse per-kind estimate).
        share = dur_ns // max(1, len(op_kinds))
        for k in op_kinds:
            self.op_kind_counts[k] += 1
            self.op_kind_ns[k] += share

    def _observe_exec_latency(self, dur_ns: int) -> None:
        dur_us = dur_ns // 1000
        for i, bound in enumerate(self.exec_latency_bounds_us):
            if dur_us <= bound:
                self.exec_latency_buckets[i] += 1
                return
        self.exec_latency_buckets[-1] += 1

    def exec_latency_percentiles_us(self) -> dict[str, int]:
        """p50/p90/p99 of whole-batch execute latency, in microseconds."""
        return {
            "p50": _percentile(self.exec_latency_buckets, self.exec_latency_bounds_us, 0.50),
            "p90": _percentile(self.exec_latency_buckets, self.exec_latency_bounds_us, 0.90),
            "p99": _percentile(self.exec_latency_buckets, self.exec_latency_bounds_us, 0.99),
        }

    def record_pipeline(self, stage: str, dur_ns: int) -> None:
        """Accumulate wall time spent in one serve-loop pipeline stage."""
        self.pipeline_ns[stage] += int(dur_ns)
        self.pipeline_counts[stage] += 1

    def record_control(self, kind: str, ok: bool) -> None:
        (self.control_ok if ok else self.control_err)[kind] += 1

    def record_error(self, code: str) -> None:
        self.error_counts[code] += 1

    @staticmethod
    def _merge_counter(target: dict[str, int], raw: object) -> None:
        """Fold a stats sub-map into a counter, ignoring malformed payloads.

        Non-mapping values and individual entries that do not coerce to int are
        skipped so a bad stats blob cannot raise out of the execute hot path.
        """
        if not isinstance(raw, dict):
            return
        for key, value in raw.items():
            target[str(key)] += _coerce_int(value)

    def record_forward_stats(self, stats: dict | None) -> None:
        if not isinstance(stats, dict):
            return
        for spec in _FORWARD_METRICS:
            raw = stats.get(spec.stats_key)
            if spec.is_map:
                self._merge_counter(getattr(self, spec.attr), raw)
            else:
                setattr(self, spec.attr, getattr(self, spec.attr) + _coerce_int(raw))

    def _forward_value(self, spec: _MetricSpec) -> dict | int:
        value = getattr(self, spec.attr)
        return dict(value) if spec.is_map else int(value)

    def snapshot(self) -> dict:
        forward = {spec.snap_key: self._forward_value(spec) for spec in _FORWARD_METRICS}
        snap = {
            "executes": self.executes,
            "ops_total": self.ops_total,
            "exec_us_total": self.exec_ns_total // 1000,
            "last_exec_us": self.last_exec_ns // 1000,
            "exec_latency_us": self.exec_latency_percentiles_us(),
            "op_kind_counts": dict(self.op_kind_counts),
            "op_kind_us": {k: v // 1000 for k, v in self.op_kind_ns.items()},
            "control_ok": dict(self.control_ok),
            "control_err": dict(self.control_err),
            "error_counts": dict(self.error_counts),
        }
        for name in _TOP_LEVEL_CUDA_GRAPH_SCALARS:
            snap[name] = int(getattr(self, name))
        snap["cuda_graph_runtime_mode_counts"] = dict(
            getattr(self, "cuda_graph_runtime_mode_counts")
        )
        snap["forward"] = forward
        snap["pipeline_us"] = {k: v // 1000 for k, v in self.pipeline_ns.items()}
        snap["pipeline_counts"] = dict(self.pipeline_counts)
        return snap
