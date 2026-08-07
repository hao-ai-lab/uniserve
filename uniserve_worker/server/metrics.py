"""Worker-side counters projected exactly onto the worker-protocol metrics shape."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any, Callable

from ..batch import WorkerForwardStats
from ..foundation.sync_detector import sync_detector

__all__ = ["MetricsService"]

_MAP_FORWARD_FIELDS = frozenset(
    {
        "mode_counts",
        "mode_tokens",
        "mode_us",
        "component_us",
        "attention_backend_counts",
        "cuda_graph_runtime_mode_counts",
        "spec_verify_path_counts",
    }
)
_FORWARD_FIELDS = tuple(field.name for field in fields(WorkerForwardStats))
_GRAPH_SCALARS = (
    "cuda_graph_captures",
    "cuda_graph_replays",
    "cuda_graph_misses",
    "cuda_graph_fallbacks",
    "cuda_graph_unpadded_tokens",
    "cuda_graph_padded_tokens",
)


def _counter_value(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return max(0, value)


class MetricsService:
    """Accumulate execute, control, failure, forward, and local pipeline timing."""

    def __init__(self, clock_ns: Callable[[], int] | None = None) -> None:
        self._clock = clock_ns or time.perf_counter_ns
        self.executes = 0
        self.operations_total = 0
        self.exec_ns_total = 0
        self.last_exec_ns = 0
        self.operation_counts: dict[str, int] = defaultdict(int)
        self.operation_ns: dict[str, int] = defaultdict(int)
        self.control_ok: dict[str, int] = defaultdict(int)
        self.control_err: dict[str, int] = defaultdict(int)
        self.error_counts: dict[str, int] = defaultdict(int)
        self.pipeline_ns: dict[str, int] = defaultdict(int)
        self.pipeline_counts: dict[str, int] = defaultdict(int)
        self.forward: dict[str, int | dict[str, int]] = {
            name: {} if name in _MAP_FORWARD_FIELDS else 0 for name in _FORWARD_FIELDS
        }

    def now_ns(self) -> int:
        return self._clock()

    def record_execute(self, duration_ns: int, variant_labels: Sequence[str]) -> None:
        duration_ns = max(0, int(duration_ns))
        self.executes += 1
        self.operations_total += len(variant_labels)
        self.exec_ns_total += duration_ns
        self.last_exec_ns = duration_ns
        share = duration_ns // max(1, len(variant_labels))
        for label in variant_labels:
            key = str(label)
            self.operation_counts[key] += 1
            self.operation_ns[key] += share

    def record_control(self, kind: str, succeeded: bool) -> None:
        (self.control_ok if succeeded else self.control_err)[str(kind)] += 1

    def record_error(self, code: str) -> None:
        self.error_counts[str(code)] += 1

    def record_pipeline(self, stage: str, duration_ns: int) -> None:
        self.pipeline_ns[str(stage)] += max(0, int(duration_ns))
        self.pipeline_counts[str(stage)] += 1

    def record_forward_stats(self, stats: WorkerForwardStats | Mapping[str, Any] | None) -> None:
        if stats is None:
            return
        source: Mapping[str, Any] = (
            stats.to_wire() if isinstance(stats, WorkerForwardStats) else stats
        )
        for name in _FORWARD_FIELDS:
            value = source.get(name)
            if name in _MAP_FORWARD_FIELDS:
                if not isinstance(value, Mapping):
                    continue
                target = self.forward[name]
                if not isinstance(target, dict):
                    raise RuntimeError("forward metric accumulator has an invalid shape")
                for key, raw in value.items():
                    target[str(key)] = target.get(str(key), 0) + _counter_value(raw)
            else:
                current = self.forward[name]
                if not isinstance(current, int):
                    raise RuntimeError("forward metric accumulator has an invalid shape")
                self.forward[name] = current + _counter_value(value)

    def snapshot(self) -> dict[str, object]:
        forward = {
            key: dict(value) if isinstance(value, dict) else int(value)
            for key, value in self.forward.items()
        }
        graph_scalars: dict[str, int] = {}
        for name in _GRAPH_SCALARS:
            value = forward[name]
            if not isinstance(value, int):
                raise RuntimeError("forward metric accumulator has an invalid graph scalar")
            graph_scalars[name] = value
        runtime_modes = forward["cuda_graph_runtime_mode_counts"]
        if not isinstance(runtime_modes, dict):
            raise RuntimeError("forward metric accumulator has invalid runtime-mode counts")
        return {
            "executes": self.executes,
            "operations_total": self.operations_total,
            "exec_us_total": self.exec_ns_total // 1000,
            "last_exec_us": self.last_exec_ns // 1000,
            "operation_counts": dict(self.operation_counts),
            "operation_us": {key: value // 1000 for key, value in self.operation_ns.items()},
            "control_ok": dict(self.control_ok),
            "control_err": dict(self.control_err),
            "error_counts": dict(self.error_counts),
            "forbidden_sync_detections": sync_detector().detections,
            **graph_scalars,
            "cuda_graph_runtime_mode_counts": dict(runtime_modes),
            "forward": forward,
        }
