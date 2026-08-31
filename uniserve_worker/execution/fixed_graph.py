"""Bounded fixed-shape CUDA graph executables for non-token workloads."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from threading import Lock
from typing import Any, Generic, TypeVar

import torch

__all__ = ["FixedShapeGraphCache", "FixedShapeGraphStats"]

_T = TypeVar("_T")
_POOL_LOCK = Lock()
_GRAPH_POOLS: dict[tuple[str, int | None], Any] = {}


def _graph_pool(device: torch.device) -> Any:
    key = (device.type, device.index)
    with _POOL_LOCK:
        pool = _GRAPH_POOLS.get(key)
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            _GRAPH_POOLS[key] = pool
        return pool


@dataclass(frozen=True, slots=True)
class FixedShapeGraphStats:
    captures: int
    replays: int
    evictions: int
    resident_bytes: int
    capture_failures: int
    entries: int


@dataclass(slots=True)
class _Executable(Generic[_T]):
    graph: torch.cuda.CUDAGraph
    output: _T
    resident_bytes: int


class FixedShapeGraphCache(Generic[_T]):
    """Capture missing legal shapes and replay resident executables with LRU bounds."""

    def __init__(self, device: torch.device | str, *, capacity: int) -> None:
        self.device = torch.device(device)
        self.capacity = int(capacity)
        if self.device.type != "cuda":
            raise ValueError("fixed-shape graph execution requires a CUDA device")
        if self.capacity < 1:
            raise ValueError("fixed-shape graph cache capacity must be positive")
        self._stream = torch.cuda.Stream(device=self.device)
        self._entries: OrderedDict[Hashable, _Executable[_T]] = OrderedDict()
        self._captures = 0
        self._replays = 0
        self._evictions = 0
        self._resident_bytes = 0
        self._capture_failures = 0

    def contains(self, shape_key: Hashable) -> bool:
        return shape_key in self._entries

    def execute(
        self,
        shape_key: Hashable,
        capture: Callable[[], _T],
        *,
        warmup: Callable[[], Any] | None = None,
    ) -> _T:
        executable = self._entries.pop(shape_key, None)
        if executable is not None:
            executable.graph.replay()
            self._replays += 1
            self._entries[shape_key] = executable
            return executable.output

        if warmup is not None:
            warmup()
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)
        before = torch.cuda.memory_allocated(self.device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(
                graph, pool=_graph_pool(self.device), stream=self._stream
            ):
                output = capture()
            graph.instantiate()
            with torch.cuda.stream(self._stream):
                graph.replay()
        except BaseException:
            self._capture_failures += 1
            current.wait_stream(self._stream)
            raise
        current.wait_stream(self._stream)
        resident_bytes = max(0, torch.cuda.memory_allocated(self.device) - before)
        executable = _Executable(graph, output, resident_bytes)
        self._entries[shape_key] = executable
        self._captures += 1
        self._resident_bytes += resident_bytes
        while len(self._entries) > self.capacity:
            _key, evicted = self._entries.popitem(last=False)
            self._resident_bytes -= evicted.resident_bytes
            self._evictions += 1
        return output

    @property
    def stats(self) -> FixedShapeGraphStats:
        return FixedShapeGraphStats(
            captures=self._captures,
            replays=self._replays,
            evictions=self._evictions,
            resident_bytes=self._resident_bytes,
            capture_failures=self._capture_failures,
            entries=len(self._entries),
        )
