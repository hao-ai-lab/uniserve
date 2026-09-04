"""Explicitly captured CUDA graph for one deployment-static operation."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any, Generic, TypeVar, cast

import torch

__all__ = ["StaticCudaGraph"]

_T = TypeVar("_T")
_POOL_LOCK = Lock()
_GRAPH_POOLS: dict[tuple[str, int | None], Any] = {}


def _graph_pool(device: torch.device) -> Any:
    """Return the process-wide static graph pool associated with one CUDA device."""

    key = (device.type, device.index)
    with _POOL_LOCK:
        pool = _GRAPH_POOLS.get(key)
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            _GRAPH_POOLS[key] = pool
        return pool


class StaticCudaGraph(Generic[_T]):
    """Capture during model warmup and replay from stable input addresses."""

    def __init__(self, device: torch.device | str) -> None:
        """Create an uncaptured graph owner with a private stream on one CUDA device."""

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA graph execution requires a CUDA device")
        self._stream = torch.cuda.Stream(device=self.device)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._output: _T | None = None

    @property
    def captured(self) -> bool:
        """Indicate whether this wrapper owns a replayable CUDA graph."""

        return self._graph is not None

    def capture(
        self,
        operation: Callable[[], _T],
        *,
        warmup: Callable[[], Any] | None = None,
    ) -> _T:
        """Warm the operation, capture it on the bound stream, and retain its graph memory pool."""

        if self._graph is not None:
            raise RuntimeError("the static CUDA graph has already been captured")
        if warmup is not None:
            warmup()
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, pool=_graph_pool(self.device), stream=self._stream):
            output = operation()
        graph.instantiate()
        with torch.cuda.stream(self._stream):
            graph.replay()
        current.wait_stream(self._stream)
        self._graph = graph
        self._output = output
        return output

    def replay(self) -> _T:
        """Replay the captured operation and return its persistent output object."""

        graph = self._graph
        if graph is None:
            raise RuntimeError("the static CUDA graph has not been captured during warmup")
        graph.replay()
        return cast(_T, self._output)
