"""Full forward capture with native CUDA graphs."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import torch

from uniserve_worker.foundation.resources import close_resources

from ..lane import verify_graph_context
from .backend import CudaGraphBackend, GraphExecutionError

T = TypeVar("T")


@dataclass(slots=True)
class _Graph(Generic[T]):
    graph: torch.cuda.CUDAGraph
    output: T
    keepalive: tuple[object, ...]


class FullCudaGraphBackend(CudaGraphBackend[T]):
    """Record complete callables on a device/context-bound capture stream.

    A supplied pool is shared only by serialized executions whose outputs are
    consumed or published before reuse. With no pool each graph has independent
    storage. Replay enqueues on the caller's stream in the same CUDA context.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        stream: torch.cuda.Stream,
        pool: Any = None,
        expected_context: int | None = None,
    ) -> None:
        self.device = device
        self._stream = stream
        self._pool = pool
        self._expected_context = expected_context
        self._graphs: dict[Hashable, _Graph[T]] = {}
        self._closed = False

    @torch.inference_mode()
    def capture_one(
        self,
        key: Hashable,
        forward: Callable[[], T],
        *,
        keepalive: tuple[object, ...] = (),
        restore: Callable[[], None] | None = None,
    ) -> None:
        if self._closed:
            raise GraphExecutionError("CUDA graph backend is closed")
        if key in self._graphs:
            raise GraphExecutionError(f"CUDA graph key already exists: {key!r}")
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                # Provider initialization may require a second eager invocation.
                # Every invocation observes the same state, including on failure.
                for _ in range(2):
                    try:
                        forward()
                    finally:
                        if restore is not None:
                            restore()
                    self._stream.synchronize()
                try:
                    with torch.cuda.graph(graph, pool=self._pool, stream=self._stream):
                        output = forward()
                    graph.instantiate()
                    verify_graph_context(graph, self._expected_context)
                finally:
                    if restore is not None:
                        restore()
                # Temporary restore snapshots can now be released by the owner.
                self._stream.synchronize()
        except BaseException as error:
            try:
                self._stream.synchronize()
                graph.reset()
                if not self._graphs and self._pool is not None:
                    self._pool = torch.cuda.graph_pool_handle()
            except BaseException as cleanup:
                error.add_note(f"CUDA graph cleanup failed: {cleanup!r}")
            error.add_note(f"CUDA graph capture device={self.device} key={key!r}")
            raise
        finally:
            current.wait_stream(self._stream)
        self._graphs[key] = _Graph(graph, output, (forward, *keepalive))

    def contains(self, key: Hashable) -> bool:
        return key in self._graphs

    def replay(self, key: Hashable) -> T:
        if self._closed:
            raise GraphExecutionError("CUDA graph backend is closed")
        try:
            record = self._graphs[key]
        except KeyError as error:
            raise GraphExecutionError(f"CUDA graph key is not resident: {key!r}") from error
        record.graph.replay()
        return record.output

    def discard(self, key: Hashable) -> None:
        record = self._graphs.pop(key, None)
        if record is not None:
            record.graph.reset()
            if not self._graphs and self._pool is not None:
                # The allocator retires a shared pool with its last graph even
                # when a consumer still retains an output allocation. A new
                # capture must use a live pool identity, preserving those outputs.
                self._pool = torch.cuda.graph_pool_handle()

    def close(self) -> None:
        if self._closed:
            return
        try:
            close_resources(*(partial(self.discard, key) for key in tuple(self._graphs)))
        finally:
            self._pool = None
            self._closed = True
