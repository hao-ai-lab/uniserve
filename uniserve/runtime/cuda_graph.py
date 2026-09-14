"""One captured CUDA executable and the numerical storage it keeps alive."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from typing import Any, Generic, TypeVar, cast

import torch

from uniserve.runtime.cuda import verify_graph_context
from uniserve.runtime.resources import close_resources

T = TypeVar("T")


def capture_pools(devices: Iterable[torch.device]) -> dict[torch.device, torch.cuda.MemPool]:
    """Own each additional capture device's allocator backing on that device.

    MemPool's lifetime reference belongs to the CUDA device selected at its
    construction. Allocation routing must use that same device to keep captured
    intermediate addresses reserved after the routing scope exits.
    """

    pools = {}
    for device in dict.fromkeys(devices):
        with torch.cuda.device(device):
            pools[device] = torch.cuda.MemPool()
    return pools


class GraphExecutionError(RuntimeError):
    """A configured CUDA graph could not execute safely."""


class CudaGraph(Generic[T]):
    """Own one executable and its borrowed output views.

    Capture warms providers twice and restores mutable numerical state after
    every invocation, including failures. Replay uses the caller's stream.
    The owner must order output consumers before replay and drain GPU use before
    close. Shared pool identities belong to that owner's serial reuse domain.
    Calls spanning GPUs require a pool for every additional device, constructed
    by capture_pools. Concurrent calls must use distinct pools on every device.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        stream: torch.cuda.Stream,
        pool: Any = None,
        expected_context: int | None = None,
        device_pools: Mapping[torch.device, torch.cuda.MemPool] | None = None,
    ) -> None:
        self.device = device
        self.stream = stream
        self.pool = pool
        self.expected_context = expected_context
        self.device_pools = dict(device_pools or {})
        if self.device in self.device_pools:
            raise ValueError("the capture device already uses the graph's primary pool")
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: T | None = None
        self.keepalive: tuple[object, ...] = ()
        self.releases: tuple[Callable[[], None], ...] = ()
        self._closed = False

    @torch.inference_mode()
    def capture(
        self,
        forward: Callable[[], T],
        *,
        keepalive: tuple[object, ...] = (),
        restore: Callable[[], None] | None = None,
    ) -> None:
        """Record exactly one computation while preserving its entry state."""

        if self._closed:
            raise GraphExecutionError("CUDA graph is closed")
        if self.graph is not None:
            raise GraphExecutionError("CUDA graph is already captured")
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with (
                ExitStack() as allocations,
                torch.cuda.device(self.device),
                torch.cuda.stream(self.stream),
            ):
                # PyTorch's graph owns allocation backing on its capture device.
                # Other devices retain explicit pools through the same executable
                # lifetime, including intermediates freed during Python capture.
                for device, pool in self.device_pools.items():
                    allocations.enter_context(torch.cuda.use_mem_pool(pool, device))
                for _ in range(2):
                    try:
                        forward()
                    finally:
                        if restore is not None:
                            restore()
                    self.stream.synchronize()
                try:
                    with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
                        output = forward()
                    graph.instantiate()
                    verify_graph_context(graph, self.expected_context)
                finally:
                    if restore is not None:
                        restore()
                self.stream.synchronize()
        except BaseException as error:
            try:
                self.stream.synchronize()
                graph.reset()
            except BaseException as cleanup:
                error.add_note(f"CUDA graph cleanup failed: {cleanup!r}")
            error.add_note(f"CUDA graph capture device={self.device}")
            raise
        finally:
            current.wait_stream(self.stream)
        self.graph = graph
        self.pool = graph.pool()
        self.output = output
        self.keepalive = (forward, *keepalive)

    def replay(self) -> T:
        """Enqueue this executable and borrow its output until the next replay."""

        if self._closed:
            raise GraphExecutionError("CUDA graph is closed")
        if self.graph is None:
            raise GraphExecutionError("CUDA graph is not captured")
        self.graph.replay()
        return cast(T, self.output)

    def close(self) -> None:
        """Release a drained executable before its numerical backing is discarded."""

        if self._closed:
            return
        self._closed = True
        try:
            actions = [] if self.graph is None else [self.graph.reset]
            close_resources(*actions, *reversed(self.releases))
        finally:
            self.graph = None
            self.output = None
            self.keepalive = ()
            self.releases = ()
            self.pool = None
            self.device_pools.clear()
