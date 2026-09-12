"""One captured CUDA executable and the numerical storage it keeps alive."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import torch

from ..foundation.resources import close_resources
from .cuda_stream import verify_graph_context

if TYPE_CHECKING:
    from .forward_batch import ForwardBatch

T = TypeVar("T")


class GraphExecutionError(RuntimeError):
    """A configured CUDA graph could not execute safely."""


class CudaGraph(Generic[T]):
    """Own one executable and its borrowed output views.

    Capture warms providers twice and restores mutable numerical state after
    every invocation, including failures. Replay uses the caller's stream.
    The owner must order output consumers before replay and drain GPU use before
    close. Shared pool identities belong to that owner's serial reuse domain.
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
        self.stream = stream
        self.pool = pool
        self.expected_context = expected_context
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: T | None = None
        self.keepalive: tuple[object, ...] = ()
        self.inputs: ForwardBatch | tuple[torch.Tensor, ...] | None = None
        # Exact input copies and paged-attention replanning borrow these fixed
        # addresses until graph teardown. Empty input leaves denote input views
        # already staged in the owner's reusable InputBuffers.
        self.input_leaves: tuple[torch.Tensor, ...] = ()
        self.attention_leaves: tuple[torch.Tensor, ...] = ()
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
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
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
            self.inputs = None
            self.input_leaves = ()
            self.attention_leaves = ()
            self.releases = ()
            self.pool = None
