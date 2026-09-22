"""Captured numerical calls and the borrowed resources they keep alive."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import Generic, TypeVar

import torch

from .cuda import (
    create_sibling_stream,
    cuda_value,
    destroy_stream,
    driver,
    verify_graph_context,
)
from .execution import ExecutionContext

ResultT = TypeVar("ResultT")


class CUDAGraphError(RuntimeError):
    """A capture or replay failure.

    A capture or replay failure retaining the original exception as its
    cause.
    """


class CUDAGraph(Generic[ResultT]):
    """Own a captured call, its output views and execution resource references.

    Warm every kernel specialization before capture. Capture invokes the call
    once and optionally restores mutable input state afterward. The caller
    orders final readers before replay, resource reuse or close. Cross-device
    calls require caller-owned MemPools for their additional device allocations.
    Construct each pool with its mapped device current, and retain it until all
    graphs and tensors using its allocations have retired.

    The call computes on the context's stream and the graph replays there, but
    capture begins and ends on a stream this graph owns, which the computation
    joins for the capture's duration. PyTorch retires the library workspaces
    cached for a graph's capture stream when the graph resets, and the
    context's stream carries every graph captured on the context plus its eager
    work, so a graph must never make the shared stream its capture stream.
    """

    def __init__(
        self,
        *,
        context: ExecutionContext,
        pools: Mapping[torch.device, torch.cuda.MemPool] | None = None,
    ):
        context._open()
        if context._device.type != "cuda":
            raise ValueError("CUDA graph execution requires a CUDA module")

        self.context = context
        self.pools = dict(pools or {})
        self._computation = context.stream or torch.cuda.Stream(
            device=context._device
        )
        # A driver stream is never one of the pool's, so no context or lane
        # stream can coincide with it.
        self._raw_capture, self._capture = create_sibling_stream(
            self._computation, "graph capture"
        )
        self._graph = None
        self._output = None
        self._call = None
        self._closed = False

    @torch.inference_mode()
    def capture(
        self,
        call: Callable[[], ResultT],
        *,
        restore: Callable[[], None] | None = None,
    ) -> None:
        """Invoke ``call`` once under capture on this graph's private stream.

        Every kernel specialization the call exercises must already be warmed.
        When given, ``restore`` runs after the capture attempt, successful or
        not, to return mutated inputs to their pre-capture state.
        """
        if self._closed or self._graph is not None:
            raise CUDAGraphError("capture requires an open uncaptured graph")
        self.context._open()

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        device = self.context._device
        current = torch.cuda.current_stream(device)
        # Capture observes all work the caller has already submitted.
        self._capture.wait_stream(current)
        try:
            with (
                ExitStack() as scope,
                torch.cuda.device(device),
                torch.cuda.stream(self._capture),
            ):
                for target, pool in self.pools.items():
                    if target != device:
                        scope.enter_context(
                            torch.cuda.use_mem_pool(pool, target)
                        )

                pool = self.pools.get(device)
                try:
                    with torch.cuda.graph(
                        graph,
                        stream=self._capture,
                        pool=None if pool is None else pool.id,
                    ):
                        # The computation stream joins the capture before
                        # the call's first launch and the capture stream
                        # rejoins it after the last, so every launch of the
                        # call, including collectives bound to the
                        # computation stream, lands in the graph.
                        self._computation.wait_stream(self._capture)
                        with (
                            torch.cuda.stream(self._computation),
                            self.context.activate(),
                        ):
                            output = call()
                        self._capture.wait_stream(self._computation)
                    graph.instantiate()

                    if self.context.stream is not None:
                        cu = driver()
                        streams = (
                            self._capture,
                            self._computation,
                            *self.context._transfers.streams.values(),
                        )
                        expected = frozenset(
                            int(
                                cuda_value(
                                    cu.cuStreamGetCtx(
                                        cu.CUstream(stream.cuda_stream)
                                    ),
                                    "query capture stream context",
                                )
                            )
                            for stream in streams
                        )
                        verify_graph_context(graph, expected)
                finally:
                    if restore is not None:
                        with self.context.activate():
                            restore()
        except BaseException as error:
            try:
                graph.reset()
            except BaseException as cleanup:
                error.add_note(f"CUDA graph cleanup failed: {cleanup!r}")
            raise CUDAGraphError(
                f"CUDA graph capture failed on {device}: {error}"
            ) from error
        finally:
            # Keep the caller's stream ordered after capture-side work.
            current.wait_stream(self._computation)
            current.wait_stream(self._capture)

        self._graph, self._output, self._call = graph, output, call

    def replay(self) -> ResultT:
        """Replay the captured call and return its retained output views."""
        if self._closed or self._graph is None:
            raise CUDAGraphError("replay requires an open captured graph")
        self.context._open()
        try:
            with self.context.activate():
                self._graph.replay()
        except BaseException as error:
            raise CUDAGraphError(
                f"CUDA graph replay failed: {error}"
            ) from error
        return self._output

    def close(self, *, aborted: bool = False) -> None:
        """Release the graph after its final readers have completed.

        Output views become invalid. Aborted close retains native resources
        without waiting; the owning process must exit before reclaiming them.
        """
        if self._closed:
            return
        self._closed = True
        if aborted:
            from .resources import retain_until_exit

            retain_until_exit(self)
            return
        try:
            if self._graph is not None:
                self._graph.reset()
        finally:
            self._graph = self._output = self._call = None
            self.pools.clear()
            self.context = None
            if self._raw_capture is not None:
                # Only capture bookkeeping was ever launched here, and the
                # caller has ordered its final readers before closing.
                self._capture.synchronize()
                destroy_stream(self._raw_capture, "graph capture")
                self._raw_capture = None

    def __enter__(self):
        if self._closed:
            raise CUDAGraphError("CUDA graph is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close(aborted=exc is not None)
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"CUDA graph cleanup failed: {error!r}")
