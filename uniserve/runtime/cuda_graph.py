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
from .resources import streams_idle

ResultT = TypeVar("ResultT")


class CUDAGraphError(RuntimeError):
    """A capture or replay failure.

    A capture or replay failure retaining the original exception as its
    cause.
    """


class _Capture:
    """Capture computation around submissions CUDA cannot put in a graph.

    Segments share a pool and always replay in capture order. Submissions
    between them must use external CUDA events for their dependencies; a
    stream fork cannot span two captures. The numerical call runs only once.
    """

    def __init__(self, capture, computation, pool):
        self.capture = capture
        self.computation = computation
        self.pool = pool
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.steps: list[Callable[[], None]] = []
        self.active = False

    def begin(self):
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        self.graphs.append(graph)
        with torch.cuda.stream(self.capture):
            graph.capture_begin(pool=self.pool)
            self.active = True
            self.computation.wait_stream(self.capture)

    def end(self):
        if not self.active:
            return
        graph = self.graphs[-1]
        try:
            with torch.cuda.stream(self.capture):
                self.capture.wait_stream(self.computation)
                graph.capture_end()
        finally:
            self.active = False
        if self.pool is None:
            self.pool = graph.pool()
        self.steps.append(graph.replay)

    def submit(self, call: Callable[[], None]):
        """Record an eager submission between the surrounding graph segments."""
        self.end()
        self.steps.append(call)
        self.begin()

    def replay(self):
        for step in self.steps:
            step()

    def reset(self):
        for graph in self.graphs:
            graph.reset()
        self.steps.clear()
        self.graphs.clear()


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

        # Close releases the borrowed context.
        self._context: ExecutionContext | None = context
        self.pools = dict(pools or {})
        if context.stream is not None:
            self._computation = context.stream.stream
        else:
            # Replays on this stream read the context's backing, so the
            # context accounts for it before releasing after a failure.
            self._computation = torch.cuda.Stream(device=context._device)
            context._graph_streams.add(self._computation)
        # A driver stream is never one of the pool's, so no context or lane
        # stream can coincide with it.
        self._raw_capture, self._capture = create_sibling_stream(
            self._computation, "graph capture"
        )
        # The captured graph, its retained output views and the captured call.
        self._captured: (
            tuple[_Capture, ResultT, Callable[[], ResultT]] | None
        ) = None
        self._closed = False

    @property
    def context(self) -> ExecutionContext:
        """Borrow the execution context whose backing this graph reads."""
        if self._context is None:
            raise CUDAGraphError("CUDA graph is closed")
        return self._context

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
        context = self._context
        if self._closed or context is None or self._captured is not None:
            raise CUDAGraphError("capture requires an open uncaptured graph")
        context._open()

        device = context._device
        device_pool = self.pools.get(device)
        captured = _Capture(
            self._capture,
            self._computation,
            None
            if device_pool is None
            else torch.cuda._POOL_HANDLE(device_pool.id),
        )
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

                try:
                    # Retire warmup work before capture. Do this once for
                    # the complete call: freeing pools between segments
                    # would defeat their shared allocation lifetime.
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                    with ExitStack() as bindings:
                        if context.weights is not None:
                            bindings.enter_context(
                                context.weights.capture(captured.submit)
                            )
                        captured.begin()
                        try:
                            with (
                                torch.cuda.stream(self._computation),
                                context.activate(),
                            ):
                                output = call()
                        finally:
                            captured.end()
                    for graph in captured.graphs:
                        graph.instantiate()

                    if context.stream is not None:
                        cu = driver()
                        transfers = context._transfers
                        streams = (
                            self._capture,
                            self._computation,
                            *(
                                transfers.streams.values()
                                if transfers is not None
                                else ()
                            ),
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
                        for graph in captured.graphs:
                            verify_graph_context(graph, expected)
                finally:
                    if restore is not None:
                        with context.activate():
                            restore()
        except BaseException as error:
            try:
                captured.reset()
            except BaseException as cleanup:
                error.add_note(f"CUDA graph cleanup failed: {cleanup!r}")
            raise CUDAGraphError(
                f"CUDA graph capture failed on {device}: {error}"
            ) from error
        finally:
            # Keep the caller's stream ordered after capture-side work.
            current.wait_stream(self._computation)
            current.wait_stream(self._capture)

        self._captured = captured, output, call

    def replay(self) -> ResultT:
        """Replay the captured call and return its retained output views."""
        context = self._context
        if self._closed or context is None or self._captured is None:
            raise CUDAGraphError("replay requires an open captured graph")
        graph, output, _ = self._captured
        context._open()
        try:
            with context.activate():
                graph.replay()
        except BaseException as error:
            raise CUDAGraphError(
                f"CUDA graph replay failed: {error}"
            ) from error
        return output

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
            if self._captured is not None:
                self._captured[0].reset()
        finally:
            self._captured = None
            self.pools.clear()
            if self._context is not None:
                self._context._graph_streams.discard(self._computation)
            self._context = None
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
            # Release normally after an exception when capture and replay work
            # has finished; unfinished work keeps the graph's backing alive.
            self.close(
                aborted=exc is not None
                and not streams_idle((self._computation, self._capture))
            )
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"CUDA graph cleanup failed: {error!r}")
