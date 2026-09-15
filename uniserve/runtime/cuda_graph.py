"""Captured numerical calls and the borrowed resources they keep alive."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import Generic, TypeVar

import torch

from .cuda import cuda_value, driver, verify_graph_context
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
        self._stream = context.stream or torch.cuda.Stream(
            device=context._device
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
        self._stream.wait_stream(current)
        try:
            with (
                ExitStack() as scope,
                torch.cuda.device(device),
                torch.cuda.stream(self._stream),
                self.context.activate(),
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
                        stream=self._stream,
                        pool=None if pool is None else pool.id,
                    ):
                        output = call()
                    graph.instantiate()

                    if self.context.stream is not None:
                        cu = driver()
                        streams = (
                            self._stream,
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
            current.wait_stream(self._stream)

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

    def close(self) -> None:
        """Release the graph and its pool references.

        Output views become invalid.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._graph is not None:
                self._graph.reset()
        finally:
            self._graph = self._output = self._call = None
            self.pools.clear()
            self.context = None

    def __enter__(self):
        if self._closed:
            raise CUDAGraphError("CUDA graph is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"CUDA graph cleanup failed: {error!r}")
