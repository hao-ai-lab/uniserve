"""Cooperative numerical microbatches on independent CUDA streams."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import ContextVar, copy_context
from functools import partial
from threading import Condition, Lock
from typing import TYPE_CHECKING, TypeVar

import torch

if TYPE_CHECKING:
    from .execution import ExecutionContext

ResultT = TypeVar("ResultT")
_yield: ContextVar[Callable[[], None] | None] = ContextVar(
    "microbatch_yield", default=None
)


def yield_microbatch() -> None:
    """Resume the next microbatch after posting an expert dispatch.

    Outside a ``Microbatches`` invocation there is no other host turn. The
    bound expert operation calls this after its send and before consuming
    the result, while the numerical model keeps its ordinary call stack.
    """
    callback = _yield.get()
    if callback is not None:
        callback()


class Microbatches:
    """Run ordinary numerical calls with cooperative expert overlap.

    Each borrowed context has its own stream, scratch and expert exchange;
    immutable model weights may be shared. Dedicated host threads preserve
    the model's local variables across yield points and retain their warmed
    CUDA library handles. All contexts and exchanges outlive this owner and
    every graph capturing its calls.

    Calls start in index order and yield after each expert dispatch. Returning
    removes a call from the rotation. A host exception wakes all suspended
    calls and is re-raised on the caller; the distributed owner still handles
    process failure and communication retirement. Invocations cannot overlap.

    The caller's current CUDA stream forks into each context stream and joins
    their work before returning. Thus the existing ``CUDAGraph`` can capture
    a complete invocation and replay it without running host threads again.
    Warm every numerical specialization through this owner before capture.
    """

    def __init__(self, contexts: Sequence[ExecutionContext]):
        self.contexts = tuple(contexts)
        # One borrowed stream per context, in context order.
        self._streams = tuple(
            context.stream
            for context in self.contexts
            if context.stream is not None
        )
        if not self.contexts or len(self._streams) != len(self.contexts):
            raise ValueError("microbatches require explicit CUDA streams")
        self.device = self._streams[0].device
        streams = [stream.stream for stream in self._streams]
        if any(stream.device != self.device for stream in streams) or len(
            {stream.cuda_stream for stream in streams}
        ) != len(streams):
            raise ValueError("microbatches require distinct streams on one GPU")

        exchanges = [
            context.experts
            for context in self.contexts
            if context.experts is not None
        ]
        if exchanges:
            exchanges[0].bind_microbatches(exchanges)

        self._condition = Condition()
        self._running = Lock()
        self._active: list[int] = []
        self._turn = 0
        self._error: BaseException | None = None
        self._closed = False
        self._threads = tuple(
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"microbatch-{i}"
            )
            for i in range(len(self.contexts))
        )
        try:
            for thread, context in zip(
                self._threads, self.contexts, strict=True
            ):
                thread.submit(self._warm, context).result()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _warm(context):
        with torch.cuda.device(context.stream.device), context.activate():
            # cuBLAS handles are thread-local. Creating one during graph
            # capture can synchronize or allocate outside the capture pool.
            torch.cuda.current_blas_handle()

    def _wait(self, index):
        self._condition.wait_for(
            lambda: self._turn == index or self._error is not None
        )
        if self._error is not None:
            raise RuntimeError("microbatch execution aborted") from self._error

    def _next(self, index):
        if self._active:
            self._turn = next(
                (other for other in self._active if other > index),
                self._active[0],
            )
        self._condition.notify_all()

    def _resume(self, index):
        with self._condition:
            self._next(index)
            self._wait(index)

    @torch.inference_mode()
    def _execute(self, index, call):
        context = self.contexts[index]
        token = _yield.set(partial(self._resume, index))
        try:
            with self._condition:
                self._wait(index)
            with torch.cuda.device(self.device), context.activate():
                return call()
        except BaseException as error:
            with self._condition:
                if self._error is None:
                    self._error = error
            raise
        finally:
            _yield.reset(token)
            with self._condition:
                self._active.remove(index)
                self._next(index)

    def __call__(self, calls: Sequence[Callable[[], ResultT]]) -> list[ResultT]:
        """Run one call per context and return results in that same order."""
        if len(calls) != len(self.contexts):
            raise ValueError("each microbatch context needs one numerical call")
        if not self._running.acquire(blocking=False):
            raise RuntimeError("microbatch invocation is already running")
        try:
            if self._closed:
                raise RuntimeError("microbatch execution is closed")
            self._active = list(range(len(calls)))
            self._turn, self._error = 0, None
            current = torch.cuda.current_stream(self.device)
            for stream in self._streams:
                stream.wait(current)
            futures = [
                thread.submit(copy_context().run, self._execute, index, call)
                for index, (thread, call) in enumerate(
                    zip(self._threads, calls, strict=True)
                )
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except BaseException as error:
                    # A caller-side interruption must also release suspended
                    # host turns. Wait for every thread before reusing state.
                    with self._condition:
                        if self._error is None:
                            self._error = error
                        self._condition.notify_all()
            # An interruption of future.result() need not mean that future
            # completed. Every host turn must retire before state is reused.
            wait(futures)
            for stream in self._streams:
                current.wait_stream(stream.stream)
            if self._error is not None:
                raise self._error
            return results
        finally:
            self._running.release()

    def close(self) -> None:
        """Release threads after invocations and captured readers retire."""
        if not self._running.acquire(blocking=False):
            raise RuntimeError("cannot close running microbatches")
        try:
            if not self._closed:
                self._closed = True
                for thread in self._threads:
                    thread.shutdown(wait=True)
        finally:
            self._running.release()
