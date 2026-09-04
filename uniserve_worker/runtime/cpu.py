"""Bounded worker-owned CPU job execution."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from threading import Lock
from typing import ParamSpec, TypeVar

from ..foundation.errors import resource_error

__all__ = ["CpuPool", "CpuTaskReservation"]

_P = ParamSpec("_P")
_T = TypeVar("_T")


class CpuPool:
    """A fixed worker set with bounded registered tasks."""

    def __init__(self, *, capacity: int, workers: int) -> None:
        """Create a bounded thread pool and its registration accounting."""

        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("CPU task capacity must be positive")
        worker_count = int(workers)
        if worker_count < 1 or worker_count > self.capacity:
            raise ValueError("CPU worker count must be within task capacity")
        self._reserved = 0
        self._lock = Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="worker-cpu",
        )
        self._completion_wake: Callable[[], None] | None = None

    def set_completion_wake(self, wake: Callable[[], None]) -> None:
        """Install the event-loop callback invoked when a submitted CPU task finishes."""

        self._completion_wake = wake

    @property
    def reserved(self) -> int:
        """Count executor slots held by submitted or not-yet-submitted reservations."""

        with self._lock:
            return self._reserved

    def reserve(self) -> CpuTaskReservation:
        """Reserve one executor slot without submitting work."""

        with self._lock:
            if self._reserved >= self.capacity:
                raise resource_error("worker CPU task capacity is exhausted")
            self._reserved += 1
        return CpuTaskReservation(self)

    def _submit(
        self,
        reservation: CpuTaskReservation,
        function: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> concurrent.futures.Future[_T]:
        """Consume a reservation, submit host work, and release capacity after completion."""

        with self._lock:
            if reservation._pool is not self or reservation._released:
                raise RuntimeError("CPU task reservation is not active")
            if reservation._submitted:
                raise RuntimeError("CPU task reservation was submitted more than once")
            reservation._submitted = True
        try:
            future = self._executor.submit(function, *args, **kwargs)
        except BaseException:
            reservation.release()
            raise

        def completed(_future: object) -> None:
            """Release task capacity and wake the completion loop exactly once."""

            reservation.release()
            wake = self._completion_wake
            if wake is not None:
                wake()

        future.add_done_callback(completed)
        return future

    def _release(self, reservation: CpuTaskReservation) -> None:
        """Return an unused CPU task reservation to bounded capacity."""

        with self._lock:
            if reservation._pool is not self or reservation._released:
                return
            reservation._released = True
            self._reserved -= 1
            if self._reserved < 0:
                raise RuntimeError("worker CPU task reservation underflow")

    def close(self) -> None:
        """Reject new reservations and shut down the bounded executor."""

        self._executor.shutdown(wait=True, cancel_futures=False)


class CpuTaskReservation:
    """One registration-visible CPU task slot with idempotent release."""

    __slots__ = ("_pool", "_submitted", "_released")

    def __init__(self, pool: CpuPool) -> None:
        """Take ownership of one reserved slot until submission or explicit release."""

        self._pool = pool
        self._submitted = False
        self._released = False

    @property
    def active(self) -> bool:
        """Indicate whether this reservation still owns one CPU executor slot."""

        return not self._released

    def submit(
        self,
        function: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> concurrent.futures.Future[_T]:
        """Consume this reservation by submitting exactly one host callable."""

        return self._pool._submit(self, function, *args, **kwargs)

    def release(self) -> None:
        """Return an unused reservation to the CPU pool."""

        self._pool._release(self)

    def abandon(self) -> None:
        """Release the slot only when no task was submitted through this reservation."""

        if not self._submitted:
            self.release()
