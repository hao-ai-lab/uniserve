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
        self._completion_wake = wake

    @property
    def reserved(self) -> int:
        with self._lock:
            return self._reserved

    def reserve(self) -> CpuTaskReservation:
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
            reservation.release()
            wake = self._completion_wake
            if wake is not None:
                wake()

        future.add_done_callback(completed)
        return future

    def _release(self, reservation: CpuTaskReservation) -> None:
        with self._lock:
            if reservation._pool is not self or reservation._released:
                return
            reservation._released = True
            self._reserved -= 1
            if self._reserved < 0:
                raise RuntimeError("worker CPU task reservation underflow")

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class CpuTaskReservation:
    """One registration-visible CPU task slot with idempotent release."""

    __slots__ = ("_pool", "_submitted", "_released")

    def __init__(self, pool: CpuPool) -> None:
        self._pool = pool
        self._submitted = False
        self._released = False

    @property
    def active(self) -> bool:
        return not self._released

    def submit(
        self,
        function: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> concurrent.futures.Future[_T]:
        return self._pool._submit(self, function, *args, **kwargs)

    def release(self) -> None:
        self._pool._release(self)

    def abandon(self) -> None:
        if not self._submitted:
            self.release()
