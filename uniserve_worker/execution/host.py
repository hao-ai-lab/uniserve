"""The rank's host lane: bounded host tasks with leased inputs.

A host task is a callable that runs on one of the lane's threads. The lane's
capacity is the one its rank advertises, so the engine's lane ledger never
hands the rank more host work than its threads serve.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from functools import partial
from queue import SimpleQueue
from threading import Lock, Thread
from typing import Any, ParamSpec, TypeVar

from uniserve.profiling import profile_range
from uniserve.runtime.resources import close_resources
from uniserve_worker.errors import resource_error

__all__ = ["HostLane", "HostTask"]

_P = ParamSpec("_P")
_T = TypeVar("_T")


class _Worker:
    """One lane thread and the tasks queued for it."""

    def __init__(self, lane: HostLane, index: int) -> None:
        self.queue: SimpleQueue[HostTask | None] = SimpleQueue()
        self.queued = 0
        self.thread = Thread(
            target=self._serve,
            args=(lane,),
            name=f"worker-host-lane-{index}",
            daemon=True,
        )
        self.thread.start()

    def _serve(self, lane: HostLane) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            try:
                if lane._aborted:
                    item._cancel()
                else:
                    item._run()
            finally:
                lane._dequeued(self, item)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()


class HostLane:
    """One rank's host execution resource.

    The lane admits at most ``max_inflight`` tasks and owns each one until it is
    cancelled before submission or actually completes. Its capacity is the one
    the rank advertises, so the engine's lane ledger never overcommits it.
    """

    def __init__(self, *, max_inflight: int, workers: int) -> None:
        self.max_inflight = int(max_inflight)
        if self.max_inflight < 1:
            raise ValueError("host lane capacity must be positive")
        if workers < 1 or workers > self.max_inflight:
            raise ValueError("host lane workers must be within its capacity")
        self._lock = Lock()
        self._closed = False
        self._aborted = False
        self._tasks: set[HostTask] = set()
        self._completion_wake: Callable[[], None] | None = None
        self._workers: list[_Worker] = []
        try:
            for index in range(int(workers)):
                self._workers.append(_Worker(self, index))
        except BaseException:
            close_resources(*(worker.close for worker in self._workers))
            raise

    def set_completion_wake(self, wake: Callable[[], None] | None) -> None:
        self._completion_wake = wake

    @property
    def reserved(self) -> int:
        """Count admitted tasks, including work whose result was abandoned."""
        with self._lock:
            return len(self._tasks)

    def reserve(self) -> HostTask:
        """Admit a task under the pool's capacity lease before its inputs.

        exist.
        """
        with self._lock:
            if self._closed:
                raise resource_error("worker host lane is closed")
            if len(self._tasks) >= self.max_inflight:
                raise resource_error("worker host lane capacity is exhausted")
            task = HostTask(self)
            self._tasks.add(task)
            return task

    def _submit(self, task: HostTask) -> None:
        with self._lock:
            if self._closed or task not in self._tasks:
                raise RuntimeError("host task is no longer admitted")
            if task._submitted:
                raise RuntimeError("host task was submitted more than once")
            if task._action is None:
                raise RuntimeError("host task has no action")
            worker = min(self._workers, key=lambda w: w.queued)
            worker.queued += 1
            task._submitted = True
        worker.queue.put(task)

    def _dequeued(self, worker: _Worker, task: HostTask) -> None:
        with self._lock:
            worker.queued -= 1

    def _remove(self, task: HostTask) -> None:
        with self._lock:
            self._tasks.discard(task)

    def abort(self) -> None:
        """Stop admission without waiting for device inputs.

        A running thread may hold a CUDA-dependent read; its resources remain
        owned by the failed worker until process exit.
        """
        with self._lock:
            self._closed = self._aborted = True
            self._completion_wake = None
        for worker in self._workers:
            worker.queue.put(None)

    def close(self) -> None:
        """Reject admission, cancel unsubmitted tasks and drain host readers."""
        with self._lock:
            self._closed = True
            unused = tuple(task for task in self._tasks if not task._submitted)
            self._tasks.difference_update(unused)
        # Cancelling a promise can wake dependent jobs that need the pool lock.
        try:
            close_resources(*(task._cancel for task in unused))
        finally:
            close_resources(*(worker.close for worker in self._workers))


class HostTask:
    """One admitted host call: its action, result promise and input lease.

    Configure it after reserving the call's lane capacity, once its input
    exists. The worker calls ``submit_if_ready`` to advance deferred actions;
    ``ready`` is pure. Abandoning submitted work neither cancels its reads nor
    returns its capacity.
    """

    def __init__(self, pool: HostLane) -> None:
        self._pool = pool
        self.promise: concurrent.futures.Future[Any] = (
            concurrent.futures.Future()
        )
        self._submitted = False
        self._action: Callable[[], Any] | None = None
        self._dependencies: tuple[concurrent.futures.Future[Any], ...] = ()
        self._input_ready: Callable[[], bool] | None = None
        self._input_completion: (
            Callable[[], concurrent.futures.Future[None]] | None
        ) = None
        self._release: Callable[[], None] | None = None
        self._profile_name = "uniserve.host"

    def configure(
        self,
        action: Callable[[], Any],
        *,
        dependencies: tuple[concurrent.futures.Future[Any], ...] = (),
        input_ready: Callable[[], bool] | None = None,
        input_completion: Callable[[], concurrent.futures.Future[None]]
        | None = None,
        release: Callable[[], None] | None = None,
        profile_name: str = "uniserve.host",
    ) -> HostTask:
        """Attach an action and transfer its input release responsibility.

        The action runs on a lane thread once its dependencies complete.
        """
        with self._pool._lock:
            if self not in self._pool._tasks or self._pool._closed:
                raise RuntimeError("host task is no longer admitted")
            if self._action is not None:
                raise RuntimeError("host task was configured more than once")
            self._action = action
            self._dependencies = dependencies
            self._input_ready = input_ready
            self._input_completion = input_completion
            self._release = release
            self._profile_name = profile_name
        return self

    def submit(
        self, function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> concurrent.futures.Future[_T]:
        """Submit an immediate action using already reserved capacity."""
        self.configure(partial(function, *args, **kwargs))
        self._pool._submit(self)
        return self.promise

    def submit_if_ready(self) -> None:
        """Submit a configured action after its input copy becomes.

        CPU-readable.
        """
        if self.promise.done() or self._submitted:
            return
        if self._input_ready is None or self._input_ready():
            self._pool._submit(self)

    def _run(self) -> None:
        """Execute on the lane thread that dequeued this task."""
        try:
            for dependency in self._dependencies:
                dependency.result()
            action = self._action
            assert action is not None
            with profile_range(self._profile_name):
                value = action()
        except BaseException as error:  # noqa: BLE001 - reported to the promise
            self._finish(error=error)
        else:
            self._finish(value=value)

    def _finish(
        self, *, value: Any = None, error: BaseException | None = None
    ) -> None:
        try:
            self._release_input()
        except BaseException as release_error:
            if error is None:
                error = release_error
        finally:
            self._action = None
            self._dependencies = ()
            self._pool._remove(self)

        if error is None:
            self.promise.set_result(value)
        else:
            self.promise.set_exception(error)

        wake = self._pool._completion_wake
        if wake is not None:
            wake()

    def _release_input(self) -> None:
        release, self._release = self._release, None
        if release is not None:
            release()
        self._input_ready = None
        self._input_completion = None

    def ready(self) -> bool:
        return self.promise.done()

    def result(self) -> Any:
        """Return the completed value without blocking the worker's event.

        loop.
        """
        if not self.ready():
            raise RuntimeError(
                "host task result was observed before completion"
            )
        return self.promise.result(timeout=0)

    def abandon(self) -> None:
        with self._pool._lock:
            if self not in self._pool._tasks or self._submitted:
                return
            self._pool._tasks.remove(self)
        self._cancel()

    def _cancel(self) -> None:
        self.promise.cancel()
        self._action = None
        self._dependencies = ()
        # Cancellation cannot authorize reuse while a preceding D2H
        # still writes.
        if self._input_completion is not None:
            completion = self._input_completion()
            if not completion.done():
                completion.add_done_callback(
                    lambda _future: self._release_input()
                )
                return
        self._release_input()
