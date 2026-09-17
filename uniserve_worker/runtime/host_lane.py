"""The rank's host lane: bounded host tasks with leased inputs."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from functools import partial
from threading import Lock
from typing import Any, ParamSpec, TypeVar

from uniserve.profiling import profile_range
from uniserve.runtime.resources import close_resources

from ..foundation.errors import resource_error

__all__ = ["HostLane", "HostTask"]

_P = ParamSpec("_P")
_T = TypeVar("_T")


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
        self._tasks: set[HostTask] = set()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="worker-host-lane"
        )
        self._completion_wake: Callable[[], None] | None = None

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
        error: BaseException | None = None
        with self._lock:
            if self._closed or task not in self._tasks:
                raise RuntimeError("host task is no longer admitted")
            if task._future is not None:
                raise RuntimeError("host task was submitted more than once")
            if task._action is None:
                raise RuntimeError("host task has no action")
            # Serialize executor admission with close; callbacks run
            # outside this lock.
            try:
                task._future = self._executor.submit(task._run)
            except BaseException as failure:
                error = failure
                self._tasks.remove(task)
        if error is not None:
            task._finish(error=error)
            raise error
        assert task._future is not None
        task._future.add_done_callback(task._completed)

    def _remove(self, task: HostTask) -> None:
        with self._lock:
            self._tasks.discard(task)

    def close(self) -> None:
        """Reject admission, cancel unsubmitted tasks and drain host readers."""
        with self._lock:
            self._closed = True
            unused = tuple(task for task in self._tasks if task._future is None)
            self._tasks.difference_update(unused)
        # Cancelling a promise can wake dependent jobs that need the pool lock.
        try:
            close_resources(*(task._cancel for task in unused))
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)


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
        self._future: concurrent.futures.Future[Any] | None = None
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

        to this task.
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
        if self.promise.done() or self._future is not None:
            return
        if self._input_ready is None or self._input_ready():
            self._pool._submit(self)

    def _run(self) -> Any:
        for dependency in self._dependencies:
            dependency.result()
        assert self._action is not None
        with profile_range(self._profile_name):
            return self._action()

    def _completed(self, future: concurrent.futures.Future[Any]) -> None:
        try:
            value = future.result()
        except BaseException as error:
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
            if self not in self._pool._tasks or self._future is not None:
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
