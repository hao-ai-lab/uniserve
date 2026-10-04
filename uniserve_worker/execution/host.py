"""The rank's host lane: bounded host tasks with leased inputs.

A host task is a callable that runs on one of the lane's threads. The lane's
capacity is the one its rank advertises, so the engine's lane ledger never
hands the rank more host work than its threads serve.

``HostLane.reserve`` admits a task, taking one unit of capacity, before its
input exists. ``HostTask.submit`` runs an immediate action; a deferred one is
attached with ``HostTask.configure`` together with its input lease, and the
worker's executor calls ``HostTask.submit_if_ready`` until the input is
CPU-readable. A lane thread runs the action and resolves ``HostTask.promise``.
Capacity returns only when an unsubmitted task is abandoned or cancelled by
``HostLane.close``, or when a task actually finishes; never when a caller
stops waiting for a submitted one.

The worker builds one lane per rank; ``CacheImports`` builds its own for KV
cache imports.
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
from uniserve_worker._uniserve_ipc import Completion
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
        # ``None`` is the stop sentinel from ``close`` or ``abort``. After an
        # abort, tasks still queued ahead of it are cancelled, not run.
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
    Submitted tasks go to the thread with the fewest queued or running tasks;
    ``workers`` must lie in ``[1, max_inflight]``.
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
        """Register the callback a lane thread calls after a task finishes.

        The worker uses it to wake result polling; ``abort`` clears it.
        """
        self._completion_wake = wake

    @property
    def reserved(self) -> int:
        """Count admitted tasks, including work whose result was abandoned."""
        with self._lock:
            return len(self._tasks)

    def reserve(self) -> HostTask:
        """Admit a task under the lane's capacity before its inputs exist.

        The returned task holds one unit of capacity until it finishes, or
        until ``abandon`` or ``close`` cancels it before submission.

        Raises:
            ResourceError: When the lane is closed or its capacity is
                exhausted.
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

        Returns without joining the lane threads. Tasks already queued are
        cancelled when a thread dequeues them, a running task finishes
        normally, and reserved but unsubmitted tasks are left unresolved.
        A running thread may hold a CUDA-dependent read; its resources remain
        owned by the failed worker until process exit.
        """
        with self._lock:
            self._closed = self._aborted = True
            self._completion_wake = None
        for worker in self._workers:
            worker.queue.put(None)

    def close(self) -> None:
        """Reject admission, cancel unsubmitted tasks and drain host readers.

        Unless the lane was aborted, submitted tasks still run to completion;
        the call returns once every lane thread has drained its queue and
        exited.
        """
        with self._lock:
            self._closed = True
            unused = tuple(task for task in self._tasks if not task._submitted)
            self._tasks.difference_update(unused)
        # As in ``HostTask.abandon``, ``_cancel`` runs after the lane lock is
        # released: it synchronously runs the promise's done-callbacks and,
        # unless the input's producer copy is still in flight, the input
        # ``release`` callback.
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

    The task's ``release`` callback runs at most once: when the task
    finishes or, on cancellation, once the input's producer copy
    (``input_completion``, when given) has completed successfully.

    Attributes:
        promise: Resolves with the action's value or error, or is cancelled
            when the task is cancelled before it runs (``abandon``, ``close``,
            or ``abort`` of a queued task).
    """

    def __init__(self, pool: HostLane) -> None:
        self._pool = pool
        self.promise: concurrent.futures.Future[Any] = (
            concurrent.futures.Future()
        )
        self._submitted = False
        self._action: Callable[[], Any] | None = None
        self._dependencies: tuple[
            concurrent.futures.Future[Any] | Completion, ...
        ] = ()
        self._input_ready: Callable[[], bool] | None = None
        self._input_completion: Callable[[], Completion] | None = None
        self._release: Callable[[], None] | None = None
        self._profile_name = "uniserve.host"

    def configure(
        self,
        action: Callable[[], Any],
        *,
        dependencies: tuple[
            concurrent.futures.Future[Any] | Completion, ...
        ] = (),
        input_ready: Callable[[], bool] | None = None,
        input_completion: Callable[[], Completion] | None = None,
        release: Callable[[], None] | None = None,
        profile_name: str = "uniserve.host",
    ) -> HostTask:
        """Attach an action and transfer its input release responsibility.

        The action runs on a lane thread once its dependencies complete.

        Args:
            action: The host work; its return value resolves ``promise``.
            dependencies: Futures the lane thread waits on before running
                ``action``; a failed or cancelled dependency fails this task
                with that error.
            input_ready: Polled by ``submit_if_ready``; the task is submitted
                once it returns True. None submits on the first poll.
            input_completion: Returns the input copy's completion signal,
                so cancellation defers ``release`` until the copy stops
                writing the leased storage.
            release: Returns the input lease; see the class docstring.
            profile_name: Profiler range name around ``action``.

        Returns:
            This task, for chaining.

        Raises:
            RuntimeError: When the task is no longer admitted, the lane is
                closed, or the task is already configured. Release
                responsibility then stays with the caller.
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
        """Submit an immediate action using already reserved capacity.

        Configures the task with ``function(*args, **kwargs)`` and no input
        lease, submits it, and returns ``promise``.
        """
        self.configure(partial(function, *args, **kwargs))
        self._pool._submit(self)
        return self.promise

    def submit_if_ready(self) -> None:
        """Submit a configured action once its input is CPU-readable.

        Does nothing when the task is already submitted or resolved, or when
        ``input_ready`` still returns False.
        """
        if self.promise.done() or self._submitted:
            return
        if self._input_ready is None or self._input_ready():
            self._pool._submit(self)

    def _run(self) -> None:
        """Execute on the lane thread that dequeued this task."""
        # A task ``cancel`` withdrew while it was queued does not run; its
        # capacity and input lease return here, on the dequeuing thread.
        if not self.promise.set_running_or_notify_cancel():
            self._action = None
            self._dependencies = ()
            try:
                self._release_input()
            finally:
                self._pool._remove(self)
            return
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
        # A release failure is reported only when the action itself succeeded.
        # Capacity returns before the promise resolves, so a caller observing
        # the result can reserve again immediately.
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
        """Return the completed value without blocking the worker's event loop.

        Re-raises the action's error, or ``CancelledError`` for a cancelled
        task.

        Raises:
            RuntimeError: When the task has not completed.
        """
        if not self.ready():
            raise RuntimeError(
                "host task result was observed before completion"
            )
        return self.promise.result(timeout=0)

    def abandon(self) -> None:
        """Cancel an unsubmitted task and return its capacity.

        A no-op for a task that is submitted or no longer admitted: submitted
        work keeps running and holds its capacity until it finishes.
        """
        with self._pool._lock:
            if self not in self._pool._tasks or self._submitted:
                return
            self._pool._tasks.remove(self)
        self._cancel()

    def cancel(self) -> None:
        """Withdraw this task unless its action has started.

        An unsubmitted task is abandoned. A submitted task still queued
        resolves ``promise`` as cancelled now and returns its capacity and
        input lease when a lane thread dequeues it, without running. A
        running or finished task is left to complete.
        """
        with self._pool._lock:
            submitted = self._submitted
        if not submitted:
            self.abandon()
        else:
            # Succeeds only while the promise is pending; ``_run`` marks it
            # running first, so exactly one of the two takes effect.
            self.promise.cancel()

    def _cancel(self) -> None:
        self.promise.cancel()
        self._action = None
        self._dependencies = ()
        # The input's device-to-host copy may still be writing the leased
        # storage; defer the release until that copy completes.
        if self._input_completion is not None:
            completion = self._input_completion()
            completion.add_done_callback(self._input_completed)
            return
        self._release_input()

    def _input_completed(self, completion: Completion) -> None:
        # Failure or cancellation cannot establish that the producer stopped
        # writing. Keep its input lease until physical completion is known.
        if completion.succeeded():
            self._release_input()
