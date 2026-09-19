"""The rank's host lane: bounded host tasks with leased inputs.

A host task is either a callable that runs on one of the lane's threads or a
codec job that the thread hands to its codec process, so no codec ever runs
inside the rank's interpreter. Mux sessions live in the codec process that
first sees them, and every later job of that session is routed to the same
process.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from queue import SimpleQueue
from threading import Lock, Thread
from typing import Any, ParamSpec, TypeVar

from uniserve.profiling import profile_range
from uniserve.runtime.resources import close_resources

from ..foundation.errors import resource_error
from ..media.codec_process import (
    CodecJob,
    CodecProcess,
    MuxClose,
    SessionKey,
    SharedMapping,
)

__all__ = ["HostLane", "HostTask"]

_P = ParamSpec("_P")
_T = TypeVar("_T")

_CODEC_JOBS = CodecJob.__args__


@dataclass(frozen=True, slots=True)
class _Discard:
    """Close a session in the codec process that owns it, outside any task."""

    session: SessionKey


class _Worker:
    """One lane thread and, when the lane runs codecs, its codec process."""

    def __init__(self, lane: HostLane, index: int, codec: bool) -> None:
        self.queue: SimpleQueue[HostTask | _Discard | None] = SimpleQueue()
        self.codec = CodecProcess() if codec else None
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
            if isinstance(item, _Discard):
                if self.codec is not None:
                    try:
                        self.codec.execute(MuxClose(item.session))
                    except Exception:  # noqa: BLE001 - a discarded session
                        pass
                continue
            try:
                item._run(self)
            finally:
                lane._dequeued(self, item)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.codec is not None:
            self.codec.close()


class HostLane:
    """One rank's host execution resource.

    The lane admits at most ``max_inflight`` tasks and owns each one until it is
    cancelled before submission or actually completes. Its capacity is the one
    the rank advertises, so the engine's lane ledger never overcommits it.
    With ``codec`` set, each of its workers owns a codec process that runs the
    lane's codec jobs.
    """

    def __init__(
        self, *, max_inflight: int, workers: int, codec: bool = False
    ) -> None:
        self.max_inflight = int(max_inflight)
        if self.max_inflight < 1:
            raise ValueError("host lane capacity must be positive")
        if workers < 1 or workers > self.max_inflight:
            raise ValueError("host lane workers must be within its capacity")
        self._lock = Lock()
        self._closed = False
        self._tasks: set[HostTask] = set()
        self._sessions: dict[SessionKey, _Worker] = {}
        self._completion_wake: Callable[[], None] | None = None
        self.codec = bool(codec)
        self._workers: list[_Worker] = []
        try:
            for index in range(int(workers)):
                self._workers.append(_Worker(self, index, self.codec))
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

    def attach(self, mapping: SharedMapping) -> None:
        """Let every codec process read media units from a shared mapping."""
        if not self.codec:
            raise RuntimeError("host lane runs no codec processes")
        for worker in self._workers:
            assert worker.codec is not None
            worker.codec.attach(mapping)

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

    def discard_session(self, session: SessionKey) -> None:
        """Drop a mux session whose request ended before its artifact.

        The close runs on the session's process after the jobs already queued
        there, so a job in flight for the session completes or fails on its
        own before the session is gone.
        """
        with self._lock:
            worker = self._sessions.pop(session, None)
        if worker is not None:
            worker.queue.put(_Discard(session))

    def _submit(self, task: HostTask) -> None:
        with self._lock:
            if self._closed or task not in self._tasks:
                raise RuntimeError("host task is no longer admitted")
            if task._submitted:
                raise RuntimeError("host task was submitted more than once")
            if task._action is None:
                raise RuntimeError("host task has no action")
            # A session's jobs share one process, which holds the session's
            # state; other work goes to the worker with the least queued.
            worker = None
            if task._session is not None:
                worker = self._sessions.get(task._session)
            if worker is None:
                worker = min(self._workers, key=lambda w: w.queued)
                if task._session is not None:
                    self._sessions[task._session] = worker
            worker.queued += 1
            task._submitted = True
        worker.queue.put(task)

    def _dequeued(self, worker: _Worker, task: HostTask) -> None:
        with self._lock:
            worker.queued -= 1
            if task._ends_session and task._session is not None:
                if self._sessions.get(task._session) is worker:
                    del self._sessions[task._session]

    def _remove(self, task: HostTask) -> None:
        with self._lock:
            self._tasks.discard(task)

    def close(self) -> None:
        """Reject admission, cancel unsubmitted tasks and drain host readers."""
        with self._lock:
            self._closed = True
            unused = tuple(task for task in self._tasks if not task._submitted)
            self._tasks.difference_update(unused)
            self._sessions.clear()
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
        self._action: Callable[[], Any] | CodecJob | None = None
        self._transform: Callable[[Any], Any] | None = None
        self._session: SessionKey | None = None
        self._ends_session = False
        self._dependencies: tuple[concurrent.futures.Future[Any], ...] = ()
        self._input_ready: Callable[[], bool] | None = None
        self._input_completion: (
            Callable[[], concurrent.futures.Future[None]] | None
        ) = None
        self._release: Callable[[], None] | None = None
        self._profile_name = "uniserve.host"

    def configure(
        self,
        action: Callable[[], Any] | CodecJob,
        *,
        dependencies: tuple[concurrent.futures.Future[Any], ...] = (),
        input_ready: Callable[[], bool] | None = None,
        input_completion: Callable[[], concurrent.futures.Future[None]]
        | None = None,
        release: Callable[[], None] | None = None,
        profile_name: str = "uniserve.host",
        session: SessionKey | None = None,
        ends_session: bool = False,
        transform: Callable[[Any], Any] | None = None,
    ) -> HostTask:
        """Attach an action and transfer its input release responsibility.

        A callable runs on a lane thread; a codec job runs in the thread's
        codec process, on the process that owns ``session`` when one is named.
        ``transform`` turns the job's result into the task's, on the thread.
        """
        if isinstance(action, _CODEC_JOBS) and not self._pool.codec:
            raise RuntimeError("host lane runs no codec processes")
        with self._pool._lock:
            if self not in self._pool._tasks or self._pool._closed:
                raise RuntimeError("host task is no longer admitted")
            if self._action is not None:
                raise RuntimeError("host task was configured more than once")
            self._action = action
            self._transform = transform
            self._session = session
            self._ends_session = bool(ends_session)
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

    def _run(self, worker: _Worker) -> None:
        """Execute on the lane thread that dequeued this task."""
        try:
            for dependency in self._dependencies:
                dependency.result()
            action = self._action
            assert action is not None
            with profile_range(self._profile_name):
                if isinstance(action, _CODEC_JOBS):
                    assert worker.codec is not None
                    value = worker.codec.execute(action)
                else:
                    value = action()
                if self._transform is not None:
                    value = self._transform(value)
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
            self._transform = None
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
