"""CPU admission and pinned-input lifetime at the public executor boundary."""

import gc
import weakref
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event

import pytest
import torch

from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.host import HostLane
from uniserve_worker.storage.output import OutputPool

pytestmark = pytest.mark.unit


def test_submitted_task_retains_capacity_until_actual_completion() -> None:
    pool = HostLane(max_inflight=1, workers=1)
    entered, finish = Event(), Event()

    def work(value: int, *, offset: int) -> int:
        entered.set()
        assert finish.wait(5)
        return value + offset

    task = pool.reserve()
    result = task.submit(work, 15, offset=2)
    try:
        assert entered.wait(5)
        task.abandon()
        with pytest.raises(WorkerError, match="capacity is exhausted"):
            pool.reserve()
        assert not task.done()
        finish.set()
        assert result.result(timeout=5) == 17
        replacement = pool.reserve()
        replacement.abandon()
        assert replacement.cancelled()
    finally:
        finish.set()
        pool.close()


def test_close_cancels_unsubmitted_dependency_and_drains_submitted_work() -> (
    None
):
    pool = HostLane(max_inflight=2, workers=1)
    predecessor = pool.reserve()
    successor = pool.reserve().configure(lambda: 2, dependencies=(predecessor,))
    successor.submit_if_ready()
    # Closing cancels unsubmitted tasks outside the admission lock. A running
    # dependent can then fail and release its capacity while shutdown waits.
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(pool.close).result(timeout=5)
    assert predecessor.cancelled()
    with pytest.raises(CancelledError):
        successor.result()
    with pytest.raises(WorkerError, match="closed"):
        pool.reserve()
    assert pool.reserved == 0


def test_abort_returns_before_running_work_and_cancels_queued_work() -> None:
    pool = HostLane(max_inflight=2, workers=1)
    entered, finish, queued_ran = Event(), Event(), Event()

    def work() -> int:
        entered.set()
        assert finish.wait(5)
        return 17

    running = pool.reserve().submit(work)
    try:
        assert entered.wait(5)
        queued = pool.reserve().submit(queued_ran.set)
        pool.abort()
        assert not running.done()
        with pytest.raises(WorkerError, match="closed"):
            pool.reserve()
        finish.set()
        assert running.result(timeout=5) == 17
        with pytest.raises(CancelledError):
            queued.result(timeout=5)
        assert not queued_ran.is_set()
    finally:
        finish.set()
        pool.close()


def test_done_does_not_submit_and_failure_releases_capacity() -> None:
    pool = HostLane(max_inflight=1, workers=1)
    called = Event()
    error = ValueError("encoding failed")

    def fail() -> None:
        called.set()
        raise error

    task = pool.reserve().configure(fail)
    try:
        assert not task.done()
        assert not called.is_set()
        task.submit_if_ready()
        with pytest.raises(ValueError, match="encoding failed") as raised:
            task.result(timeout=5)
        assert raised.value is error
        assert task.exception() is error
        assert pool.reserved == 0
    finally:
        pool.close()


def test_queued_cancellation_returns_unused_capacity() -> None:
    pool = HostLane(max_inflight=2, workers=1)
    entered, finish, queued_ran, released = (Event() for _ in range(4))

    def work() -> None:
        entered.set()
        assert finish.wait(5)

    pool.reserve().submit(work)
    try:
        assert entered.wait(5)
        queued = pool.reserve().configure(queued_ran.set, release=released.set)
        queued.submit_if_ready()
        assert queued.cancel()
        with pytest.raises(CancelledError):
            queued.result(timeout=0)
        assert released.is_set()
        replacement = pool.reserve()
        replacement.abandon()
    finally:
        finish.set()
        pool.close()

    assert released.is_set()
    assert not queued_ran.is_set()
    assert pool.reserved == 0


def test_result_observers_can_reserve_returned_capacity() -> None:
    pool = HostLane(max_inflight=1, workers=1)
    finish, notified = Event(), Event()
    observed: list[int] = []

    def work() -> int:
        assert finish.wait(5)
        return 23

    def observe(task) -> None:
        replacement = pool.reserve()
        replacement.abandon()
        observed.append(task.result())
        notified.set()

    task = pool.reserve().submit(work)
    try:
        task.add_done_callback(observe)
        with pytest.raises(TimeoutError):
            task.result(timeout=0)
        finish.set()
        task.result(timeout=5)
        assert notified.wait(5)
    finally:
        finish.set()
        pool.close()

    assert observed == [23]
    task.add_done_callback(
        lambda completed: observed.append(completed.result())
    )
    assert observed == [23, 23]


def test_completed_result_cycles_release_their_payload() -> None:
    class Payload:
        task = None

    pool = HostLane(max_inflight=1, workers=1)
    value = Payload()
    reference = weakref.ref(value)
    try:
        task = pool.reserve().submit(lambda payload: payload, value)
        value.task = task
        assert task.result(timeout=5) is value
    finally:
        pool.close()

    del task, value
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("outcome", ("success", "failure", "cancelled"))
def test_cancel_preserves_input_until_its_producer_completes(
    outcome: str,
) -> None:
    pool = HostLane(max_inflight=1, workers=1)
    copied: Completion = Completion()
    released = Event()
    task = pool.reserve().configure(
        lambda: 1,
        input_ready=copied.done,
        input_completion=lambda: copied,
        release=released.set,
    )
    try:
        task.submit_if_ready()
        task.abandon()
        assert task.cancelled()
        assert not released.is_set()
        if outcome == "success":
            copied.set_result(None)
        elif outcome == "failure":
            copied.set_exception(RuntimeError("device completion unknown"))
        else:
            copied.cancel()
        assert released.is_set() == (outcome == "success")
    finally:
        pool.close()


def test_abandoned_output_remains_readable_until_cpu_reader_finishes() -> None:
    events = EventPool()
    outputs = OutputPool(capacity=1, max_words=8, event_pool=events)
    pool = HostLane(max_inflight=1, workers=1)
    entered, finish = Event(), Event()
    buffer = outputs.acquire(1, token_capacity=8)
    capture = buffer.capture_bytes(torch.tensor([3, 5, 7], dtype=torch.uint8))
    buffer.seal()

    def read() -> list[int]:
        entered.set()
        assert finish.wait(5)
        return capture.tolist()

    task = pool.reserve().configure(
        read,
        input_ready=buffer.ready,
        input_completion=buffer.completion,
        release=buffer.retain_cpu_reader(),
    )
    try:
        task.submit_if_ready()
        assert entered.wait(5)
        task.abandon()
        buffer.abandon()
        with pytest.raises(WorkerError, match="leases are active"):
            outputs.acquire(1, token_capacity=8)
        finish.set()
        assert task.result(timeout=5) == [3, 5, 7]
        replacement = outputs.acquire(1, token_capacity=8)
        replacement.abandon()
    finally:
        finish.set()
        pool.close()
        outputs.close()
        events.close()
