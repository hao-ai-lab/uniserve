"""Storage completions across producer and consumer threads."""

import gc
import weakref
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event

import pytest

from uniserve_worker._uniserve_ipc import Completion

pytestmark = pytest.mark.unit


def test_completion_wakes_waiters_and_allows_reentrant_observers() -> None:
    completion = Completion()
    entered, observed = Event(), Event()

    def observe(done: Completion) -> None:
        done.result(timeout=0)
        done.add_done_callback(lambda _: observed.set())

    completion.add_done_callback(observe)

    def wait() -> None:
        entered.set()
        completion.result(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as workers:
        waiter = workers.submit(wait)
        assert entered.wait(5)
        completion.set_result(None)
        assert waiter.result(timeout=5) is None

    assert observed.is_set()
    assert completion.exception(timeout=0) is None
    assert not completion.cancel()
    with pytest.raises(RuntimeError, match="already completed"):
        completion.set_result(None)


def test_failure_and_cancellation_preserve_the_terminal_outcome() -> None:
    completion = Completion()
    with pytest.raises(TimeoutError):
        completion.result(timeout=0)
    assert not completion.done()

    error = RuntimeError("device completion unknown")
    completion.set_exception(error)
    assert completion.exception() is error
    with pytest.raises(RuntimeError) as raised:
        completion.result()
    assert raised.value is error

    cancelled = Completion()
    assert cancelled.cancel()
    assert cancelled.done() and cancelled.cancelled()
    with pytest.raises(CancelledError):
        cancelled.result()
    with pytest.raises(CancelledError):
        cancelled.exception()


def test_unreachable_observers_release_retained_storage() -> None:
    class Storage:
        def __init__(self) -> None:
            self.completion = Completion()
            self.completion.add_done_callback(self.finished)

        def finished(self, completion: Completion) -> None:
            completion.result()

    storage = Storage()
    retained = weakref.ref(storage)
    del storage
    gc.collect()
    assert retained() is None
