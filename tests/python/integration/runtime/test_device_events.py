"""Pooled CUDA event ownership, timing, and shutdown through the public API."""

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve.runtime import EventPool, EventPoolError
from uniserve_worker._uniserve_ipc import StreamSignal

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_rejected_release_preserves_the_lease_and_deferred_owner():
    pool = EventPool()
    event = pool.acquire("cuda:0")
    pool.retain(event, "cuda:0")
    owner = torch.ones(1)
    retained = weakref.ref(owner)
    completed = []

    def release_completed():
        # A callback may immediately use this pool again.
        successor = pool.acquire("cuda:0")
        pool.retain(successor, "cuda:0")
        pool.record(successor, "cuda:0")
        successor.synchronize()
        pool.release(successor)
        completed.append(True)

    try:
        with blocked_stream("cuda:0") as stream:
            with torch.cuda.stream(stream):
                pool.record(event, "cuda:0")
            with pytest.raises(EventPoolError, match="query-ready"):
                pool.release(event)

            pool.defer_release((event,), owner, completed=release_completed)
            del owner
            pool.reap()
            assert retained() is not None
            assert completed == []

        pool.reap()
        assert retained() is None
        assert completed == [True]
    finally:
        pool.close()


def test_close_waits_outside_the_pool_lock_and_releases_the_gil():
    pool = EventPool()
    signal = StreamSignal()
    pool.set_completion_wake(signal.schedule)
    event = pool.acquire("cuda:0")
    pool.retain(event, "cuda:0")
    entering_close = threading.Event()

    def close():
        entering_close.set()
        pool.close()

    with ThreadPoolExecutor(max_workers=1) as threads:
        with blocked_stream("cuda:0") as stream:
            with torch.cuda.stream(stream):
                pool.record(event, "cuda:0")
            pool.schedule_completion_wake("cuda:0", event)
            closing = threads.submit(close)
            assert entering_close.wait(5)
            assert not closing.done()
            assert not event.query()
            # This takes the pool lock while close waits for the device.
            pool.defer_release((event,), torch.ones(1))

        closing.result(timeout=5)
    signal.consume()
    with pytest.raises(EventPoolError, match="closed"):
        pool.acquire("cuda:0")


def test_native_event_timing_uses_cuda_milliseconds():
    pool = EventPool()
    stream = torch.cuda.Stream(device=0)
    start = pool.acquire("cuda:0", timing=True)
    end = pool.acquire("cuda:0", timing=True)
    pool.retain(start, "cuda:0")
    pool.retain(end, "cuda:0")
    outer_start, inner_start, inner_end, outer_end = (
        torch.cuda.Event(enable_timing=True) for _ in range(4)
    )

    try:
        with torch.cuda.stream(stream):
            outer_start.record(stream)
            pool.record(start, "cuda:0")
            inner_start.record(stream)
            torch.cuda._sleep(1_000_000)
            inner_end.record(stream)
            pool.record(end, "cuda:0")
            outer_end.record(stream)
        end.synchronize()
        outer_end.synchronize()

        # Nested intervals on one stream bound the duration independently of
        # device clock rate and the amount of host time spent submitting work.
        assert (
            inner_start.elapsed_time(inner_end)
            <= start.elapsed_time(end)
            <= outer_start.elapsed_time(outer_end)
        )
        pool.release(start)
        pool.release(end)
    finally:
        pool.close()
