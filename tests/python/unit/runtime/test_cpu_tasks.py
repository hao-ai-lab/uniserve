from __future__ import annotations

import time
from threading import Event

import pytest

from uniserve_worker.foundation.errors import ResourceError
from uniserve_worker.server.cpu_tasks import BoundedCpuTaskPool

pytestmark = pytest.mark.unit


def test_cpu_task_slot_reclaims_after_the_registered_future_finishes() -> None:
    pool = BoundedCpuTaskPool(capacity=1, workers=1)
    started = Event()
    release = Event()
    reservation = pool.reserve()

    def execute() -> int:
        started.set()
        release.wait(timeout=2.0)
        return 41

    future = reservation.submit(execute)
    assert started.wait(timeout=2.0)
    with pytest.raises(ResourceError):
        pool.reserve()

    release.set()
    assert future.result(timeout=2.0) == 41
    deadline = time.monotonic() + 2.0
    while pool.reserved and time.monotonic() < deadline:
        time.sleep(0.001)
    successor = pool.reserve()
    assert successor.active
    successor.abandon()
    pool.close()


def test_abandoned_cpu_task_registration_reclaims_its_slot() -> None:
    pool = BoundedCpuTaskPool(capacity=1, workers=1)
    reservation = pool.reserve()
    reservation.abandon()

    successor = pool.reserve()
    assert successor.active
    successor.abandon()
    pool.close()


def test_cpu_worker_set_runs_registered_tasks_concurrently() -> None:
    pool = BoundedCpuTaskPool(capacity=2, workers=2)
    first_started = Event()
    second_started = Event()
    release = Event()

    def execute(started: Event) -> int:
        started.set()
        release.wait(timeout=2.0)
        return 1

    first = pool.reserve().submit(execute, first_started)
    second = pool.reserve().submit(execute, second_started)
    assert first_started.wait(timeout=2.0)
    assert second_started.wait(timeout=2.0)
    release.set()
    assert first.result(timeout=2.0) + second.result(timeout=2.0) == 2
    pool.close()
