"""Read cancellation and failures through the transfer submission contract."""

from threading import Event

import pytest
import torch

from uniserve.runtime import EventPool
from uniserve_worker.errors import WorkerError
from uniserve_worker.transport.pool import TransferCapacity, TransferPool

pytestmark = pytest.mark.unit


def test_queued_cancellation_returns_credit_without_waiting_for_another_read():
    events = EventPool()
    capacity = TransferCapacity(8, 2)
    pool = TransferPool(
        workers=1,
        capacity=capacity,
        name="transfer-cancellation",
        event_pool=events,
    )
    source = torch.tensor([7.0])
    started, proceed = Event(), Event()

    def waiting_source(ticket, destination):
        started.set()
        assert proceed.wait(5), "source was never made available"
        pool.copy(ticket, source, destination)

    try:
        first_value = torch.zeros_like(source)
        first = pool.submit(
            waiting_source, first_value, nbytes=4, destination=first_value
        )
        assert started.wait(5), "first read never started"
        cancelled_value = torch.zeros_like(source)
        cancelled = pool.submit(
            pool.copy,
            source,
            cancelled_value,
            nbytes=4,
            destination=cancelled_value,
        )
        retired = Event()
        cancelled.add_retirement_callback(retired.set)
        cancelled.cancel()
        assert retired.wait(5), "queued cancellation did not return its credit"
        assert cancelled.ready() and cancelled.retirement_ready()
        with pytest.raises(WorkerError, match="cancelled"):
            cancelled.result()
        assert torch.equal(cancelled_value, torch.zeros_like(source))

        # Its replacement is admitted while the independent first read still
        # holds the only worker thread and its own byte/read reservation.
        replacement_value = torch.zeros_like(source)
        replacement = pool.submit(
            pool.copy,
            source,
            replacement_value,
            nbytes=4,
            destination=replacement_value,
        )
        proceed.set()
        pool.close()
        assert torch.equal(first.result(), source)
        assert torch.equal(replacement.result(), source)
        assert first.retirement_ready() and replacement.retirement_ready()
        assert capacity.used == 0
    finally:
        proceed.set()
        pool.close()
        events.close()


def test_late_read_failure_retires_storage_and_rejects_further_submissions():
    events = EventPool()
    capacity = TransferCapacity(4, 1)
    pool = TransferPool(
        workers=1, capacity=capacity, name="transfer-failure", event_pool=events
    )
    source = torch.tensor([3.0])
    destination = torch.zeros_like(source)

    def failing_source(ticket):
        pool.copy(ticket, source, destination)
        raise RuntimeError("source failed after delivering its views")

    try:
        ticket = pool.submit(failing_source, nbytes=4, destination=destination)
        retired = Event()
        ticket.add_retirement_callback(retired.set)
        assert retired.wait(5), "drained failed read did not retire"
        assert torch.equal(destination, source)
        assert capacity.used == 0
        with pytest.raises(RuntimeError, match="source failed"):
            ticket.result()
        with pytest.raises(RuntimeError, match="source failed"):
            pool.submit(pool.copy, source, destination, nbytes=4)
        with pytest.raises(RuntimeError, match="source failed"):
            pool.close()
    finally:
        try:
            pool.close()
        except RuntimeError:
            pass
        events.close()
