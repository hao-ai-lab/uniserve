"""Batch input observers and cancellation over native resource owners."""

from concurrent.futures import CancelledError
from threading import Event

import pytest
import torch

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import BatchInputs, Completion
from uniserve_worker.execution.host import HostLane
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.storage.buffer_pool import BufferPool
from uniserve_worker.storage.output import OutputPool
from uniserve_worker.storage.tensor_store import TensorStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store():
    buffers = BufferPool(byte_capacity=16, devices=("cpu",))
    value = TensorStore(capacity=1, buffer_pool=buffers)
    try:
        yield value
    finally:
        value.close()
        buffers.close()
        value.event_pool.close()


@pytest.mark.parametrize("immediate", (False, True))
def test_ready_observer_can_close_its_inputs(store, immediate: bool) -> None:
    inputs = BatchInputs()
    first, second = Completion(), Completion()
    first.set_result(None)
    inputs.set_dependencies((first, second))
    inputs.submitted = True
    observed = []

    def consume() -> None:
        observed.append(inputs.ready())
        inputs.require_storage()
        inputs.close(store, None, None)

    if immediate:
        second.set_result(None)
    inputs.on_ready(consume)
    if not immediate:
        assert not observed
        assert not inputs.ready()
        second.set_result(None)

    assert observed == [True]
    assert inputs.closed


def test_closed_inputs_cancel_queued_preparation_and_suppress_wakes(
    store,
) -> None:
    lane = HostLane(max_inflight=2, workers=1)
    entered, release, notified = Event(), Event(), Event()

    def occupy_lane() -> None:
        entered.set()
        assert release.wait(5)

    running = lane.reserve().configure(occupy_lane)
    running.submit_if_ready()
    assert entered.wait(5)

    inputs = BatchInputs()
    dependency = Completion()
    inputs.set_dependencies((dependency,))
    queued = lane.reserve().configure(lambda: None)
    inputs.add_image(CallId(1, 0), queued)
    queued.submit_if_ready()
    inputs.submitted = True
    inputs.on_ready(notified.set)
    try:
        inputs.close(store, None, None)
        with pytest.raises(CancelledError):
            queued.result(timeout=5)
        dependency.set_result(None)
        assert not notified.is_set()
        inputs.close(store, None, None)
    finally:
        release.set()
        lane.close()


def test_failed_storage_wakes_its_consumer_and_preserves_the_error(
    store,
) -> None:
    inputs = BatchInputs()
    dependency = Completion()
    inputs.set_dependencies((dependency,))
    inputs.submitted = True
    notified = Event()
    inputs.on_ready(notified.set)

    failure = ValueError("input storage failed")
    dependency.set_exception(failure)
    assert notified.is_set()
    assert inputs.ready()
    with pytest.raises(ValueError) as observed:
        inputs.require_storage()
    assert observed.value is failure
    inputs.close(store, None, None)


@pytest.mark.gpu
def test_predicate_observer_can_close_after_device_readback(store) -> None:
    events = EventPool()
    outputs = OutputPool(capacity=1, max_words=1, event_pool=events)
    inputs = BatchInputs()
    inputs.submitted = True
    value = torch.tensor([1], device="cuda:0")
    try:
        buffer = outputs.acquire(1, token_capacity=1, devices=("cuda:0",))
        inputs.predicate = buffer
        with blocked_stream("cuda:0") as stream, torch.cuda.stream(stream):
            buffer.capture(value)
            buffer.seal()
            inputs.on_ready(lambda: inputs.close(store, None, None))
            assert not inputs.ready()

        stream.synchronize()
        assert inputs.ready()
        events.reap()
        assert inputs.closed
    finally:
        inputs.close(store, None, None)
        outputs.close()
        events.close()
