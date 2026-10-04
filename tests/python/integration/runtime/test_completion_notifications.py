"""A stalled producer must not delay another stream's completion signal."""

import select

import pytest
import torch

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import StreamSignal

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("device_index", [0, 1])
def test_independent_producer_notifies_while_another_producer_is_blocked(
    device_index: int,
) -> None:
    if torch.cuda.device_count() <= device_index:
        pytest.skip("requires the selected CUDA device")

    device = f"cuda:{device_index}"
    fast = torch.cuda.Stream(device=device)
    signal = StreamSignal()
    slow_signal = StreamSignal()
    events = EventPool()
    slow_done = events.acquire(device)
    fast_done = events.acquire(device)
    events.retain(slow_done, device)
    events.retain(fast_done, device)

    try:
        with blocked_stream(device) as slow:
            events.set_completion_wake(slow_signal.schedule)
            with torch.cuda.stream(slow):
                events.record(slow_done, device)
            events.schedule_completion_wake(device, slow_done)

            events.set_completion_wake(signal.schedule)
            with torch.cuda.stream(fast):
                events.record(fast_done, device)
            events.schedule_completion_wake(device, fast_done)
            readable, _, _ = select.select([signal.fileno()], [], [], 5)
            assert readable, "independent completion did not notify its reader"
            signal.consume()
            assert fast_done.query()
            assert not slow_done.query()

        events.release(slow_done)
        events.release(fast_done)
    finally:
        fast.synchronize()
        events.close()
