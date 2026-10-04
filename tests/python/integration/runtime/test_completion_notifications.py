"""A stalled producer must not delay another stream's completion signal."""

import ctypes
import select

import pytest
import torch

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
    driver = ctypes.CDLL("libcuda.so.1")
    for name in ("cuStreamWaitValue32_v2", "cuStreamWriteValue32_v2"):
        function = getattr(driver, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int

    gate = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.current_stream(device).synchronize()
    slow = torch.cuda.Stream(device=device)
    fast = torch.cuda.Stream(device=device)
    signal = StreamSignal()
    slow_signal = StreamSignal()
    events = EventPool()
    events.set_completion_wake(slow_signal.schedule)
    slow_done = events.acquire(device)
    fast_done = events.acquire(device)
    events.retain(slow_done, device)
    events.retain(fast_done, device)

    try:
        # A device storage wait controls producer completion without occupying
        # CUDA's host callback thread, which also delivers native notifications.
        assert (
            driver.cuStreamWaitValue32_v2(
                slow.cuda_stream, gate.data_ptr(), 1, 1
            )
            == 0
        )
        with torch.cuda.stream(slow):
            events.record(slow_done, device)
        events.schedule_completion_wake(device, slow_done)
        events.set_completion_wake(signal.schedule)
        with torch.cuda.stream(fast):
            events.record(fast_done, device)
        events.schedule_completion_wake(device, fast_done)
        readable, _, _ = select.select([signal.fileno()], [], [], 5)
        assert readable, (
            "independent completion did not reach the native notification "
            "descriptor"
        )
        signal.consume()
        assert fast_done.query()
        assert not slow_done.query()
    finally:
        assert (
            driver.cuStreamWriteValue32_v2(
                fast.cuda_stream, gate.data_ptr(), 1, 0
            )
            == 0
        )
        slow.synchronize()
        fast.synchronize()
        events.release(slow_done)
        events.release(fast_done)
        events.close()
