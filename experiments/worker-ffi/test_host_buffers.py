"""Host input tensors retain their allocation and copy ordering through FFI."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest
import torch
import tvm_ffi
from bindings import HostBuffers

from tests.python.fixtures.cuda_stream import blocked_stream


@pytest.mark.parametrize("device", (None, 0))
def test_host_inputs_preserve_storage_and_copy_values(device):
    buffers = HostBuffers(
        [torch.empty(16, dtype=torch.int64, pin_memory=device is not None)],
        device,
    )
    target = "cpu" if device is None else f"cuda:{device}"
    output = torch.empty(16, dtype=torch.int64, device=target)
    try:
        for value in (7, 13, 29):
            slot, source = buffers.acquire()
            host = torch.from_dlpack(source)
            host.fill_(value)
            output.copy_(host, non_blocking=device is not None)
            buffers.record_copy(slot)
            del source, host
            _, source = buffers.acquire()
            host = torch.from_dlpack(source)
            assert host.tolist() == [value] * 16
            host.fill_(value + 1)
            assert output.cpu().tolist() == [value] * 16
    finally:
        buffers.close()

    # A returned view retains its own allocation after its pool closes.
    assert host.tolist() == [30] * 16
    with pytest.raises(RuntimeError, match="closed"):
        buffers.acquire()


@pytest.mark.timeout(30)
def test_close_drains_copy_without_holding_python_gil():
    buffers = HostBuffers([torch.full((32,), 17, pin_memory=True)], 0)
    output = torch.empty(32, dtype=torch.int64, device="cuda:0")
    entered = Event()

    def close():
        entered.set()
        buffers.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            with blocked_stream("cuda:0") as stream:
                slot, source = buffers.acquire()
                with (
                    torch.cuda.stream(stream),
                    tvm_ffi.use_torch_stream(stream),
                ):
                    output.copy_(torch.from_dlpack(source), non_blocking=True)
                    buffers.record_copy(slot)
                del source

                pending = threads.submit(close)
                assert entered.wait(5)
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
            pending.result(timeout=5)
        assert output.cpu().tolist() == [17] * 32
    finally:
        buffers.close()


def test_empty_ring_is_rejected():
    with pytest.raises(RuntimeError, match="positive"):
        HostBuffers([], None)


def test_foreign_owner_keeps_the_ring_open():
    buffers = HostBuffers([torch.tensor([7])], None)
    retained = tvm_ffi.Array([buffers])
    del buffers

    # Destroying one Python wrapper must not close another owner's buffers.
    owner = retained[0]
    try:
        _, source = owner.acquire()
        assert torch.from_dlpack(source).tolist() == [7]
    finally:
        owner.close()
