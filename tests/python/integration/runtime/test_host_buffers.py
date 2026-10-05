"""Host input reuse and release follow completion of their own CUDA copies."""

import weakref
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest
import torch

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve_worker.storage.host_buffers import HostBuffers

pytestmark = pytest.mark.integration


def test_cpu_ring_preserves_values_and_releases_storage():
    buffers = HostBuffers((2, 3), dtype=torch.int64, depth=2, device="cpu")
    first_slot, first = buffers.acquire()
    second_slot, second = buffers.acquire()
    assert first_slot != second_slot
    assert first.shape == second.shape == (2, 3)
    assert first.dtype == second.dtype == torch.int64
    first.fill_(7)
    second.fill_(11)
    buffers.record_copy(first_slot)
    buffers.record_copy(second_slot)

    slot, reused = buffers.acquire()
    assert slot == first_slot
    assert reused.tolist() == [[7, 7, 7], [7, 7, 7]]
    retained = [weakref.ref(first), weakref.ref(second)]
    del first, second, reused
    assert all(reference() is not None for reference in retained)
    buffers.close()
    assert all(reference() is None for reference in retained)
    buffers.close()
    with pytest.raises(RuntimeError, match="closed"):
        buffers.acquire()


def test_empty_ring_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        HostBuffers(3, dtype=torch.int32, depth=0, device="cpu")


@pytest.mark.gpu
@pytest.mark.timeout(30)
@pytest.mark.parametrize("operation", ("acquire", "close", "drop"))
def test_copy_finishes_before_source_reuse_or_release(operation):
    buffers = HostBuffers(32, dtype=torch.int64, depth=2, device="cuda:0")
    output = torch.empty(32, dtype=torch.int64, device="cuda:0")
    entered = Event()

    def reuse_or_close():
        nonlocal buffers
        entered.set()
        if operation == "close":
            buffers.close()
        elif operation == "drop":
            buffers = None
        else:
            _, host = buffers.acquire()
            host.fill_(29)

    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            with blocked_stream("cuda:0") as stream:
                slot, host = buffers.acquire()
                assert host.is_pinned()
                host.fill_(13)
                with torch.cuda.stream(stream):
                    output.copy_(host, non_blocking=True)
                    buffers.record_copy(slot)
                del host

                # Another slot is writable while this copy is still pending.
                next_slot, other = buffers.acquire()
                assert next_slot != slot
                other.fill_(5)
                del other

                pending = threads.submit(reuse_or_close)
                assert entered.wait(5)
                if operation == "drop":
                    # Destruction may wait or defer allocation release. Both
                    # must let Python run while the GPU copy is pending.
                    try:
                        pending.result(timeout=0.1)
                    except TimeoutError:
                        pass
                else:
                    with pytest.raises(TimeoutError):
                        pending.result(timeout=0.1)

            pending.result(timeout=5)
        assert output.cpu().tolist() == [13] * 32
    finally:
        if buffers is not None:
            buffers.close()


@pytest.mark.gpu
def test_completed_copy_does_not_wait_for_unrelated_stream():
    buffers = HostBuffers(8, dtype=torch.int32, depth=1, device="cuda:0")
    output = torch.empty(8, dtype=torch.int32, device="cuda:0")
    try:
        slot, host = buffers.acquire()
        host.fill_(17)
        output.copy_(host, non_blocking=True)
        buffers.record_copy(slot)
        torch.cuda.current_stream().synchronize()

        with ThreadPoolExecutor(max_workers=1) as threads:
            with blocked_stream("cuda:0"):
                _, host = threads.submit(buffers.acquire).result(timeout=5)
                host.fill_(23)
                threads.submit(buffers.close).result(timeout=5)
        assert output.cpu().tolist() == [17] * 8
    finally:
        buffers.close()
