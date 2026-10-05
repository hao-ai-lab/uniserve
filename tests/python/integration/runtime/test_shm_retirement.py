"""Failed readers and refused exports release their segment and quota."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import pytest
import torch

from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import atomic_load_u32
from uniserve_worker.errors import ResourceError
from uniserve_worker.transport import make_transports, segment
from uniserve_worker.transport.shared_storage import open_shared_storage

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backend", ("local", "shm"))
def test_retirement_observer_can_publish_into_returned_capacity(backend):
    events = EventPool()
    producer = make_transports(
        (backend,), byte_capacity=4, ticket_capacity=1, event_pool=events
    )[backend]
    consumer = make_transports(
        (backend,), byte_capacity=4, ticket_capacity=1, event_pool=events
    )[backend]
    replacement = []
    try:
        locator = producer.export(torch.tensor([1.0]))
        retirement = producer.retirement(locator)
        retirement.add_done_callback(
            lambda _: replacement.append(producer.export(torch.tensor([2.0])))
        )
        producer.release(locator)

        ticket = consumer.fetch(replacement[0], device=torch.device("cpu"))
        ready = threading.Event()
        ticket.add_done_callback(ready.set)
        assert ready.wait(5)
        torch.testing.assert_close(ticket.result(), torch.tensor([2.0]))
        ticket.close()
    finally:
        consumer.close()
        producer.close()
        events.close()


@pytest.mark.parametrize("ending", ("cancel", "failed_fetch", "failed_borrow"))
def test_failed_reader_returns_export_capacity(ending):
    events = EventPool()
    producer = make_transports(
        ("shm",),
        byte_capacity=4,
        ticket_capacity=1,
        event_pool=events,
        host_slots=(0,),
    )["shm"]
    consumer = make_transports(
        ("shm",),
        byte_capacity=4,
        ticket_capacity=1,
        event_pool=events,
        acknowledgment_slot=0,
    )["shm"]
    locator = producer.export(torch.tensor([1.0]), consumers=(0,))
    header = shared_memory.SharedMemory(name=locator.transport.name)
    try:
        # Model a publisher whose external readiness word is still pending.
        segment.set_state(header.buf, segment.PENDING)
        with ThreadPoolExecutor(max_workers=1) as reader:
            if ending == "failed_borrow":
                borrowed = reader.submit(consumer.borrow, locator)
            else:
                ticket = consumer.fetch(locator, device=torch.device("cpu"))
                retired = threading.Event()
                ticket.add_retirement_callback(retired.set)
            deadline = time.monotonic() + 5
            while (
                atomic_load_u32(header.buf, segment.ack_offset(0))
                != segment.CLAIMED
            ):
                assert time.monotonic() < deadline, (
                    "reader did not claim export"
                )
                time.sleep(0.001)
            if ending == "cancel":
                ticket.cancel()
            else:
                segment.set_state(header.buf, segment.FAILED)
            if ending == "failed_borrow":
                with pytest.raises(Exception, match="fail"):
                    borrowed.result(timeout=5)
            else:
                assert retired.wait(5), "failed read did not retire"
                ticket.close()
        segment.set_state(header.buf, segment.READY)
        completion = producer.release(locator)
        producer.reap()
        completion.result(timeout=5)
        replacement = producer.export(torch.tensor([2.0]), consumers=(0,))
        value = consumer.borrow(replacement)
        assert value.nbytes == 4
        value.release()
        producer.release(replacement)
    finally:
        segment.set_state(header.buf, segment.FAILED)
        header.close()
        consumer.close()
        producer.release(locator)
        producer.reap()
        producer.close()
        events.close()


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="a device export requires a CUDA device",
                ),
            ),
        ),
    ),
)
def test_a_refused_export_returns_its_segment_and_quota(device, monkeypatch):
    """A export the table refuses reports why and keeps nothing.

    The refusal is backpressure the caller acts on, so it must surface as
    the resource error itself. The segment made for the product is unlinked
    and its bytes return to the rank's budget, or every refusal would shrink
    the budget all backends share.
    """
    # The segment a refused export made is only observable in the
    # shared-memory namespace, so the names of new segments are recorded.
    created = []

    class RecordedSegment(shared_memory.SharedMemory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self.name)

    monkeypatch.setattr(shared_memory, "SharedMemory", RecordedSegment)

    events = EventPool()
    byte_capacity = 1 << 20
    producer = make_transports(
        ("shm",),
        byte_capacity=byte_capacity,
        ticket_capacity=1,
        event_pool=events,
    )["shm"]
    published = []
    try:
        # Host exports that nothing retires fill the export table.
        while True:
            try:
                published.append(producer.export(torch.tensor([1.0])))
            except ResourceError:
                break
            assert len(published) * 4 < byte_capacity, (
                "the export table never filled"
            )

        before = len(created)
        with pytest.raises(ResourceError):
            producer.export(torch.ones(4, device=device))
        refused = created[before:]
        assert refused, "the refused export made no segment"
        for name in refused:
            with pytest.raises(FileNotFoundError):
                open_shared_storage(name, 1)

        # Once the table's own exports retire, the whole budget is
        # available again.
        for locator in published:
            producer.release(locator)
        published.clear()
        producer.reap()
        whole = producer.export(torch.zeros(byte_capacity // 4))
        producer.release(whole)
    finally:
        for locator in published:
            producer.release(locator)
        producer.reap()
        producer.close()
        events.close()
