"""Failed readers release their real producer publication and quota."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import pytest
import torch

from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import atomic_load_u32
from uniserve_worker.transfer import segment
from uniserve_worker.transfer.tickets import make_transports

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("ending", ("cancel", "failed_fetch", "failed_borrow"))
def test_failed_reader_returns_publication_capacity(ending):
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
    locator = producer.publish(torch.tensor([1.0]), consumers=(0,))
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
                    "reader did not claim publication"
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
        replacement = producer.publish(torch.tensor([2.0]), consumers=(0,))
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
