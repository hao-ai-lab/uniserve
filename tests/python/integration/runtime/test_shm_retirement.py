"""Shared tensor readiness, borrowed views and physical retirement."""

import ctypes
import multiprocessing as mp
import os
import select
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

import pytest
import torch

from tests.python.fixtures import segment
from tests.python.fixtures.cuda_stream import blocked_stream
from tests.python.fixtures.shared_storage import open_shared_storage
from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import (
    SharedBuffer,
    SharedRead,
    atomic_load_u32,
    atomic_store_u32,
)
from uniserve_worker.errors import ResourceError, WorkerError
from uniserve_worker.protocol.transfer import Locator
from uniserve_worker.transport import make_transports

pytestmark = pytest.mark.integration


def test_shared_read_views_retain_their_claim_and_selected_bytes():
    storage = SharedBuffer(16, (0,))
    payload = torch.frombuffer(
        memoryview(storage)[segment.HEADER_BYTES :], dtype=torch.int32
    )
    payload.copy_(torch.arange(4, dtype=torch.int32))
    storage.mark_ready()
    try:
        read = SharedRead(storage.name, 12, 0, offset=4)
        read.truncate(8)
        with pytest.raises(WorkerError, match="borrowed range"):
            read.truncate(12)
        view = torch.frombuffer(read, dtype=torch.int32)
        del read
        assert not storage.settled()
        torch.testing.assert_close(
            view, torch.tensor([1, 2], dtype=torch.int32)
        )
        del view
        assert storage.settled()
    finally:
        storage.close()


@pytest.mark.parametrize("ending", ("timeout", "failure", "short_segment"))
def test_refused_shared_read_leaves_no_reader_claim(ending):
    storage = SharedBuffer(4, (0,))
    try:
        if ending == "failure":
            segment.set_state(memoryview(storage), segment.FAILED)
        size = 8 if ending == "short_segment" else 4
        with pytest.raises(WorkerError, match="readiness|shorter"):
            SharedRead(storage.name, size, 0, timeout=0)
        storage.mark_ready()
        assert storage.settled()
    finally:
        storage.close()


def test_borrowed_tensor_survives_shared_buffer_close():
    storage = SharedBuffer(16, ())
    name = storage.name
    view = memoryview(storage)[segment.HEADER_BYTES :]
    tensor = torch.frombuffer(view, dtype=torch.float32)
    direct = torch.frombuffer(
        storage, dtype=torch.float32, offset=segment.HEADER_BYTES
    )
    tensor.copy_(torch.arange(4, dtype=torch.float32))
    storage.mark_ready()
    storage.close()

    with pytest.raises(FileNotFoundError):
        open_shared_storage(name, 1)
    del storage, view

    # PyTorch may release Py_buffer immediately while retaining its exporter.
    # The tensor must remain writable through its last borrowed reference.
    direct.add_(1)
    torch.testing.assert_close(tensor, torch.arange(1, 5, dtype=torch.float32))


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
def test_failed_shared_read_returns_source_and_destination_capacity(device):
    events = EventPool()
    producer, consumer = (
        make_transports(
            ("shm",), byte_capacity=32, ticket_capacity=1, event_pool=events
        )["shm"]
        for _ in range(2)
    )
    source = torch.arange(8, dtype=torch.float32)
    location = producer.export(source, consumers=(0,))
    try:
        # Without a supplied destination, shape validation runs on the read
        # thread. Its failure must end the source claim as well as the task.
        failed = consumer.fetch(
            location, device=torch.device(device), region=(slice(0, 9),)
        )
        retired = threading.Event()
        failed.add_retirement_callback(retired.set)
        assert retired.wait(5), "failed read retained destination capacity"
        with pytest.raises(WorkerError, match="region exceeds"):
            failed.result()
        retirement = producer.release(location)
        producer.reap()
        retirement.result(timeout=5)

        # Both endpoints have exactly one payload's capacity. Another full
        # transfer must succeed after the failed read, using the same owners.
        location = producer.export(source + 1, consumers=(0,))
        read = consumer.fetch(location, device=torch.device(device))
        ready = threading.Event()
        read.add_done_callback(ready.set)
        assert ready.wait(5), "replacement read did not complete"
        torch.testing.assert_close(read.result().cpu(), source + 1)
        read.close()
    finally:
        producer.release(location)
        consumer.close()
        producer.close()
        events.close()


@pytest.mark.gpu
def test_reaping_a_cuda_export_does_not_block_the_execution_thread():
    events = EventPool()
    producer = make_transports(
        ("shm",), byte_capacity=4096, ticket_capacity=1, event_pool=events
    )["shm"]
    source = torch.arange(1024, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()

    def reap(locator):
        retirement = producer.release(locator)
        producer.reap()
        return retirement

    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            with blocked_stream("cuda:0") as later:
                with blocked_stream("cuda:0") as earlier:
                    stream.wait_stream(earlier)
                    with torch.cuda.stream(stream):
                        locator = producer.export(source)
                        copied = torch.cuda.Event()
                        copied.record(stream)
                    stream.wait_stream(later)
                copied.synchronize()
                # Even once the producer copy is done, host unregistration
                # may wait for later stream work. Reaping must still return
                # control to execution so it can submit independent work.
                retirement = threads.submit(reap, locator).result(timeout=5)
            retirement.result(timeout=5)
    finally:
        producer.close()
        events.close()


def _export_with_gil_held(channel):
    """Hold the producer GIL while the receiver releases its device copy."""
    from cuda.bindings import driver

    events = EventPool()
    producer = make_transports(
        ("shm",), byte_capacity=4096, ticket_capacity=1, event_pool=events
    )["shm"]
    stream = torch.cuda.Stream()
    source = torch.arange(1024, device="cuda", dtype=torch.float32)
    stream.wait_stream(torch.cuda.current_stream())

    # A separate registered word gates the device before the export copy.
    # The receiver writes it only after the producer is inside a GIL-held C
    # call, so neither readiness nor Python notification can precede the hold.
    gate = SharedBuffer(4, (), (0, int(stream.cuda_stream)))
    gate_word = torch.frombuffer(
        memoryview(gate)[segment.HEADER_BYTES :], dtype=torch.int32
    )
    status, pointer = driver.cuMemHostGetDevicePointer(gate_word.data_ptr(), 0)
    assert status == driver.CUresult.CUDA_SUCCESS
    (status,) = driver.cuStreamWaitValue32(
        stream.cuda_stream,
        pointer,
        1,
        driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
    )
    assert status == driver.CUresult.CUDA_SUCCESS

    try:
        with torch.cuda.stream(stream):
            locator = producer.export(source)
        channel.send((locator.to_mapping(), gate.name))

        # system holds the GIL through the shell's entire lifetime. Its ready
        # byte proves entry into that call; read ends only when the receiver
        # has inspected the tensor. Standard input/output are this private
        # child connection, not the test runner's terminal.
        os.dup2(channel.fileno(), 0)
        os.dup2(channel.fileno(), 1)
        system = ctypes.PyDLL(None).system
        system.argtypes = (ctypes.c_char_p,)
        system.restype = ctypes.c_int
        assert system(b"printf r; read -r release") == 0
    finally:
        stream.synchronize()
        gate.close()
        producer.close()
        events.close()
        channel.close()


@pytest.mark.gpu
def test_cuda_export_becomes_readable_while_producer_holds_gil():
    context = mp.get_context("spawn")
    channel, child = context.Pipe()
    process = context.Process(target=_export_with_gil_held, args=(child,))
    process.start()
    child.close()
    gate = None

    try:
        assert channel.poll(30), "producer did not export its tensor"
        value, gate_name = channel.recv()
        locator = Locator.from_mapping(value)
        gate = open_shared_storage(gate_name, segment.HEADER_BYTES + 4)
        assert select.select([channel.fileno()], [], [], 10)[0]
        assert os.read(channel.fileno(), 1) == b"r"
        atomic_store_u32(gate, segment.HEADER_BYTES, 1)

        with open_shared_storage(
            locator.transport.name, segment.HEADER_BYTES + locator.nbytes
        ) as mapping:
            header = memoryview(mapping)
            try:
                segment.await_ready(header, timeout=10)
                observed = torch.frombuffer(
                    header[segment.HEADER_BYTES :], dtype=torch.float32
                ).clone()
                torch.testing.assert_close(
                    observed, torch.arange(1024, dtype=torch.float32)
                )
            finally:
                header.release()
    finally:
        if gate is not None:
            atomic_store_u32(gate, segment.HEADER_BYTES, 1)
            gate.close()
        os.write(channel.fileno(), b"release\n")
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        channel.close()
    assert process.exitcode == 0


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
def test_a_refused_export_returns_its_quota(device):
    """Backpressure must not consume capacity needed by the next export."""
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

        with pytest.raises(ResourceError):
            producer.export(torch.ones(4, device=device))

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
