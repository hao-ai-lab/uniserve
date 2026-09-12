"""Physical transfer ordering across process and CUDA stream boundaries."""

from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.shm_publication import serve_pending_publication
from uniserve_worker.execution.batch import (
    BufferAllocation,
    ComputationId,
    DType,
    Locator,
    RequestKey,
    ShapeBound,
    StaticDim,
    TensorRef,
    WorkerEndpoint,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.device_events import DeviceEventPool
from uniserve_worker.runtime.device_products import DeviceProducts
from uniserve_worker.runtime.encoder_cache import EncoderCache
from uniserve_worker.runtime.latent_pool import LatentPool
from uniserve_worker.runtime.persistent_buffers import PersistentBuffers
from uniserve_worker.transfer.layout import TensorRegion, region_view
from uniserve_worker.transfer.tickets import make_transport

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _await_ticket(ticket):
    ready = threading.Event()
    ticket.add_done_callback(ready.set)
    assert ready.wait(30), "transfer did not expose a consumable result"
    return ticket.result()


@pytest.mark.parametrize("backend", ("local", "shm", "cuda_ipc"))
def test_transfer_writes_only_the_reserved_destination(backend: str) -> None:
    device = torch.device("cuda:0")
    events = DeviceEventPool()
    transport = make_transport(backend, byte_capacity=16384, ticket_capacity=2, event_pool=events)
    source = torch.arange(1024, dtype=torch.float32, device=device)
    storage = torch.full((1026,), -1.0, device=device)
    destination = storage[1:-1]
    locator = transport.publish(source)
    try:
        with pytest.raises(WorkerError, match="destination disagrees"):
            transport.fetch(locator, device=device, destination=destination[:-1])
        # The owner finishes initialization before granting this range to an
        # independent transport stream.
        torch.cuda.synchronize(device)
        ticket = transport.fetch(locator, device=device, destination=destination)
        _await_ticket(ticket)
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
        torch.testing.assert_close(storage[[0, -1]], torch.full((2,), -1.0, device=device))
    finally:
        transport.release(locator)
        transport.close()
        events.close()


@pytest.mark.parametrize("backend", ("local", "shm", "cuda_ipc"))
def test_transfer_scatters_exactly_into_disjoint_page_spans(backend: str) -> None:
    device = torch.device("cuda:0")
    events = DeviceEventPool()
    transport = make_transport(backend, byte_capacity=16384, ticket_capacity=1, event_pool=events)
    source = torch.arange(44, dtype=torch.float32, device=device).reshape(11, 4)
    storage = torch.full((5, 4, 4), -1.0, device=device)
    destination = (storage[3], storage[1], storage[4, :3])
    locator = transport.publish(source)
    try:
        with pytest.raises(WorkerError, match="destination disagrees"):
            transport.fetch(locator, device=device, destination=destination[:-1])
        with pytest.raises(WorkerError, match="spans overlap"):
            transport.fetch(
                locator, device=device, destination=(storage[1], storage[1], storage[4, :3])
            )
        torch.cuda.synchronize(device)
        ticket = transport.fetch(locator, device=device, destination=destination)
        values = _await_ticket(ticket)
        torch.testing.assert_close(torch.cat(values), source, rtol=0, atol=0)
        torch.testing.assert_close(storage[0], torch.full_like(storage[0], -1.0), rtol=0, atol=0)
        torch.testing.assert_close(storage[2], torch.full_like(storage[2], -1.0), rtol=0, atol=0)
        torch.testing.assert_close(
            storage[4, 3], torch.full_like(storage[4, 3], -1.0), rtol=0, atol=0
        )
    finally:
        transport.release(locator)
        transport.close()
        events.close()


def _read_cuda_publications(channel) -> None:
    torch.cuda.set_device(0)
    event_pool = DeviceEventPool()
    transport = make_transport(
        "cuda_ipc", byte_capacity=8 << 20, ticket_capacity=2, event_pool=event_pool
    )
    try:
        warmup = Locator.from_mapping(channel.recv())
        value = _await_ticket(transport.fetch(warmup, device=torch.device("cuda:0")))
        torch.cuda.synchronize()
        del value
        channel.send("ready")
        locator = Locator.from_mapping(channel.recv())
        stream = torch.cuda.Stream()
        pending = torch.cuda.Event()
        destination = (torch.empty(511, device="cuda:0"), torch.empty(513, device="cuda:0"))
        with torch.cuda.stream(stream):
            # The fetch must return while earlier work on the consuming stream
            # is pending. Checking the device fence avoids a host-time threshold.
            torch.cuda._sleep(1_000_000_000)
            pending.record(stream)
            ticket = transport.fetch(
                locator, device=torch.device("cuda:0"), destination=destination
            )
            asynchronous = not pending.query()
            value = _await_ticket(ticket)
            actual = torch.cat(value).cpu()
        channel.send(
            (asynchronous, bool(torch.equal(actual, torch.arange(1024, dtype=torch.float32))))
        )
    finally:
        transport.close()
        event_pool.close()
        channel.close()


def test_cuda_ipc_read_does_not_wait_for_consumer_stream() -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    event_pool = DeviceEventPool()
    transport = make_transport(
        "cuda_ipc", byte_capacity=8 << 20, ticket_capacity=2, event_pool=event_pool
    )
    process = context.Process(target=_read_cuda_publications, args=(child,))
    publications = []
    try:
        warmup = transport.publish(torch.ones(1024, device="cuda:0"))
        locator = transport.publish(torch.arange(1024, dtype=torch.float32, device="cuda:0"))
        publications.extend((warmup, locator))
        process.start()
        child.close()
        parent.send(warmup.to_mapping())
        assert parent.poll(60), "CUDA IPC consumer did not become ready"
        assert parent.recv() == "ready"
        parent.send(locator.to_mapping())
        assert parent.poll(30), "CUDA IPC consumer did not finish its read"
        asynchronous, correct = parent.recv()
        process.join(30)
        assert process.exitcode == 0
        assert correct, "CUDA IPC read returned the wrong published value"
        assert asynchronous, "CUDA IPC fetch waited for the consumer stream"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        for locator in publications:
            transport.release(locator)
        transport.close()
        event_pool.close()


def _consume_fanout(channel, device_index: int) -> None:
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    event_pool = DeviceEventPool()
    transport = make_transport(
        "cuda_ipc", byte_capacity=8 << 20, ticket_capacity=2, event_pool=event_pool
    )
    try:
        channel.send("ready")
        locator = Locator.from_mapping(channel.recv())
        value = _await_ticket(transport.fetch(locator, device=device))
        channel.send("readable")
        assert channel.recv() == "consume"
        correct = torch.equal(value.cpu(), torch.arange(1024, dtype=torch.float32))
        transport.close()
        channel.send((correct, str(value.device)))
    finally:
        transport.close()
        event_pool.close()
        channel.close()


def test_cuda_ipc_publication_fits_the_existing_device_allocation() -> None:
    device = torch.device("cuda:0")
    events = DeviceEventPool()
    source = torch.arange(65536, dtype=torch.float32, device=device).view(16, 4096).T
    destination = torch.empty(source.shape, dtype=source.dtype, device=device)
    transport = make_transport(
        "cuda_ipc",
        byte_capacity=2 * source.numel() * source.element_size(),
        ticket_capacity=1,
        event_pool=events,
    )
    locator = None
    try:
        warmup = transport.publish(source)
        transport.release(warmup)
        torch.cuda.synchronize(device)
        events.reap()
        resident_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        locator = transport.publish(source)
        # Publication's GPU payload budget is already occupied by its source.
        # An additional tensor-sized copy exceeds that public resource contract.
        assert torch.cuda.max_memory_allocated(device) == resident_bytes
        _await_ticket(transport.fetch(locator, device=device, destination=destination))
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
    finally:
        if locator is not None:
            transport.release(locator)
        transport.close()
        events.close()


def test_cuda_ipc_retirement_preserves_pending_fanout_and_reclaims_capacity() -> None:
    context = mp.get_context("spawn")
    event_pool = DeviceEventPool()
    transport = make_transport(
        "cuda_ipc", byte_capacity=8192, ticket_capacity=2, event_pool=event_pool
    )
    readers = []
    locator = None
    replacement = None
    try:
        # Warm native export before measuring the ordering of a pending fence.
        warmup = transport.publish(torch.ones(1024, device="cuda:0"))
        torch.cuda.synchronize(0)
        transport.release(warmup)
        for device in (0, 1):
            parent, child = context.Pipe()
            process = context.Process(target=_consume_fanout, args=(child, device))
            process.start()
            child.close()
            readers.append((parent, process, device))
        for channel, _process, _device in readers:
            assert channel.poll(60), "CUDA IPC fan-out consumer did not start"
            assert channel.recv() == "ready"

        stream = torch.cuda.Stream(device=0)
        published = torch.cuda.Event()
        with torch.cuda.stream(stream):
            source = torch.arange(1024, dtype=torch.float32, device="cuda:0")
            torch.cuda._sleep(8_000_000_000)
            locator = transport.publish(source)
            published.record(stream)
        for channel, _process, _device in readers:
            channel.send(locator.to_mapping())
        for channel, _process, _device in readers:
            assert channel.poll(30), "CUDA IPC fan-out read did not become consumable"
            assert channel.recv() == "readable"
        assert not published.query(), "read tickets waited for producer device completion"

        transport.release(locator)
        with pytest.raises(WorkerError, match="capacity"):
            transport.publish(torch.empty(2048, device="cuda:0"))
        for channel, _process, _device in readers:
            channel.send("consume")
        for channel, process, device in readers:
            assert channel.poll(30), "CUDA IPC fan-out consumer did not complete"
            assert channel.recv() == (True, f"cuda:{device}")
            process.join(30)
            assert process.exitcode == 0

        with pytest.raises(WorkerError, match="retired"):
            _await_ticket(transport.fetch(locator, device=torch.device("cuda:0")))
        replacement = transport.publish(torch.ones(2048, device="cuda:0"))
    finally:
        for channel, process, _device in readers:
            if process.is_alive():
                process.terminate()
                process.join(30)
            channel.close()
        if locator is not None:
            transport.release(locator)
        if replacement is not None:
            transport.release(replacement)
        transport.close()
        event_pool.close()


def test_cuda_ipc_rejects_a_changed_registered_view() -> None:
    event_pool = DeviceEventPool()
    transport = make_transport(
        "cuda_ipc", byte_capacity=8 << 20, ticket_capacity=2, event_pool=event_pool
    )
    locator = transport.publish(torch.arange(1024, device="cuda:0"))
    try:
        changed = replace(
            locator,
            shape=(512,),
            nbytes=locator.nbytes // 2,
            transport=replace(locator.transport, span_lengths=(512,)),
        )
        with pytest.raises(WorkerError, match="invalid"):
            _await_ticket(transport.fetch(changed, device=torch.device("cuda:0")))
    finally:
        transport.release(locator)
        transport.close()
        event_pool.close()


def test_local_read_keeps_its_producer_fence_after_publication_retirement() -> None:
    device = torch.device("cuda:0")
    events = DeviceEventPool()
    transport = make_transport("local", byte_capacity=8192, ticket_capacity=2, event_pool=events)
    source = torch.zeros(1024, device=device)
    unrelated = torch.full_like(source, 9)
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    producer.wait_stream(torch.cuda.current_stream(device))
    locator = None
    replacement = None
    try:
        with torch.cuda.stream(producer):
            torch.cuda._sleep(1_000_000_000)
            source.fill_(7)
            locator = transport.publish(source)
        unused = transport.fetch(locator, device=device)
        retained = transport.fetch(locator, device=device)
        del unused
        retirement = transport.release(locator)
        assert retirement is not None and not retirement.done()
        replacement = transport.publish(unrelated)
        with torch.cuda.stream(consumer):
            value = retained.result(consumer).cpu()
        retained.close()
        consumer.synchronize()
        events.reap()
        retirement.result(timeout=5)
        assert torch.equal(value, torch.full((1024,), 7, dtype=torch.float32))
    finally:
        if locator is not None:
            transport.release(locator)
        if replacement is not None:
            transport.release(replacement)
        transport.close()
        events.close()


def _serve_unacknowledged_cuda_read(channel, invalid_handle: bool = False) -> None:
    """Serve a real device allocation and reject retirement after the copy finishes."""

    import socket
    import uuid

    from uniserve_kernel.peer_memory import export_ipc

    from uniserve_worker.execution.batch import CudaIpcTransfer

    torch.cuda.set_device(0)
    source = torch.arange(1024, dtype=torch.float32, device="cuda:0")
    event = torch.cuda.Event(interprocess=True)
    event.record()
    handle, capacity, offset = export_ipc(source)
    endpoint = f"uniserve-test-read-{uuid.uuid4().hex}"
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
        listener.bind("\0" + endpoint)
        listener.listen(1)
        locator = Locator(
            source=WorkerEndpoint.local("publisher"),
            transport=CudaIpcTransfer(
                endpoint=endpoint,
                publication_id=uuid.uuid4().hex,
                storage_handle=bytes(len(handle)) if invalid_handle else handle,
                storage_size_bytes=capacity,
                storage_offsets_bytes=(offset,),
                span_lengths=(source.shape[0],),
                span_counts=(1,),
                tensor_stride=tuple(source.stride()),
                ready_event_handle=event.ipc_handle(),
            ),
            nbytes=source.numel() * source.element_size(),
            dtype="float32",
            shape=tuple(source.shape),
            offset=(0,) * source.ndim,
            device="cuda:0",
        )
        channel.send(locator.to_mapping())
        connection, _address = listener.accept()
        with connection:
            assert len(connection.recv(128)) == 64
            connection.sendall(b"G")
            assert connection.recv(1) == b"A"
            channel.send("released")
            assert channel.recv() == "reject"
            connection.sendall(b"E")
        # Keep the allocation alive until the receiver has observed retirement.
        assert channel.recv() == "close"
    channel.close()


def test_cuda_ipc_reports_retirement_failure_after_result_is_consumable() -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_serve_unacknowledged_cuda_read, args=(child,))
    events = DeviceEventPool()
    transport = make_transport("cuda_ipc", byte_capacity=8192, ticket_capacity=1, event_pool=events)
    retirement_rejected = False
    process.start()
    child.close()
    try:
        assert parent.poll(60), "CUDA IPC source did not start"
        locator = Locator.from_mapping(parent.recv())
        ticket = transport.fetch(locator, device=torch.device("cuda:0"))
        actual = _await_ticket(ticket).cpu()
        torch.testing.assert_close(actual, torch.arange(1024, dtype=torch.float32), rtol=0, atol=0)
        assert parent.poll(30), "CUDA IPC reader did not finish copying"
        assert parent.recv() == "released"
        retired = threading.Event()
        ticket.add_retirement_callback(retired.set)
        assert not ticket.retired()
        assert not retired.is_set()
        parent.send("reject")
        retirement_rejected = True
        with pytest.raises(WorkerError, match="reader acknowledgement"):
            transport.close()
        with pytest.raises(WorkerError, match="reader acknowledgement"):
            ticket.result()
        assert retired.wait(5)
        assert ticket.retired()
    finally:
        if process.is_alive() and not retirement_rejected:
            parent.send("reject")
        try:
            transport.close()
        except WorkerError:
            pass
        if process.is_alive():
            parent.send("close")
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        events.close()
    assert process.exitcode == 0


def test_cuda_ipc_reports_import_failure_before_acknowledgement() -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_serve_unacknowledged_cuda_read, args=(child, True))
    events = DeviceEventPool()
    transport = make_transport("cuda_ipc", byte_capacity=8192, ticket_capacity=1, event_pool=events)
    retirement_rejected = False
    process.start()
    child.close()
    try:
        assert parent.poll(60), "CUDA IPC source did not start"
        locator = Locator.from_mapping(parent.recv())
        ticket = transport.fetch(locator, device=torch.device("cuda:0"))
        ready = threading.Event()
        ticket.add_done_callback(ready.set)
        assert ready.wait(30), "import failure waited for the source acknowledgement"
        with pytest.raises(RuntimeError, match="IPC"):
            ticket.result()
        assert parent.poll(30), "failed import did not release its source grant"
        assert parent.recv() == "released"
        assert not ticket.retired()
        parent.send("reject")
        retirement_rejected = True
        transport.close()
        assert ticket.retired()
        with pytest.raises(RuntimeError, match="IPC"):
            ticket.result()
    finally:
        if process.is_alive() and not retirement_rejected:
            parent.send("reject")
        transport.close()
        if process.is_alive():
            parent.send("close")
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        events.close()
    assert process.exitcode == 0


def _read_granted_shm_publication(channel) -> None:
    """External host reader that opens storage only after the producer retires it."""

    import hashlib
    import json
    import mmap
    import os
    import socket

    channel.send("ready")
    locator = channel.recv()
    digest = hashlib.sha256(
        json.dumps(locator, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    key = hashlib.sha256(locator["name"].encode()).digest()
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.connect("\0" + locator["endpoint"])
        connection.sendall(key + digest)
        assert connection.recv(1) == b"G"
        channel.send("granted")
        assert channel.recv() == "consume"
        descriptor = os.open("/dev/shm/" + locator["name"], os.O_RDONLY)
        try:
            with mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ) as storage:
                data = bytearray(storage[: locator["nbytes"]])
        finally:
            os.close(descriptor)
        actual = torch.frombuffer(data, dtype=torch.float32)
        correct = torch.equal(actual, torch.arange(1024, dtype=torch.float32))
        connection.sendall(b"A")
        assert connection.recv(1) == b"D"
        channel.send(correct)
    channel.close()


@pytest.mark.parametrize("source_device", ["cpu", "cuda:0"])
def test_shm_retirement_preserves_granted_fanout_and_reclaims_capacity(source_device: str) -> None:
    context = mp.get_context("spawn")
    events = DeviceEventPool()
    transport = make_transport("shm", byte_capacity=4096, ticket_capacity=2, event_pool=events)
    consumer = make_transport("shm", byte_capacity=4096, ticket_capacity=1, event_pool=events)
    readers = []
    locator = None
    replacement = None
    try:
        for _ in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=_read_granted_shm_publication, args=(child,))
            process.start()
            child.close()
            readers.append((parent, process))
        for channel, _process in readers:
            assert channel.poll(60), "shared-memory reader did not start"
            assert channel.recv() == "ready"
        source = torch.arange(1024, dtype=torch.float32, device=source_device)
        if source.is_cuda:
            stream = torch.cuda.Stream(device=source.device)
            stream.wait_stream(torch.cuda.current_stream(source.device))
            completed = torch.cuda.Event()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1_000_000_000)
                locator = transport.publish(source)
                completed.record(stream)
            assert not completed.query(), "shared-memory publication waited for device completion"
        else:
            locator = transport.publish(source)
        for channel, _process in readers:
            channel.send(locator.to_mapping())
        for channel, _process in readers:
            assert channel.poll(30), "shared-memory reader did not acquire its source"
            assert channel.recv() == "granted"
        retirement = transport.release(locator)
        assert retirement is not None and not retirement.done()
        for channel, process in readers:
            with pytest.raises(WorkerError, match="capacity"):
                transport.publish(source)
            channel.send("consume")
            assert channel.poll(30), "shared-memory reader did not complete"
            assert channel.recv() is True
            process.join(30)
            assert process.exitcode == 0
        retirement.result(timeout=5)
        replacement = transport.publish(source)
        with pytest.raises(WorkerError, match="retired"):
            _await_ticket(consumer.fetch(locator, device=torch.device("cpu")))
    finally:
        for channel, process in readers:
            if process.is_alive():
                process.terminate()
                process.join(30)
            channel.close()
        if locator is not None:
            transport.release(locator)
        if replacement is not None:
            transport.release(replacement)
        consumer.close()
        transport.close()
        events.close()


def test_cuda_ipc_retirement_retains_capacity_until_producer_completion() -> None:
    device = torch.device("cuda:0")
    events = DeviceEventPool()
    transport = make_transport("cuda_ipc", byte_capacity=4096, ticket_capacity=1, event_pool=events)
    source = torch.ones(1024, device=device)
    locator = None
    replacement = None
    try:
        warmup = transport.publish(source)
        torch.cuda.synchronize(device)
        transport.release(warmup)
        stream = torch.cuda.Stream(device=device)
        completed = torch.cuda.Event()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            locator = transport.publish(source)
            completed.record(stream)
        assert not completed.query(), "producer completed before the retirement check"
        retirement = transport.release(locator)
        assert retirement is not None and not retirement.done()
        with pytest.raises(WorkerError, match="capacity"):
            transport.publish(source)
        completed.synchronize()
        events.reap()
        retirement.result(timeout=5)
        replacement = transport.publish(source)
    finally:
        if locator is not None:
            transport.release(locator)
        if replacement is not None:
            transport.release(replacement)
        transport.close()
        events.close()


def test_shm_source_loss_wakes_pending_read_and_preserves_independent_reads() -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=serve_pending_publication, args=(child,))
    events = DeviceEventPool()
    consumer = make_transport("shm", byte_capacity=16384, ticket_capacity=4, event_pool=events)
    producer = make_transport("shm", byte_capacity=4096, ticket_capacity=2, event_pool=events)
    healthy = None
    process.start()
    child.close()
    try:
        assert parent.poll(60), "shared-memory publisher did not start"
        locator = Locator.from_mapping(parent.recv())
        pending = consumer.fetch(locator, device=torch.device("cpu"))
        ready = threading.Event()
        pending.add_done_callback(ready.set)
        assert parent.poll(30), "publisher did not receive the pending read"
        assert parent.recv() == "pending"
        assert not ready.is_set(), "reader exposed bytes before producer readiness"

        healthy = producer.publish(torch.tensor([7.0]))
        actual = _await_ticket(consumer.fetch(healthy, device=torch.device("cpu")))
        torch.testing.assert_close(actual, torch.tensor([7.0]), rtol=0, atol=0)
        parent.send("exit")
        process.join(30)
        assert process.exitcode == 0
        assert ready.wait(5), "publisher loss did not wake its pending reader"
        with pytest.raises(WorkerError, match="endpoint was lost before readiness"):
            pending.result()
        with pytest.raises(WorkerError, match="endpoint was lost before readiness"):
            _await_ticket(consumer.fetch(locator, device=torch.device("cpu")))
        actual = _await_ticket(consumer.fetch(healthy, device=torch.device("cpu")))
        torch.testing.assert_close(actual, torch.tensor([7.0]), rtol=0, atol=0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        if healthy is not None:
            producer.release(healthy)
        consumer.close()
        producer.close()
        events.close()


@pytest.mark.parametrize("owner", ("encoder", "device", "latent"))
def test_cancelled_shard_reads_retain_destination_and_capacity_until_physical_retirement(
    owner: str,
) -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    shape = (256, 4) if owner == "latent" else (1024,)
    process = context.Process(target=serve_pending_publication, args=(child, shape))
    events = DeviceEventPool()
    consumer = make_transport("shm", byte_capacity=4096, ticket_capacity=1, event_pool=events)
    producer = make_transport("shm", byte_capacity=4096, ticket_capacity=1, event_pool=events)
    buffers = PersistentBuffers(byte_capacity=4096, devices=("cpu",))
    if owner == "encoder":
        store = EncoderCache(
            entry_capacity=1,
            max_entry_bytes=4096,
            devices=("cpu",),
            persistent_buffers=buffers,
            event_pool=events,
        )
    elif owner == "device":
        store = DeviceProducts(
            capacity=1, byte_capacity=4096, persistent_buffers=buffers, event_pool=events
        )
    else:
        store = LatentPool(
            request_pool_size=2,
            num_pages=5,
            page_units=64,
            latent_width=4,
            dtype=torch.float32,
            device="cpu",
        )
    product = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=ComputationId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound(tuple(StaticDim(dimension) for dimension in shape)),
    )
    replacement = replace(product, generation=2)
    allocation = BufferAllocation(product.buffer_id, 0, 4096)
    replacement_allocation = replace(allocation, buffer=replacement.buffer_id)

    def reserve(product, allocation):
        if isinstance(store, LatentPool):
            return store.reserve_import(
                product, request_pool_idx=1, page_table=(4, 2, 1, 3), latent_units=256
            )
        return store.bind_outputs(
            ((product, "cpu"),), buffer_allocations={product.buffer_id: allocation}
        )[0]

    def abandon(binding):
        if isinstance(store, LatentPool):
            store.abandon_import(binding)
        else:
            store.abandon_writes((binding,))

    binding = reserve(product, allocation)
    if isinstance(store, LatentPool):
        target = binding.spans
    elif isinstance(store, EncoderCache):
        target = binding.buffer_binding.tensor
    else:
        target = store.producer_write_views((binding,))[0]
    shard_shape = (shape[0] // 2, *shape[1:])
    first_region = TensorRegion((0,) * len(shape), shard_shape)
    second_region = TensorRegion((shard_shape[0], *(0 for _ in shape[1:])), shard_shape)
    healthy = None
    process.start()
    child.close()
    try:
        assert parent.poll(60), "shared-memory publisher did not start"
        locator = Locator.from_mapping(parent.recv())
        completed_source = producer.publish(torch.full(shard_shape, 3.0))
        completed = consumer.fetch(
            completed_source,
            device=torch.device("cpu"),
            destination=region_view(target, first_region),
        )
        store.retain_transfer(binding, completed)
        _await_ticket(completed)
        completed_retired = threading.Event()
        completed.add_retirement_callback(completed_retired.set)
        assert completed_retired.wait(5), "completed shard did not retire"
        producer.release(completed_source).result(timeout=5)

        ticket = consumer.fetch(
            locator,
            device=torch.device("cpu"),
            destination=region_view(target, second_region),
            region=second_region,
        )
        store.retain_transfer(binding, ticket)
        assert parent.poll(30), "publisher did not receive the read"
        assert parent.recv() == "pending"
        retired = threading.Event()
        ticket.add_retirement_callback(retired.set)
        ticket.cancel()
        if isinstance(store, LatentPool):
            store.cancel_imports((product.request_key,))
        abandon(binding)
        assert ticket.ready()
        with pytest.raises(WorkerError, match="cancelled"):
            ticket.result()
        assert not ticket.retired()
        assert not retired.is_set()
        with pytest.raises(WorkerError):
            reserve(replacement, replacement_allocation)
        if isinstance(store, LatentPool):
            assert not store.retirement_ready((product.request_key,))
            with pytest.raises(WorkerError, match="owned"):
                store.reserve_import(
                    replace(replacement, request_key=RequestKey(2, 1, 1)),
                    request_pool_idx=2,
                    page_table=(4, 2, 1, 3),
                    latent_units=256,
                )

        healthy = producer.publish(torch.tensor([7.0]))
        with pytest.raises(WorkerError, match="capacity"):
            consumer.fetch(healthy, device=torch.device("cpu"))
        parent.send("exit")
        process.join(30)
        assert process.exitcode == 0
        assert retired.wait(5), "cancelled physical read did not retire after source loss"
        assert ticket.retired()
        if isinstance(store, LatentPool):
            assert store.retirement_ready((product.request_key,))
        reused = reserve(replacement, replacement_allocation)
        abandon(reused)
        actual = _await_ticket(consumer.fetch(healthy, device=torch.device("cpu")))
        torch.testing.assert_close(actual, torch.tensor([7.0]), rtol=0, atol=0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        if healthy is not None:
            producer.release(healthy)
        consumer.close()
        producer.close()
        store.close()
        buffers.close()
        events.close()


@pytest.mark.parametrize("backend", ("local", "shm", "cuda_ipc"))
def test_publication_rejects_changed_producer_identity(backend: str) -> None:
    device = torch.device("cuda:0")
    events = DeviceEventPool()
    endpoint = WorkerEndpoint.local("encoder-0", rank=2)
    transport = make_transport(
        backend, byte_capacity=16384, ticket_capacity=2, event_pool=events, source=endpoint
    )
    source = torch.arange(32, dtype=torch.float32, device=device)
    destination = torch.empty_like(source)
    locator = transport.publish(source)
    try:
        assert locator.source == endpoint
        for field, value in (
            ("worker_id", "encoder-1"),
            ("rank", 1),
            ("node", "another-node"),
            ("address_space", "another-process"),
            ("incarnation", "another-incarnation"),
        ):
            changed = replace(locator, source=replace(endpoint, **{field: value}))
            with pytest.raises(WorkerError):
                _await_ticket(transport.fetch(changed, device=device, destination=destination))
        _await_ticket(transport.fetch(locator, device=device, destination=destination))
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
    finally:
        transport.release(locator)
        transport.close()
        events.close()


def test_local_delivery_between_workers_retains_the_publisher_until_consumption() -> None:
    device = torch.device("cuda:0")
    producer_events, consumer_events = DeviceEventPool(), DeviceEventPool()
    producer = make_transport(
        "local",
        byte_capacity=128,
        ticket_capacity=1,
        event_pool=producer_events,
        source=WorkerEndpoint.local("encoder"),
    )
    consumer = make_transport(
        "local",
        byte_capacity=128,
        ticket_capacity=1,
        event_pool=consumer_events,
        source=WorkerEndpoint.local("denoiser"),
    )
    source = torch.arange(32, dtype=torch.float32, device=device)
    locator = producer.publish(source)
    try:
        ticket = consumer.fetch(locator, device=device)
        value = _await_ticket(ticket)
        assert value.data_ptr() == source.data_ptr()
        retirement = producer.release(locator)
        assert retirement is not None and not retirement.done()
        with pytest.raises(WorkerError):
            consumer.fetch(locator, device=device)
        # Already-granted consumption remains valid after semantic release.
        torch.testing.assert_close(value, source, rtol=0, atol=0)
        ticket.close()
        torch.cuda.synchronize(device)
        producer_events.reap()
        retirement.result(timeout=30)
        successor = producer.publish(source)
        producer.release(successor)
    finally:
        consumer.close()
        producer.close()
        consumer_events.close()
        producer_events.close()
