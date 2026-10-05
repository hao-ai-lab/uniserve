"""Physical transfer ordering across process and CUDA stream boundaries."""

from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.shm_export import serve_pending_export
from tests.python.fixtures.transport import make_transport
from uniserve.runtime import EventPool
from uniserve_worker.errors import WorkerError, WorkerErrorCode
from uniserve_worker.protocol.batch import BufferAllocation
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.tensor import (
    DType,
    ShapeBound,
    StaticDim,
    TensorRef,
)
from uniserve_worker.protocol.transfer import Locator, WorkerEndpoint
from uniserve_worker.storage.buffer_pool import BufferPool
from uniserve_worker.storage.latent_pool import LatentPool
from uniserve_worker.storage.tensor_store import TensorStore
from uniserve_worker.transport.layout import region_view

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _await_ticket(ticket):
    ready = threading.Event()
    ticket.add_done_callback(ready.set)
    assert ready.wait(30), "transfer did not expose a consumable result"
    return ticket.result()


def test_exhausted_vmm_pool_delivers_host_fallback_and_restores_quota():
    from uniserve_kernels.peer_storage import allocation_granularity

    from uniserve_worker.transport import make_transports
    from uniserve_worker.transport.exports import export_tensor

    device = torch.device("cuda:0")
    page = allocation_granularity(device)
    events = EventPool()
    transports = make_transports(
        ("cuda_vmm", "shm"),
        byte_capacity=page,
        ticket_capacity=2,
        event_pool=events,
        host_slots=(0,),
    )
    consumer = make_transports(
        ("shm",),
        byte_capacity=page,
        ticket_capacity=2,
        event_pool=events,
        acknowledgment_slot=0,
    )["shm"]
    # One physical page fits the payload but cannot also hold the VMM chunk's
    # acknowledgment header. Its logical byte lease must survive fallback.
    source = torch.arange(page // 4, dtype=torch.float32, device=device)
    try:
        for _ in range(2):
            retirements = []
            locations = export_tensor(
                transports,
                source,
                retain=retirements.append,
                consumers=(0,),
            )
            (location,) = locations
            ticket = consumer.fetch(location, device=torch.device("cpu"))
            observed = _await_ticket(ticket)
            torch.testing.assert_close(observed, source.cpu(), rtol=0, atol=0)
            retired = threading.Event()
            ticket.add_retirement_callback(retired.set)
            ticket.close()
            assert retired.wait(5)
            transports[location.backend].release(location)
            transports[location.backend].reap()
            for retirement in retirements:
                retirement.result(timeout=5)
    finally:
        consumer.close()
        for transport in transports.values():
            transport.close()
        events.close()


def test_exhausted_vmm_pool_without_another_mechanism_refuses_the_product():
    from uniserve_kernels.peer_storage import allocation_granularity

    from uniserve_worker.errors import ResourceError
    from uniserve_worker.transport import make_transports
    from uniserve_worker.transport.exports import export_tensor

    device = torch.device("cuda:0")
    page = allocation_granularity(device)
    events = EventPool()
    # A rank whose only export mechanism is CUDA VMM, as an edge that
    # names only a device mechanism binds it.
    transports = make_transports(
        ("cuda_vmm",), byte_capacity=page, ticket_capacity=2, event_pool=events
    )
    # One physical page fits the payload but not also the chunk's header.
    source = torch.arange(page // 4, dtype=torch.float32, device=device)
    try:
        # The refusal is backpressure on this product alone, and it returns
        # the product's byte lease, so a second attempt is refused the same
        # way rather than for an exhausted budget.
        for _ in range(2):
            with pytest.raises(ResourceError, match="does not fit its VMM"):
                export_tensor(
                    transports,
                    source,
                    retain=lambda future: None,
                    consumers=(1,),
                )
    finally:
        for transport in transports.values():
            transport.close()
        events.close()


@pytest.mark.parametrize("backend", ("local", "shm", "cuda_vmm"))
def test_transfer_writes_only_the_reserved_destination(backend: str) -> None:
    device = torch.device("cuda:0")
    events = EventPool()
    transport = make_transport(
        backend, byte_capacity=16384, ticket_capacity=2, event_pool=events
    )
    source = torch.arange(1024, dtype=torch.float32, device=device)
    storage = torch.full((1026,), -1.0, device=device)
    destination = storage[1:-1]
    locator = transport.export(source)
    try:
        with pytest.raises(WorkerError, match="destination disagrees"):
            transport.fetch(
                locator, device=device, destination=destination[:-1]
            )
        # The owner finishes initialization before granting this range to an
        # independent transport stream.
        torch.cuda.synchronize(device)
        ticket = transport.fetch(
            locator, device=device, destination=destination
        )
        _await_ticket(ticket)
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
        torch.testing.assert_close(
            storage[[0, -1]], torch.full((2,), -1.0, device=device)
        )
    finally:
        transport.release(locator)
        transport.close()
        events.close()


@pytest.mark.parametrize("backend", ("local", "shm", "cuda_vmm"))
def test_transfer_scatters_exactly_into_disjoint_page_spans(
    backend: str,
) -> None:
    device = torch.device("cuda:0")
    events = EventPool()
    transport = make_transport(
        backend, byte_capacity=16384, ticket_capacity=1, event_pool=events
    )
    source = torch.arange(44, dtype=torch.float32, device=device).reshape(11, 4)
    storage = torch.full((5, 4, 4), -1.0, device=device)
    destination = (storage[3], storage[1], storage[4, :3])
    locator = transport.export(source)
    try:
        with pytest.raises(WorkerError, match="destination disagrees"):
            transport.fetch(
                locator, device=device, destination=destination[:-1]
            )
        with pytest.raises(WorkerError, match="spans overlap"):
            transport.fetch(
                locator,
                device=device,
                destination=(storage[1], storage[1], storage[4, :3]),
            )
        torch.cuda.synchronize(device)
        ticket = transport.fetch(
            locator, device=device, destination=destination
        )
        values = _await_ticket(ticket)
        torch.testing.assert_close(torch.cat(values), source, rtol=0, atol=0)
        torch.testing.assert_close(
            storage[0], torch.full_like(storage[0], -1.0), rtol=0, atol=0
        )
        torch.testing.assert_close(
            storage[2], torch.full_like(storage[2], -1.0), rtol=0, atol=0
        )
        torch.testing.assert_close(
            storage[4, 3], torch.full_like(storage[4, 3], -1.0), rtol=0, atol=0
        )
    finally:
        transport.release(locator)
        transport.close()
        events.close()


def _read_cuda_exports(channel) -> None:
    torch.cuda.set_device(0)
    event_pool = EventPool()
    transport = make_transport(
        "cuda_vmm",
        byte_capacity=8 << 20,
        ticket_capacity=2,
        event_pool=event_pool,
    )
    try:
        warmup = Locator.from_mapping(channel.recv())
        value = _await_ticket(
            transport.fetch(warmup, device=torch.device("cuda:0"))
        )
        torch.cuda.synchronize()
        del value
        channel.send("ready")
        locator = Locator.from_mapping(channel.recv())
        stream = torch.cuda.Stream()
        pending = torch.cuda.Event()
        destination = (
            torch.empty(511, device="cuda:0"),
            torch.empty(513, device="cuda:0"),
        )
        with torch.cuda.stream(stream):
            # The fetch must return while earlier work on the consuming stream
            # is pending. Checking the device fence avoids a host-time
            # threshold.
            torch.cuda._sleep(1_000_000_000)
            pending.record(stream)
            ticket = transport.fetch(
                locator, device=torch.device("cuda:0"), destination=destination
            )
            asynchronous = not pending.query()
            value = _await_ticket(ticket)
            actual = torch.cat(value).cpu()
        channel.send(
            (
                asynchronous,
                bool(
                    torch.equal(actual, torch.arange(1024, dtype=torch.float32))
                ),
            )
        )
    finally:
        transport.close()
        event_pool.close()
        channel.close()


def test_cuda_vmm_read_does_not_wait_for_consumer_stream() -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    event_pool = EventPool()
    transport = make_transport(
        "cuda_vmm",
        byte_capacity=8 << 20,
        ticket_capacity=2,
        event_pool=event_pool,
    )
    process = context.Process(target=_read_cuda_exports, args=(child,))
    exports = []
    try:
        warmup = transport.export(torch.ones(1024, device="cuda:0"))
        locator = transport.export(
            torch.arange(1024, dtype=torch.float32, device="cuda:0")
        )
        exports.extend((warmup, locator))
        process.start()
        child.close()
        parent.send(warmup.to_mapping())
        assert parent.poll(60), "CUDA VMM consumer did not become ready"
        assert parent.recv() == "ready"
        parent.send(locator.to_mapping())
        assert parent.poll(30), "CUDA VMM consumer did not finish its read"
        asynchronous, correct = parent.recv()
        process.join(30)
        assert process.exitcode == 0
        assert correct, "CUDA VMM read returned the wrong published value"
        assert asynchronous, "CUDA VMM fetch waited for the consumer stream"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        for locator in exports:
            transport.release(locator)
        transport.close()
        event_pool.close()


def _consume_fanout(channel, device_index: int) -> None:
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    event_pool = EventPool()
    transport = make_transport(
        "cuda_vmm",
        byte_capacity=8 << 20,
        ticket_capacity=2,
        event_pool=event_pool,
    )
    try:
        channel.send("ready")
        locator = Locator.from_mapping(channel.recv())
        value = _await_ticket(transport.fetch(locator, device=device))
        channel.send("readable")
        assert channel.recv() == "consume"
        correct = torch.equal(
            value.cpu(), torch.arange(1024, dtype=torch.float32)
        )
        transport.close()
        channel.send((correct, str(value.device)))
    finally:
        transport.close()
        event_pool.close()
        channel.close()


def test_cuda_vmm_export_fits_the_existing_device_allocation() -> None:
    device = torch.device("cuda:0")
    events = EventPool()
    source = (
        torch.arange(65536, dtype=torch.float32, device=device).view(16, 4096).T
    )
    destination = torch.empty(source.shape, dtype=source.dtype, device=device)
    transport = make_transport(
        "cuda_vmm",
        byte_capacity=2 * source.numel() * source.element_size(),
        ticket_capacity=1,
        event_pool=events,
    )
    locator = None
    try:
        warmup = transport.export(source)
        transport.release(warmup)
        torch.cuda.synchronize(device)
        events.reap()
        resident_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        locator = transport.export(source)
        # Export's GPU payload budget is already occupied by its source.
        # An additional tensor-sized copy exceeds that public resource contract.
        assert torch.cuda.max_memory_allocated(device) == resident_bytes
        _await_ticket(
            transport.fetch(locator, device=device, destination=destination)
        )
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
    finally:
        if locator is not None:
            transport.release(locator)
        transport.close()
        events.close()


def test_cuda_vmm_serves_a_fanout_and_refuses_a_retired_export() -> None:
    """One export is read by consumers on two devices, then retires.

    A chunk is addressed by offset in the producing device's pool, so each
    consumer imports the same allocation handle and reads the same bytes.
    """
    context = mp.get_context("spawn")
    event_pool = EventPool()
    transport = make_transport(
        "cuda_vmm", byte_capacity=8192, ticket_capacity=2, event_pool=event_pool
    )
    readers = []
    locator = None
    replacement = None
    try:
        # Warm native export before measuring the ordering of a pending fence.
        warmup = transport.export(torch.ones(1024, device="cuda:0"))
        torch.cuda.synchronize(0)
        transport.release(warmup)
        for device in (0, 1):
            parent, child = context.Pipe()
            process = context.Process(
                target=_consume_fanout, args=(child, device)
            )
            process.start()
            child.close()
            readers.append((parent, process, device))
        for channel, _process, _device in readers:
            assert channel.poll(60), "CUDA VMM fan-out consumer did not start"
            assert channel.recv() == "ready"

        source = torch.arange(1024, dtype=torch.float32, device="cuda:0")
        locator = transport.export(source)
        for channel, _process, _device in readers:
            channel.send(locator.to_mapping())
        for channel, _process, _device in readers:
            assert channel.poll(30), (
                "CUDA VMM fan-out read did not become consumable"
            )
            assert channel.recv() == "readable"

        transport.release(locator)
        for channel, _process, _device in readers:
            channel.send("consume")
        for channel, process, device in readers:
            assert channel.poll(30), (
                "CUDA VMM fan-out consumer did not complete"
            )
            assert channel.recv() == (True, f"cuda:{device}")
            process.join(30)
            assert process.exitcode == 0

        with pytest.raises(WorkerError) as failure:
            _await_ticket(
                transport.fetch(locator, device=torch.device("cuda:0"))
            )
        assert failure.value.code == WorkerErrorCode.INVALID_DESCRIPTOR
        replacement = transport.export(torch.ones(2048, device="cuda:0"))
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


def test_cuda_vmm_rejects_a_changed_registered_view() -> None:
    event_pool = EventPool()
    transport = make_transport(
        "cuda_vmm",
        byte_capacity=8 << 20,
        ticket_capacity=2,
        event_pool=event_pool,
    )
    locator = transport.export(torch.arange(1024, device="cuda:0"))
    try:
        changed = replace(
            locator,
            shape=(512,),
            nbytes=locator.nbytes // 2,
            transport=replace(locator.transport, span_lengths=(512,)),
        )
        with pytest.raises(WorkerError) as failure:
            _await_ticket(
                transport.fetch(changed, device=torch.device("cuda:0"))
            )
        assert failure.value.code == WorkerErrorCode.INVALID_DESCRIPTOR
    finally:
        transport.release(locator)
        transport.close()
        event_pool.close()


@pytest.mark.parametrize("ending", ("close", "drop"))
def test_local_read_keeps_its_producer_fence_after_retirement(
    ending: str,
) -> None:
    device = torch.device("cuda:0")
    events = EventPool()
    transport = make_transport(
        "local", byte_capacity=8192, ticket_capacity=2, event_pool=events
    )
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
            locator = transport.export(source)
        unused = transport.fetch(locator, device=device)
        retained = transport.fetch(locator, device=device)
        del unused
        retirement = transport.release(locator)
        assert retirement is not None and not retirement.done()
        replacement = transport.export(unrelated)
        with torch.cuda.stream(consumer):
            value = retained.result(consumer).clone()
        if ending == "close":
            retained.close()
        else:
            del retained
        consumer.synchronize()
        events.reap()
        retirement.result(timeout=5)
        assert torch.equal(
            value.cpu(), torch.full((1024,), 7, dtype=torch.float32)
        )
    finally:
        if locator is not None:
            transport.release(locator)
        if replacement is not None:
            transport.release(replacement)
        transport.close()
        events.close()


def _serve_unusable_cuda_handle(channel) -> None:
    """Publish a locator whose handle names no allocation, and hold it."""
    import uuid

    from uniserve_kernels.peer_storage import empty, export_handle

    from uniserve_worker.protocol.transfer import CudaVmmTransfer

    torch.cuda.set_device(0)
    source = empty((1024,), dtype=torch.float32, device=torch.device("cuda:0"))
    source.copy_(torch.arange(1024, dtype=torch.float32, device="cuda:0"))
    event = torch.cuda.Event(interprocess=True)
    event.record()
    exported, capacity, offset = export_handle(source)
    # A handle of the right length that names no allocation: the consumer
    # imports from the export, so this is what an unusable handle is.
    locator = Locator(
        source=WorkerEndpoint.local("publisher"),
        transport=CudaVmmTransfer(
            endpoint=f"uniserve-test-read-{uuid.uuid4().hex}",
            export_id=uuid.uuid4().hex,
            storage_size_bytes=capacity,
            storage_offsets_bytes=(offset,),
            span_lengths=(source.shape[0],),
            span_counts=(1,),
            tensor_stride=tuple(source.stride()),
            ready_event_handle=event.ipc_handle(),
            allocation_handle=bytes(len(exported)),
            acknowledgment_offset=-1,
        ),
        nbytes=source.numel() * source.element_size(),
        dtype="float32",
        shape=tuple(source.shape),
        offset=(0,) * source.ndim,
        device="cuda:0",
    )
    channel.send(locator.to_mapping())
    # Keep the allocation alive until the receiver has observed the failure.
    assert channel.recv() == "close"
    channel.close()


def test_cuda_vmm_reports_an_unusable_handle_to_its_reader() -> None:
    """A read whose handle cannot be imported fails at once.

    Nothing connects a consumer back to the producing rank, so the failure
    is the consumer's own and does not wait on anything the producer holds.
    """
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_serve_unusable_cuda_handle, args=(child,))
    events = EventPool()
    transport = make_transport(
        "cuda_vmm", byte_capacity=8192, ticket_capacity=1, event_pool=events
    )
    process.start()
    child.close()
    try:
        assert parent.poll(60), "CUDA VMM source did not start"
        locator = Locator.from_mapping(parent.recv())
        ticket = transport.fetch(locator, device=torch.device("cuda:0"))
        ready = threading.Event()
        ticket.add_done_callback(ready.set)
        assert ready.wait(30), "import failure waited on the producer"
        with pytest.raises(RuntimeError, match="allocation"):
            ticket.result()
        transport.close()
        assert ticket.retired()
        with pytest.raises(RuntimeError, match="allocation"):
            ticket.result()
    finally:
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


def _read_shm_export(channel, slot: int) -> None:
    """External reader that acknowledges the segment only when told to.

    The reader waits on readiness and claims its acknowledgment word before
    its first read, as the transport does,
    and holds the claim across the producer's release so retirement is
    observed to wait for it.
    """
    from tests.python.fixtures import segment
    from tests.python.fixtures.shared_storage import open_shared_storage

    channel.send("ready")
    locator = Locator.from_mapping(channel.recv())
    storage = open_shared_storage(
        locator.transport.name, segment.HEADER_BYTES + locator.nbytes
    )
    try:
        header = memoryview(storage)
        segment.claim(header, slot)
        segment.await_ready(header)
        payload = segment.HEADER_BYTES
        data = bytearray(storage[payload : payload + locator.nbytes])
        actual = torch.frombuffer(data, dtype=torch.float32)
        correct = torch.equal(actual, torch.arange(1024, dtype=torch.float32))
        channel.send("readable")
        assert channel.recv() == "consume"
        segment.acknowledge(header, slot)
        header.release()
    finally:
        storage.close()
    channel.send(correct)
    channel.close()


@pytest.mark.parametrize("source_device", ["cpu", "cuda:0"])
def test_shm_retirement_waits_for_its_consumers_and_reclaims_capacity(
    source_device: str,
) -> None:
    """A segment returns once every consumer that claimed it has acknowledged.

    A device-sourced export returns before its bytes have landed; the
    readers wait on the segment's readiness word instead. Retirement holds the
    segment and its capacity until both readers, each of which claimed its
    word before reading, have acknowledged.
    """
    context = mp.get_context("spawn")
    events = EventPool()
    transport = make_transport(
        "shm", byte_capacity=4096, ticket_capacity=2, event_pool=events
    )
    consumer = make_transport(
        "shm", byte_capacity=4096, ticket_capacity=1, event_pool=events
    )
    readers = []
    locator = None
    replacement = None
    try:
        for slot in (1, 2):
            parent, child = context.Pipe()
            process = context.Process(
                target=_read_shm_export, args=(child, slot)
            )
            process.start()
            child.close()
            readers.append((parent, process))
        for channel, _process in readers:
            assert channel.poll(60), "shared-storage reader did not start"
            assert channel.recv() == "ready"
        source = torch.arange(1024, dtype=torch.float32, device=source_device)
        if source.is_cuda:
            stream = torch.cuda.Stream(device=source.device)
            stream.wait_stream(torch.cuda.current_stream(source.device))
            completed = torch.cuda.Event()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1_000_000_000)
                locator = transport.export(source, consumers=(1, 2))
                completed.record(stream)
            assert not completed.query(), (
                "shared-storage export waited for device completion"
            )
        else:
            locator = transport.export(source, consumers=(1, 2))
        for channel, _process in readers:
            channel.send(locator.to_mapping())
        for channel, _process in readers:
            assert channel.poll(30), (
                "shared-storage reader did not find the segment ready"
            )
            assert channel.recv() == "readable"
        retirement = transport.release(locator)
        assert retirement is not None and not retirement.done()
        for channel, process in readers:
            with pytest.raises(WorkerError, match="capacity"):
                transport.export(source)
            channel.send("consume")
            assert channel.poll(30), "shared-storage reader did not complete"
            assert channel.recv() is True
            process.join(30)
            assert process.exitcode == 0
        transport.reap()
        retirement.result(timeout=5)
        replacement = transport.export(source)
        with pytest.raises(WorkerError) as retired:
            _await_ticket(consumer.fetch(locator, device=torch.device("cpu")))
        assert retired.value.code is WorkerErrorCode.INVALID_DESCRIPTOR
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


def test_cuda_vmm_export_read_from_another_host_carries_no_fence() -> None:
    """A chunk whose readers are elsewhere is readable when it is published.

    An interprocess event handle does not reach another host, and imported VMM
    storage admits no device-side wait on current drivers, so the producer
    synchronizes after copying into the chunk instead and the export
    carries nothing for a consumer to wait on.
    """
    device = torch.device("cuda:0")
    events = EventPool()
    transport = make_transport(
        "cuda_vmm",
        byte_capacity=1 << 20,
        ticket_capacity=1,
        event_pool=events,
        cross_host_consumers=True,
    )
    locator = None
    try:
        # Exportable storage, so the export is the source itself: a
        # crossing decides the fence, not where the product is materialized.
        from uniserve_kernels.peer_storage import empty

        source = empty((1024,), dtype=torch.float32, device=device)
        source.fill_(1.0)
        stream = torch.cuda.Stream(device=device)
        submitted = torch.cuda.Event()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            locator = transport.export(source)
            submitted.record(stream)

        assert submitted.query(), "export returned before its copy had landed"
        assert not locator.transport.ready_event_handle, (
            "a pool export carries a fence a consumer cannot import"
        )
    finally:
        if locator is not None:
            transport.release(locator)
        transport.close()
        events.close()


def test_cuda_vmm_export_on_this_host_does_not_stall_its_producer() -> None:
    """Readiness within a host is an event, which costs the producer nothing.

    Every consumer on this host can wait on an interprocess event, so a
    export hands them one rather than draining the producing stream. A
    synchronize here would be a bubble the placement does not require, and on
    a single-host instance every export would pay it.
    """
    device = torch.device("cuda:0")
    events = EventPool()
    transport = make_transport(
        "cuda_vmm", byte_capacity=1 << 20, ticket_capacity=1, event_pool=events
    )
    locator = None
    try:
        source = torch.ones(1024, device=device)
        stream = torch.cuda.Stream(device=device)
        submitted = torch.cuda.Event()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            locator = transport.export(source)
            submitted.record(stream)

        assert not submitted.query(), "export drained the producing stream"
        assert len(locator.transport.ready_event_handle) == 64, (
            "an export read on this host carries the fence to wait on"
        )
    finally:
        if locator is not None:
            transport.release(locator)
        transport.close()
        events.close()


def test_cuda_vmm_retirement_retains_capacity_until_it_completes() -> None:
    """Byte capacity is held from export to retirement.

    Capacity bounds what a rank can have published at once, so it is returned
    when the export retires rather than when its copy completes.
    """
    device = torch.device("cuda:0")
    events = EventPool()
    transport = make_transport(
        "cuda_vmm", byte_capacity=4096, ticket_capacity=1, event_pool=events
    )
    source = torch.ones(1024, device=device)
    locator = None
    replacement = None
    try:
        locator = transport.export(source)
        with pytest.raises(WorkerError, match="capacity"):
            transport.export(source)

        retirement = transport.release(locator)
        assert retirement is not None
        events.reap()
        retirement.result(timeout=5)
        replacement = transport.export(source)
    finally:
        if locator is not None:
            transport.release(locator)
        if replacement is not None:
            transport.release(replacement)
        transport.close()
        events.close()


def test_shm_producer_failure_fails_its_pending_read_and_preserves_independent_reads() -> (  # noqa: E501
    None
):
    """A pending read waits on the segment's readiness word.

    A producer that fails marks its segment failed and unlinks it, so the
    waiting read fails and a later read finds no export, while reads of
    a healthy export are unaffected.
    """
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=serve_pending_export, args=(child,))
    events = EventPool()
    consumer = make_transport(
        "shm", byte_capacity=16384, ticket_capacity=4, event_pool=events
    )
    producer = make_transport(
        "shm", byte_capacity=4096, ticket_capacity=2, event_pool=events
    )
    healthy = None
    process.start()
    child.close()
    try:
        assert parent.poll(60), "shared-storage publisher did not start"
        locator = Locator.from_mapping(parent.recv())
        pending = consumer.fetch(locator, device=torch.device("cpu"))
        ready = threading.Event()
        pending.add_done_callback(ready.set)
        assert not ready.wait(0.5), (
            "reader exposed bytes before producer readiness"
        )

        healthy = producer.export(torch.tensor([7.0]))
        actual = _await_ticket(
            consumer.fetch(healthy, device=torch.device("cpu"))
        )
        torch.testing.assert_close(actual, torch.tensor([7.0]), rtol=0, atol=0)
        parent.send("exit")
        process.join(30)
        assert process.exitcode == 0
        assert ready.wait(5), (
            "publisher failure did not wake its pending reader"
        )
        with pytest.raises(WorkerError, match="failed before readiness"):
            pending.result()
        with pytest.raises(WorkerError) as retired:
            _await_ticket(consumer.fetch(locator, device=torch.device("cpu")))
        assert retired.value.code is WorkerErrorCode.INVALID_DESCRIPTOR
        actual = _await_ticket(
            consumer.fetch(healthy, device=torch.device("cpu"))
        )
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
def test_cancelled_pending_shards_release_destination_after_read_retirement(
    owner: str,
) -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    shape = (256, 4) if owner == "latent" else (1024,)
    process = context.Process(target=serve_pending_export, args=(child, shape))
    events = EventPool()
    consumer = make_transport(
        "shm", byte_capacity=4096, ticket_capacity=1, event_pool=events
    )
    producer = make_transport(
        "shm", byte_capacity=4096, ticket_capacity=1, event_pool=events
    )
    buffers = BufferPool(byte_capacity=4096, devices=("cpu",))
    if owner == "encoder":
        store = TensorStore(
            max_feature_bytes=4096,
            devices=("cpu",),
            buffer_pool=buffers,
            event_pool=events,
        )
    elif owner == "device":
        store = TensorStore(
            capacity=1,
            byte_capacity=4096,
            buffer_pool=buffers,
            event_pool=events,
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
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound(
            tuple(StaticDim(dimension) for dimension in shape)
        ),
    )
    replacement = replace(product, generation=2)
    allocation = BufferAllocation(product.buffer_id, 0, 4096)
    replacement_allocation = replace(allocation, buffer=replacement.buffer_id)

    def reserve(product, allocation):
        if isinstance(store, LatentPool):
            return store.reserve_import(
                product,
                request_pool_idx=1,
                page_table=(4, 2, 1, 3),
                latent_units=256,
            )
        reserve_tensor = (
            store.reserve_features if owner == "encoder" else store.bind_outputs
        )
        return reserve_tensor(
            ((product, "cpu"),),
            buffer_allocations={product.buffer_id: allocation},
        )[0]

    def abandon(binding):
        if isinstance(store, LatentPool):
            store.abandon_import(binding)
        else:
            store.abandon_writes((binding,))

    binding = reserve(product, allocation)
    if isinstance(store, LatentPool):
        target = binding.spans
    else:
        target = store.producer_write_views((binding,))[0]
    shard_shape = (shape[0] // 2, *shape[1:])
    first_region = tuple(
        slice(start, start + extent)
        for start, extent in zip((0,) * len(shape), shard_shape, strict=True)
    )
    second_region = tuple(
        slice(start, start + extent)
        for start, extent in zip(
            (shard_shape[0], *(0 for _ in shape[1:])), shard_shape, strict=True
        )
    )
    healthy = None
    process.start()
    child.close()
    try:
        assert parent.poll(60), "shared-storage publisher did not start"
        locator = Locator.from_mapping(parent.recv())
        completed_source = producer.export(torch.full(shard_shape, 3.0))
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
        pending = threading.Event()
        ticket.add_done_callback(pending.set)
        assert not pending.wait(0.5), "read completed before the producer"
        retired = threading.Event()
        ticket.add_retirement_callback(retired.set)
        ticket.cancel()
        if isinstance(store, LatentPool):
            store.cancel_imports((product.request_key,))
        abandon(binding)
        assert ticket.ready()
        with pytest.raises(WorkerError, match="cancelled"):
            ticket.result()
        # Waiting for unread source bytes is cancellable. A retired read
        # releases its destination, capacity and producer claim together.
        assert retired.wait(5), "cancelled unread transfer did not retire"
        parent.send("settled")
        assert parent.poll(5) and parent.recv()
        parent.send("exit")
        process.join(30)
        assert process.exitcode == 0
        assert ticket.retired()
        if isinstance(store, LatentPool):
            assert store.retirement_ready((product.request_key,))
        reused = reserve(replacement, replacement_allocation)
        abandon(reused)
        healthy = producer.export(torch.tensor([7.0]))
        actual = _await_ticket(
            consumer.fetch(healthy, device=torch.device("cpu"))
        )
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


def test_local_delivery_between_workers_retains_the_publisher_until_consumption() -> (  # noqa: E501
    None
):
    device = torch.device("cuda:0")
    producer_events, consumer_events = EventPool(), EventPool()
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
    locator = producer.export(source)
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
        successor = producer.export(source)
        producer.release(successor)
    finally:
        consumer.close()
        producer.close()
        consumer_events.close()
        producer_events.close()


def test_cuda_vmm_source_retires_before_its_consumers_acknowledge() -> None:
    """A published source is reusable as soon as its own producer fence drains.

    Publishing copies the product into a pool chunk, so the source and the
    chunk have separate lifetimes. Holding the source until consumers finished
    reading would stall the producing rank's slot reset behind a rank that has
    not yet run the batch that reads the product.
    """
    device = torch.device("cuda:0")
    events = EventPool()
    transport = make_transport(
        "cuda_vmm", byte_capacity=1 << 20, ticket_capacity=2, event_pool=events
    )
    source = torch.ones(1024, device=device)
    try:
        # A consumer that never acknowledges: the chunk stays out of the pool.
        locator = transport.export(source, consumers=(1,))
        torch.cuda.synchronize(device)

        retirement = transport.release(locator)
        assert retirement is not None
        events.reap()
        # No consumer wrote its word, yet the source is the producer's again.
        retirement.result(timeout=5)
    finally:
        transport.close()
        events.close()


@pytest.mark.parametrize("storage", ("direct", "pooled"))
def test_cuda_vmm_local_copy_retains_its_source_until_device_completion(
    storage,
):
    from uniserve_kernels.peer_storage import empty

    device = torch.device("cuda:0")
    events = EventPool()
    producer = make_transport(
        "cuda_vmm", byte_capacity=1 << 20, ticket_capacity=1, event_pool=events
    )
    consumer = make_transport(
        "cuda_vmm", byte_capacity=1 << 20, ticket_capacity=1, event_pool=events
    )
    allocate = empty if storage == "direct" else torch.empty
    source = allocate((1024,), dtype=torch.float32, device=device)
    source.fill_(7)
    locator = producer.export(source)
    destination = torch.empty_like(source)
    initializer = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    try:
        with torch.cuda.stream(initializer):
            # Readiness can be exposed while device access still waits for
            # the destination's previous work. The source must stay held.
            torch.cuda._sleep(1_000_000_000)
            destination.zero_()
            ticket = consumer.fetch(
                locator, device=device, destination=destination
            )
        _await_ticket(ticket)
        retirement = producer.release(locator)
        events.reap()
        assert retirement is not None and not retirement.done()

        retired = threading.Event()
        ticket.add_retirement_callback(retired.set)
        assert retired.wait(10)
        events.reap()
        retirement.result(timeout=5)
        torch.testing.assert_close(destination, torch.full_like(source, 7))
    finally:
        consumer.close()
        producer.close()
        events.close()


def _read_late_filled_export(channel) -> None:
    """External consumer reading only once the producer says it is filled."""
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    event_pool = EventPool()
    transport = make_transport(
        "cuda_vmm",
        byte_capacity=8 << 20,
        ticket_capacity=2,
        event_pool=event_pool,
    )
    try:
        channel.send("ready")
        locator = Locator.from_mapping(channel.recv())
        assert channel.recv() == "filled"
        value = _await_ticket(transport.fetch(locator, device=device))
        channel.send(bool(torch.equal(value.cpu(), torch.full((256,), 7.0))))
    finally:
        transport.close()
        event_pool.close()
        channel.close()


def test_cuda_vmm_publishes_a_row_whose_bytes_arrive_later() -> None:
    """A export may name storage its producer has not written yet.

    An encoded media unit is published with the batch that reserves its row and
    filled when the encode completes; the engine schedules its consumer only
    after that. Publishing therefore has to name the row rather than snapshot
    it, or the consumer reads whatever the row held beforehand.
    """
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    events = EventPool()
    transport = make_transport(
        "cuda_vmm", byte_capacity=8 << 20, ticket_capacity=2, event_pool=events
    )
    process = context.Process(target=_read_late_filled_export, args=(child,))
    locator = None
    try:
        process.start()
        child.close()
        assert parent.poll(60), "late-fill consumer did not start"
        assert parent.recv() == "ready"

        # The row lives in the worker's exportable arena, as a reserved media
        # unit row does; that is what lets it be published where it lies.
        from uniserve_kernels.peer_storage import empty

        row = empty((256,), dtype=torch.float32, device=torch.device("cuda:0"))
        row.zero_()
        locator = transport.export(row)
        parent.send(locator.to_mapping())

        # The producer writes the row after it is published, as the encode
        # round does, and only then admits the read.
        row.fill_(7.0)
        torch.cuda.synchronize(0)
        parent.send("filled")

        assert parent.poll(60), "late-fill consumer did not finish"
        assert parent.recv() is True, (
            "the consumer read the row as it was before the producer wrote it"
        )
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        if locator is not None:
            transport.release(locator)
        transport.close()
        events.close()


def test_cuda_vmm_refuses_a_descriptor_handle_from_another_host() -> None:
    """A device product crosses hosts as a fabric handle and not otherwise.

    A process descriptor names an allocation only within the host that
    exported it, so a consumer elsewhere is told that rather than being left
    to fail inside the driver. A fabric handle carries no such restriction,
    which is what lets a rank read a product produced on another machine.
    """
    from dataclasses import replace

    device = torch.device("cuda:0")
    events = EventPool()
    transport = make_transport(
        "cuda_vmm", byte_capacity=1 << 20, ticket_capacity=1, event_pool=events
    )
    locator = None
    try:
        locator = transport.export(torch.ones(256, device=device))
        elsewhere = replace(
            locator, source=replace(locator.source, node="another-host")
        )

        handle = elsewhere.transport
        if len(handle.allocation_handle) == 64:
            # This device exports fabric handles, so the crossing is allowed
            # and the read fails for want of a live producer there instead.
            with pytest.raises(WorkerError):
                _await_ticket(transport.fetch(elsewhere, device=device))
        else:
            with pytest.raises(WorkerError, match="fabric handle"):
                _await_ticket(transport.fetch(elsewhere, device=device))
    finally:
        if locator is not None:
            transport.release(locator)
        transport.close()
        events.close()
