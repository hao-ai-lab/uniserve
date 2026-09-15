"""Logical product coverage delivered through real local, SHM and CUDA IPC endpoints."""

from __future__ import annotations

import errno
import multiprocessing as mp
import os
import threading

import pytest
import torch

from uniserve.runtime import EventPool
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.protocol.batch import BufferAllocation
from uniserve_worker.protocol.identity import ComputationId, RequestKey
from uniserve_worker.protocol.tensor import (
    DeviceDim,
    DType,
    ShapeBound,
    StaticDim,
    TensorRef,
)
from uniserve_worker.protocol.transfer import TensorTransfer, WorkerEndpoint
from uniserve_worker.runtime.buffer_pool import BufferPool
from uniserve_worker.runtime.tensor_store import FeatureMetadata, TensorStore
from uniserve_worker.transfer.layout import fetch_tensor
from uniserve_worker.transfer.tickets import make_transport, make_transports

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _consume(tickets) -> None:
    for ticket in tickets:
        ready = threading.Event()
        ticket.add_done_callback(ready.set)
        assert ready.wait(30), "tensor read did not become consumable"
        ticket.result()


@pytest.mark.parametrize(
    "backend,device",
    (
        ("local", "cpu"),
        ("shm", "cpu"),
        ("local", "cuda:0"),
        ("shm", "cuda:0"),
        ("cuda_ipc", "cuda:0"),
    ),
)
@pytest.mark.parametrize("shard_axis", (0, 1))
@torch.inference_mode()
def test_tensor_resharding_preserves_values_and_destination_bounds(
    backend: str, device: str, shard_axis: int
) -> None:
    events = [EventPool() for _ in range(3)]
    endpoints = [WorkerEndpoint.local("encoder", rank=rank) for rank in range(2)]
    producers = [
        make_transport(
            backend,
            byte_capacity=16384,
            ticket_capacity=4,
            event_pool=events[rank],
            source=endpoint,
        )
        for rank, endpoint in enumerate(endpoints)
    ]
    consumer = make_transport(
        backend,
        byte_capacity=16384,
        ticket_capacity=4,
        event_pool=events[2],
        source=WorkerEndpoint.local("denoiser"),
    )
    expected = torch.arange(48, dtype=torch.float32, device=device).reshape(6, 8)
    pieces = expected.chunk(2, dim=shard_axis)
    offsets = [(0, 0), (3, 0) if shard_axis == 0 else (0, 4)]
    reference = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=ComputationId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(6), StaticDim(8))),
    )
    arenas = [
        BufferPool(byte_capacity=piece.numel() * piece.element_size(), devices=(device,))
        for piece in pieces
    ]
    stores = [
        TensorStore(capacity=1, byte_capacity=16, buffer_pool=arena, event_pool=event)
        for arena, event in zip(arenas, events[:2], strict=True)
    ]
    locations = []
    try:
        for producer, piece, offset, store in zip(producers, pieces, offsets, stores, strict=True):
            region = tuple(
                slice(start, start + extent)
                for start, extent in zip(offset, tuple(piece.shape), strict=True)
            )
            write = store.bind_outputs(
                ((reference, device),),
                buffer_allocations={
                    reference.buffer_id: BufferAllocation(
                        reference.buffer_id, 0, piece.numel() * piece.element_size()
                    )
                },
                regions={reference: region},
            )[0]
            physical = store.publish_write(write, piece)
            store.commit_writes((write,))
            location = producer.publish(physical, offset=tuple(axis.start for axis in region))
            store.retain_publication(write, producer.publication_retirement(location))
            locations.append(location)
            store.release_requests((reference.request_key,))
        tensor = TensorTransfer(shape=(6, 8), locations=tuple(locations))
        # The consumer needs a region spanning both source shards. Its storage
        # has guard rows and columns outside the authorized write range.
        storage = torch.full((6, 6), -1.0, device=device)
        destination = storage[1:5, 1:5]
        region = (slice(1, 5), slice(2, 6))
        _consume(
            fetch_tensor(
                tensor,
                destination,
                bindings={(endpoint, backend): consumer for endpoint in endpoints},
                region=region,
            )
        )
        torch.testing.assert_close(destination, expected[1:5, 2:6], rtol=0, atol=0)
        assert bool(torch.all(storage[0] == -1)) and bool(torch.all(storage[-1] == -1))
        assert bool(torch.all(storage[:, 0] == -1)) and bool(torch.all(storage[:, -1] == -1))
    finally:
        consumer.close()
        for producer, location in zip(producers, locations):
            producer.release(location)
        for producer in producers:
            producer.close()
        for store in stores:
            store.close()
        for arena in arenas:
            arena.close()
        for pool in events:
            pool.close()


def test_missing_coverage_is_rejected_before_destination_writes() -> None:
    events = EventPool()
    transport = make_transport("local", byte_capacity=16384, ticket_capacity=2, event_pool=events)
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    location = transport.publish(source)
    destination = torch.full((6, 4), -1.0)
    try:
        tensor = TensorTransfer(shape=(6, 4), locations=(location,))
        with pytest.raises(WorkerError, match="do not cover"):
            fetch_tensor(
                tensor, destination, bindings={(location.source, location.backend): transport}
            )
        assert bool(torch.all(destination == -1))
    finally:
        transport.release(location)
        transport.close()
        events.close()


def test_shard_reads_reject_aliasing_destination_pages_before_writing() -> None:
    events = EventPool()
    transport = make_transport("local", byte_capacity=16384, ticket_capacity=2, event_pool=events)
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    locations = (transport.publish(source[:3]), transport.publish(source[3:], offset=(3, 0)))
    storage = torch.full((3, 4), -1.0)
    try:
        tensor = TensorTransfer(shape=(6, 4), locations=locations)
        with pytest.raises(WorkerError, match="spans overlap"):
            fetch_tensor(
                tensor, (storage, storage), bindings={(transport.source, transport.name): transport}
            )
        assert bool(torch.all(storage == -1))
    finally:
        for location in locations:
            transport.release(location)
        transport.close()
        events.close()


def test_explicit_replica_binding_can_use_an_independent_live_publication() -> None:
    events = EventPool()
    first = make_transport(
        "shm",
        byte_capacity=16384,
        ticket_capacity=2,
        event_pool=events,
        source=WorkerEndpoint.local("encoder", rank=0),
    )
    second = make_transport(
        "shm",
        byte_capacity=16384,
        ticket_capacity=2,
        event_pool=events,
        source=WorkerEndpoint.local("encoder", rank=1),
    )
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    unavailable, available = first.publish(expected), second.publish(expected)
    first.release(unavailable)
    first.close()
    destination = torch.empty_like(expected)
    try:
        tensor = TensorTransfer(shape=(3, 4), locations=(unavailable, available))
        _consume(
            fetch_tensor(
                tensor, destination, bindings={(available.source, available.backend): second}
            )
        )
        torch.testing.assert_close(destination, expected, rtol=0, atol=0)
    finally:
        second.release(available)
        second.close()
        events.close()


def test_shm_cuda_region_fits_its_reserved_device_allocation() -> None:
    events = EventPool()
    transport = make_transport("shm", byte_capacity=16384, ticket_capacity=2, event_pool=events)
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    location = transport.publish(source)
    destination = torch.full((4, 3), -1.0, device="cuda:0")
    try:
        tensor = TensorTransfer(shape=(4, 6), locations=(location,))
        resident_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _consume(
            fetch_tensor(
                tensor,
                destination,
                bindings={(location.source, location.backend): transport},
                region=(slice(0, 4), slice(1, 4)),
            )
        )
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() == resident_bytes
        torch.testing.assert_close(destination.cpu(), source[:, 1:4], rtol=0, atol=0)
    finally:
        transport.release(location)
        transport.close()
        events.close()


def _receive_tensor_shards(channel, backends: tuple[str, ...]) -> None:
    events = EventPool()
    consumers = make_transports(backends, byte_capacity=16384, ticket_capacity=4, event_pool=events)
    try:
        tensor = TensorTransfer.from_mapping(channel.recv())
        destination = torch.empty(tensor.shape, device="cuda:0")
        _consume(
            fetch_tensor(
                tensor,
                destination,
                bindings={
                    (location.source, location.backend): consumers[location.backend]
                    for location in tensor.locations
                },
            )
        )
        channel.send(destination.cpu().tolist())
    finally:
        for consumer in consumers.values():
            consumer.close()
        events.close()
        channel.close()


@pytest.mark.parametrize("fragmented", (False, True))
@pytest.mark.parametrize("backends", (("cuda_ipc",), ("shm", "cuda_ipc")))
def test_tensor_delivery_gathers_shards_across_processes(
    fragmented: bool, backends: tuple[str, ...]
) -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    events = EventPool()
    producers = make_transports(backends, byte_capacity=16384, ticket_capacity=4, event_pool=events)
    source = torch.arange(48, dtype=torch.float32, device="cuda:0").reshape(6, 8)
    order = (4, 1, 5, 2, 0, 3) if fragmented else tuple(range(6))
    expected = source[list(order)].cpu().tolist()
    locations = [
        producers[backends[0]].publish(
            tuple(source[index : index + 1, :4] for index in order) if fragmented else source[:, :4]
        ),
        producers[backends[-1]].publish(
            tuple(source[index : index + 1, 4:] for index in order)
            if fragmented
            else source[:, 4:],
            offset=(0, 4),
        ),
    ]
    process = context.Process(target=_receive_tensor_shards, args=(child, backends))
    try:
        process.start()
        parent.send(TensorTransfer(shape=(6, 8), locations=tuple(locations)).to_mapping())
        assert parent.poll(45), "cross-process tensor delivery did not complete"
        assert parent.recv() == expected
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        for location in locations:
            producers[location.backend].release(location)
        for producer in producers.values():
            producer.close()
        events.close()
        parent.close()
        child.close()


def _receive_independent_shards(channel) -> None:
    events = EventPool()
    consumer = make_transport(
        "cuda_ipc", byte_capacity=12 << 20, ticket_capacity=4, event_pool=events
    )
    try:
        tensor = TensorTransfer.from_mapping(channel.recv())
        destination = torch.empty(tensor.shape, device="cuda:0")
        independent = torch.cuda.Stream()
        pending = torch.cuda.Event()
        torch.cuda.synchronize()
        with torch.cuda.stream(independent):
            torch.cuda._sleep(2_000_000_000)
            pending.record()
        tickets = fetch_tensor(
            tensor,
            destination,
            bindings={(location.source, "cuda_ipc"): consumer for location in tensor.locations},
        )
        _consume(tickets)
        actual = destination.cpu()
        progressed = not pending.query()
        expected = torch.arange(3, dtype=torch.float32)[:, None, None].expand_as(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        independent.synchronize()
        consumer.close()
        channel.send(
            {"independent_pending": progressed, "retired": all(t.retired() for t in tickets)}
        )
    finally:
        consumer.close()
        events.close()
        channel.close()


def test_sharded_delivery_progresses_during_independent_device_work() -> None:
    # Each published shard has its own allocation. Delivering their complete
    # logical tensor must not depend on an unrelated consumer-GPU computation.
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    events = EventPool()
    producer = make_transport(
        "cuda_ipc", byte_capacity=12 << 20, ticket_capacity=4, event_pool=events
    )
    sources = tuple(
        torch.full((1, 1024, 1024), float(index), device="cuda:1") for index in range(3)
    )
    locations = tuple(
        producer.publish(source, offset=(index, 0, 0)) for index, source in enumerate(sources)
    )
    process = context.Process(target=_receive_independent_shards, args=(child,))
    try:
        process.start()
        parent.send(TensorTransfer(shape=(3, 1024, 1024), locations=locations).to_mapping())
        assert parent.poll(45), "cross-process tensor delivery did not complete"
        result = parent.recv()
        process.join(30)
        assert process.exitcode == 0
        assert result["independent_pending"], "sharded delivery waited for independent device work"
        assert result["retired"]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        for location in locations:
            producer.release(location)
        producer.close()
        events.close()
        parent.close()
        child.close()


def test_backends_share_the_rank_byte_budget_until_publications_retire() -> None:
    events = EventPool()
    transports = make_transports(
        ("local", "shm"), byte_capacity=64, ticket_capacity=2, event_pool=events
    )
    source = torch.arange(8, dtype=torch.float32)
    local = transports["local"].publish(source)
    shared = transports["shm"].publish(source)
    try:
        with pytest.raises(WorkerError, match="byte capacity is exhausted"):
            transports["local"].publish(source)
        retired = transports["shm"].release(shared)
        assert retired is not None
        retired.result(timeout=10)
        replacement = transports["local"].publish(source)
        transports["local"].release(replacement)
    finally:
        transports["local"].release(local)
        transports["shm"].release(shared)
        for transport in transports.values():
            transport.close()
        events.close()


def test_read_ticket_capacity_is_shared_across_backends() -> None:
    events = EventPool()
    transports = make_transports(
        ("local", "shm"), byte_capacity=1024, ticket_capacity=1, event_pool=events
    )
    source = torch.arange(8, dtype=torch.float32)
    local = transports["local"].publish(source)
    shared = transports["shm"].publish(source)
    borrowed = transports["local"].fetch(local, device=torch.device("cpu"))
    destination = torch.full_like(source, -1)
    try:
        with pytest.raises(WorkerError, match="ticket capacity is exhausted"):
            transports["shm"].fetch(shared, device=destination.device, destination=destination)
        assert bool(torch.all(destination == -1))
        borrowed.close()
        _consume(
            (transports["shm"].fetch(shared, device=destination.device, destination=destination),)
        )
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
    finally:
        borrowed.close()
        for location in (local, shared):
            transports[location.backend].release(location)
        for transport in transports.values():
            transport.close()
        events.close()


@pytest.mark.parametrize(
    "backend,device",
    (
        ("local", "cpu"),
        ("shm", "cpu"),
        ("local", "cuda:0"),
        ("shm", "cuda:0"),
        ("cuda_ipc", "cuda:0"),
    ),
)
def test_fragmented_layer_pages_use_one_read_into_reserved_pages(backend: str, device: str) -> None:
    events = EventPool()
    transport = make_transport(backend, byte_capacity=32768, ticket_capacity=1, event_pool=events)
    source = torch.arange(3 * 7 * 4 * 2 * 3, dtype=torch.float32, device=device).reshape(
        3, 7, 4, 2, 3
    )
    spans = tuple(
        source[:, page, start : start + length].permute(1, 0, 2, 3)
        for page, start, length in ((5, 2, 2), (1, 0, 4), (4, 0, 1))
    )
    expected = torch.cat(spans)
    storage = torch.full((3, 8, 4, 2, 3), -1.0, device=device)
    destinations = (
        storage[:, 6, 1:].permute(1, 0, 2, 3),
        storage[:, 2].permute(1, 0, 2, 3),
    )
    location = None
    try:
        if device.startswith("cuda"):
            resident_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        location = transport.publish(spans)
        _consume(
            fetch_tensor(
                TensorTransfer(shape=tuple(expected.shape), locations=(location,)),
                destinations,
                bindings={(location.source, location.backend): transport},
            )
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_allocated() == resident_bytes
        torch.testing.assert_close(torch.cat(destinations), expected, rtol=0, atol=0)
        untouched = storage[:, (0, 1, 3, 4, 5, 7)]
        assert bool(torch.all(untouched == -1))
        assert bool(torch.all(storage[:, 6, 0] == -1))
    finally:
        if location is not None:
            transport.release(location)
        transport.close()
        events.close()


@pytest.mark.parametrize(
    "backend,device", (("local", "cpu"), ("shm", "cpu"), ("cuda_ipc", "cuda:0"))
)
@pytest.mark.parametrize("shard_axis", (0, 1))
@torch.inference_mode()
@pytest.mark.parametrize("feature", (False, True))
def test_resident_shard_materialization_preserves_readers_and_shared_consumers(
    backend: str, device: str, shard_axis: int, feature: bool
) -> None:
    events = EventPool()
    transport = make_transport(
        backend,
        byte_capacity=4096,
        ticket_capacity=4,
        event_pool=events,
        source=WorkerEndpoint.local("denoiser", rank=1),
    )
    expected = torch.arange(48, dtype=torch.float32, device=device).reshape(6, 8)
    peer, resident = expected.chunk(2, dim=shard_axis)
    offset = (3, 0) if shard_axis == 0 else (0, 4)
    region = tuple(
        slice(start, start + extent)
        for start, extent in zip(offset, tuple(resident.shape), strict=True)
    )
    reference = TensorRef(
        RequestKey(1, 1, 1),
        ComputationId(1, 0),
        0,
        1,
        DType.F32,
        ShapeBound((DeviceDim(12), StaticDim(8))),
    )
    allocation = BufferAllocation(reference.buffer_id, 0, reference.max_bytes)
    arena = BufferPool(byte_capacity=reference.max_bytes, devices=(device,))
    store = TensorStore(
        capacity=1,
        byte_capacity=16,
        entry_capacity=1,
        max_entry_bytes=reference.max_bytes,
        devices=(device,),
        buffer_pool=arena,
        event_pool=events,
    )
    metadata = FeatureMetadata(6, 8) if feature else None
    reserve = store.reserve_features if feature else store.bind_outputs
    location = transport.publish(peer)
    # The descriptor supplies only the missing half: successfully consuming the
    # whole value therefore requires preserving the resident region.
    representation = TensorTransfer(shape=(6, 8), locations=(location,))
    imports = []
    try:
        write = reserve(
            ((reference, device),),
            buffer_allocations={reference.buffer_id: allocation},
            regions={reference: region},
            shapes={reference: (6, 8)},
        )[0]
        store.publish_write(write, resident, metadata=metadata)
        store.commit_writes((write,))
        earlier = store.consume(reference, consumer_op_id=ComputationId(2, 0), device=device)
        for _ in range(2):
            imports.append(
                store.import_tensor(
                    reference,
                    representation,
                    device=device,
                    bindings={(location.source, backend): transport},
                    metadata=metadata,
                    request_slots={},
                    buffer_allocations={reference.buffer_id: allocation},
                )
            )
        store.complete_reads((imports[0],))
        assert imports[1].imported is not None
        _consume(imports[1].imported.tickets)
        store.complete_import(imports[1])
        store.complete_reads((imports[1],))
        complete = store.consume(reference, consumer_op_id=ComputationId(3, 0), device=device)
        assert earlier.region == region and complete.region is None
        torch.testing.assert_close(earlier.tensor, resident, rtol=0, atol=0)
        torch.testing.assert_close(complete.tensor, expected, rtol=0, atol=0)

        # Full resident coverage can be consumed again without a physical edge.
        repeated = store.import_tensor(
            reference,
            representation,
            device=device,
            bindings={},
            metadata=metadata,
            request_slots={},
            buffer_allocations={reference.buffer_id: allocation},
        )
        imports.append(repeated)
        store.complete_import(repeated)
        torch.testing.assert_close(repeated.tensor, expected, rtol=0, atol=0)
        store.complete_reads((repeated,))
        store.release_requests((reference.request_key,))
        assert not store.retirement_ready(
            buffers=frozenset(), requests=frozenset((reference.request_key,))
        )
        store.complete_reads((earlier, complete))
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        events.reap()
        assert store.retirement_ready(
            buffers=frozenset(), requests=frozenset((reference.request_key,))
        )
    finally:
        for imported in imports:
            store.complete_reads((imported,))
        transport.release(location)
        transport.close()
        store.close()
        arena.close()
        events.close()


def test_full_region_publishes_complete_bounded_tensor() -> None:
    expected = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    reference = TensorRef(
        RequestKey(1, 1, 1),
        ComputationId(1, 0),
        0,
        1,
        DType.F32,
        ShapeBound((DeviceDim(12), StaticDim(8))),
    )
    allocation = BufferAllocation(reference.buffer_id, 0, reference.max_bytes)
    arena = BufferPool(byte_capacity=reference.max_bytes, devices=("cpu",))
    store = TensorStore(capacity=1, byte_capacity=16, buffer_pool=arena)
    try:
        (write,) = store.bind_outputs(
            ((reference, "cpu"),),
            buffer_allocations={reference.buffer_id: allocation},
            shapes={reference: (6, 8)},
            regions={reference: (slice(0, 6), slice(0, 8))},
        )
        store.publish_write(write, expected)
        store.commit_writes((write,))
        read = store.consume(reference, consumer_op_id=ComputationId(2, 0), device="cpu")
        assert read.region is None
        torch.testing.assert_close(read.tensor, expected, rtol=0, atol=0)
        store.complete_reads((read,))
    finally:
        store.close()
        arena.close()


def test_shm_allocation_failure_preserves_publication_capacity(monkeypatch) -> None:
    events = EventPool()
    transport = make_transport("shm", byte_capacity=16, ticket_capacity=1, event_pool=events)
    consumer = make_transport("shm", byte_capacity=16, ticket_capacity=1, event_pool=events)
    source = torch.arange(4, dtype=torch.float32)
    locator = None

    def exhausted_filesystem(*_args):
        raise OSError(errno.ENOSPC, "shared-memory filesystem is full")

    try:
        with monkeypatch.context() as filesystem:
            filesystem.setattr(os, "posix_fallocate", exhausted_filesystem)
            with pytest.raises(OSError) as failure:
                transport.publish(source)
            assert failure.value.errno == errno.ENOSPC
        locator = transport.publish(source)
        destination = torch.empty_like(source)
        ticket = consumer.fetch(locator, device=torch.device("cpu"), destination=destination)
        _consume((ticket,))
        torch.testing.assert_close(destination, source, rtol=0, atol=0)
    finally:
        if locator is not None:
            transport.release(locator)
        consumer.close()
        transport.close()
        events.close()


@pytest.mark.parametrize("backend", ("shm", "cuda_ipc"))
def test_transfer_orders_destination_writes_before_its_copy(backend: str) -> None:
    events = EventPool()
    transport = make_transport(backend, byte_capacity=4096, ticket_capacity=1, event_pool=events)
    source = torch.arange(12, dtype=torch.float32, device="cpu" if backend == "shm" else "cuda:0")
    locator = transport.publish(source)
    destination = torch.empty(12, device="cuda:0")
    initializer = torch.cuda.Stream(device=0)
    tickets = ()
    try:
        with torch.cuda.stream(initializer):
            # The destination belongs to this stream until fetch hands it to the
            # transport. Delaying its initialization exposes a missing handoff.
            torch.cuda._sleep(1_000_000_000)
            destination.fill_(-1)
            tickets = fetch_tensor(
                TensorTransfer(shape=(12,), locations=(locator,)),
                destination,
                bindings={(locator.source, backend): transport},
            )
            _consume(tickets)
        initializer.synchronize()
        torch.testing.assert_close(
            destination.cpu(), torch.arange(12, dtype=torch.float32), rtol=0, atol=0
        )
    finally:
        initializer.synchronize()
        for ticket in tickets:
            ticket.close()
        transport.release(locator)
        transport.close()
        events.close()
