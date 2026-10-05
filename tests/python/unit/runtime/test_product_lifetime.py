"""Physical product ownership from read acquisition.

Ownership runs through consumer completion.
"""

import gc
import weakref
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest
import torch

from tests.python.fixtures.cache import mha_pool
from tests.python.fixtures.transport import make_transport
from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker.errors import WorkerError, WorkerErrorCode
from uniserve_worker.execution.host import HostLane
from uniserve_worker.protocol.batch import (
    BufferAllocation,
    LatentParams,
)
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.tensor import (
    DType,
    ShapeBound,
    StaticDim,
    TensorRef,
)
from uniserve_worker.storage.block_tables import GroupTable
from uniserve_worker.storage.buffer_pool import BufferPool
from uniserve_worker.storage.latent_pool import LatentPool, LatentUpdate
from uniserve_worker.storage.output import OutputPool
from uniserve_worker.storage.tensor_store import FeatureMetadata, TensorStore

pytestmark = pytest.mark.unit


def test_abandoned_output_job_releases_capacity_and_terminates_dependent_work() -> (  # noqa: E501
    None
):
    pool = HostLane(max_inflight=2, workers=1)
    predecessor = pool.reserve().configure(
        lambda: 1, profile_name="output.predecessor"
    )
    successor = pool.reserve().configure(
        lambda: 2,
        dependencies=(predecessor,),
        profile_name="output.successor",
    )
    try:
        successor.submit_if_ready()
        predecessor.abandon()
        with pytest.raises(CancelledError):
            successor.result(timeout=5)
        assert predecessor.cancelled()
    finally:
        predecessor.abandon()
        successor.abandon()
        pool.close()
    assert pool.reserved == 0


def test_compact_persistent_buffers_remap_live_logical_allocations() -> None:
    reference = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(32),)),
    )
    second = replace(reference, producer_call_id=CallId(2, 0))
    third = replace(reference, producer_call_id=CallId(3, 0))
    buffers = BufferPool(byte_capacity=512, devices=("cpu",), compact=True)
    first_binding = buffers.bind(
        reference,
        BufferAllocation(reference.buffer_id, 4096, 512),
        device="cpu",
        dtype=torch.float32,
        shape=(32,),
    )
    second_binding = buffers.bind(
        second,
        BufferAllocation(second.buffer_id, 8192, 512),
        device="cpu",
        dtype=torch.float32,
        shape=(32,),
    )
    first_binding.tensor.fill_(1)
    second_binding.tensor.fill_(2)

    try:
        torch.testing.assert_close(
            first_binding.tensor,
            torch.full_like(first_binding.tensor, 1),
            rtol=0,
            atol=0,
        )
        with pytest.raises(WorkerError, match="exceeds the worker buffer pool"):
            buffers.bind(
                third,
                BufferAllocation(third.buffer_id, 12288, 512),
                device="cpu",
                dtype=torch.float32,
                shape=(32,),
            )
        buffers.release(first_binding)
        third_binding = buffers.bind(
            third,
            BufferAllocation(third.buffer_id, 12288, 512),
            device="cpu",
            dtype=torch.float32,
            shape=(32,),
        )
        third_binding.tensor.fill_(3)
        torch.testing.assert_close(
            second_binding.tensor,
            torch.full_like(second_binding.tensor, 2),
            rtol=0,
            atol=0,
        )
        buffers.release(third_binding)
    finally:
        buffers.release(second_binding)
        buffers.close()


def test_persistent_buffer_release_requires_issuing_pool_and_generation():
    reference = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4),)),
    )
    allocation = BufferAllocation(reference.buffer_id, 0, 16)
    pools = [BufferPool(byte_capacity=16, devices=("cpu",)) for _ in range(2)]
    try:
        bindings = [
            pool.bind(
                reference,
                allocation,
                device="cpu",
                dtype=torch.float32,
                shape=(4,),
            )
            for pool in pools
        ]
        with pytest.raises(WorkerError) as failure:
            pools[0].release(bindings[1])
        assert failure.value.code is WorkerErrorCode.INVARIANT_VIOLATION
        assert failure.value.fatal
        with pytest.raises(WorkerError, match="already bound"):
            pools[0].bind(
                reference,
                allocation,
                device="cpu",
                dtype=torch.float32,
                shape=(4,),
            )

        pools[0].release(bindings[0])
        replacement = pools[0].bind(
            reference,
            allocation,
            device="cpu",
            dtype=torch.float32,
            shape=(4,),
        )
        with pytest.raises(WorkerError, match="stale persistent buffer"):
            pools[0].release(bindings[0])
        replacement.tensor.fill_(3)
        torch.testing.assert_close(
            replacement.tensor, torch.full((4,), 3.0), rtol=0, atol=0
        )
        pools[0].release(replacement)
        pools[1].release(bindings[1])
    finally:
        for pool in pools:
            pool.close()


def test_concurrent_buffer_bindings_preserve_independent_values():
    reference = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4),)),
    )
    pool = BufferPool(byte_capacity=64, devices=("cpu",))

    def bind(index):
        output = replace(reference, output_index=index)
        binding = pool.bind(
            output,
            BufferAllocation(output.buffer_id, index * 16, 16),
            device="cpu",
            dtype=torch.float32,
            shape=(4,),
        )
        binding.tensor.fill_(index)
        return binding

    try:
        with ThreadPoolExecutor(max_workers=4) as threads:
            bindings = list(threads.map(bind, range(4)))
        for index, binding in enumerate(bindings):
            torch.testing.assert_close(
                binding.tensor,
                torch.full((4,), float(index)),
                rtol=0,
                atol=0,
            )
            pool.release(binding)
    finally:
        pool.close()


def _table(pool, units):
    """Return a slot table of whole pages of ``units`` in the one group."""
    shape = pool.shapes[0]
    return GroupTable(shape, 0, tuple(units), len(units) * shape.page_tokens)


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
@pytest.mark.parametrize("abandoned", (False, True))
def test_kv_computation_retains_pages_through_output_completion_and_reuse(
    device: str, abandoned: bool
) -> None:
    cache = mha_pool(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        total_layers=1,
        total_kv_heads=1,
        num_pages=3,
        page_size=4,
        device=device,
    )
    events = EventPool()
    outputs = OutputPool(capacity=1, max_words=8, event_pool=events)
    request = RequestKey(1, 1, 1)
    values = torch.arange(1, 5, dtype=torch.float32, device=device).reshape(
        4, 1, 1
    )
    independent = torch.full_like(values, 9)
    stream = (
        torch.cuda.Stream(device=device) if device.startswith("cuda") else None
    )
    try:
        cache.cache.state(cache.cache.groups[0].layers[0]).write(
            (1,), start=0, key=values, value=-values
        )
        cache.cache.state(cache.cache.groups[0].layers[0]).write(
            (2,), start=0, key=independent, value=independent
        )
        output = outputs.acquire(1, token_capacity=8, devices=(device,))
        completion = output.completion()
        cache.retain_execution(
            request, _table(cache, (1,)), length=3, completion=completion
        )
        with pytest.raises(WorkerError, match="executing producer or consumer"):
            cache.zero_units((1,))
        cache.require_reusable(_table(cache, (1,)).spans(3, 1))
        for actual in cache.cache.state(cache.cache.groups[0].layers[0]).read(
            (2,), start=0, length=4
        ):
            torch.testing.assert_close(actual, independent, rtol=0, atol=0)
        assert not cache.retirement_ready(requests=(request,))
        # Releasing another product of this request does not retire its KV
        # computation. The execution fence still protects page reuse above.
        assert cache.retirement_ready(
            buffers=(BufferId(request, CallId(2, 0), 0, 2),)
        )

        source = cache.cache.state(
            cache.cache.groups[0].layers[0]
        ).transfer_blocks((1,), start=0, length=3)["key.values"][0]
        if stream is not None:
            # First-use reduction initialization can finish prior device work
            # while the host is still submitting it. Initialize before the
            # delay.
            source.sum().to(dtype=torch.long)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1_000_000_000)
                capture = output.capture(source.sum().reshape(1))
                output.seal()
            assert not output.ready(), (
                "consumer completed before its pending fence was checked"
            )
            with pytest.raises(
                WorkerError, match="executing producer or consumer"
            ):
                cache.zero_units((1,))
        else:
            capture = output.capture(source.sum().reshape(1))
            output.seal()
        if abandoned:
            output.abandon()
        if stream is not None:
            stream.synchronize()
        events.reap()
        # Device completion must make pages reusable even when the host has
        # not read or abandoned the call's output report.
        assert cache.retirement_ready(requests=(request,))
        if not abandoned:
            assert output.ready()
            assert output.read_tokens(*capture) == (6,)
            output.abandon()
        assert completion.done()
        assert cache.retirement_ready(requests=(request,))
        cache.zero_units((1,))
        for actual in cache.cache.state(cache.cache.groups[0].layers[0]).read(
            (1,), start=0, length=4
        ):
            assert torch.count_nonzero(actual).item() == 0

        # A subsequent lease over the same bounded output storage must retain
        # its own computation even though the preceding future is complete.
        next_output = outputs.acquire(1, token_capacity=8, devices=(device,))
        next_completion = next_output.completion()
        cache.retain_execution(
            request, _table(cache, (1,)), length=4, completion=next_completion
        )
        assert completion.done()
        assert not next_completion.done()
        with pytest.raises(WorkerError, match="executing producer or consumer"):
            cache.zero_units((1,))
        outputs.close()
        assert next_completion.done()
        cache.zero_units((1,))
    finally:
        outputs.close()
        cache.close()
        events.close()


def test_published_kv_prefix_allows_append_and_waits_for_every_reader_before_reuse(  # noqa: E501
) -> None:
    events = EventPool()
    outputs = OutputPool(capacity=1, max_words=8, event_pool=events)
    transport = make_transport(
        "local", byte_capacity=4096, ticket_capacity=2, event_pool=events
    )
    pool = mha_pool(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        total_layers=1,
        total_kv_heads=1,
        num_pages=5,
        page_size=4,
        device="cpu",
    )
    buffer = BufferId(
        owner=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
    )
    pages = (3, 1, 4)
    prefix = torch.arange(12, dtype=torch.float32).view(3, 1, 4)
    locations = []
    readers = []
    try:
        pool.cache.state(pool.cache.groups[0].layers[0]).write(
            pages, start=0, key=prefix, value=-prefix
        )
        source = pool.reserve_export(buffer, _table(pool, pages).spans(0, 3))
        for tensor in pool.cache.state(pool.cache.groups[0].layers[0]).read(
            pages, start=0, length=3
        ):
            assert tensor is not None
            location = transport.export(tensor)
            locations.append(location)
            pool.retain_export(source, transport.retirement(location))
            readers.append(
                transport.fetch(location, device=torch.device("cpu"))
            )

        # Appending touches the remaining token of the same page and the next
        # page. Neither call changes the retained prefix's logical value.
        suffix = torch.full((5, 1, 4), 7.0)
        pool.require_writable(
            _table(pool, pages), start=3, length=suffix.shape[0]
        )
        pool.cache.state(pool.cache.groups[0].layers[0]).write(
            pages, start=3, key=suffix, value=-suffix
        )
        torch.testing.assert_close(readers[0].result(), prefix, rtol=0, atol=0)
        torch.testing.assert_close(readers[1].result(), -prefix, rtol=0, atol=0)
        pool.zero_units((2,))
        with pytest.raises(WorkerError, match="published version"):
            pool.require_writable(_table(pool, pages), start=2, length=1)

        pool.release_buffers((buffer,))
        for location in locations:
            transport.release(location)
        readers[0].close()
        assert not pool.retirement_ready(buffers=(buffer,))
        with pytest.raises(WorkerError, match="published version"):
            pool.zero_units((3,))
        torch.testing.assert_close(readers[1].result(), -prefix, rtol=0, atol=0)
        dependencies = pool.write_dependencies(_table(pool, pages).spans(0, 3))
        output = outputs.acquire(1, token_capacity=8, devices=("cpu",))
        pool.retain_execution(
            buffer.owner,
            _table(pool, pages),
            length=3,
            completion=output.completion(),
        )
        readers[1].close()
        for future in dependencies:
            future.result(timeout=5)
        assert pool.retirement_ready(buffers=(buffer,))
        assert not pool.retirement_ready(requests=(buffer.owner,))
        with pytest.raises(WorkerError, match="executing producer or consumer"):
            pool.zero_units((3,))
        output.seal()
        output.abandon()
        pool.zero_units((3,))
        keys, values = pool.cache.state(pool.cache.groups[0].layers[0]).read(
            pages, start=0, length=3
        )
        torch.testing.assert_close(
            keys, torch.zeros_like(prefix), rtol=0, atol=0
        )
        torch.testing.assert_close(
            values, torch.zeros_like(prefix), rtol=0, atol=0
        )
    finally:
        for reader in readers:
            reader.close()
        for location in locations:
            transport.release(location)
        transport.close()
        outputs.close()
        pool.close()
        events.close()


@pytest.mark.parametrize("storage", ("encoder", "tensor"))
def test_free_retains_an_acquired_consumer_until_it_records_completion(
    storage: str,
) -> None:
    events = EventPool()
    buffers = BufferPool(byte_capacity=16, devices=("cpu",))
    store = (
        TensorStore(
            max_feature_bytes=16,
            devices=("cpu",),
            buffer_pool=buffers,
            event_pool=events,
        )
        if storage == "encoder"
        else TensorStore(
            capacity=1, byte_capacity=1, buffer_pool=buffers, event_pool=events
        )
    )
    product = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4),)),
    )
    replacement = replace(product, generation=2)
    value = torch.arange(4, dtype=torch.float32)
    try:
        reserve = (
            store.reserve_features
            if storage == "encoder"
            else store.bind_outputs
        )
        write = reserve(
            ((product, "cpu"),),
            buffer_allocations={
                product.buffer_id: BufferAllocation(product.buffer_id, 0, 16)
            },
        )[0]
        store.write(
            write,
            value,
            metadata=FeatureMetadata(height=1, width=4)
            if storage == "encoder"
            else None,
        )
        store.commit_writes((write,))
        store.release_requests(
            (product.request_key,), retained=frozenset((product.buffer_id,))
        )
        read = store.consume(product, consumer_call_id=CallId(2, 0))
        store.release_buffers((product.buffer_id,))
        with pytest.raises(WorkerError):
            reserve(
                ((replacement, "cpu"),),
                buffer_allocations={
                    replacement.buffer_id: BufferAllocation(
                        replacement.buffer_id, 0, 16
                    )
                },
            )
        torch.testing.assert_close(read.tensor, value, rtol=0, atol=0)
        store.complete_reads((read,))
        # Completion recording is idempotent, including the ordinary lane
        # cleanup that follows a read already fenced by its output producer.
        store.complete_reads((read,))
        reused = reserve(
            ((replacement, "cpu"),),
            buffer_allocations={
                replacement.buffer_id: BufferAllocation(
                    replacement.buffer_id, 0, 16
                )
            },
        )
        store.abandon_writes(reused)
    finally:
        store.close()
        buffers.close()
        events.close()


def test_shared_tensor_reads_survive_collection_and_release_cycles() -> None:
    buffers = BufferPool(byte_capacity=16, devices=("cpu",))
    store = TensorStore(capacity=1, buffer_pool=buffers)
    reference = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4),)),
    )
    value = torch.arange(4, dtype=torch.float32)

    try:
        writes = store.bind_outputs(
            ((reference, "cpu"),),
            buffer_allocations={
                reference.buffer_id: BufferAllocation(
                    reference.buffer_id, 0, 16
                )
            },
        )
        store.write(writes[0], value)
        store.commit_writes(writes)
        reads = tuple(
            store.consume(reference, consumer_call_id=CallId(batch, 0))
            for batch in (2, 3)
        )
        observed = tuple(weakref.ref(read.tensor) for read in reads)
        for read in reads:
            read.tensor.consumer = read
        del read

        store.release_buffers((reference.buffer_id,))
        del writes
        gc.collect()
        for read in reads:
            torch.testing.assert_close(read.tensor, value, rtol=0, atol=0)
        del read

        store.complete_reads(reads)
        del reads
        gc.collect()
        assert all(view() is None for view in observed)
    finally:
        store.close()
        buffers.close()
        store.event_pool.close()


def test_a_retired_segment_no_consumer_began_reading_returns_at_once() -> None:
    """A export released before its consumer read retires unread.

    The engine releases a buffer only once every consuming call has resolved
    or will never be submitted, as for a request cancelled before its
    encoder ran, so a named consumer that never claimed the segment is not
    waited for.
    """
    events = EventPool()
    transport = make_transport(
        "shm", byte_capacity=4096, ticket_capacity=2, event_pool=events
    )
    try:
        value = torch.arange(8, dtype=torch.float32)
        locator = transport.export(value, consumers=(1, 2))
        retirement = transport.retirement(locator)
        transport.release(locator)
        transport.reap()
        retirement.result(timeout=5)
        assert not transport.awaiting_acknowledgment()
    finally:
        transport.close()
        events.close()


def test_a_deferred_write_commits_when_its_host_work_publishes_it() -> None:
    """A write host work fills later packs without it and commits with it."""
    buffers = BufferPool(byte_capacity=16, devices=("cpu",))
    store = TensorStore(
        capacity=2,
        request_capacity=0,
        relay_depth=0,
        buffer_pool=buffers,
    )
    rows = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(1),)),
    )
    other = replace(rows, output_index=1)
    consumer = CallId(2, 0)
    try:
        writes = store.bind_outputs(
            ((rows, "cpu"), (other, "cpu")),
            buffer_allocations={
                reference.buffer_id: BufferAllocation(
                    reference.buffer_id, index * 4, 4
                )
                for index, reference in enumerate((rows, other))
            },
        )
        deferred, immediate = writes
        store.defer_write(deferred)
        store.write(immediate, torch.tensor([7.0]))
        # The call's completion packs and commits with the immediate write
        # only; the deferred one is neither published nor exposed yet.
        store.validate_writes(writes)
        store.commit_writes(writes)
        read = store.consume(other, consumer_call_id=consumer)
        torch.testing.assert_close(
            read.tensor, torch.tensor([7.0]), rtol=0, atol=0
        )
        store.complete_reads((read,))
        with pytest.raises(WorkerError):
            store.consume(rows, consumer_call_id=consumer)
        # Host work publishes and commits it later; then it reads like any
        # other product.
        store.write(deferred, torch.tensor([3.0]))
        store.commit_writes((deferred,))
        read = store.consume(rows, consumer_call_id=consumer)
        torch.testing.assert_close(
            read.tensor, torch.tensor([3.0]), rtol=0, atol=0
        )
        store.complete_reads((read,))
        with pytest.raises(WorkerError):
            store.defer_write(deferred)
    finally:
        store.close()
        buffers.close()


def test_resident_bytes_cover_only_storage_the_store_allocates() -> None:
    """Resident bytes count the store's relay arenas, not borrowed storage.

    They are compared with the store's own device-product byte bound, so a
    persistent product, whose storage the `BufferPool` owns and the worker
    layout counts, adds nothing, while the first relay binding adds its
    whole arena.
    """
    buffers = BufferPool(byte_capacity=64, devices=("cpu",))
    request_slots, relay_depth = 1, 2
    store = TensorStore(
        capacity=1,
        byte_capacity=32,
        request_capacity=request_slots,
        relay_depth=relay_depth,
        buffer_pool=buffers,
    )
    persistent = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4),)),
    )
    relay = replace(
        persistent,
        producer_call_id=CallId(2, 0),
        shape_bound=ShapeBound((StaticDim(1),)),
    )
    try:
        store.bind_outputs(
            ((persistent, "cpu"),),
            buffer_allocations={
                persistent.buffer_id: BufferAllocation(
                    persistent.buffer_id, 0, 16
                )
            },
        )
        assert store.resident_bytes("cpu") == 0

        store.bind_outputs(
            ((relay, "cpu"),), request_slots={relay.request_key: 1}
        )
        # One F32 element per (request slot, lane), with row zero unused.
        assert store.resident_bytes("cpu") == (request_slots + 1) * (
            relay_depth * 4
        )
    finally:
        store.close()
        buffers.close()
        store.event_pool.close()


@pytest.mark.parametrize("relay", (False, True))
def test_tensor_export_is_atomic_and_preserves_generation_ownership(
    relay: bool,
) -> None:
    buffers = BufferPool(byte_capacity=16, devices=("cpu",))
    store = TensorStore(
        capacity=2,
        byte_capacity=32 if relay else 16,
        request_capacity=1 if relay else 0,
        relay_depth=2 if relay else 0,
        buffer_pool=buffers,
    )
    first = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(1),)),
    )
    second = replace(first, output_index=1)
    consumer = CallId(2, 0)

    def reserve(references):
        return store.bind_outputs(
            tuple((reference, "cpu") for reference in references),
            request_slots={first.request_key: 1} if relay else None,
            buffer_allocations=None
            if relay
            else {
                reference.buffer_id: BufferAllocation(
                    reference.buffer_id, index * 4, 4
                )
                for index, reference in enumerate(references)
            },
        )

    try:
        with pytest.raises(WorkerError):
            reserve((first, first))
        writes = reserve((first, second))
        store.write(writes[0], torch.tensor([3.0]))
        with pytest.raises(WorkerError):
            reserve((first,))
        with pytest.raises(WorkerError):
            store.commit_writes(writes)
        # A batch containing an unfinished producer exposes neither product.
        for reference in (first, second):
            with pytest.raises(WorkerError):
                store.consume(reference, consumer_call_id=consumer)
        store.write(writes[1], torch.tensor([7.0]))
        store.commit_writes(writes)
        with pytest.raises(WorkerError):
            store.consume_batch(
                (
                    (first, consumer, None),
                    (replace(second, generation=2), consumer, None),
                )
            )
        for reference, expected in ((first, 3.0), (second, 7.0)):
            with pytest.raises(WorkerError):
                store.consume(
                    replace(reference, generation=2), consumer_call_id=consumer
                )
            read = store.consume(reference, consumer_call_id=consumer)
            torch.testing.assert_close(
                read.tensor, torch.tensor([expected]), rtol=0, atol=0
            )
            store.complete_reads((read, read))

        store.release_buffers((first.buffer_id, second.buffer_id))
        replacement = replace(first, generation=2)
        (write,) = reserve((replacement,))
        with pytest.raises(WorkerError):
            store.producer_write_views((writes[0],))
        store.write(write, torch.tensor([11.0]))
        store.commit_writes((write,))
        with pytest.raises(WorkerError):
            store.consume(first, consumer_call_id=consumer)
        read = store.consume(replacement, consumer_call_id=consumer)
        torch.testing.assert_close(
            read.tensor, torch.tensor([11.0]), rtol=0, atol=0
        )
        store.complete_reads((read,))
    finally:
        store.close()
        buffers.close()
        store.event_pool.close()


def test_tensor_export_enforces_its_logical_region_and_representation() -> (  # noqa: E501
    None
):
    buffers = BufferPool(byte_capacity=24, devices=("cpu",))
    events = EventPool()
    store = TensorStore(
        capacity=1, byte_capacity=16, buffer_pool=buffers, event_pool=events
    )
    reference = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4), StaticDim(4))),
    )
    allocation = {
        reference.buffer_id: BufferAllocation(reference.buffer_id, 0, 24)
    }
    region = (slice(2, 4), slice(1, 4))
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    try:
        with pytest.raises(WorkerError, match="logical bounds"):
            store.bind_outputs(
                ((reference, "cpu"),),
                buffer_allocations=allocation,
                regions={reference: (slice(3, 5), slice(1, 4))},
            )
        write = store.bind_outputs(
            ((reference, "cpu"),),
            buffer_allocations=allocation,
            regions={reference: region},
        )[0]
        with pytest.raises(WorkerError, match="dtype"):
            store.write(write, expected.to(torch.float16))
        with pytest.raises(WorkerError, match="shape"):
            store.write(write, expected.reshape(3, 2))
        store.write(write, expected)
        store.commit_writes((write,))
        read = store.consume(reference, consumer_call_id=CallId(2, 0))
        assert read.region == region
        torch.testing.assert_close(read.tensor, expected, rtol=0, atol=0)
        store.complete_reads((read,))
    finally:
        store.close()
        buffers.close()
        events.close()


def test_latent_import_preserves_page_order_and_committed_metadata() -> None:
    events = EventPool()
    transport = make_transport(
        "local", byte_capacity=4096, ticket_capacity=1, event_pool=events
    )
    pool = LatentPool(
        request_pool_size=2,
        num_pages=5,
        page_units=4,
        latent_width=4,
        dtype=torch.float32,
        device="cpu",
    )
    product = TensorRef(
        request_key=RequestKey(1, 1, 1),
        producer_call_id=CallId(2, 0),
        output_index=0,
        generation=3,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(11), StaticDim(4))),
    )
    expected = torch.arange(44, dtype=torch.float32).reshape(11, 4)
    locator = transport.export(expected)
    try:
        write = pool.reserve_import(
            product, request_pool_idx=1, page_table=(3, 1, 4), latent_units=11
        )
        staging = pool.stage(((3, 1, 4),), (11,))[0]
        with pytest.raises(WorkerError, match="committed trajectory"):
            pool.gather_current(
                1,
                staging,
                generation=3,
                step=2,
                latent_units=11,
                height=16,
                width=176,
            )
        ticket = transport.fetch(
            locator, device=torch.device("cpu"), destination=write.spans
        )
        pool.retain_transfer(write, ticket)
        ready = Event()
        ticket.add_done_callback(ready.set)
        assert ready.wait(5)
        pool.adopt_import(write, generation=3, step=2, height=16, width=176)
        actual = pool.gather_current(
            1,
            staging,
            generation=3,
            step=2,
            latent_units=11,
            height=16,
            width=176,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with pytest.raises(WorkerError, match="page table does not match"):
            pool.gather_current(
                1,
                pool.stage(((1, 3, 4),), (11,))[0],
                generation=3,
                step=2,
                latent_units=11,
                height=16,
                width=176,
            )
        retired = Event()
        ticket.add_retirement_callback(retired.set)
        assert retired.wait(5)
        pool.release_slots((1,))
        reused = pool.reserve_import(
            replace(product, request_key=RequestKey(2, 1, 1)),
            request_pool_idx=2,
            page_table=(4, 3, 1),
            latent_units=11,
        )
        pool.abandon_import(reused)
    finally:
        transport.release(locator)
        transport.close()
        pool.close()
        events.close()


@pytest.fixture
def latent_output():
    """Prepare public latent visibility updates without execution state."""

    def create(slot, pages, units, height, width):
        key = RequestKey(1, slot, 1)
        return LatentUpdate(
            request_pool_idx=slot,
            params=LatentParams(
                key, CallId(1, 0), pages, units, height, width, 0, 0
            ),
            generation=1,
        )

    return create


@pytest.mark.parametrize("committed", (False, True))
def test_published_latent_bank_waits_for_every_reader_before_reuse(
    committed: bool, latent_output
) -> None:

    events = EventPool()
    transport = make_transport(
        "local", byte_capacity=4096, ticket_capacity=4, event_pool=events
    )
    pool = LatentPool(
        request_pool_size=2,
        num_pages=5,
        page_units=4,
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
        shape_bound=ShapeBound((StaticDim(11), StaticDim(4))),
    )
    pages = (3, 1, 4)
    staging = pool.stage((pages,), (11,))[0]
    commit = latent_output(1, pages, 11, 16, 176)
    locations = []
    readers = []
    try:
        staging.value.fill_(1)
        pool.initialize(1, staging, latent_units=11)
        if committed:
            pool.validate_updates((commit,))
            pool.apply_updates((commit,))
            source = pool.reserve_current_export(
                product,
                request_pool_idx=1,
                page_table=pages,
                generation=1,
                step=0,
                latent_units=11,
                height=16,
                width=176,
            )
        else:
            source = pool.reserve_export(
                product, request_pool_idx=1, page_table=pages, latent_units=11
            )
        offset = 0
        for span in source.spans:
            location = transport.export(span, offset=(offset, 0))
            locations.append(location)
            pool.retain_export(source, transport.retirement(location))
            readers.append(
                transport.fetch(location, device=torch.device("cpu"))
            )
            offset += span.shape[0]
        borrowed = tuple(reader.result() for reader in readers)
        if not committed:
            pool.validate_updates((commit,))
            pool.apply_updates((commit,))

        # The next step uses the other bank and can proceed during fan-out.
        staging.value.fill_(2)
        pool.write_inactive(
            1,
            staging,
            expected_step=0,
            expected_generation=1,
            latent_units=11,
            height=16,
            width=176,
        )
        second = latent_output(1, pages, 11, 16, 176)
        second.expected_generation = 1
        second.generation = 2
        second.step = 1
        pool.validate_updates((second,))
        pool.apply_updates((second,))
        for value in borrowed:
            torch.testing.assert_close(
                value, torch.ones_like(value), rtol=0, atol=0
            )
        torch.testing.assert_close(
            pool.gather_current(
                1,
                staging,
                generation=2,
                step=1,
                latent_units=11,
                height=16,
                width=176,
            ),
            torch.full((11, 4), 2.0),
            rtol=0,
            atol=0,
        )

        pool.release_buffers((product.buffer_id,))
        for location in locations:
            transport.release(location)
        for reader in readers[:-1]:
            reader.close()
        dependencies = pool.write_dependencies(1, pages)
        assert any(not dependency.done() for dependency in dependencies)
        staging.value.fill_(3)
        with pytest.raises(WorkerError, match="published version"):
            pool.write_inactive(
                1,
                staging,
                expected_step=1,
                expected_generation=2,
                latent_units=11,
                height=16,
                width=176,
            )
        torch.testing.assert_close(
            borrowed[-1], torch.ones_like(borrowed[-1]), rtol=0, atol=0
        )
        readers[-1].close()
        for dependency in dependencies:
            dependency.result(timeout=5)
        pool.write_inactive(
            1,
            staging,
            expected_step=1,
            expected_generation=2,
            latent_units=11,
            height=16,
            width=176,
        )
        third = latent_output(1, pages, 11, 16, 176)
        third.expected_generation = 2
        third.expected_step = 1
        third.generation = 3
        third.step = 2
        pool.validate_updates((third,))
        pool.apply_updates((third,))
        torch.testing.assert_close(
            pool.gather_current(
                1,
                staging,
                generation=3,
                step=2,
                latent_units=11,
                height=16,
                width=176,
            ),
            torch.full((11, 4), 3.0),
            rtol=0,
            atol=0,
        )
    finally:
        for reader in readers:
            reader.close()
        for location in locations:
            transport.release(location)
        transport.close()
        pool.close()
        events.close()


def test_unacknowledged_latent_export_retains_its_pages_without_poisoning_other_requests(  # noqa: E501
    latent_output,
) -> None:
    """A retired export holds its pages while its consumer reads.

    The consumer named on the export has claimed its word and not yet
    acknowledged, so the pages stay owned and the request is not
    retirement-ready, while an independent request proceeds. Once the word
    lands, the next sweep returns the pages.
    """
    from contextlib import suppress

    from uniserve_worker.transport import segment
    from uniserve_worker.transport.shared_storage import open_shared_storage

    events = EventPool()
    transport = make_transport(
        "shm", byte_capacity=4096, ticket_capacity=2, event_pool=events
    )
    pool = LatentPool(
        request_pool_size=2,
        num_pages=5,
        page_units=4,
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
        shape_bound=ShapeBound((StaticDim(4), StaticDim(4))),
    )
    staging = pool.stage(((1,),), (4,))[0]
    staging.value.fill_(1)
    pool.initialize(1, staging, latent_units=4)
    source = pool.reserve_export(
        product, request_pool_idx=1, page_table=(1,), latent_units=4
    )
    locator = transport.export(source.spans[0], consumers=(1,))
    retirement = transport.retirement(locator)
    pool.retain_export(source, retirement)
    commit = latent_output(1, (1,), 4, 16, 64)
    pool.validate_updates((commit,))
    pool.apply_updates((commit,))
    # The consumer begins reading before the engine retires the export.
    storage = open_shared_storage(
        locator.transport.name, segment.HEADER_BYTES + locator.nbytes
    )
    try:
        header = memoryview(storage)
        segment.claim(header, 1)
        header.release()
    finally:
        storage.close()
    try:
        transport.release(locator)
        pool.release_buffers((product.buffer_id,))
        transport.reap()
        assert not retirement.done(), (
            "export retired before its consumer acknowledged"
        )
        assert not pool.retirement_ready((product.request_key,))
        pool.release_slots((1,))
        with pytest.raises(WorkerError, match="owned"):
            pool.reserve_import(
                replace(product, request_key=RequestKey(1, 2, 1)),
                request_pool_idx=2,
                page_table=(1,),
                latent_units=4,
            )

        independent = pool.stage(((2,),), (4,))[0]
        independent.value.fill_(7)
        pool.initialize(2, independent, latent_units=4)
        next_commit = latent_output(2, (2,), 4, 16, 64)
        pool.validate_updates((next_commit,))
        pool.apply_updates((next_commit,))
        torch.testing.assert_close(
            pool.gather_current(
                2,
                independent,
                generation=1,
                step=0,
                latent_units=4,
                height=16,
                width=64,
            ),
            torch.full((4, 4), 7.0),
            rtol=0,
            atol=0,
        )
        assert pool.retirement_ready((RequestKey(1, 2, 1),))

        # The consumer writes its word; the producer's next sweep returns the
        # segment and with it the pages.
        storage = open_shared_storage(
            locator.transport.name, segment.HEADER_BYTES + locator.nbytes
        )
        try:
            header = memoryview(storage)
            segment.acknowledge(header, 1)
            header.release()
        finally:
            storage.close()
        transport.reap()
        retirement.result(timeout=5)
        assert pool.retirement_ready((product.request_key,))
    finally:
        with suppress(WorkerError):
            transport.close()
        with suppress(WorkerError):
            pool.close()
        events.close()


@pytest.mark.parametrize("ending", ("failed", "cancelled"))
def test_unknown_latent_reader_completion_retains_only_its_pages(
    ending, latent_output
) -> None:
    pool = LatentPool(
        request_pool_size=2,
        num_pages=3,
        page_units=4,
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
        shape_bound=ShapeBound((StaticDim(4), StaticDim(4))),
    )
    source = pool.reserve_export(
        product, request_pool_idx=1, page_table=(1,), latent_units=4
    )
    retirement: Completion = Completion()
    pool.retain_export(source, retirement)
    if ending == "failed":
        retirement.set_exception(RuntimeError("reader completion unknown"))
        error = RuntimeError
    else:
        retirement.cancel()
        error = CancelledError

    pool.release_buffers((product.buffer_id,))
    pool.release_slots((1,))
    with pytest.raises(WorkerError) as held:
        pool.initial_bank(2, (1,), latent_units=4)
    assert held.value.code is WorkerErrorCode.INVALID_DESCRIPTOR
    with pytest.raises(error):
        pool.retirement_ready((product.request_key,))

    # An independent trajectory can still commit and expose its values.
    staging = pool.stage(((2,),), (4,))[0]
    staging.value.fill_(7)
    pool.initialize(2, staging, latent_units=4)
    update = latent_output(2, (2,), 4, 16, 64)
    pool.validate_updates((update,))
    pool.apply_updates((update,))
    actual = pool.gather_current(
        2, staging, generation=1, step=0, latent_units=4, height=16, width=64
    )
    torch.testing.assert_close(actual, torch.full((4, 4), 7.0), rtol=0, atol=0)
    assert pool.retirement_ready((update.params.request_key,))
    with pytest.raises(WorkerError) as closing:
        pool.close()
    assert closing.value.code is WorkerErrorCode.RESOURCE_ERROR


def test_latent_staging_preserves_live_trajectories(latent_output) -> None:
    pool = LatentPool(
        request_pool_size=2,
        num_pages=5,
        page_units=4,
        latent_width=4,
        dtype=torch.float32,
        device="cpu",
    )
    try:
        first = pool.stage(((3, 1),), (7,))[0]
        first.value.fill_(7)
        second = pool.stage(((4, 2),), (7,), occupied=(first,))[0]
        second.value.fill_(9)
        with pytest.raises(WorkerError, match="overlap"):
            pool.stage(((1,),), (4,), occupied=(first, second))

        # Both consumers run after both inputs were staged. Their original
        # values and page order must survive staging an independent call.
        pool.initialize(1, first, latent_units=7)
        pool.initialize(2, second, latent_units=7)
        updates = (
            latent_output(1, (3, 1), 7, 16, 112),
            latent_output(2, (4, 2), 7, 16, 112),
        )
        pool.validate_updates(updates)
        pool.apply_updates(updates)
        for slot, staging, expected in ((1, first, 7), (2, second, 9)):
            actual = pool.gather_current(
                slot,
                staging,
                step=0,
                generation=1,
                latent_units=7,
                height=16,
                width=112,
            )
            torch.testing.assert_close(
                actual,
                torch.full((7, 4), expected),
                rtol=0,
                atol=0,
                check_dtype=False,
            )
    finally:
        pool.close()


def test_fp8_export_preserves_values_before_a_later_block_scale_growth():
    pool = mha_pool(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        store_dtype=torch.float8_e4m3fn,
        total_layers=1,
        total_kv_heads=1,
        num_pages=2,
        page_size=4,
        device="cpu",
        table_width=1,
    )
    events = EventPool()
    transport = make_transport(
        "local", byte_capacity=4096, ticket_capacity=4, event_pool=events
    )
    source = BufferId(RequestKey(1, 1, 1), CallId(1, 0), 0, 1)
    state = pool.cache.state(pool.cache.groups[0].layers[0])
    prefix = torch.tensor([[[1.0, 0.111]]])
    scale = torch.tensor(1.0) / 448
    expected = (
        (prefix / scale).to(torch.float8_e4m3fn).float() * scale
    ).unsqueeze(1)
    locations, readers = [], []
    try:
        pool.block_tables.install(((1, 0, 0, (1,), 4),))
        state.write((1,), start=0, key=prefix, value=-prefix)
        export = pool.export(
            request_pool_idx=1,
            visible_length=1,
            destination="consumer",
            buffer=source,
            transports={"local": transport},
        )
        locations.extend(
            location for field in export.tensors for location in field.locations
        )
        pool.validate_exports(((source, export),), ())
        pool.apply_exports(((source, export),), ())
        # The import can begin after another invocation appends to the same
        # physical block. Its BufferId still denotes the earlier exact value.
        suffix = torch.full_like(prefix, 896)
        pool.require_writable(_table(pool, (1,)), start=1, length=1)
        state.write((1,), start=1, key=suffix, value=-suffix)
        for index, field in enumerate(export.tensors[:2]):
            reader = transport.fetch(
                field.locations[0], device=torch.device("cpu")
            )
            readers.append(reader)
            scale_reader = transport.fetch(
                export.tensors[2].locations[index],
                device=torch.device("cpu"),
            )
            readers.append(scale_reader)
            values = torch.cat(reader.result()).float() * torch.cat(
                scale_reader.result()
            ).reshape(1, 1, 1, 1)
            torch.testing.assert_close(
                values, expected if index == 0 else -expected, rtol=0, atol=0
            )
    finally:
        for reader in readers:
            reader.close()
        for location in locations:
            transport.release(location)
        pool.release_buffers((source,))
        transport.close()
        pool.close()
        events.close()


def test_a_host_product_is_published_where_its_consumers_are() -> None:
    """Each host mechanism carries a product only to consumers it reaches."""
    from uniserve_worker.transport import make_transports, segment
    from uniserve_worker.transport.exports import export_tensor
    from uniserve_worker.transport.shared_storage import open_shared_storage

    events = EventPool()
    # Slots 0 and 1 share this host; slot 2 is on another host.
    transports = make_transports(
        ("local", "shm", "channel"),
        byte_capacity=1 << 20,
        ticket_capacity=8,
        event_pool=events,
        acknowledgment_slot=0,
        host_slots=(0, 1),
    )
    value = torch.arange(16, dtype=torch.int16).reshape(8, 2)
    try:
        cases = {
            (1,): {"local", "shm"},
            (2,): {"local", "channel"},
            (1, 2): {"local", "shm", "channel"},
            (): {"local", "shm", "channel"},
        }
        for consumers, expected in cases.items():
            locations = export_tensor(
                transports,
                value,
                retain=lambda future: None,
                consumers=consumers,
                host=True,
            )
            try:
                assert {
                    location.backend for location in locations
                } == expected, consumers
            finally:
                # A segment retires once its named consumers have written
                # their words; the channel and local copies retire with the
                # release alone.
                for location in locations:
                    if location.backend == "shm":
                        mapping = open_shared_storage(
                            location.transport.name,
                            segment.HEADER_BYTES + location.nbytes,
                        )
                        try:
                            for slot in consumers:
                                segment.acknowledge(memoryview(mapping), slot)
                        finally:
                            mapping.close()
                    transports[location.backend].release(location)
                transports["shm"].reap()
    finally:
        for transport in transports.values():
            transport.close()
