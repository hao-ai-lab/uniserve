"""Physical product ownership from read acquisition through consumer completion."""

from dataclasses import replace
from threading import Event

import pytest
import torch

from uniserve_worker.execution.batch import (
    BufferAllocation,
    DType,
    PointRange,
    ProductKind,
    ProductRef,
    RequestKey,
    ShapeBound,
    StaticDim,
    StorageClass,
)
from uniserve_worker.execution.output import OutputPool
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.cache_pool import CachePool
from uniserve_worker.runtime.device_events import DeviceEventPool
from uniserve_worker.runtime.device_products import DeviceProducts
from uniserve_worker.runtime.encoder_cache import EncoderCache, EncoderMetadata
from uniserve_worker.runtime.latent_pool import LatentPool
from uniserve_worker.runtime.persistent_buffers import PersistentBuffers
from uniserve_worker.transfer.layout import TensorRegion
from uniserve_worker.transfer.tickets import make_transport


def test_compact_persistent_buffers_remap_live_logical_allocations() -> None:
    reference = ProductRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=1,
        output_index=0,
        generation=1,
        kind=ProductKind.TENSOR,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(32),)),
        point_range=PointRange(),
    )
    second = replace(reference, producer_op_id=2)
    third = replace(reference, producer_op_id=3)
    buffers = PersistentBuffers(byte_capacity=512, devices=("cpu",), compact=True)
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


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)))
@pytest.mark.parametrize("abandoned", (False, True))
def test_kv_computation_retains_pages_through_output_completion_and_reuse(
    device: str, abandoned: bool
) -> None:
    cache = CachePool(
        num_layers=1,
        num_pages=3,
        page_size=4,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        device=device,
    )
    events = DeviceEventPool()
    outputs = OutputPool(capacity=1, max_words=8, event_pool=events)
    request = RequestKey(1, 1, 1)
    values = torch.arange(1, 5, dtype=torch.float32, device=device).reshape(4, 1, 1)
    independent = torch.full_like(values, 9)
    stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None
    try:
        cache.write(0, (1,), start=0, k=values, v=-values)
        cache.write(0, (2,), start=0, k=independent, v=independent)
        output = outputs.acquire(1, token_capacity=8, devices=(device,))
        completion = output.completion_future()
        cache.retain_execution(request, (1,), group=0, length=3, completion=completion)
        with pytest.raises(WorkerError, match="executing producer or consumer"):
            cache.zero_pages(0, (1,))
        cache.require_reusable((1,), group=0, start=3, length=1)
        for actual in cache.read(0, (2,), start=0, length=4):
            torch.testing.assert_close(actual, independent, rtol=0, atol=0)
        assert not cache.retirement_ready(requests=(request,))

        source = cache.transfer_views((1,), group=0, start=0, length=3)[0][0]
        if stream is not None:
            # First-use reduction initialization can finish prior device work
            # while the host is still submitting it. Initialize before the delay.
            source.sum().to(dtype=torch.long)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1_000_000_000)
                capture = output.capture(source.sum().reshape(1))
                output.seal()
            assert not output.ready(), "consumer completed before its pending fence was checked"
            with pytest.raises(WorkerError, match="executing producer or consumer"):
                cache.zero_pages(0, (1,))
        else:
            capture = output.capture(source.sum().reshape(1))
            output.seal()
        if abandoned:
            output.abandon()
        if stream is not None:
            stream.synchronize()
        events.reap()
        # Device completion must make pages reusable even when the host has
        # not read or abandoned the operation's output report.
        assert cache.retirement_ready(requests=(request,))
        if not abandoned:
            assert output.ready()
            assert output.read_tokens(capture) == (6,)
            output.abandon()
        assert completion.done()
        assert cache.retirement_ready(requests=(request,))
        cache.zero_pages(0, (1,))
        for actual in cache.read(0, (1,), start=0, length=4):
            assert torch.count_nonzero(actual).item() == 0

        # A subsequent lease over the same bounded output storage must retain
        # its own computation even though the preceding future is complete.
        next_output = outputs.acquire(1, token_capacity=8, devices=(device,))
        next_completion = next_output.completion_future()
        cache.retain_execution(request, (1,), group=0, length=4, completion=next_completion)
        assert completion.done()
        assert not next_completion.done()
        with pytest.raises(WorkerError, match="executing producer or consumer"):
            cache.zero_pages(0, (1,))
        outputs.close()
        assert next_completion.done()
        cache.zero_pages(0, (1,))
    finally:
        outputs.close()
        cache.close()
        events.close()


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)))
def test_fp8_kv_append_preserves_page_scales_until_page_reuse(device: str) -> None:
    pool = CachePool(
        num_layers=2,
        num_pages=4,
        page_size=4,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        store_dtype=torch.float8_e4m3fn,
        device=device,
    )
    pages = (3, 1)
    prefix = torch.tensor((112.0, 224.0, 448.0), device=device).view(3, 1, 1)
    suffix = torch.tensor((896.0, -896.0), device=device).view(2, 1, 1)
    try:
        for layer in range(2):
            factor = 2**layer
            pool.write(layer, pages, start=0, k=prefix * factor, v=-prefix * factor / 2)
            pool.write(layer, pages, start=3, k=suffix * factor, v=-suffix * factor / 2)
            keys, values = pool.read(layer, pages, start=0, length=5)
            # E4M3's maximum magnitude is 448. The first page keeps its scale,
            # while the second page derives its own scale from its first write.
            expected = torch.tensor((112.0, 224.0, 448.0, 448.0, -896.0), device=device)
            expected = expected.view(5, 1, 1) * factor
            torch.testing.assert_close(keys, expected, rtol=0, atol=0)
            torch.testing.assert_close(values, -expected / 2, rtol=0, atol=0)

        pool.zero_pages(0, (3,))
        for layer in range(2):
            replacement = prefix * 8 * 2**layer
            pool.write(layer, (3,), start=0, k=replacement, v=-replacement)
            keys, values = pool.read(layer, (3,), start=0, length=3)
            torch.testing.assert_close(keys, replacement, rtol=0, atol=0)
            torch.testing.assert_close(values, -replacement, rtol=0, atol=0)
            keys, values = pool.read(layer, pages, start=4, length=1)
            torch.testing.assert_close(keys, suffix[1:] * 2**layer, rtol=0, atol=0)
            torch.testing.assert_close(values, -suffix[1:] * 2**layer / 2, rtol=0, atol=0)
    finally:
        pool.close()


def test_published_kv_prefix_allows_append_and_waits_for_every_reader_before_reuse() -> None:
    events = DeviceEventPool()
    transport = make_transport("local", byte_capacity=4096, ticket_capacity=2, event_pool=events)
    pool = CachePool(
        num_layers=1,
        num_pages=5,
        page_size=4,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    product = ProductRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=1,
        output_index=0,
        generation=1,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
        dtype=DType.U8,
        shape_bound=ShapeBound((StaticDim(4096),)),
        point_range=PointRange(),
    )
    pages = (3, 1, 4)
    prefix = torch.arange(12, dtype=torch.float32).view(3, 1, 4)
    locations = []
    readers = []
    try:
        pool.write(0, pages, start=0, k=prefix, v=-prefix)
        source = pool.reserve_publication(product, pages, group=0, start=0, length=3)
        for tensor in pool.read(0, pages, start=0, length=3):
            assert tensor is not None
            location = transport.publish(tensor)
            locations.append(location)
            pool.retain_publication(source, transport.publication_retirement(location))
            readers.append(transport.fetch(location, device=torch.device("cpu")))

        # Appending touches the remaining token of the same page and the next
        # page. Neither operation changes the retained prefix's logical value.
        suffix = torch.full((5, 1, 4), 7.0)
        pool.write(0, pages, start=3, k=suffix, v=-suffix)
        torch.testing.assert_close(readers[0].result(), prefix, rtol=0, atol=0)
        torch.testing.assert_close(readers[1].result(), -prefix, rtol=0, atol=0)
        pool.zero_pages(0, (2,))
        with pytest.raises(WorkerError, match="published version"):
            pool.write(0, pages, start=2, k=suffix[:1], v=-suffix[:1])

        pool.release_buffers((product.buffer_id,))
        for location in locations:
            transport.release(location)
        readers[0].close()
        assert not pool.retirement_ready(buffers=(product.buffer_id,))
        with pytest.raises(WorkerError, match="published version"):
            pool.zero_pages(0, (3,))
        torch.testing.assert_close(readers[1].result(), -prefix, rtol=0, atol=0)
        dependencies = pool.write_dependencies(pages, group=0, start=0, length=3)
        readers[1].close()
        for future in dependencies:
            future.result(timeout=5)
        assert pool.retirement_ready(buffers=(product.buffer_id,))
        pool.zero_pages(0, (3,))
        keys, values = pool.read(0, pages, start=0, length=3)
        torch.testing.assert_close(keys, torch.zeros_like(prefix), rtol=0, atol=0)
        torch.testing.assert_close(values, torch.zeros_like(prefix), rtol=0, atol=0)
    finally:
        for reader in readers:
            reader.close()
        for location in locations:
            transport.release(location)
        transport.close()
        pool.close()
        events.close()


@pytest.mark.parametrize(
    "kind", (ProductKind.VISION_FEATURE, ProductKind.ARTIFACT, ProductKind.TENSOR)
)
def test_free_retains_an_acquired_consumer_until_it_records_completion(kind: ProductKind) -> None:
    events = DeviceEventPool()
    buffers = PersistentBuffers(byte_capacity=16, devices=("cpu",))
    store = (
        EncoderCache(
            entry_capacity=1,
            max_entry_bytes=16,
            devices=("cpu",),
            persistent_buffers=buffers,
            event_pool=events,
        )
        if kind is ProductKind.VISION_FEATURE
        else DeviceProducts(
            capacity=1, byte_capacity=16, persistent_buffers=buffers, event_pool=events
        )
    )
    product = ProductRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=1,
        output_index=0,
        generation=1,
        kind=kind,
        storage_class=(
            StorageClass.DEVICE_TENSOR if kind is ProductKind.TENSOR else StorageClass.LATENT_ARENA
        ),
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4),)),
        point_range=PointRange(),
    )
    replacement = replace(product, generation=2)
    value = torch.arange(4, dtype=torch.float32)
    try:
        write = store.bind_outputs(
            ((product, "cpu"),),
            buffer_allocations={product.buffer_id: BufferAllocation(product.buffer_id, 0, 16)},
        )[0]
        if isinstance(store, EncoderCache):
            store.publish(write, value, EncoderMetadata(height=1, width=4))
        else:
            store.publish_write(write, value)
        store.commit_writes((write,))
        store.release_requests((product.request_key,), retained=frozenset((product.buffer_id,)))
        read = store.consume(product, consumer_op_id=2)
        store.release_buffers((product.buffer_id,))
        with pytest.raises(WorkerError):
            store.bind_outputs(
                ((replacement, "cpu"),),
                buffer_allocations={
                    replacement.buffer_id: BufferAllocation(replacement.buffer_id, 0, 16)
                },
            )
        torch.testing.assert_close(read.tensor, value, rtol=0, atol=0)
        store.record_readers((read,))
        # Completion recording is idempotent, including the ordinary lane
        # cleanup that follows a read already fenced by its output producer.
        store.record_readers((read,))
        reused = store.bind_outputs(
            ((replacement, "cpu"),),
            buffer_allocations={
                replacement.buffer_id: BufferAllocation(replacement.buffer_id, 0, 16)
            },
        )
        store.abandon_writes(reused)
    finally:
        store.close()
        buffers.close()
        events.close()


def test_tensor_publication_enforces_its_logical_region_and_representation() -> None:
    buffers = PersistentBuffers(byte_capacity=24, devices=("cpu",))
    events = DeviceEventPool()
    store = DeviceProducts(
        capacity=1, byte_capacity=16, persistent_buffers=buffers, event_pool=events
    )
    reference = ProductRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=1,
        output_index=0,
        generation=1,
        kind=ProductKind.TENSOR,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4), StaticDim(4))),
        point_range=PointRange(),
    )
    allocation = {reference.buffer_id: BufferAllocation(reference.buffer_id, 0, 24)}
    region = TensorRegion(offset=(2, 1), shape=(2, 3))
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    try:
        with pytest.raises(WorkerError, match="logical bounds"):
            store.bind_outputs(
                ((reference, "cpu"),),
                buffer_allocations=allocation,
                regions={reference: TensorRegion(offset=(3, 1), shape=(2, 3))},
            )
        write = store.bind_outputs(
            ((reference, "cpu"),), buffer_allocations=allocation, regions={reference: region}
        )[0]
        with pytest.raises(WorkerError, match="dtype"):
            store.publish_write(write, expected.to(torch.float16))
        with pytest.raises(WorkerError, match="shape"):
            store.publish_write(write, expected.reshape(3, 2))
        store.publish_write(write, expected)
        store.commit_writes((write,))
        read = store.consume(reference, consumer_op_id=2)
        assert read.region == region
        torch.testing.assert_close(read.tensor, expected, rtol=0, atol=0)
        store.record_readers((read,))
    finally:
        store.close()
        buffers.close()
        events.close()


def test_latent_import_preserves_page_order_and_committed_metadata() -> None:
    events = DeviceEventPool()
    transport = make_transport("local", byte_capacity=4096, ticket_capacity=1, event_pool=events)
    pool = LatentPool(
        request_pool_size=2,
        num_pages=5,
        page_units=4,
        latent_width=4,
        dtype=torch.float32,
        device="cpu",
    )
    product = ProductRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=2,
        output_index=0,
        generation=3,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(11), StaticDim(4))),
        point_range=PointRange(),
    )
    expected = torch.arange(44, dtype=torch.float32).reshape(11, 4)
    locator = transport.publish(expected)
    try:
        write = pool.reserve_import(
            product, request_pool_idx=1, page_table=(3, 1, 4), latent_units=11
        )
        with pytest.raises(WorkerError, match="no committed trajectory"):
            pool.snapshot(request_pool_idx=1, page_table=(3, 1, 4))
        ticket = transport.fetch(locator, device=torch.device("cpu"), destination=write.spans)
        pool.retain_transfer(write, ticket)
        ready = Event()
        ticket.add_done_callback(ready.set)
        assert ready.wait(5)
        pool.adopt_import(write, generation=3, step=2, height=16, width=176)
        snapshot = pool.snapshot(request_pool_idx=1, page_table=(3, 1, 4))
        assert (snapshot.generation, snapshot.step, snapshot.latent_units) == (3, 2, 11)
        assert (snapshot.height, snapshot.width) == (16, 176)
        torch.testing.assert_close(snapshot.value, expected, rtol=0, atol=0)
        with pytest.raises(WorkerError, match="page table does not match"):
            pool.snapshot(request_pool_idx=1, page_table=(1, 3, 4))
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


@pytest.mark.parametrize("committed", (False, True))
def test_published_latent_bank_waits_for_every_reader_before_reuse(committed: bool) -> None:
    from uniserve_worker.runtime.latent_pool import LatentPublication

    events = DeviceEventPool()
    transport = make_transport("local", byte_capacity=4096, ticket_capacity=4, event_pool=events)
    pool = LatentPool(
        request_pool_size=2,
        num_pages=5,
        page_units=4,
        latent_width=4,
        dtype=torch.float32,
        device="cpu",
    )
    product = ProductRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=1,
        output_index=0,
        generation=1,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(11), StaticDim(4))),
        point_range=PointRange(),
    )
    pages = (3, 1, 4)
    staging = pool.stage((pages,), (11,))[0]
    commit = LatentPublication(1, pages, 0, 0, 1, 0, 11, 16, 176)
    locations = []
    readers = []
    try:
        staging.value.fill_(1)
        pool.initialize(1, staging, latent_units=11)
        if committed:
            pool.validate_commit((commit,), ())
            pool.apply_commit((commit,), ())
            source = pool.reserve_current_publication(
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
            source = pool.reserve_publication(
                product, request_pool_idx=1, page_table=pages, latent_units=11
            )
        offset = 0
        for span in source.spans:
            location = transport.publish(span, offset=(offset, 0))
            locations.append(location)
            pool.retain_publication(source, transport.publication_retirement(location))
            readers.append(transport.fetch(location, device=torch.device("cpu")))
            offset += span.shape[0]
        borrowed = tuple(reader.result() for reader in readers)
        if not committed:
            pool.validate_commit((commit,), ())
            pool.apply_commit((commit,), ())

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
        second = replace(commit, expected_generation=1, generation=2, step=1)
        pool.validate_commit((second,), ())
        pool.apply_commit((second,), ())
        for value in borrowed:
            torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)
        torch.testing.assert_close(
            pool.snapshot(request_pool_idx=1, page_table=pages).value,
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
        torch.testing.assert_close(borrowed[-1], torch.ones_like(borrowed[-1]), rtol=0, atol=0)
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
        third = replace(second, expected_generation=2, expected_step=1, generation=3, step=2)
        pool.validate_commit((third,), ())
        pool.apply_commit((third,), ())
        torch.testing.assert_close(
            pool.snapshot(request_pool_idx=1, page_table=pages).value,
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


def test_failed_latent_publication_retains_its_pages_without_poisoning_other_requests() -> None:
    import hashlib
    import json
    import socket
    from contextlib import suppress
    from multiprocessing import shared_memory

    from uniserve_worker.runtime.latent_pool import LatentPublication

    events = DeviceEventPool()
    transport = make_transport("shm", byte_capacity=4096, ticket_capacity=2, event_pool=events)
    pool = LatentPool(
        request_pool_size=2,
        num_pages=5,
        page_units=4,
        latent_width=4,
        dtype=torch.float32,
        device="cpu",
    )
    product = ProductRef(
        request_key=RequestKey(1, 1, 1),
        producer_op_id=1,
        output_index=0,
        generation=1,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4), StaticDim(4))),
        point_range=PointRange(),
    )
    staging = pool.stage(((1,),), (4,))[0]
    staging.value.fill_(1)
    pool.initialize(1, staging, latent_units=4)
    source = pool.reserve_publication(product, request_pool_idx=1, page_table=(1,), latent_units=4)
    locator = transport.publish(source.spans[0])
    retirement = transport.publication_retirement(locator)
    pool.retain_publication(source, retirement)
    commit = LatentPublication(1, (1,), 0, 0, 1, 0, 4, 16, 64)
    pool.validate_commit((commit,), ())
    pool.apply_commit((commit,), ())
    try:
        descriptor = locator.to_mapping()
        digest = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        key = hashlib.sha256(descriptor["name"].encode()).digest()
        failed = Event()
        retirement.add_done_callback(lambda _future: failed.set())
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as reader:
            reader.settimeout(5)
            reader.connect("\0" + descriptor["endpoint"])
            reader.sendall(key + digest)
            assert reader.recv(1) == b"G"
            # Lost acknowledgement cannot prove physical completion.
        assert failed.wait(5)
        with pytest.raises(WorkerError, match="without physical completion"):
            retirement.result()
        transport.release(locator)
        pool.release_buffers((product.buffer_id,))
        with pytest.raises(WorkerError, match="without physical completion"):
            pool.retirement_ready((product.request_key,))
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
        next_commit = replace(commit, request_pool_idx=2, page_table=(2,))
        pool.validate_commit((next_commit,), ())
        pool.apply_commit((next_commit,), ())
        torch.testing.assert_close(
            pool.snapshot(request_pool_idx=2, page_table=(2,)).value,
            torch.full((4, 4), 7.0),
            rtol=0,
            atol=0,
        )
        assert pool.retirement_ready((RequestKey(1, 2, 1),))
    finally:
        with suppress(WorkerError):
            transport.close()
        with suppress(WorkerError):
            pool.close()
        events.close()
        # Both sides of this host-only test have stopped. Remove the retained
        # external segment without treating it as an acknowledged publication.
        with suppress(FileNotFoundError):
            segment = shared_memory.SharedMemory(name=locator.transport.name)
            segment.close()
            segment.unlink()
