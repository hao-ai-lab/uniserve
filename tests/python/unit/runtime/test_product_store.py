from __future__ import annotations

import pytest
import torch

from uniserve_worker.batch import (
    DeviceDim,
    DType,
    PointRange,
    ProductKind,
    ProductRef,
    RequestKey,
    ShapeBound,
    StorageClass,
)
from uniserve_worker.runtime.product_capacity import device_product_scalar_arena_bytes
from uniserve_worker.runtime.product_store import (
    DeviceProductTable,
    ProductRecord,
    ProductStore,
    VisionFeatureProduct,
)

pytestmark = pytest.mark.unit


def test_device_product_byte_contract_covers_every_scalar_storage_arena() -> None:
    capacity = 7
    table = DeviceProductTable(
        capacity=capacity,
        byte_capacity=device_product_scalar_arena_bytes(capacity, 1),
    )

    for index, dtype in enumerate(DType, start=1):
        reference = _device_ref(op_id=index, generation=index, dtype=dtype)
        (write,) = table.bind_outputs(((reference, f"{index:064x}", "cpu"),))
        table.abandon_writes((write,))

    assert table.allocated_bytes == device_product_scalar_arena_bytes(capacity, 1)


def _vision_record(handle: int, session_id: int) -> ProductRecord:
    return ProductRecord(
        handle=handle,
        session_id=session_id,
        payload=VisionFeatureProduct(
            features=torch.full((2, 3), float(handle)),
            height=16,
            width=32,
            source_base64=f"image-{handle}",
        ),
    )


def _commit(store: ProductStore, record: ProductRecord) -> None:
    transaction = store.begin_step({record.session_id})
    transaction.stage(record)
    transaction.prepare()
    transaction.publish()
    transaction.finalize()


def _device_ref(
    *,
    op_id: int,
    generation: int,
    dtype: DType = DType.U32,
    output_index: int = 0,
    kind: ProductKind = ProductKind.TOKEN,
    elements: int = 1,
) -> ProductRef:
    return ProductRef(
        request_key=RequestKey(1, 7, 3),
        producer_op_id=op_id,
        output_index=output_index,
        generation=generation,
        kind=kind,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=dtype,
        shape_bound=(ShapeBound() if elements == 1 else ShapeBound((DeviceDim(elements),))),
        point_range=PointRange(),
    )


def test_encoder_product_lifetime_is_owned_by_explicit_handle_release() -> None:
    store = ProductStore(
        encoder_cache_budget=1,
        device_product_capacity=1,
        device_product_byte_capacity=1 << 20,
    )
    reference = ProductRef(
        request_key=RequestKey(1, 1, 3),
        producer_op_id=11,
        output_index=0,
        generation=11,
        kind=ProductKind.VISION_FEATURE,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(6),)),
        point_range=PointRange(),
    )
    (write,) = store.device_products.bind_outputs(((reference, "ab" * 32, "cpu"),))
    resident = store.device_products.publish_write(
        write,
        torch.full((2, 3), 11.0, dtype=torch.bfloat16),
    )
    cached = _vision_record(11, 1)
    cached = ProductRecord(
        handle=cached.handle,
        session_id=cached.session_id,
        payload=VisionFeatureProduct(
            features=resident,
            height=cached.payload.height,
            width=cached.payload.width,
            source_base64=cached.payload.source_base64,
        ),
    )
    _commit(store, cached)
    original_physical_generation = store.device_products.physical_generation(reference)

    store.drop(1)

    assert store.require(11) == cached
    read = store.device_products.consume(reference, consumer_op_id=12)
    store.device_products.record_reader(read)
    assert read.tensor.reshape(-1).tolist() == [11.0] * 6

    store.release((11,))
    replacement_reference = ProductRef(
        request_key=RequestKey(1, 2, 3),
        producer_op_id=12,
        output_index=0,
        generation=12,
        kind=ProductKind.VISION_FEATURE,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(6),)),
        point_range=PointRange(),
    )
    store.device_products.bind_outputs(((replacement_reference, "cd" * 32, "cpu"),))
    store.device_products.publish(
        replacement_reference,
        torch.full((2, 3), 12.0, dtype=torch.bfloat16),
    )
    replacement = _vision_record(12, 2)
    _commit(store, replacement)

    assert store.require(12) == replacement
    assert store.encoder_output_count() == 1
    assert (
        store.device_products.physical_generation(replacement_reference)
        != original_physical_generation
    )


def test_device_product_access_validates_logical_and_physical_generations() -> None:
    table = DeviceProductTable(capacity=1, byte_capacity=1 << 20)
    reference = _device_ref(op_id=11, generation=5)
    table.bind_outputs(((reference, "ab" * 32, "cpu"),))
    published = table.publish(reference, torch.tensor([41], dtype=torch.long))

    read = table.consume(
        reference,
        consumer_op_id=12,
        producer_plan_digest="ab" * 32,
    )
    table.record_reader(read)

    assert published.tolist() == [41]
    assert read.tensor.tolist() == [41]
    assert table.physical_generation(reference) == read.physical_generation

    stale = ProductRef(
        request_key=reference.request_key,
        producer_op_id=reference.producer_op_id,
        output_index=reference.output_index,
        generation=reference.generation + 1,
        kind=reference.kind,
        storage_class=reference.storage_class,
        dtype=reference.dtype,
        shape_bound=reference.shape_bound,
        point_range=reference.point_range,
    )
    with pytest.raises(Exception, match="stale device-product logical generation"):
        table.consume(stale, consumer_op_id=12)


def test_device_product_slot_reuse_advances_the_physical_generation() -> None:
    table = DeviceProductTable(capacity=1, byte_capacity=1 << 20)
    first = _device_ref(op_id=21, generation=8)
    (first_write,) = table.bind_outputs(((first, "cd" * 32, "cpu"),))
    table.publish(first, torch.tensor([7], dtype=torch.long))
    first_generation = table.physical_generation(first)
    table.release_operation(first.request_key, first.producer_op_id)

    second = _device_ref(op_id=22, generation=9)
    table.bind_outputs(((second, "ef" * 32, "cpu"),))
    table.publish(second, torch.tensor([13], dtype=torch.long))

    assert table.physical_generation(second) != first_generation
    assert table.consume(second, consumer_op_id=23).tensor.tolist() == [13]
    with pytest.raises(Exception, match="stale device-product physical generation"):
        table.producer_write_views((first_write,))


def test_device_product_registration_failure_preserves_atomic_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = DeviceProductTable(capacity=2, byte_capacity=1 << 20)
    first = _device_ref(op_id=31, generation=10)
    second = _device_ref(op_id=32, generation=11, dtype=DType.U8)
    allocate = torch.empty
    calls = 0

    def fail_second_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("injected allocation failure")
        return allocate(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_second_allocation)
    with pytest.raises(torch.OutOfMemoryError):
        table.bind_outputs(
            (
                (first, "ab" * 32, "cpu"),
                (second, "cd" * 32, "cpu"),
            )
        )

    monkeypatch.setattr(torch, "empty", allocate)
    table.bind_outputs(
        (
            (first, "ab" * 32, "cpu"),
            (second, "cd" * 32, "cpu"),
        )
    )
    table.publish(first, torch.tensor([17], dtype=torch.long))
    table.publish(second, torch.tensor([1], dtype=torch.uint8))

    assert table.consume(first, consumer_op_id=41).tensor.tolist() == [17]
    assert table.consume(second, consumer_op_id=42).tensor.tolist() == [1]


def test_device_product_byte_exhaustion_preserves_capacity_and_reclaims_slots() -> None:
    table = DeviceProductTable(capacity=2, byte_capacity=16)
    oversized = _device_ref(op_id=41, generation=12, elements=3)

    with pytest.raises(Exception, match="device-product byte credit is exhausted"):
        table.bind_outputs(((oversized, "ab" * 32, "cpu"),))

    assert table.allocated_bytes == 0

    first = _device_ref(op_id=42, generation=13)
    second = _device_ref(op_id=43, generation=14)
    writes = table.bind_outputs(
        (
            (first, "cd" * 32, "cpu"),
            (second, "ef" * 32, "cpu"),
        )
    )
    table.publish_batch((first, second), torch.tensor([19, 23], dtype=torch.long))

    assert table.allocated_bytes == 16
    assert {write.slot.index for write in writes} == {0, 1}
    assert table.consume(first, consumer_op_id=51).tensor.tolist() == [19]
    assert table.consume(second, consumer_op_id=52).tensor.tolist() == [23]


def test_compatible_product_batch_binds_one_contiguous_producer_range() -> None:
    table = DeviceProductTable(capacity=6, byte_capacity=1 << 20)
    retained = tuple(_device_ref(op_id=51 + index, generation=20 + index) for index in range(4))
    table.bind_outputs(tuple((reference, "ab" * 32, "cpu") for reference in retained))
    table.publish_batch(retained, torch.tensor([1, 2, 3, 4], dtype=torch.long))
    table.release_operations(
        (
            (retained[0].request_key, retained[0].producer_op_id),
            (retained[2].request_key, retained[2].producer_op_id),
        )
    )

    current = tuple(_device_ref(op_id=61 + index, generation=30 + index) for index in range(2))
    writes = table.bind_outputs(tuple((reference, "cd" * 32, "cpu") for reference in current))
    producer_batch = table.producer_scalar_batch(writes)

    assert producer_batch is not None
    assert producer_batch.tensor.shape == (2,)


def test_scalar_output_groups_retain_independent_direct_producer_ranges() -> None:
    table = DeviceProductTable(capacity=6, byte_capacity=1 << 20)
    finish = tuple(
        _device_ref(
            op_id=71 + index,
            generation=40 + index,
            dtype=DType.U8,
            output_index=1,
            kind=ProductKind.FINISH,
        )
        for index in range(3)
    )
    completion = tuple(
        _device_ref(
            op_id=71 + index,
            generation=50 + index,
            dtype=DType.U8,
            output_index=2,
            kind=ProductKind.COMPLETION,
        )
        for index in range(3)
    )

    groups = table.bind_output_groups(
        (
            tuple((reference, "ab" * 32, "cpu") for reference in finish),
            tuple((reference, "ab" * 32, "cpu") for reference in completion),
        )
    )

    assert len(groups) == 2
    assert all(group.scalar is not None for group in groups)
    assert tuple(group.scalar.tensor.shape for group in groups if group.scalar is not None) == (
        (3,),
        (3,),
    )


def test_row_output_batch_publishes_each_registered_tensor_shape() -> None:
    table = DeviceProductTable(capacity=3, byte_capacity=1 << 20)
    references = tuple(
        _device_ref(op_id=91 + index, generation=60 + index, elements=3) for index in range(3)
    )
    writes = table.bind_outputs(tuple((reference, "ab" * 32, "cpu") for reference in references))

    table.publish_rows(
        writes,
        torch.tensor(
            (
                (1, 2, 3),
                (4, 5, 6),
                (7, 8, 9),
            ),
            dtype=torch.long,
        ),
    )
    reads = table.consume_batch(
        tuple(
            (reference, 101 + index, "ab" * 32, "cpu") for index, reference in enumerate(references)
        )
    )

    assert tuple(read.tensor.tolist() for read in reads) == (
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    )
    assert tuple(write.actual_shape for write in writes) == ((3,), (3,), (3,))


def test_mixed_shape_generations_reuse_compatible_resident_storage() -> None:
    table = DeviceProductTable(capacity=4, byte_capacity=1 << 20)
    first = tuple(
        _device_ref(
            op_id=101 + index,
            generation=70 + index,
            output_index=index,
            elements=elements,
        )
        for index, elements in enumerate((2, 4, 2, 4))
    )
    first_writes = table.bind_outputs(tuple((reference, "ab" * 32, "cpu") for reference in first))
    for write, elements in zip(first_writes, (2, 4, 2, 4), strict=True):
        table.publish_write(write, torch.arange(elements, dtype=torch.long))
    first_storage = {
        elements: {
            write.slot.tensor.untyped_storage().data_ptr()
            for write in first_writes
            if write.slot.tensor is not None and write.actual_extent == elements
        }
        for elements in (2, 4)
    }
    table.release_operations(
        (reference.request_key, reference.producer_op_id) for reference in first
    )
    assert table.reclaim_ready() == 4

    second = tuple(
        _device_ref(
            op_id=111 + index,
            generation=80 + index,
            output_index=index,
            elements=elements,
        )
        for index, elements in enumerate((4, 2, 4, 2))
    )
    second_writes = table.bind_outputs(tuple((reference, "cd" * 32, "cpu") for reference in second))
    second_storage = {
        elements: {
            write.slot.tensor.untyped_storage().data_ptr()
            for write in second_writes
            if write.slot.tensor is not None and write.slot.tensor.numel() == elements
        }
        for elements in (2, 4)
    }

    assert second_storage == first_storage


def test_operation_release_covers_continuation_and_regular_outputs() -> None:
    table = DeviceProductTable(capacity=3, byte_capacity=1 << 20)
    parent = _device_ref(op_id=71, generation=40)
    (parent_write,) = table.bind_outputs(((parent, "ab" * 32, "cpu"),))
    table.publish_write(parent_write, torch.tensor([5], dtype=torch.long))

    token = _device_ref(op_id=72, generation=41)
    finish = _device_ref(
        op_id=72,
        generation=42,
        dtype=DType.U8,
        output_index=1,
        kind=ProductKind.FINISH,
    )
    continuation = table.bind_scalar_continuation(
        outputs=((token, "cd" * 32),),
        parents=((parent, token.producer_op_id, "ab" * 32),),
        device="cpu",
    )
    (finish_write,) = table.bind_outputs(((finish, "cd" * 32, "cpu"),))
    table.publish_continuation(continuation, torch.tensor([7], dtype=torch.long))
    table.publish_scalar_write(finish_write, False)

    table.release_operation(token.request_key, token.producer_op_id)

    assert table.reclaim_ready() == 2


def test_continuation_accepts_complete_split_publication() -> None:
    table = DeviceProductTable(capacity=2, byte_capacity=1 << 20)
    parent = _device_ref(op_id=81, generation=50)
    (parent_write,) = table.bind_outputs(((parent, "ab" * 32, "cpu"),))
    table.publish_write(parent_write, torch.tensor([5], dtype=torch.long))

    token = _device_ref(op_id=82, generation=51)
    continuation = table.bind_scalar_continuation(
        outputs=((token, "cd" * 32),),
        parents=((parent, token.producer_op_id, "ab" * 32),),
        device="cpu",
    )
    token_batch = continuation.scalar
    assert token_batch is not None
    token_batch.tensor.fill_(7)
    table.publish_scalar_batch(token_batch)

    table.finish_continuation(continuation)
    table.validate_continuation(continuation)

    (read,) = table.consume_batch(((token, 83, "cd" * 32, "cpu"),))
    assert int(read.tensor.item()) == 7
    table.record_readers((read,), device="cpu")
