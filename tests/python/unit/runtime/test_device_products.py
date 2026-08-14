"""Behavioral contracts for bounded immutable device products."""

from __future__ import annotations

from dataclasses import replace

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
from uniserve_worker.foundation.errors import ResourceError, WorkerError
from uniserve_worker.runtime.device_products import (
    DeviceProductMetadata,
    DeviceProducts,
    ImageRange,
)

pytestmark = pytest.mark.unit


def _reference(
    *,
    session_id: int,
    op_id: int,
    generation: int,
    output_index: int = 0,
    kind: ProductKind = ProductKind.TOKEN,
    storage_class: StorageClass = StorageClass.DEVICE_TENSOR,
    dtype: DType = DType.U32,
    elements: int = 1,
) -> ProductRef:
    return ProductRef(
        request_key=RequestKey(1, session_id, 3),
        producer_op_id=op_id,
        output_index=output_index,
        generation=generation,
        kind=kind,
        storage_class=storage_class,
        dtype=dtype,
        shape_bound=(
            ShapeBound() if elements == 1 else ShapeBound((DeviceDim(elements),))
        ),
        point_range=PointRange(),
    )


def test_exact_generation_release_reclaims_only_the_selected_product() -> None:
    products = DeviceProducts(capacity=2, byte_capacity=1 << 20)
    first = _reference(session_id=7, op_id=11, generation=5)
    retained = _reference(session_id=8, op_id=12, generation=6)
    writes = products.bind_outputs(
        (
            (first, "ab" * 32, "cpu"),
            (retained, "cd" * 32, "cpu"),
        )
    )
    products.publish_writes(writes, torch.tensor((41, 53), dtype=torch.long))

    reads = products.consume_batch(
        (
            (first, 21, "ab" * 32, "cpu"),
            (retained, 22, "cd" * 32, "cpu"),
        )
    )
    products.record_readers(reads)
    assert tuple(int(read.tensor.item()) for read in reads) == (41, 53)

    products.release_generations((first.generation,))
    replacement = _reference(session_id=9, op_id=13, generation=7)
    (replacement_write,) = products.bind_outputs(
        ((replacement, "ef" * 32, "cpu"),)
    )
    products.publish_write(replacement_write, torch.tensor([67], dtype=torch.long))

    assert products.consume(retained, consumer_op_id=23).tensor.item() == 53
    assert products.consume(replacement, consumer_op_id=24).tensor.item() == 67
    with pytest.raises(WorkerError, match="unknown device-product reference"):
        products.consume(first, consumer_op_id=25)


def test_logical_generation_and_producer_digest_are_exact() -> None:
    products = DeviceProducts(capacity=1, byte_capacity=1 << 20)
    reference = _reference(session_id=7, op_id=31, generation=9)
    (write,) = products.bind_outputs(((reference, "ab" * 32, "cpu"),))
    products.publish_write(write, torch.tensor([73], dtype=torch.long))

    stale = replace(reference, generation=reference.generation + 1)
    with pytest.raises(WorkerError, match="stale device-product logical generation"):
        products.consume(stale, consumer_op_id=32)
    with pytest.raises(WorkerError, match="plan digest"):
        products.consume(
            reference,
            consumer_op_id=32,
            producer_plan_digest="cd" * 32,
        )
    assert products.consume(
        reference,
        consumer_op_id=32,
        producer_plan_digest="ab" * 32,
    ).tensor.item() == 73


def test_resident_artifact_preserves_image_geometry() -> None:
    products = DeviceProducts(capacity=1, byte_capacity=1 << 20)
    reference = _reference(
        session_id=7,
        op_id=41,
        generation=10,
        kind=ProductKind.ARTIFACT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.F32,
        elements=12,
    )
    value = torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 2, 2)
    metadata = DeviceProductMetadata(
        height=2,
        width=2,
        value_range=ImageRange.SIGNED_UNIT,
    )
    (write,) = products.bind_outputs(((reference, "ab" * 32, "cpu"),))
    products.publish_write(write, value, metadata=metadata)

    read = products.consume(reference, consumer_op_id=42)
    assert read.metadata == metadata
    torch.testing.assert_close(read.tensor, value)


@pytest.mark.parametrize(
    ("kind", "storage_class"),
    (
        (ProductKind.KV, StorageClass.PAGED_KV),
        (ProductKind.LATENT, StorageClass.LATENT_ARENA),
        (ProductKind.VISION_FEATURE, StorageClass.LATENT_ARENA),
        (ProductKind.LOGPROB, StorageClass.COMPLETION_ARENA),
        (ProductKind.TOKEN, StorageClass.HOST_STAGING),
    ),
)
def test_non_device_owner_kinds_are_rejected(
    kind: ProductKind,
    storage_class: StorageClass,
) -> None:
    products = DeviceProducts(capacity=1, byte_capacity=1 << 20)
    reference = _reference(
        session_id=7,
        op_id=51,
        generation=11,
        kind=kind,
        storage_class=storage_class,
    )

    with pytest.raises(WorkerError, match="does not belong|storage class"):
        products.bind_outputs(((reference, "ab" * 32, "cpu"),))


def test_byte_and_slot_exhaustion_return_bounded_backpressure() -> None:
    products = DeviceProducts(capacity=1, byte_capacity=8)
    oversized = _reference(
        session_id=7,
        op_id=61,
        generation=12,
        elements=2,
    )
    with pytest.raises(ResourceError, match="byte capacity"):
        products.bind_outputs(((oversized, "ab" * 32, "cpu"),))

    resident = _reference(session_id=7, op_id=62, generation=13)
    (write,) = products.bind_outputs(((resident, "ab" * 32, "cpu"),))
    products.publish_write(write, torch.tensor([1], dtype=torch.long))
    blocked = _reference(session_id=8, op_id=63, generation=14)
    with pytest.raises(ResourceError, match="no query-ready free generation"):
        products.bind_outputs(((blocked, "cd" * 32, "cpu"),))
