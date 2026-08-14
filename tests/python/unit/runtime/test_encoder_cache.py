"""Behavioral contracts for immutable reusable encoder features."""

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
from uniserve_worker.runtime.encoder_cache import EncoderCache, EncoderMetadata

pytestmark = pytest.mark.unit


def _feature(
    *,
    session_id: int,
    op_id: int,
    generation: int,
    kind: ProductKind = ProductKind.VISION_FEATURE,
    storage_class: StorageClass = StorageClass.LATENT_ARENA,
    elements: int = 6,
) -> ProductRef:
    return ProductRef(
        request_key=RequestKey(1, session_id, 2),
        producer_op_id=op_id,
        output_index=0,
        generation=generation,
        kind=kind,
        storage_class=storage_class,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(elements),)),
        point_range=PointRange(),
    )


def test_feature_value_geometry_and_cross_request_read_are_immutable() -> None:
    cache = EncoderCache(
        entry_capacity=1,
        max_entry_bytes=64,
        devices=("cpu",),
    )
    reference = _feature(session_id=1, op_id=11, generation=5)
    metadata = EncoderMetadata(height=16, width=32)
    value = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    (write,) = cache.bind_outputs(((reference, "ab" * 32, "cpu"),))
    cache.publish(write, value, metadata)

    read = cache.consume(
        reference,
        consumer_op_id=21,
        producer_plan_digest="ab" * 32,
    )
    cache.record_readers((read,))
    assert read.metadata == metadata
    torch.testing.assert_close(read.tensor, value)


def test_release_and_reuse_preserve_exact_feature_identity() -> None:
    cache = EncoderCache(
        entry_capacity=2,
        max_entry_bytes=64,
        devices=("cpu",),
    )
    first = _feature(session_id=1, op_id=31, generation=7)
    retained = _feature(session_id=2, op_id=32, generation=8)
    writes = cache.bind_outputs(
        (
            (first, "ab" * 32, "cpu"),
            (retained, "cd" * 32, "cpu"),
        )
    )
    cache.publish(writes[0], torch.full((2, 3), 3.0, dtype=torch.bfloat16), EncoderMetadata(8, 8))
    cache.publish(writes[1], torch.full((2, 3), 5.0, dtype=torch.bfloat16), EncoderMetadata(8, 8))

    cache.release_generations((first.generation,))
    replacement = _feature(session_id=3, op_id=33, generation=9)
    (replacement_write,) = cache.bind_outputs(
        ((replacement, "ef" * 32, "cpu"),)
    )
    cache.publish(
        replacement_write,
        torch.full((2, 3), 7.0, dtype=torch.bfloat16),
        EncoderMetadata(8, 8),
    )

    assert cache.consume(retained, consumer_op_id=34).tensor[0, 0].item() == 5.0
    assert cache.consume(replacement, consumer_op_id=35).tensor[0, 0].item() == 7.0
    with pytest.raises(WorkerError, match="unknown encoder feature"):
        cache.consume(first, consumer_op_id=36)


def test_feature_generation_and_producer_digest_are_exact() -> None:
    cache = EncoderCache(
        entry_capacity=1,
        max_entry_bytes=64,
        devices=("cpu",),
    )
    reference = _feature(session_id=1, op_id=41, generation=10)
    (write,) = cache.bind_outputs(((reference, "ab" * 32, "cpu"),))
    cache.publish(write, torch.ones((2, 3), dtype=torch.bfloat16), EncoderMetadata(4, 4))

    with pytest.raises(WorkerError, match="stale encoder feature generation"):
        cache.consume(replace(reference, generation=11), consumer_op_id=42)
    with pytest.raises(WorkerError, match="plan digest"):
        cache.consume(
            reference,
            consumer_op_id=42,
            producer_plan_digest="cd" * 32,
        )


@pytest.mark.parametrize(
    ("kind", "storage_class"),
    (
        (ProductKind.TOKEN, StorageClass.DEVICE_TENSOR),
        (ProductKind.KV, StorageClass.PAGED_KV),
        (ProductKind.VISION_FEATURE, StorageClass.DEVICE_TENSOR),
    ),
)
def test_non_encoder_owner_products_are_rejected(
    kind: ProductKind,
    storage_class: StorageClass,
) -> None:
    cache = EncoderCache(
        entry_capacity=1,
        max_entry_bytes=64,
        devices=("cpu",),
    )
    reference = _feature(
        session_id=1,
        op_id=51,
        generation=12,
        kind=kind,
        storage_class=storage_class,
    )
    with pytest.raises(WorkerError, match="non-feature|storage class"):
        cache.bind_outputs(((reference, "ab" * 32, "cpu"),))


def test_entry_and_byte_capacity_return_bounded_backpressure() -> None:
    cache = EncoderCache(
        entry_capacity=1,
        max_entry_bytes=16,
        devices=("cpu",),
    )
    oversized = _feature(
        session_id=1,
        op_id=61,
        generation=13,
        elements=16,
    )
    with pytest.raises(ResourceError, match="entry byte capacity"):
        cache.bind_outputs(((oversized, "ab" * 32, "cpu"),))

    resident = _feature(session_id=1, op_id=62, generation=14)
    (write,) = cache.bind_outputs(((resident, "ab" * 32, "cpu"),))
    cache.publish(write, torch.ones((2, 3), dtype=torch.bfloat16), EncoderMetadata(4, 4))
    blocked = _feature(session_id=2, op_id=63, generation=15)
    with pytest.raises(ResourceError, match="entry capacity"):
        cache.bind_outputs(((blocked, "cd" * 32, "cpu"),))
