"""CUDA stream-ordering behavior for reusable encoder features."""

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
from uniserve_worker.foundation.errors import ResourceError
from uniserve_worker.runtime.encoder_cache import EncoderCache, EncoderMetadata

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _feature(op_id: int, generation: int) -> ProductRef:
    return ProductRef(
        request_key=RequestKey(1, 9, 2),
        producer_op_id=op_id,
        output_index=0,
        generation=generation,
        kind=ProductKind.VISION_FEATURE,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(6),)),
        point_range=PointRange(),
    )


def test_feature_consumer_waits_for_its_producer_stream() -> None:
    device = torch.device("cuda:0")
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    cache = EncoderCache(
        entry_capacity=1,
        max_entry_bytes=64,
        devices=(device,),
    )
    reference = _feature(11, 5)
    (write,) = cache.bind_outputs(((reference, "ab" * 32, device),))
    observed = torch.empty((2, 3), dtype=torch.bfloat16, device=device)

    with torch.cuda.stream(producer):
        torch.cuda._sleep(50_000_000)
        cache.publish(
            write,
            torch.full((2, 3), 7.0, dtype=torch.bfloat16, device=device),
            EncoderMetadata(16, 16),
        )
        cache.commit_writes((write,))
    with torch.cuda.stream(consumer):
        read = cache.consume(
            reference,
            consumer_op_id=12,
            producer_plan_digest="ab" * 32,
            device=device,
        )
        observed.copy_(read.tensor)
        cache.record_readers((read,))

    consumer.synchronize()
    torch.testing.assert_close(observed, torch.full_like(observed, 7.0))


def test_feature_reuse_waits_for_the_last_reader() -> None:
    device = torch.device("cuda:0")
    producer = torch.cuda.Stream(device=device)
    delayed_consumer = torch.cuda.Stream(device=device)
    cache = EncoderCache(
        entry_capacity=1,
        max_entry_bytes=64,
        devices=(device,),
    )
    first = _feature(21, 8)
    (write,) = cache.bind_outputs(((first, "ab" * 32, device),))
    with torch.cuda.stream(producer):
        cache.publish(
            write,
            torch.full((2, 3), 11.0, dtype=torch.bfloat16, device=device),
            EncoderMetadata(8, 8),
        )
        cache.commit_writes((write,))
    with torch.cuda.stream(delayed_consumer):
        read = cache.consume(first, consumer_op_id=22, device=device)
        observed = read.tensor.clone()
        torch.cuda._sleep(50_000_000)
        cache.record_readers((read,))

    producer.synchronize()
    cache.release_generations((first.generation,))
    second = _feature(23, 9)
    with pytest.raises(ResourceError, match="entry capacity"):
        cache.bind_outputs(((second, "cd" * 32, device),))

    delayed_consumer.synchronize()
    (second_write,) = cache.bind_outputs(((second, "cd" * 32, device),))
    with torch.cuda.stream(producer):
        cache.publish(
            second_write,
            torch.full((2, 3), 13.0, dtype=torch.bfloat16, device=device),
            EncoderMetadata(8, 8),
        )
        cache.commit_writes((second_write,))
    producer.synchronize()

    torch.testing.assert_close(observed, torch.full_like(observed, 11.0))
    assert cache.consume(second, consumer_op_id=24, device=device).tensor[0, 0].item() == 13.0
