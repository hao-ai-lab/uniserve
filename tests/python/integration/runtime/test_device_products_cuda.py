"""CUDA stream-ordering behavior for immutable device products."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.batch import (
    DType,
    PointRange,
    ProductKind,
    ProductRef,
    RequestKey,
    ShapeBound,
    StorageClass,
)
from uniserve_worker.foundation.errors import ResourceError
from uniserve_worker.runtime.device_products import DeviceProducts

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _reference(op_id: int, generation: int) -> ProductRef:
    return ProductRef(
        request_key=RequestKey(1, 9, 2),
        producer_op_id=op_id,
        output_index=0,
        generation=generation,
        kind=ProductKind.TOKEN,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U32,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )


def test_consumer_stream_observes_the_exact_producer_generation() -> None:
    device = torch.device("cuda:0")
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    products = DeviceProducts(capacity=1, byte_capacity=1 << 20)
    reference = _reference(11, 5)
    (write,) = products.bind_outputs(((reference, "ab" * 32, device),))
    observed = torch.empty(1, dtype=torch.long, device=device)

    with torch.cuda.stream(producer):
        torch.cuda._sleep(50_000_000)
        products.publish_write(write, torch.tensor([73], dtype=torch.long, device=device))
        products.commit_writes((write,))
    with torch.cuda.stream(consumer):
        read = products.consume(
            reference,
            consumer_op_id=12,
            producer_plan_digest="ab" * 32,
            device=device,
        )
        observed.copy_(read.tensor)
        products.record_reader(read, device=device)

    consumer.synchronize()
    assert observed.item() == 73


def test_reuse_waits_until_every_consumer_stream_retires() -> None:
    device = torch.device("cuda:0")
    producer = torch.cuda.Stream(device=device)
    delayed_consumer = torch.cuda.Stream(device=device)
    products = DeviceProducts(capacity=1, byte_capacity=1 << 20)
    first = _reference(21, 8)
    (write,) = products.bind_outputs(((first, "cd" * 32, device),))
    with torch.cuda.stream(producer):
        products.publish_write(write, torch.tensor([17], dtype=torch.long, device=device))
        products.commit_writes((write,))
    with torch.cuda.stream(delayed_consumer):
        read = products.consume(first, consumer_op_id=22, device=device)
        observed = read.tensor.clone()
        torch.cuda._sleep(50_000_000)
        products.record_reader(read, device=device)

    producer.synchronize()
    products.release_operation(first.request_key, first.producer_op_id)
    second = _reference(23, 9)
    with pytest.raises(ResourceError, match="no query-ready free generation"):
        products.bind_outputs(((second, "ef" * 32, device),))

    delayed_consumer.synchronize()
    (second_write,) = products.bind_outputs(((second, "ef" * 32, device),))
    with torch.cuda.stream(producer):
        products.publish_write(
            second_write,
            torch.tensor([29], dtype=torch.long, device=device),
        )
        products.commit_writes((second_write,))
    producer.synchronize()

    assert observed.item() == 17
    assert products.consume(second, consumer_op_id=24, device=device).tensor.item() == 29
