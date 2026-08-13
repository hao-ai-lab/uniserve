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
from uniserve_worker.runtime.product_store import DeviceProductTable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _reference(op_id: int, generation: int, *, elements: int = 1) -> ProductRef:
    return ProductRef(
        request_key=RequestKey(1, 9, 2),
        producer_op_id=op_id,
        output_index=0,
        generation=generation,
        kind=ProductKind.TOKEN,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U32,
        shape_bound=(ShapeBound() if elements == 1 else ShapeBound((DeviceDim(elements),))),
        point_range=PointRange(),
    )


def test_consumer_submission_precedes_producer_host_observation() -> None:
    device = torch.device("cuda:0")
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    table = DeviceProductTable(capacity=1, byte_capacity=1 << 20)
    reference = _reference(11, 5)
    table.bind_outputs(((reference, "ab" * 32, device),))
    source = torch.tensor([73], dtype=torch.long, device=device)
    observed = torch.empty_like(source)

    with torch.cuda.stream(producer):
        torch.cuda._sleep(1_000_000_000)
        table.publish(reference, source)
        producer_done = torch.cuda.Event(blocking=False)
        producer_done.record(producer)

    with torch.cuda.stream(consumer):
        read = table.consume(
            reference,
            consumer_op_id=12,
            producer_plan_digest="ab" * 32,
            device=device,
        )
        observed.copy_(read.tensor)
        table.record_reader(read, device=device)

    assert not producer_done.query()
    consumer.synchronize()
    assert observed.item() == 73


def test_slot_reuse_waits_for_every_recorded_reader_event() -> None:
    device = torch.device("cuda:0")
    producer = torch.cuda.Stream(device=device)
    fast_consumer = torch.cuda.Stream(device=device)
    delayed_consumer = torch.cuda.Stream(device=device)
    table = DeviceProductTable(capacity=1, byte_capacity=1 << 20)
    first = _reference(21, 8)
    table.bind_outputs(((first, "cd" * 32, device),))

    with torch.cuda.stream(producer):
        table.publish(first, torch.tensor([17], dtype=torch.long, device=device))
    with torch.cuda.stream(fast_consumer):
        fast_read = table.consume(first, consumer_op_id=22, device=device)
        fast_observed = fast_read.tensor.clone()
        table.record_reader(fast_read, device=device)
    with torch.cuda.stream(delayed_consumer):
        delayed_read = table.consume(first, consumer_op_id=23, device=device)
        delayed_observed = delayed_read.tensor.clone()
        torch.cuda._sleep(50_000_000)
        table.record_reader(delayed_read, device=device)

    producer.synchronize()
    fast_consumer.synchronize()
    table.release_operation(first.request_key, first.producer_op_id)
    second = _reference(23, 9)
    with pytest.raises(Exception, match="no query-ready free generation"):
        table.bind_outputs(((second, "ef" * 32, device),))

    delayed_consumer.synchronize()
    table.bind_outputs(((second, "ef" * 32, device),))
    with torch.cuda.stream(producer):
        table.publish(second, torch.tensor([29], dtype=torch.long, device=device))
    producer.synchronize()

    assert fast_observed.item() == 17
    assert delayed_observed.item() == 17
    assert table.consume(second, consumer_op_id=24, device=device).tensor.item() == 29


def test_batched_producer_writes_registered_scalar_storage_directly() -> None:
    configured_device = torch.device("cuda")
    device = torch.device("cuda", torch.cuda.current_device())
    table = DeviceProductTable(capacity=8, byte_capacity=1 << 20)
    references = tuple(_reference(31 + index, 11 + index) for index in range(4))
    writes = table.bind_outputs(
        tuple((reference, "ab" * 32, configured_device) for reference in references)
    )
    producer_batch = table.producer_scalar_batch(writes)
    assert producer_batch is not None
    destination = producer_batch.tensor
    logits = torch.tensor(
        (
            (0.0, 1.0, 4.0, 2.0),
            (8.0, 1.0, 0.0, 2.0),
            (3.0, 7.0, 4.0, 2.0),
            (0.0, 1.0, 2.0, 9.0),
        ),
        device=device,
    )

    torch.argmax(logits, dim=-1, out=destination)
    table.publish_scalar_batch(producer_batch)
    reads = table.consume_batch(
        tuple(
            (reference, 41 + index, "ab" * 32, configured_device)
            for index, reference in enumerate(references)
        )
    )
    table.record_readers(reads, device=configured_device)

    torch.cuda.synchronize(device)
    assert tuple(int(read.tensor.item()) for read in reads) == (2, 0, 1, 3)

def test_row_product_publication_uses_one_completion_event() -> None:
    device = torch.device("cuda:0")
    table = DeviceProductTable(capacity=3, byte_capacity=1 << 20)
    references = tuple(_reference(71 + index, 31 + index, elements=4) for index in range(3))
    writes = table.bind_outputs(tuple((reference, "ab" * 32, device) for reference in references))
    values = torch.tensor(
        (
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
        ),
        dtype=torch.long,
        device=device,
    )

    table.publish_rows(writes, values)
    event = writes[0].producer_event
    assert event is not None
    assert all(write.producer_event is event for write in writes)
    reads = table.consume_batch(
        tuple(
            (reference, 81 + index, "ab" * 32, device) for index, reference in enumerate(references)
        )
    )
    observed = torch.stack(tuple(read.tensor for read in reads))
    table.record_readers(reads, device=device)

    torch.testing.assert_close(observed, values)
