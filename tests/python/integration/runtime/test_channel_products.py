"""Host products carried on the rank channel's data path."""

from __future__ import annotations

import threading

import pytest
import torch

from tests.python.fixtures.transport import make_transport
from uniserve.runtime import EventPool
from uniserve_worker.protocol.transfer import Locator

pytestmark = [pytest.mark.integration]


def _await_ticket(ticket):
    ready = threading.Event()
    ticket.add_done_callback(ready.set)
    assert ready.wait(30), "channel transfer did not expose a result"
    return ticket.result()


@pytest.fixture
def events() -> EventPool:
    pool = EventPool()
    yield pool
    pool.close()


def test_a_channel_product_carries_its_own_bytes(events: EventPool) -> None:
    """The locator is the product, so it needs nothing of the producer.

    Shared storage names a segment in one host's namespace. A locator that
    carries its bytes reaches a consumer wherever the rank channel does,
    because it travels in the producing rank's result and the consuming rank's
    batch like any other value on that channel.
    """
    transport = make_transport(
        "channel", byte_capacity=1 << 20, ticket_capacity=2, event_pool=events
    )
    try:
        source = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        locator = transport.export(source)

        # A round trip through the wire mapping is the journey the product
        # actually makes: rank, head, rank.
        delivered = Locator.from_mapping(locator.to_mapping())
        # The consumer picks its transport by the mechanism the locator names.
        assert delivered.backend == "channel"
        destination = torch.empty(3, 4, dtype=torch.bfloat16)
        _await_ticket(
            transport.fetch(
                delivered, device=torch.device("cpu"), destination=destination
            )
        )

        assert torch.equal(destination, source)
    finally:
        transport.close()


def test_a_channel_export_leaves_its_producer_nothing_to_retire(
    events: EventPool,
) -> None:
    """The source is the rank's own again as soon as the locator exists.

    Publishing copies the product out of the rank's storage, so no consumer
    can be waiting on that storage and the head owns the bytes from there.
    """
    transport = make_transport(
        "channel", byte_capacity=1 << 20, ticket_capacity=2, event_pool=events
    )
    try:
        locator = transport.export(torch.ones(256, dtype=torch.float32))

        assert transport.retirement(locator).done()
        # Capacity is the buffer, not the product, so publishing again
        # does not need the first export to be released.
        assert transport.export(torch.ones(256, dtype=torch.float32))
    finally:
        transport.close()


def test_a_channel_locator_from_another_endpoint_is_refused(
    events: EventPool,
) -> None:
    """A rank retires only what it published.

    The bytes are indistinguishable once they travel, so the endpoint is what
    identifies whose export a locator is.
    """
    from uniserve_worker.errors import WorkerError

    producer = make_transport(
        "channel", byte_capacity=1 << 20, ticket_capacity=2, event_pool=events
    )
    other = make_transport(
        "channel", byte_capacity=1 << 20, ticket_capacity=2, event_pool=events
    )
    try:
        locator = producer.export(torch.ones(16, dtype=torch.float32))
        with pytest.raises(WorkerError, match="another endpoint"):
            other.release(locator)
    finally:
        producer.close()
        other.close()


def test_a_channel_product_of_media_size_fits_its_export_bound(
    events: EventPool,
) -> None:
    """The bytes are the product, not the handle.

    A media unit's PCM or encoded bytes are far larger than a transfer handle
    may be, and they travel on the rank channel whose message caps bound them;
    the export's handle bound counts the locator, not the payload.
    """
    from uniserve_worker.protocol.batch import TensorExport
    from uniserve_worker.protocol.identity import CallId, RequestKey
    from uniserve_worker.protocol.tensor import (
        DType,
        ShapeBound,
        StaticDim,
        TensorRef,
    )
    from uniserve_worker.protocol.transfer import (
        DeviceProductTransferValue,
        TensorTransfer,
    )

    transport = make_transport(
        "channel", byte_capacity=1 << 21, ticket_capacity=2, event_pool=events
    )
    try:
        samples = 5 * 32000
        pcm = torch.zeros(samples, 2, dtype=torch.int16)
        locator = transport.export(pcm)
        export = TensorExport(
            product=TensorRef(
                request_key=RequestKey(1, 1, 1),
                producer_call_id=CallId(1, 0),
                output_index=0,
                generation=1,
                dtype=DType.I16,
                shape_bound=ShapeBound((StaticDim(samples), StaticDim(2))),
            ),
            value=DeviceProductTransferValue(
                height=0,
                width=0,
                value_range="",
                tensor=TensorTransfer(shape=(samples, 2), locations=(locator,)),
            ),
        )
        assert export.encoded_size_bound() < pcm.numel() * 2
    finally:
        transport.close()


@pytest.mark.gpu
def test_a_channel_product_reaches_a_device_destination(
    events: EventPool,
) -> None:
    """The bytes arrive pageable and still land in a device tensor."""
    transport = make_transport(
        "channel", byte_capacity=1 << 20, ticket_capacity=2, event_pool=events
    )
    try:
        source = torch.arange(96, dtype=torch.int16).reshape(48, 2)
        delivered = Locator.from_mapping(transport.export(source).to_mapping())
        destination = torch.empty(48, 2, dtype=torch.int16, device="cuda:0")
        _await_ticket(
            transport.fetch(
                delivered,
                device=torch.device("cuda:0"),
                destination=destination,
            )
        )
        assert torch.equal(destination.cpu(), source)
    finally:
        transport.close()
