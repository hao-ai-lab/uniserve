from __future__ import annotations

from multiprocessing import shared_memory

import pytest
import torch

from uniserve_worker.foundation.errors import ResourceError, WorkerError
from uniserve_worker.foundation.product_transfer import (
    MAX_TRANSFER_DESCRIPTOR_BYTES,
    TRANSFER_DESCRIPTOR_PREFIX,
)
from uniserve_worker.transfer.tickets import (
    LocalTransport,
    ShmTransport,
    decode_transfer_descriptor,
    encode_transfer_descriptor,
)


def test_transfer_descriptor_round_trips_exact_canonical_provenance() -> None:
    digest = "a" * 64
    for kind in ("encoder", "device_product", "kv", "latent"):
        encoded = encode_transfer_descriptor(kind, {"height": 16, "width": 24}, digest)
        assert decode_transfer_descriptor(encoded) == (
            kind,
            {"height": 16, "width": 24},
            digest,
        )


def test_transfer_descriptor_rejects_noncanonical_or_unbounded_frames() -> None:
    digest = "b" * 64
    canonical = encode_transfer_descriptor("kv", {"snapshot": {}}, digest)
    noncanonical = canonical.replace(b'"kind":"kv"', b'"kind": "kv"')
    oversized = TRANSFER_DESCRIPTOR_PREFIX + b"{" + b" " * MAX_TRANSFER_DESCRIPTOR_BYTES

    with pytest.raises(WorkerError, match="not canonical"):
        decode_transfer_descriptor(noncanonical)
    with pytest.raises(WorkerError, match="descriptor bound"):
        decode_transfer_descriptor(oversized)
    with pytest.raises(WorkerError, match="descriptor bound"):
        encode_transfer_descriptor(
            "device_product", {"payload": "x" * MAX_TRANSFER_DESCRIPTOR_BYTES}, digest
        )


def test_local_transport_capacity_is_atomic_and_reclaimable() -> None:
    transport = LocalTransport(byte_capacity=16)
    try:
        first = transport.publish(torch.arange(3, dtype=torch.float32))
        with pytest.raises(ResourceError, match="byte capacity"):
            transport.publish(torch.arange(2, dtype=torch.float32))
        transport.release(first)
        successor = transport.publish(torch.arange(4, dtype=torch.float32))
        torch.testing.assert_close(transport.fetch(successor), torch.arange(4, dtype=torch.float32))
    finally:
        transport.close()


def test_shm_async_publication_round_trips_cpu_storage() -> None:
    transport = ShmTransport(byte_capacity=1 << 20, ticket_capacity=256)
    source = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    try:
        locator = transport.publish_async(source)
        actual = transport.fetch(locator)
        torch.testing.assert_close(actual, source, rtol=0, atol=0)
        transport.release(locator)
    finally:
        transport.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_shm_async_publication_exposes_an_inflight_ticket_and_exact_snapshot() -> None:
    transport = ShmTransport(byte_capacity=1 << 20, ticket_capacity=256)
    source = torch.zeros((1024,), dtype=torch.float32, device="cuda")
    torch.cuda._sleep(500_000_000)
    source.add_(1)
    try:
        locator = transport.publish_async(source)
        segment = shared_memory.SharedMemory(name=locator.handle.decode())
        try:
            assert segment.buf[0] == 0
        finally:
            segment.close()
        actual = transport.fetch(locator)
        torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)
        transport.release(locator)
    finally:
        transport.close()
