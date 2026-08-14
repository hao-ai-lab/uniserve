from __future__ import annotations

import time
from multiprocessing import shared_memory
from threading import Event

import pytest
import torch

from uniserve_worker.foundation.errors import ResourceError, WorkerError
from uniserve_worker.foundation.product_transfer import (
    MAX_TRANSFER_DESCRIPTOR_BYTES,
    TRANSFER_DESCRIPTOR_PREFIX,
)
from uniserve_worker.runtime.transfer import (
    ShmTransport,
    _BoundedTransferPool,
    decode_transfer_descriptor,
    encode_transfer_descriptor,
)


def test_transfer_descriptor_round_trips_exact_canonical_provenance() -> None:
    digest = "a" * 64
    for kind in ("tensor", "kv", "latent"):
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
            "tensor", {"payload": "x" * MAX_TRANSFER_DESCRIPTOR_BYTES}, digest
        )


def test_transfer_entry_capacity_is_nonblocking_and_reclaimable() -> None:
    release = Event()
    pool = _BoundedTransferPool(
        workers=1,
        capacity=2,
        byte_capacity=16,
        name="transfer-entry-test",
    )

    def blocked(value: int) -> torch.Tensor:
        assert release.wait(timeout=5.0)
        return torch.tensor([value])

    try:
        first = pool.submit(blocked, 1, nbytes=8)
        second = pool.submit(blocked, 2, nbytes=8)
        with pytest.raises(ResourceError):
            pool.submit(blocked, 3, nbytes=1)
        release.set()
        deadline = time.monotonic() + 5.0
        while not (first.ready() and second.ready()) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert first.ready() and second.ready()
        assert first.result().tolist() == [1]
        assert second.result().tolist() == [2]
        successor = pool.submit(blocked, 3, nbytes=8)
        deadline = time.monotonic() + 5.0
        while not successor.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert successor.ready()
        assert successor.result().tolist() == [3]
    finally:
        release.set()
        pool.close()


def test_transfer_byte_capacity_is_atomic_and_reclaimable() -> None:
    release = Event()
    pool = _BoundedTransferPool(
        workers=1,
        capacity=3,
        byte_capacity=16,
        name="transfer-byte-test",
    )

    def blocked() -> torch.Tensor:
        assert release.wait(timeout=5.0)
        return torch.tensor([1])

    try:
        first = pool.submit(blocked, nbytes=12)
        with pytest.raises(ResourceError, match="byte capacity"):
            pool.submit(blocked, nbytes=8)
        release.set()
        deadline = time.monotonic() + 5.0
        while not first.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert first.ready()
        successor = pool.submit(blocked, nbytes=16)
        deadline = time.monotonic() + 5.0
        while not successor.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert successor.ready()
    finally:
        release.set()
        pool.close()


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
