"""Startup readiness checks the process's device footprint against its grant."""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.device_storage import (
    DeviceStorage,
    install_device_storage,
)
from tests.python.fixtures.worker_config import stub_worker_config
from uniserve_worker.bootstrap.capacity import check_startup_storage
from uniserve_worker.errors import WorkerError, WorkerErrorCode
from uniserve_worker.storage.buffer_pool import BufferPool
from uniserve_worker.storage.tensor_store import TensorStore

pytestmark = pytest.mark.unit

# A 0.9 share of a 100_000-byte device grants 90_000 bytes.
TOTAL_BYTES = 100_000
FRACTION = 0.9


def _startup(monkeypatch: pytest.MonkeyPatch, *, held: int, reserved: int):
    storage = DeviceStorage(total=TOTAL_BYTES, free=TOTAL_BYTES - held)
    storage.held = held
    install_device_storage(monkeypatch, storage)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: reserved)

    config = stub_worker_config(max_batch_tokens=8).replace(
        device="cuda:0",
        kv_storage_fraction=FRACTION,
    )
    # No product is resident yet, so every product byte is still pending.
    store = TensorStore(
        capacity=1,
        buffer_pool=BufferPool(byte_capacity=16, devices=("cpu",)),
    )
    return config, store


def test_startup_refuses_storage_held_outside_the_caching_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The caching allocator's 60_000 bytes and the 5_000 pending product
    # bytes fit the grant; everything the process holds does not.
    config, store = _startup(monkeypatch, held=88_000, reserved=60_000)

    with pytest.raises(WorkerError) as refused:
        check_startup_storage(config, 5_000, store)

    assert refused.value.code == WorkerErrorCode.UNSUPPORTED_SETUP
    assert "requires 93000 bytes" in str(refused.value)
    assert "of a 90000-byte grant" in str(refused.value)


def test_startup_accepts_a_footprint_that_leaves_room_for_pending_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store = _startup(monkeypatch, held=85_000, reserved=60_000)

    check_startup_storage(config, 5_000, store)
