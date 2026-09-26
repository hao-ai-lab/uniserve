from __future__ import annotations

import os

import pytest
import torch

from tests.python.fixtures.device_storage import (
    DeviceStorage,
    install_device_storage,
)
from uniserve.runtime.device import device_storage_budget

pytestmark = pytest.mark.unit


def test_device_budget_charges_storage_held_outside_the_caching_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The process holds 20_000 bytes on the device, of which the caching
    # allocator reserved only 5_000; another process holds 40_000.
    storage = DeviceStorage(total=100_000, free=40_000)
    storage.held = 20_000
    storage.processes[os.getpid() + 1] = 40_000
    install_device_storage(monkeypatch, storage)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 5_000)

    grant, free = device_storage_budget("cuda:0", 0.5)

    assert grant == 30_000
    assert free == 40_000


def test_device_budget_never_exceeds_physical_free_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DeviceStorage(total=100_000, free=10_000)
    storage.held = 5_000
    install_device_storage(monkeypatch, storage)

    grant, free = device_storage_budget("cuda:0", 0.8)

    assert grant == 10_000
    assert free == 10_000


@pytest.mark.parametrize("reported", [{}, {os.getpid(): None}])
def test_device_budget_refuses_a_process_nvml_does_not_account(
    monkeypatch: pytest.MonkeyPatch, reported: dict[int, int | None]
) -> None:
    storage = DeviceStorage(total=100_000, free=40_000)
    storage.processes = {os.getpid() + 1: 40_000, **reported}
    install_device_storage(monkeypatch, storage)

    with pytest.raises(RuntimeError, match="NVML reports no device storage"):
        device_storage_budget("cuda:0", 0.5)
