from __future__ import annotations

import pytest
import torch

from uniserve.runtime.device import device_storage_budget

pytestmark = pytest.mark.unit


def _cuda_storage(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free: int,
    total: int,
    process_reserved: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (free, total)
    )
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda device: process_reserved
    )


def test_device_budget_charges_only_the_current_worker_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_storage(
        monkeypatch, free=40_000, total=100_000, process_reserved=20_000
    )

    grant, free = device_storage_budget("cuda:0", 0.5)

    assert grant == 30_000
    assert free == 40_000


def test_device_budget_never_exceeds_physical_free_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_storage(
        monkeypatch, free=10_000, total=100_000, process_reserved=5_000
    )

    grant, free = device_storage_budget("cuda:0", 0.8)

    assert grant == 10_000
    assert free == 10_000
