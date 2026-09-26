"""One CUDA device whose storage a test controls.

Storage grants read a device's capacity and free bytes from CUDA and each
process's usage from NVML's per-process accounting. Both are environmental
boundaries, which ``install_device_storage`` replaces with a
``DeviceStorage`` the test mutates, so no CUDA context is created.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace

import pynvml
import pytest
import torch

DEVICE_UUID = uuid.UUID("5f0c1a2b-3c4d-4e5f-8a9b-0c1d2e3f4a5b")


@dataclass
class DeviceStorage:
    """Capacity, free bytes and NVML per-process usage of one device.

    Attributes:
        total: Device capacity in bytes.
        free: Bytes the device reports free.
        processes: Device bytes NVML reports per process ID; ``None`` stands
            for a device that does not report per-process usage.
    """

    total: int
    free: int
    processes: dict[int, int | None] = field(default_factory=dict)

    @property
    def held(self) -> int | None:
        """Bytes NVML reports for the test process."""
        return self.processes.get(os.getpid())

    @held.setter
    def held(self, value: int | None) -> None:
        self.processes[os.getpid()] = value


def install_device_storage(
    monkeypatch: pytest.MonkeyPatch, storage: DeviceStorage
) -> None:
    """Serve every CUDA device and its NVML accounting from ``storage``.

    NVML resolves the device only by the ``GPU-`` UUID CUDA reports for it.
    """
    handle = object()

    def handle_by_uuid(identity: str) -> object:
        if identity != f"GPU-{DEVICE_UUID}":
            raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_FOUND)
        return handle

    def running_processes(device: object) -> list[SimpleNamespace]:
        assert device is handle
        return [
            SimpleNamespace(pid=pid, usedGpuMemory=used)
            for pid, used in storage.processes.items()
        ]

    def properties(device: object) -> SimpleNamespace:
        return SimpleNamespace(
            uuid=str(DEVICE_UUID), total_memory=storage.total
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=None: (storage.free, storage.total),
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByUUID", handle_by_uuid)
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetComputeRunningProcesses", running_processes
    )
