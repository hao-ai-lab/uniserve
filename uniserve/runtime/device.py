"""Device identity, storage accounting and host integer staging."""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from typing import Any

import pynvml
import torch

__all__ = [
    "canonical_device",
    "fill_cpu_bools",
    "fill_cpu_ints",
    "process_device_bytes",
]

_np: Any | None
try:
    import numpy as _numpy_module
except Exception:
    _np = None
else:
    _np = _numpy_module

_NUMPY_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


def process_device_bytes(device: torch.device | str) -> int:
    """Return the storage this process holds on one CUDA device.

    The value is NVML's per-process usage for this process on the device,
    the figure a CUDA out-of-memory report gives as the memory "this process
    has in use". It covers the caching allocator's segments, graph pools
    included, and everything allocated outside that allocator: VMM arenas
    such as product and transfer pools, symmetric-memory and communicator
    buffers, instantiated graph executables, loaded modules and the context.
    The device is synchronized and the allocator's unused cached blocks are
    released first, so reusable cache does not count.

    Raises:
        ValueError: ``device`` is not a CUDA device.
        RuntimeError: NVML attributes no usage on the device to this process.
            Where NVML reports host process IDs, a process in another process
            ID namespace, as in a container without the host's, is not
            found; a device without per-process accounting reports none.
    """
    target = canonical_device(device)
    if target.type != "cuda":
        raise ValueError("process device storage requires a CUDA device")

    torch.cuda.synchronize(target)
    torch.cuda.empty_cache()

    # NVML enumerates physical devices, while CUDA ordinals follow
    # CUDA_VISIBLE_DEVICES, so the device is matched by its UUID.
    identity = uuid.UUID(str(torch.cuda.get_device_properties(target).uuid))
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{identity}")

    pid = os.getpid()
    usage = [
        process.usedGpuMemory
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if process.pid == pid
    ]
    if not usage or any(used is None for used in usage):
        raise RuntimeError(
            f"NVML reports no device storage for process {pid} on {target}; "
            "storage grants are measured from NVML's per-process usage, "
            "which requires the process to run in the process ID namespace "
            "NVML reports"
        )
    return sum(usage)


def device_storage_budget(
    device: torch.device | str, fraction: float
) -> tuple[int, int]:
    """Return this process's remaining grant and the device's free bytes.

    The configured fraction is a per-process device share. This distinction
    matters when independent WorkerGroups share a GPU: each process may size
    its own storage up to its share, while the physical free-storage bound
    keeps their aggregate allocations honest.

    The grant is charged with everything the process already holds on the
    device (``process_device_bytes``), not only the caching allocator's
    reservation: storage the worker allocates outside that allocator and the
    CUDA libraries' own allocations occupy the same device.

    Raises:
        ValueError: ``device`` is not CUDA or ``fraction`` is outside (0, 1].
        RuntimeError: ``process_device_bytes`` cannot attribute usage to
            this process.
    """
    target = canonical_device(device)
    if target.type != "cuda" or not 0 < fraction <= 1:
        raise ValueError(
            "device storage sizing requires CUDA and a fraction in (0, 1]"
        )

    process_bytes = process_device_bytes(target)
    free, total = torch.cuda.mem_get_info(target)
    process_grant = max(0, int(total * fraction) - process_bytes)
    return min(process_grant, free), free


def canonical_device(device: torch.device | str) -> torch.device:
    """Resolve a bare ``cuda`` device to the current explicit CUDA index."""
    if isinstance(device, torch.device) and (
        device.type != "cuda" or device.index is not None
    ):
        return device

    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def fill_cpu_ints(cpu: torch.Tensor, values: Sequence[int]) -> None:
    """Bulk-fill a CPU integer tensor from a Python int sequence."""
    if len(values) == 0:
        return

    if _np is not None and cpu.dtype in _NUMPY_DTYPES:
        cpu[: len(values)].copy_(
            torch.from_numpy(
                _np.asarray(values, dtype=_NUMPY_DTYPES[cpu.dtype])
            )
        )
        return

    cpu[: len(values)].copy_(torch.as_tensor(values, dtype=cpu.dtype))


def fill_cpu_bools(cpu: torch.Tensor, values: Sequence[bool]) -> None:
    """Bulk-fill a CPU boolean tensor from a Python boolean sequence."""
    if cpu.dtype is not torch.bool:
        raise TypeError("boolean staging requires a torch.bool destination")

    if len(values) == 0:
        return

    if _np is not None:
        cpu[: len(values)].copy_(
            torch.from_numpy(_np.asarray(values, dtype="bool"))
        )
        return

    cpu[: len(values)].copy_(torch.as_tensor(values, dtype=torch.bool))
