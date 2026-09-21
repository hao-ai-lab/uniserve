"""Device identity and host integer staging for physical resource owners."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

__all__ = [
    "canonical_device",
    "fill_cpu_bools",
    "fill_cpu_ints",
]

_np: Any | None
try:
    import numpy as _numpy_module
except Exception:
    _np = None
else:
    _np = _numpy_module

_NUMPY_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


def device_memory_budget(
    device: torch.device | str, fraction: float
) -> tuple[int, int]:
    """Return this process's pool grant and the device's total free bytes.

    The configured fraction is a per-process device share. This distinction
    matters when independent WorkerGroups share a GPU: each process may size
    its own pool up to its share, while the physical free-memory bound keeps
    their aggregate allocations honest.

    Empty cached allocations before measuring so ``memory_reserved`` describes
    live tensors and runtime allocations owned by this process. CUDA-library
    allocations that PyTorch does not track remain covered by the physical
    free-memory bound.
    """
    target = canonical_device(device)
    if target.type != "cuda" or not 0 < fraction <= 1:
        raise ValueError(
            "device memory sizing requires CUDA and a fraction in (0, 1]"
        )

    torch.cuda.synchronize(target)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(target)
    process_bytes = torch.cuda.memory_reserved(target)
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
