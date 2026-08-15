"""Device identity and host integer staging for physical resource owners."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

__all__ = [
    "canonical_device",
    "cpu_int_staging_buffer",
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


def canonical_device(device: torch.device | str) -> torch.device:
    """Resolve a bare ``cuda`` device to the current explicit CUDA index."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def cpu_int_staging_buffer(
    numel: int,
    *,
    dtype: torch.dtype,
    pin: bool,
) -> torch.Tensor:
    """Allocate a CPU integer staging tensor with pinned storage when available."""
    if pin:
        try:
            return torch.empty(int(numel), dtype=dtype, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(int(numel), dtype=dtype)


def fill_cpu_ints(cpu: torch.Tensor, values: Sequence[int]) -> None:
    """Bulk-fill a CPU integer tensor from a Python int sequence."""
    if len(values) == 0:
        return
    if _np is not None and cpu.dtype in _NUMPY_DTYPES:
        cpu[: len(values)].copy_(
            torch.from_numpy(_np.asarray(values, dtype=_NUMPY_DTYPES[cpu.dtype]))
        )
        return
    cpu[: len(values)].copy_(torch.as_tensor(values, dtype=cpu.dtype))
