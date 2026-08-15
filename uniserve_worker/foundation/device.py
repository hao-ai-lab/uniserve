"""Host-to-device staging primitives for integer sidecar tensors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..foundation.torch_compat import torch_is_compiling

_np: Any | None
try:  # Optional fast host packing path; minimal environments may not carry numpy.
    import numpy as _numpy_module
except Exception:  # pragma: no cover - availability depends on worker image.
    _np = None
else:  # pragma: no cover
    _np = _numpy_module

__all__ = [
    "canonical_device",
    "copy_cpu_to_device",
    "cpu_int_staging_buffer",
    "fill_cpu_ints",
    "is_pinned",
    "torch_is_compiling",
]

_NUMPY_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


def canonical_device(device: torch.device | str) -> torch.device:
    """Resolve a bare ``cuda`` device to the current explicit CUDA index."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def is_pinned(tensor: torch.Tensor) -> bool:
    return bool(getattr(tensor, "is_pinned", lambda: False)())


def cpu_int_staging_buffer(
    numel: int,
    *,
    dtype: torch.dtype,
    pin: bool,
) -> torch.Tensor:
    """Allocate a CPU integer staging tensor with pinned storage when available."""
    if pin and not torch_is_compiling():
        try:
            return torch.empty(int(numel), dtype=dtype, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(int(numel), dtype=dtype)


def fill_cpu_ints(cpu: torch.Tensor, values: Sequence[int]) -> None:
    """Bulk-fill a CPU integer tensor from a Python int sequence.

    One bulk host->host conversion for every size: a Python list of a few
    hundred ints converts in a single C call, which is cheaper than the N
    individual tensor ``__setitem__`` ATen ops a scalar loop issues on the
    per-forward staging path.
    """
    if len(values) == 0:
        return
    if _np is not None and cpu.dtype in _NUMPY_DTYPES:
        cpu[: len(values)].copy_(
            torch.from_numpy(_np.asarray(values, dtype=_NUMPY_DTYPES[cpu.dtype]))
        )
        return
    cpu[: len(values)].copy_(torch.as_tensor(values, dtype=cpu.dtype))


def copy_cpu_to_device(
    cpu: torch.Tensor,
    *,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    """Copy a staged CPU tensor to ``device``."""
    device = canonical_device(device)
    return cpu.to(device=device, non_blocking=non_blocking)
