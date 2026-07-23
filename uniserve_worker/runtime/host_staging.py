"""Shared host->device staging primitives for integer sidecar tensors.

Both H2D staging seams — token/position staging and the `KvStore` block-table
and sequence-length staging — need the same four
mechanics: canonical CUDA device resolution, pinned-host buffer acquisition
(optionally recycled through a stager slot), a bulk CPU fill that avoids
per-element ATen ``__setitem__`` calls, and a device copy that reuses a slot's
device buffer when one is available. These are dtype-parameterized here so the
two seams share one implementation instead of drifting copies.
"""
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
    slot: Any | None = None,
    name: str = "buffer",
) -> torch.Tensor:
    """Acquire a CPU staging tensor, recycling through ``slot`` when present.

    ``slot`` supplies reusable host and device integer buffers.
    (``int_buffer`` for int32; the concrete ``TextTensorStagingSlot`` also
    provides ``long_buffer`` for int64).
    """
    if slot is not None:
        if dtype == torch.long:
            return slot.long_buffer(name, numel, pin=pin)
        return slot.int_buffer(name, numel, pin=pin)
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
    slot: Any | None,
    name: str,
) -> torch.Tensor:
    """Copy a staged CPU tensor to ``device``, reusing a slot device buffer."""
    device = canonical_device(device)
    if slot is None or device.type != "cuda":
        return cpu.to(device=device, non_blocking=non_blocking)
    out = slot.device_buffer(name, int(cpu.numel()), dtype=cpu.dtype, device=device)
    out.copy_(cpu, non_blocking=non_blocking)
    return out
