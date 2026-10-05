"""Allocate host input tensors for the native buffer ring.

Rust owns slot selection and waits for outstanding copies before buffer reuse
or release. Python supplies the tensor shape, dtype and pinned allocation.
"""

from __future__ import annotations

import torch

from uniserve_worker._uniserve_ipc import HostBuffers

__all__ = ["HostBuffers"]


def _allocate(
    shape: tuple[int, ...] | int,
    dtype: torch.dtype,
    depth: int,
    pinned: bool,
) -> list[torch.Tensor]:
    return [
        torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pinned)
        for _ in range(depth)
    ]
