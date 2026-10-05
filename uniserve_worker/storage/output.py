"""Numerical copies into Rust-owned output storage.

The native pool owns capacity, result rows, device fences and CPU readers.
These functions allocate host tensors and submit numerical copies.
"""

from __future__ import annotations

import torch

from uniserve_worker._uniserve_ipc import OutputBuffer, OutputPool

__all__ = ["OutputBuffer", "OutputPool"]


def _allocate(words: int, pinned: bool) -> torch.Tensor:
    return torch.empty(
        words, dtype=torch.int64, device="cpu", pin_memory=pinned
    )


def _tokens(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1).to(dtype=torch.int64)


def _bytes(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.uint8:
        raise ValueError("byte captures require a uint8 tensor")
    return value.detach().contiguous()


def _copy_tokens(host: torch.Tensor, offset: int, value: torch.Tensor) -> None:
    host[offset : offset + value.numel()].copy_(
        value, non_blocking=value.is_cuda
    )


def _copy_bytes(
    host: torch.Tensor, offset: int, value: torch.Tensor
) -> torch.Tensor:
    capture = host.view(torch.uint8)[offset : offset + value.numel()].view(
        value.shape
    )
    capture.copy_(value, non_blocking=value.is_cuda)
    return capture
