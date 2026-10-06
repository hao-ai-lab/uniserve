"""Numerical packing of rank-channel payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniserve.profiling import profile_range
from uniserve_worker.transport.layout import copy_pairs, resolve_dtype

if TYPE_CHECKING:
    import torch


def _export_payload(
    source: torch.Tensor | tuple[torch.Tensor, ...],
    shape: tuple[int, ...],
) -> memoryview:
    """Copy physical tensor values into a contiguous host payload."""
    import torch

    first = source[0] if isinstance(source, tuple) else source
    packed = torch.empty(shape, dtype=first.dtype, device="cpu")
    for target, value in copy_pairs(source, packed):
        target.copy_(value)

    if first.is_cuda:
        # The bytes leave with the result; a downstream reader cannot wait
        # on the producer's stream.
        with profile_range("channel_export_synchronize"):
            torch.cuda.current_stream(first.device).synchronize()

    with profile_range("channel_export_payload"):
        return memoryview(packed.flatten().view(torch.uint8).numpy())


def _allocate_payload(
    shape: tuple[int, ...], dtype: str, device: torch.device
) -> tuple[torch.Tensor, memoryview]:
    """Allocate a private host tensor and expose its writable bytes to Rust."""
    import torch

    with profile_range("channel_fetch_payload"):
        # The copy stream retains pinned storage until its DMA completes.
        payload = torch.empty(
            shape,
            dtype=resolve_dtype(dtype),
            device="cpu",
            pin_memory=device.type == "cuda",
        )
        return payload, memoryview(payload.flatten().view(torch.uint8).numpy())
