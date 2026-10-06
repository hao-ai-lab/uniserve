"""Numerical packing of rank-channel payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from uniserve.profiling import profile_range
from uniserve_worker.protocol.transfer import ChannelTransfer, Locator
from uniserve_worker.transport.layout import copy_pairs, resolve_dtype

if TYPE_CHECKING:
    import torch


def _export_payload(
    source: torch.Tensor | tuple[torch.Tensor, ...],
    shape: tuple[int, ...],
) -> bytes:
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
        return bytes(packed.flatten().view(torch.uint8).numpy())


def _copy_payload(locator: Locator, device: torch.device) -> torch.Tensor:
    """Make a private tensor from channel bytes, pinned for device DMA."""
    import torch

    handle = cast(ChannelTransfer, locator.transport)
    with profile_range("channel_fetch_payload"):
        payload = torch.frombuffer(
            bytearray(handle.payload),
            dtype=resolve_dtype(locator.dtype),
        ).reshape(locator.shape)
        if device.type != "cuda":
            return payload

        # The copy stream needs pinned storage until its DMA completes.
        carried = torch.empty_like(payload, pin_memory=True)
        carried.copy_(payload)
        return carried
