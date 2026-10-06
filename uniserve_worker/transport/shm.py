"""Numerical tensor copies to and from native shared storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniserve_worker._uniserve_ipc import (
    SHM_HEADER_BYTES,
    SharedBuffer,
    SharedRead,
)
from uniserve_worker.protocol.transfer import Locator
from uniserve_worker.transport.layout import copy_pairs, resolve_dtype

if TYPE_CHECKING:
    import torch


def _export_payload(
    source: torch.Tensor | tuple[torch.Tensor, ...],
    shape: tuple[int, ...],
    storage: SharedBuffer,
    stream: torch.cuda.Stream | None,
) -> None:
    """Copy spans into the producer's mapping on its selected stream."""
    import torch

    first = source[0] if isinstance(source, tuple) else source
    packed = torch.frombuffer(
        memoryview(storage)[SHM_HEADER_BYTES:], dtype=first.dtype
    ).reshape(shape)
    if stream is None:
        for target, value in copy_pairs(source, packed):
            target.copy_(value)
        return

    from uniserve_kernels.peer_storage import copy_host_device

    for target, value in copy_pairs(source, packed):
        copy_host_device(target, value, stream)
        value.record_stream(stream)


def _copy_payload(
    read: SharedRead, locator: Locator, device: torch.device
) -> torch.Tensor:
    """Copy mapped bytes into private host storage, pinned for device DMA.

    The caller holds the source claim through this synchronous host copy.
    The returned tensor owns its storage independently of the mapping.
    """
    import torch

    view = torch.frombuffer(read, dtype=resolve_dtype(locator.dtype)).reshape(
        locator.shape
    )
    source = torch.empty(
        view.shape, dtype=view.dtype, pin_memory=device.type == "cuda"
    )
    source.copy_(view)
    return source
