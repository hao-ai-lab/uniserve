"""Numerical views and copies of CUDA virtual storage."""

from __future__ import annotations

from functools import cache
from itertools import groupby, repeat
from typing import TYPE_CHECKING

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import CudaVmmTransfer, Locator
from uniserve_worker.transport.layout import copy_pairs
from uniserve_worker.transport.vmm_pool import ACK_WORD_BYTES

if TYPE_CHECKING:
    import torch


@cache
def _can_access_peer(device: str, peer: str) -> bool:
    """Return the process-stable CUDA peer relation for two visible devices."""
    import torch

    return torch.cuda.can_device_access_peer(
        torch.device(device), torch.device(peer)
    )


def _export_spans(source: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
    """Require device spans with one allocation and a common stride."""
    spans = source if isinstance(source, tuple) else (source,)
    first = spans[0]
    if not first.is_cuda:
        raise invalid_descriptor("cuda_vmm transport requires a CUDA tensor")
    if any(
        span.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
        or span.stride() != first.stride()
        for span in spans
    ):
        raise invalid_descriptor(
            "CUDA VMM export spans require one allocation and stride"
        )


def _copy_export(
    source: torch.Tensor | tuple[torch.Tensor, ...],
    destination: torch.Tensor,
) -> None:
    """Copy only the logical spans into a reserved pool chunk."""
    for target, value in copy_pairs(source, destination):
        target.copy_(value, non_blocking=True)


def _export_layout(
    source: torch.Tensor | tuple[torch.Tensor, ...], storage_offset: int
) -> dict[str, tuple[int, ...]]:
    """Describe physical spans relative to the exported allocation."""
    spans = source if isinstance(source, tuple) else (source,)
    first = spans[0]
    lengths = tuple(
        (length, sum(1 for _ in values))
        for length, values in groupby(int(span.shape[0]) for span in spans)
    )
    return {
        "storage_offsets_bytes": tuple(
            storage_offset + span.data_ptr() - first.data_ptr()
            for span in spans
        ),
        "span_lengths": tuple(length for length, _ in lengths),
        "span_counts": tuple(count for _, count in lengths),
        "tensor_stride": tuple(first.stride()),
    }


def _import_views(
    locator: Locator,
    destination: torch.Tensor | tuple[torch.Tensor, ...],
    device: torch.device,
    exported: bytes,
    slot: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    """Map a granted handle and bind its numerical spans and reader word.

    Every returned view retains the same allocation. The caller owns the
    descriptor through import and the mapping through all device accesses.
    """
    import torch
    from uniserve_kernels.peer_storage import import_handle

    handle = locator.transport
    assert isinstance(handle, CudaVmmTransfer)
    prototype = (
        destination[0] if isinstance(destination, tuple) else destination
    )
    itemsize = prototype.element_size()
    if any(offset % itemsize for offset in handle.storage_offsets_bytes):
        raise invalid_descriptor("CUDA VMM span offset is not element aligned")
    if prototype.device != device:
        prototype = torch.empty(0, dtype=prototype.dtype, device=device)

    allocation = import_handle(prototype, exported, handle.storage_size_bytes)
    lengths = (
        length
        for length, count in zip(
            handle.span_lengths, handle.span_counts, strict=True
        )
        for length in repeat(length, count)
    )
    mapped = tuple(
        allocation.as_strided(
            (length, *locator.shape[1:]),
            handle.tensor_stride,
            byte_offset // itemsize,
        )
        for byte_offset, length in zip(
            handle.storage_offsets_bytes, lengths, strict=True
        )
    )
    acknowledgment = None
    if handle.acknowledgment_offset >= 0:
        start = handle.acknowledgment_offset + slot * ACK_WORD_BYTES
        acknowledgment = allocation.view(torch.uint8)[
            start : start + ACK_WORD_BYTES
        ].view(torch.int32)
    return mapped, acknowledgment
