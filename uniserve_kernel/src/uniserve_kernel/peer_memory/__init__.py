"""CUDA allocation, mapping, and asynchronous host-storage primitives."""

from functools import lru_cache
from math import prod
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    return load(
        "uniserve_peer_memory",
        sources=[str(Path(__file__).parent / "csrc" / "peer_memory.cpp")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_ldflags=["-lcuda"],
        with_cuda=True,
    )


def allocation_granularity(device: torch.device) -> int:
    """Return the CUDA device allocation granularity in bytes."""
    if device.type != "cuda":
        raise ValueError("peer tensor mappings require CUDA")
    index = (
        torch.cuda.current_device() if device.index is None else device.index
    )
    return _extension().allocation_granularity(index)


def allocate(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
):
    """Create exactly page-aligned physical storage for one peer's tensor.

    The returned allocation exports a POSIX descriptor and maps an ordered
    descriptor list into one tensor via ``map_peers``. Descriptor transport,
    publication, reuse, and retirement belong to the distributed runtime.
    """
    return _extension().PeerAllocation(
        torch.empty(0, dtype=dtype, device=device), list(shape)
    )


def empty(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Allocate exportable CUDA storage.

    Physical backing is rounded up to CUDA pages. The tensor owns its
    allocation and mapping. Its owner must retain it until all local device
    accesses and remote grants have retired. Logical shape and storage size
    remain distinct; views retain the complete physical backing.
    """
    elements = prod(shape)
    if elements == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    itemsize = torch.empty((), dtype=dtype).element_size()
    page_bytes = allocation_granularity(device)
    nbytes = ((elements * itemsize + page_bytes - 1) // page_bytes) * page_bytes
    owner = allocate((nbytes // itemsize,), dtype=dtype, device=device)
    storage = owner.map_local()
    return storage[:elements].view(shape)


def export_handle(tensor: torch.Tensor) -> tuple[bytes, int, int] | None:
    """Export shared storage as a shareable handle, byte capacity and offset.

    The handle is the device's probed type: a fabric handle where the driver
    exports one, which another host inside the fabric domain can import, and a
    process descriptor otherwise. Both travel as bytes so one publication
    shape carries either.

    Return None for storage without exportable physical backing. The caller
    retains the source through every reader grant. Other CUDA failures are
    raised.
    """
    return _extension().export_handle(tensor)


def import_handle(
    prototype: torch.Tensor, exported: bytes, allocation_bytes: int
) -> torch.Tensor:
    """Map a granted allocation on the prototype device as a flat typed tensor.

    The returned tensor retains the imported physical handle. Retain its
    mapping until all device reads complete, then release it before
    acknowledging the source grant. This primitive does not synchronize
    consumer streams.
    """
    return _extension().import_handle(prototype, exported, allocation_bytes)


def copy_host_device(
    destination: torch.Tensor, source: torch.Tensor, stream: torch.cuda.Stream
) -> None:
    """Enqueue an exact strided copy between pinned host and CUDA storage.

    The caller owns both views through stream completion. Shape and dtype must
    agree; this primitive performs no conversion or GPU packing allocation.
    Destination elements must be disjoint, as validated by the transfer owner.
    """
    device = source.device if source.is_cuda else destination.device
    if device != stream.device:
        raise ValueError("host/device copy stream belongs to another device")
    _extension().copy_host_device(destination, source, int(stream.cuda_stream))


def record_host_usage(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Retain pinned storage through a native asynchronous DMA submission.

    Call after native code enqueues a host/device copy outside PyTorch's
    copy operator. The pinned allocator delays storage reuse until this
    stream retires the copy; the owner must leave the submitted contents
    immutable meanwhile.
    """
    _extension().record_host_usage(
        tensor, stream.device_index, int(stream.cuda_stream)
    )
