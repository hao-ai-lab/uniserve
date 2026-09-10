"""CUDA allocation, mapping, and asynchronous host-storage primitives."""

from functools import lru_cache
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
    index = torch.cuda.current_device() if device.index is None else device.index
    return _extension().allocation_granularity(index)


def allocate(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device):
    """Create exactly page-aligned physical storage for one peer's tensor.

    The returned allocation exports a POSIX descriptor and maps an ordered
    descriptor list into one tensor via ``map_peers``. Descriptor transport,
    publication, reuse, and retirement belong to the distributed runtime.
    """

    return _extension().PeerAllocation(torch.empty(0, dtype=dtype, device=device), list(shape))


def export_ipc(tensor: torch.Tensor) -> tuple[bytes, int, int]:
    """Export an allocation handle, byte capacity, and tensor byte offset.

    The caller owns the source and must keep its published contents immutable
    until all mapped readers have completed and closed their mappings.
    """

    return _extension().export_ipc(tensor)


def import_ipc(
    prototype: torch.Tensor,
    handle: bytes,
    allocation_bytes: int,
    byte_offset: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> torch.Tensor:
    """Map a bounded tensor on the prototype's CUDA device and dtype.

    The returned tensor closes its mapping when released. Its transfer lease
    must retain it through the last GPU read and acknowledge the producer only
    after releasing it. This primitive does not synchronize consumer streams.
    """

    return _extension().import_ipc(
        prototype, handle, allocation_bytes, byte_offset, list(shape), list(strides)
    )


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

    Call after native code enqueues a host/device copy outside PyTorch's copy
    operator. The pinned allocator delays storage reuse until this stream retires
    the copy; the owner must leave the submitted contents immutable meanwhile.
    """

    _extension().record_host_usage(tensor, stream.device_index, int(stream.cuda_stream))
