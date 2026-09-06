"""Virtual tensor views over runtime-owned CUDA peer allocations."""

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
