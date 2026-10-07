"""Native execution streams and numerical views of SM partitions."""

import torch

from uniserve_worker._uniserve_ipc import CUDAStream as CUDAStream

from .cuda import CUDAError


def partition_streams(
    device: torch.device,
    sm_counts: tuple[int, ...],
    *,
    event_slots: int | tuple[int, ...] = 2,
) -> tuple[CUDAStream, ...]:
    """Allocate disjoint SM partitions and expose their PyTorch stream views."""
    if not sm_counts:
        return ()
    if device.type != "cuda":
        raise CUDAError("execution lanes require a CUDA device")

    torch.cuda.init()
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    slots = (
        (event_slots,) * len(sm_counts)
        if isinstance(event_slots, int)
        else event_slots
    )
    return tuple(CUDAStream.partition(index, sm_counts, slots))


__all__ = ["CUDAStream", "partition_streams"]
