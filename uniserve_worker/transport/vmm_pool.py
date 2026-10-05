"""Numerical storage for native VMM export pools.

Rust owns chunk placement, producer dependencies and asynchronous reader
retirement. These helpers allocate exportable backing and borrow tensor views.
"""

from __future__ import annotations

import torch

from uniserve_worker._uniserve_ipc import PoolChunk, PoolExhaustedError, VmmPool

__all__ = ["PoolChunk", "PoolExhaustedError", "VmmPool"]

# Remote readers address their own word within the exported chunk header.
ACK_WORD_BYTES = 4
MAX_ACKNOWLEDGMENT_SLOTS = 64
HEADER_BYTES = ACK_WORD_BYTES * MAX_ACKNOWLEDGMENT_SLOTS
UNCLAIMED = 0
CLAIMED = 1
ACKNOWLEDGED = 2


def _allocate(
    device: torch.device, capacity: int
) -> tuple[torch.Tensor, bytes]:
    from uniserve_kernels.peer_storage import allocate, allocation_granularity

    page = allocation_granularity(device)
    capacity = ((capacity + page - 1) // page) * page
    allocation = allocate((capacity,), dtype=torch.uint8, device=device)
    return allocation.map_local(), allocation.export_handle()


def _views(
    storage: torch.Tensor, offset: int, nbytes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = offset + HEADER_BYTES
    return (
        storage[payload : payload + nbytes],
        storage[offset:payload].view(torch.int32),
    )
