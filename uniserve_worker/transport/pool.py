"""Numerical copies and errors for the native bounded transfer pool.

Rust owns rank credits, read reservations, task submission, stream ordering,
cancellation, and physical retirement. These helpers copy borrowed tensor
views and supply the immutable pinned words used by the VMM protocol.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from uniserve_worker._uniserve_ipc import (
    ReadReservation as ReadReservation,
)
from uniserve_worker._uniserve_ipc import (
    TransferCapacity as TransferCapacity,
)
from uniserve_worker._uniserve_ipc import (
    TransferPool as TransferPool,
)
from uniserve_worker.errors import ResourceError
from uniserve_worker.transport.layout import copy_pairs

if TYPE_CHECKING:
    import torch


class ReadBackpressureError(ResourceError):
    """Too few of the rank's read tickets are free for a fetch right now.

    A ticket returns when its read physically retires, which happens once the
    read's copy drains, whatever its submitter does next. A caller therefore
    retries after a return of ``capacity``'s tickets: ``returns`` is its
    return count when the fetch was refused, which
    `TransferCapacity.notify_reads_returned` compares against so that no
    return between the refusal and the request for a notification is missed.
    """

    def __init__(
        self,
        message: str,
        *,
        capacity: TransferCapacity,
        returns: int,
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.capacity = capacity
        self.returns = returns


@cache
def chunk_word(state: int) -> torch.Tensor:
    """Return the pinned host word a consumer writes into a chunk's header.

    A claim precedes the consumer's first read of the chunk and an
    acknowledgment follows its last, so a producing rank sweeping a retired
    publication can tell a consumer that is still reading from one that never
    began.

    One tensor is cached per state value and shared by every thread and read
    in the process, so it is only ever a copy source and must never be
    written.
    """
    import torch

    return torch.full((1,), state, dtype=torch.int32).pin_memory()


def _copy_tensors(
    source: torch.Tensor | tuple[torch.Tensor, ...],
    destination: torch.Tensor | tuple[torch.Tensor, ...],
    stream: torch.cuda.Stream | None,
) -> None:
    """Copy matching spans on the caller's already-selected device stream."""
    for target, value in copy_pairs(source, destination):
        if stream is None:
            target.copy_(value)
        elif value.device.type == "cpu":
            from uniserve_kernels.peer_storage import copy_host_device

            copy_host_device(target, value, stream)
        else:
            target.copy_(value, non_blocking=True)
