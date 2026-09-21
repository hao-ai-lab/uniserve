"""Collective ownership of a contiguous virtual tensor over local CUDA peers."""

from __future__ import annotations

import array
import os
import socket
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from uniserve.distributed.mesh import Communicator


@dataclass(frozen=True)
class SymmetricMemory:
    """Runtime-owned allocation with ordered peer views.

    Runtime-owned allocation with peer views ordered by logical membership.
    """

    coordinator: Communicator
    local: torch.Tensor
    peers: tuple[torch.Tensor, ...]
    handle: Any

    @property
    def rank(self) -> int:
        return self.coordinator.rank

    @property
    def size(self) -> int:
        return self.coordinator.size

    def fence(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """Publish arrival on the allocation's group with stream ordering."""
        if tuple(input.shape) != (1,) or tuple(output.shape) != (self.size,):
            raise ValueError(
                "symmetric-memory fence buffers do not match group membership"
            )
        self.coordinator._all_gather_into_tensor(output, input)


def _peer_identities(
    group: Communicator,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    address: str,
) -> list[tuple[str, str, tuple[int, ...], str]]:
    """Exchange and validate what each rank of the group is allocating.

    Peers map one another's physical memory, so every rank must be on one host
    and allocating the same shape. The result is ordered by the group's ranks.
    """
    identity = (socket.gethostname(), address, shape, str(dtype))
    identities: list[tuple[str, str, tuple[int, ...], str] | None] = [
        None
    ] * group.size
    dist.all_gather_object(identities, identity, group=group._require())
    if any(
        peer is None
        or (peer[0], peer[2], peer[3]) != (identity[0], shape, str(dtype))
        for peer in identities
    ):
        raise ValueError("peer tensors require matching shapes on one host")

    backend_ranks = sorted(group.ranks)
    ordered = [identities[backend_ranks.index(rank)] for rank in group.ranks]
    assert all(peer is not None for peer in ordered)
    return ordered  # type: ignore[return-value]


def _gathered_handles(
    group: Communicator,
    handle: bytes,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> list[bytes]:
    """Collect every owner's fabric handle, in rank order.

    A fabric handle names memory rather than a file this process holds open, so
    it is meaningful in any process that receives it and travels as an ordinary
    payload.
    """
    _peer_identities(group, shape, dtype, "")
    handles: list[bytes | None] = [None] * group.size
    dist.all_gather_object(handles, handle, group=group._require())
    backend_ranks = sorted(group.ranks)
    ordered = [handles[backend_ranks.index(rank)] for rank in group.ranks]
    if any(peer is None for peer in ordered):
        raise ValueError("a peer reported no allocation handle")
    return ordered  # type: ignore[return-value]


@contextmanager
def _rotated_descriptors(
    group: Communicator,
    handle: bytes,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> Iterator[list[bytes]]:
    """Hold every owner's descriptor open, in rank order, for the body.

    Where a device exports a POSIX descriptor rather than a fabric handle, the
    handle names a file this process holds open, so it reaches a peer only
    through `SCM_RIGHTS` on a Unix-domain socket. Descriptors rotate around the
    logical rank ring: each hop a rank forwards the descriptor it just received
    to its successor and accepts its predecessor's, so after `size - 1` hops
    every rank holds one descriptor per owner. A descriptor is only meaningful
    while it is open, so the collected list is yielded rather than returned and
    the ring closes once the caller has imported from it. All setup finishes
    before CUDA graph capture.
    """
    width = len(handle)
    received_descriptors: list[int] = []
    with tempfile.TemporaryDirectory(prefix="uniserve-peer-") as directory:
        address = os.path.join(directory, "memory.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.bind(address)
            channel.settimeout(
                dist.constants.default_pg_timeout.total_seconds()
            )
            ordered = _peer_identities(group, shape, dtype, address)
            destination = ordered[(group.rank + 1) % group.size]

            try:
                current = int.from_bytes(handle, sys.byteorder)
                owners = {group.rank: handle}
                for hop in range(1, group.size):
                    channel.sendmsg(
                        [b"v"],
                        [
                            (
                                socket.SOL_SOCKET,
                                socket.SCM_RIGHTS,
                                array.array("i", [current]),
                            )
                        ],
                        0,
                        destination[1],
                    )
                    _payload, received, _flags, _address = socket.recv_fds(
                        channel, 1, 1
                    )
                    if len(received) != 1:
                        raise RuntimeError(
                            "peer allocation exchange requires one descriptor"
                        )
                    received_descriptors.extend(received)
                    current = received[0]
                    owners[(group.rank - hop) % group.size] = current.to_bytes(
                        width, sys.byteorder
                    )
                # The descriptors stay open for the caller: each one names a
                # file this process holds, and mapping a peer's shard reads it.
                yield [owners[owner] for owner in range(group.size)]
            finally:
                for descriptor in received_descriptors:
                    os.close(descriptor)


def allocate_peer_tensor(
    group: Communicator,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate one physical shard and map ordered peers into one tensor.

    Every rank allocates its own shard and exports a handle to it; each rank
    then maps every owner's shard, in rank order, into one virtual tensor. How
    a handle reaches a peer depends on what the device exports, which is what
    the two collection paths below distinguish.
    """
    from uniserve_kernel.peer_memory import allocate, exports_fabric_handles

    allocation = allocate(shape, dtype=dtype, device=group.device)
    handle = allocation.export_handle()
    device = torch.device(group.device)
    fabric = exports_fabric_handles(device.index or 0)
    try:
        if group.size == 1:
            return allocation.map_peers([handle])
        if fabric:
            return allocation.map_peers(
                _gathered_handles(group, handle, shape, dtype)
            )
        with _rotated_descriptors(group, handle, shape, dtype) as descriptors:
            return allocation.map_peers(descriptors)
    finally:
        # A descriptor export opens a file this process owns, and the mapping
        # keeps its own reference to the physical memory. Without this close,
        # every peer allocation would leak one descriptor for the lifetime of
        # the worker. A fabric handle opens nothing.
        if not fabric:
            os.close(int.from_bytes(handle, sys.byteorder))


def allocate_collective_buffer(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Own VMM backing suitable for NCCL registration, without mapping peers.

    Collective communicators register their own windows. Transfers do not
    require Python-visible peer tensors or a second rendezvous communicator.
    """
    import torch.distributed._symmetric_memory as symm_mem

    if symm_mem.get_backend(device) != "NCCL":
        symm_mem.set_backend("NCCL")
    return symm_mem.empty(shape, dtype=dtype, device=device)


def allocate_symmetric_memory(
    group: Communicator,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> SymmetricMemory:
    """Allocate stable peer views in logical rank order.

    The caller retains the handle.
    """
    peers: tuple[torch.Tensor, ...]
    if group.size == 1:
        local = torch.empty(shape, dtype=dtype, device=group.device)
        handle = None
        peers = (local,)
    else:
        if dist.get_backend(group._require()) != "nccl":
            raise RuntimeError(
                "symmetric peer memory requires the NCCL backend"
            )
        import torch.distributed._symmetric_memory as symm_mem

        local = allocate_collective_buffer(
            shape, dtype=dtype, device=group.device
        )
        handle = symm_mem.rendezvous(local, group._require())
        backend_ranks = sorted(group.ranks)
        peers = tuple(
            handle.get_buffer(backend_ranks.index(rank), shape, dtype)
            for rank in group.ranks
        )
    workspace = SymmetricMemory(group, local, peers, handle)
    return workspace
