"""Collective ownership of a contiguous virtual tensor over local CUDA peers."""

from __future__ import annotations

import array
import os
import socket
import tempfile

import torch
import torch.distributed as dist

from ..nn.mesh import Communicator


def allocate_peer_tensor(
    group: Communicator,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate one physical shard and map ordered peers into one virtual tensor.

    POSIX descriptors are transferred through Unix-domain sockets using the
    standard SCM_RIGHTS protocol. The existing process group exchanges socket
    addresses and validates matching geometry and a shared host before any
    descriptor transfer. All setup finishes before CUDA graph capture.
    """

    from uniserve_kernel.peer_memory import allocate

    allocation = allocate(shape, dtype=dtype, device=group.device)
    descriptors = [allocation.export_fd()]
    try:
        if group.world_size == 1:
            return allocation.map_peers(descriptors)
        with tempfile.TemporaryDirectory(prefix="uniserve-peer-") as directory:
            address = os.path.join(directory, "memory.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
                channel.bind(address)
                channel.settimeout(dist.constants.default_pg_timeout.total_seconds())
                identity = (socket.gethostname(), address, shape, str(dtype))
                identities: list[tuple[str, str, tuple[int, ...], str] | None] = [
                    None
                ] * group.world_size
                dist.all_gather_object(identities, identity, group=group._require())
                if any(
                    peer is None or (peer[0], peer[2], peer[3]) != (identity[0], shape, str(dtype))
                    for peer in identities
                ):
                    raise ValueError("peer tensors require matching geometry on one host")
                backend_ranks = sorted(group.ranks)
                ordered = [identities[backend_ranks.index(rank)] for rank in group.ranks]
                destination = ordered[(group.rank_in_group + 1) % group.world_size]
                assert destination is not None
                owner_descriptors = {group.rank_in_group: descriptors[0]}
                current = descriptors[0]
                for hop in range(1, group.world_size):
                    channel.sendmsg(
                        [b"v"],
                        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [current]))],
                        0,
                        destination[1],
                    )
                    _payload, received, _flags, _address = socket.recv_fds(channel, 1, 1)
                    descriptors.extend(received)
                    if len(received) != 1:
                        raise RuntimeError("peer allocation exchange requires one descriptor")
                    current = received[0]
                    owner_descriptors[(group.rank_in_group - hop) % group.world_size] = current
                return allocation.map_peers(
                    [owner_descriptors[owner] for owner in range(group.world_size)]
                )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
