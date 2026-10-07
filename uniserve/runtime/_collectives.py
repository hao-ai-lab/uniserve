"""Native stream communication and process-wide NCCL capture policy."""

import os

from uniserve_worker._uniserve_ipc import GatherPool as GatherPool
from uniserve_worker._uniserve_ipc import NcclCommunicator as NcclCommunicator
from uniserve_worker._uniserve_ipc import (
    StreamCommunication as StreamCommunication,
)


def _forbid_implicit_registration() -> None:
    """Keep NCCL from registering borrowed buffers of captured collectives.

    By default NCCL registers the send and receive buffers of a collective
    captured in a CUDA graph and maps the physical allocations behind them
    into its peers, which then write to them directly on every replay, and
    it reuses a registration for later captures by address. Borrowed buffers
    are caching-allocator storage: with expandable segments a tensor spans
    several 20 MiB physical chunks that the allocator maps, unmaps and hands
    to other tensors, so no registration of it stays valid for the graphs
    that replay it. Where NCCL can import those chunks (fabric handles across
    GB200 hosts, POSIX handles on PCIe hosts), captured collectives on such
    registrations write outside mapped memory. Only ``register_buffers``
    windows, whose storage the stream owns, are addressed by peers.

    NCCL reads the setting when it first enqueues a captured collective, so
    setting it before any capture governs the whole process. The setting is
    process-wide and overrides the environment: no UniServe collective may
    run with it enabled.
    """
    os.environ["NCCL_GRAPH_REGISTER"] = "0"
