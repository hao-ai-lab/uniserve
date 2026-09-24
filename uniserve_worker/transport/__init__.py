"""Construct bounded transport backends for one worker endpoint.

The transport package moves a rank's published products (tensors) to the
ranks that read them. Each backend is one physical mechanism:

- `local`: a publication read within the producer's own address space.
- `shm`: host bytes in a POSIX shared-memory segment, for readers on the
  producer's host.
- `cuda_vmm`: device storage exported as a CUDA VMM shareable handle, either
  where it lies or copied into the device's pool.
- `channel`: host bytes carried inside the locator itself, through the head,
  for readers on another host.

`publication.publish_tensor` chooses the mechanisms for one product,
`fetch.fetch_tensor` assembles a consumer's region from the published
locations, and `exports` operates on each storage owner's record of what it
has published.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from uniserve.runtime import EventPool
from uniserve_worker.errors import (
    invalid_descriptor,
    unsupported_setup,
)
from uniserve_worker.protocol.transfer import WorkerEndpoint
from uniserve_worker.transport.channel import ChannelTransport
from uniserve_worker.transport.cuda_vmm import CudaVmmTransport
from uniserve_worker.transport.interface import TRANSPORTS, Transport
from uniserve_worker.transport.local import LocalTransport
from uniserve_worker.transport.pool import TransferCapacity
from uniserve_worker.transport.shm import ShmTransport


def make_transports(
    names: Sequence[str],
    *,
    byte_capacity: int,
    ticket_capacity: int,
    event_pool: EventPool,
    source: WorkerEndpoint | None = None,
    acknowledgment_slot: int = 0,
    host_slots: Sequence[int] = (),
    cross_host_consumers: bool = False,
) -> dict[str, Transport]:
    """Construct configured backends against one rank resource budget.

    `acknowledgment_slot` is this rank's own word, which it writes in the
    header of every chunk or segment it reads. `host_slots` are the words of
    the ranks on this host, which decide whether a host product's consumers
    are reached over shared storage or over the rank channel.
    `cross_host_consumers` says whether a rank on another host reads this
    rank's products, which decides how a device publication states
    readiness. Which ranks read a given product is stated on the call that
    produces it.

    Raises:
        WorkerError: `invalid_descriptor` for empty, duplicate or unknown
            names, and `unsupported_setup` for a non-positive capacity. When
            a backend fails to construct, the ones already built are closed
            and its error propagates.
    """
    if not names or len(set(names)) != len(names):
        raise invalid_descriptor(
            "transport bindings must be nonempty and unique"
        )
    if any(name not in TRANSPORTS for name in names):
        raise invalid_descriptor(
            f"unknown transport binding; expected names from {TRANSPORTS}"
        )
    if min(byte_capacity, ticket_capacity) < 1:
        raise unsupported_setup(
            "transport byte and ticket capacities must be positive"
        )
    # One budget shared by every backend: bytes or read tickets reserved by
    # one backend are unavailable to the others.
    capacity = TransferCapacity(byte_capacity, ticket_capacity)
    endpoint = source or WorkerEndpoint.local()
    constructors: Mapping[str, Callable[..., Transport]] = {
        "local": LocalTransport,
        "shm": ShmTransport,
        "cuda_vmm": CudaVmmTransport,
        "channel": ChannelTransport,
    }
    transports: dict[str, Transport] = {}
    try:
        for name in names:
            arguments = {
                "capacity": capacity,
                "event_pool": event_pool,
                "source": endpoint,
            }
            if name == "cuda_vmm":
                arguments["acknowledgment_slot"] = acknowledgment_slot
                arguments["cross_host_consumers"] = cross_host_consumers
            if name == "shm":
                arguments["acknowledgment_slot"] = acknowledgment_slot
                arguments["host_slots"] = host_slots
            if name == "channel":
                arguments["host_slots"] = host_slots
            transports[name] = constructors[name](**arguments)
    except BaseException:
        # A partial construction must not leave the threads, sockets or
        # endpoint registrations of the backends already built behind.
        for transport in transports.values():
            transport.close()
        raise
    return transports
