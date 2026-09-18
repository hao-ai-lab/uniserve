"""Single-backend transport construction for transfer tests.

Production code always binds a rank's complete backend set at once through
``make_transports``. Tests that exercise one backend in isolation use this
helper so the convenience form stays out of the worker's public surface.
"""

from __future__ import annotations

from collections.abc import Sequence

from uniserve.runtime import EventPool
from uniserve_worker.protocol.transfer import WorkerEndpoint
from uniserve_worker.transfer.tickets import (
    Transport,
    TransportKind,
    make_transports,
)


def make_transport(
    name: str | TransportKind,
    *,
    byte_capacity: int,
    ticket_capacity: int,
    event_pool: EventPool,
    source: WorkerEndpoint | None = None,
    consumers: Sequence[int] = (),
    acknowledgment_slot: int = 0,
) -> Transport:
    """Construct one bounded physical backend for a standalone endpoint."""
    return make_transports(
        (str(name),),
        byte_capacity=byte_capacity,
        ticket_capacity=ticket_capacity,
        event_pool=event_pool,
        source=source,
        consumers=consumers,
        acknowledgment_slot=acknowledgment_slot,
    )[str(name)]
