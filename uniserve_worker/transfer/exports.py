"""Calls on transport registrations held by their storage owners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import Future

from ..protocol.identity import BufferId, RequestKey
from ..protocol.transfer import Locator
from .tickets import Transport

ExportLocations = tuple[tuple[Transport, Locator], ...]


def validate_exports(
    resident: Mapping[BufferId, ExportLocations],
    candidates: Mapping[BufferId, ExportLocations],
) -> None:
    """Reject conflicting registrations.

    Registrations are rejected before a completion group's writes commit.
    """
    for buffer, locations in candidates.items():
        existing = resident.get(buffer)
        if existing is not None and existing != locations:
            raise RuntimeError(
                "committed transport publication identity was reused"
            )


def retiring_exports(
    exports: Mapping[BufferId, ExportLocations],
    *,
    buffers: frozenset[BufferId] = frozenset(),
    requests: frozenset[RequestKey] = frozenset(),
    retained: frozenset[BufferId] = frozenset(),
) -> tuple[BufferId, ...]:
    """Select local registrations whose allocation ownership is ending."""
    return tuple(
        buffer
        for buffer in exports
        if buffer in buffers
        or (buffer.owner in requests and buffer not in retained)
    )


def release_exports(
    exports: Mapping[BufferId, ExportLocations],
    retirements: dict[BufferId, tuple[Future[None], ...]],
    buffers: Iterable[BufferId],
) -> tuple[Future[None], ...]:
    """Revoke acquisition and retain the exact physical retirement futures.

    Storage records can outlive request state. Keep registrations and futures
    together in the storage owner until every already admitted reader retires.
    Remote locators from imports are never inserted into this directory.
    """
    pending: list[Future[None]] = []
    for buffer in buffers:
        locations = exports.get(buffer)
        if locations is None:
            continue
        # Retirement futures are recorded once per buffer so repeated releases
        # observe the exact same physical completion, never a fresh revocation.
        if buffer not in retirements:
            retirements[buffer] = tuple(
                future
                for transport, locator in locations
                if (future := transport.release(locator)) is not None
            )
        pending.extend(retirements[buffer])
    return tuple(pending)


def forget_exports(
    exports: dict[BufferId, ExportLocations],
    retirements: dict[BufferId, tuple[Future[None], ...]],
    buffers: Iterable[BufferId],
) -> None:
    """Forget only registrations whose physical release succeeded."""
    for buffer in buffers:
        if buffer not in exports:
            continue
        futures = retirements.get(buffer)
        if futures is None or any(not future.done() for future in futures):
            raise RuntimeError("transport registration has not retired")
        # Propagate any retirement failure before dropping the registration.
        for future in futures:
            future.result()
        del exports[buffer]
        del retirements[buffer]
