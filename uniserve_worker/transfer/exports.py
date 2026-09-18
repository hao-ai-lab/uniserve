"""Calls on transport registrations held by their storage owners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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
    buffers: Iterable[BufferId],
) -> None:
    """Revoke acquisition of the named buffers' registrations.

    Revocation rejects new readers. The publication's physical retirement runs
    on the owning transport: a device chunk returns to its pool once every
    consumer has acknowledged it, and the storage owner refuses to reclaim a
    write while any publication it retained is still live. Remote locators from
    imports are never inserted into this directory.
    """
    for buffer in buffers:
        locations = exports.get(buffer)
        if locations is None:
            continue
        for transport, locator in locations:
            transport.release(locator)


def forget_exports(
    exports: dict[BufferId, ExportLocations],
    buffers: Iterable[BufferId],
) -> None:
    """Forget the named buffers' registrations after their revocation."""
    for buffer in buffers:
        exports.pop(buffer, None)
