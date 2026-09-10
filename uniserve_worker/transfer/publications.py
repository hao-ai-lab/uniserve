"""Logical publication identities and the lifetime of their transport registrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future

from ..execution.batch import BufferId, Locator, RequestKey
from .tickets import Transport


class TransferPublications:
    """Retain registrations until physical retirement permits their identity to disappear."""

    def __init__(self, transports: Mapping[str, Transport]) -> None:
        self._transports = transports
        self._publications: dict[BufferId, tuple[Locator, ...]] = {}

    def validate(self, publications: Mapping[BufferId, tuple[Locator, ...]]) -> None:
        """Reject a conflicting identity before any cross-resource commit begins."""

        for identity, locators in publications.items():
            existing = self._publications.get(identity)
            if existing is not None and existing != locators:
                raise RuntimeError("committed transport publication identity was reused")

    def commit(self, publications: Mapping[BufferId, tuple[Locator, ...]]) -> None:
        """Publish registrations validated by the caller's resource transaction."""

        self._publications.update(publications)

    def retiring(
        self,
        *,
        buffers: frozenset[BufferId] = frozenset(),
        requests: frozenset[RequestKey] = frozenset(),
        retained: frozenset[BufferId] = frozenset(),
    ) -> tuple[BufferId, ...]:
        """Select registrations whose buffer or owning request is retiring."""

        return tuple(
            buffer
            for buffer in self._publications
            if buffer in buffers or (buffer.owner in requests and buffer not in retained)
        )

    def release(self, buffers: Sequence[BufferId]) -> tuple[Future[None], ...]:
        """Revoke new acquisition while retaining identities through existing readers."""

        return tuple(
            future
            for buffer in buffers
            for locator in self._publications.get(buffer, ())
            if (future := self._transports[locator.backend].release(locator)) is not None
        )

    def forget(self, buffers: Sequence[BufferId]) -> None:
        """Remove registrations after their physical retirement has been observed."""

        for buffer in buffers:
            self._publications.pop(buffer, None)

    def drop_request(self, request_id: int, retained: frozenset[BufferId]) -> tuple[BufferId, ...]:
        """Release the remaining registrations when a request's execution state is dropped."""

        selected = tuple(
            buffer
            for buffer in self._publications
            if int(buffer.owner.request_id) == request_id and buffer not in retained
        )
        self.release(selected)
        self.forget(selected)
        return selected

    def clear(self) -> None:
        """Discard identities after the owning transports have stopped."""

        self._publications.clear()
