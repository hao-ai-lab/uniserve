"""Registered publications of one address space and their retirement.

A publication is registered when its producer exposes it and retired when
the engine releases its buffer. Nothing here talks to a consumer: readiness
and identity travel inside the published storage, and a consumer says it has
finished by writing its acknowledgment word there. The producing rank sweeps
those words when it retires the publication, so a consumer on another host
retires a product the same way one on this host does.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..foundation.errors import invalid_descriptor, resource_error
from ..protocol.transfer import CudaVmmTransfer, Locator, PosixShmTransfer

Source = TypeVar("Source")


def locator_digest(locator: Locator) -> bytes:
    """Bind a read to the registered view, including its producer fence."""
    encoded = json.dumps(
        locator.to_mapping(),
        sort_keys=True,
        separators=(",", ":"),
        default=bytes.hex,
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def publication_key(locator: Locator) -> bytes:
    """Return the fixed-width key that identifies a publication."""
    handle = locator.transport
    if isinstance(handle, CudaVmmTransfer):
        return handle.publication_id.encode("ascii")
    if isinstance(handle, PosixShmTransfer):
        return hashlib.sha256(handle.name.encode("utf-8")).digest()
    raise invalid_descriptor("process publication requires a shared transport")


@dataclass(slots=True)
class _Publication(Generic[Source]):
    """One registered source and where it stands towards reclamation."""

    locator: Locator
    source: Source
    digest: bytes
    pending: bool  # the producer may still be writing the published bytes
    retired: bool = False
    error: BaseException | None = None
    retirement: Future[None] = field(default_factory=Future)
    reclaiming: bool = False  # physical hand-back to the owner has started


class Publications(Generic[Source]):
    """Own the registered sources of one address space.

    A source is handed back to its owner once the engine has retired the
    publication, the producer has finished writing it, and no consumer the
    head named is still reading it. A consumer claims its word in the
    published storage before its first read and acknowledges after its last,
    so one that never began holds nothing. A producer that fails with unknown
    physical completion keeps its source registered, so nothing reuses storage
    a device may still be writing.
    """

    def __init__(
        self,
        *,
        capacity: int,
        reclaim: Callable[[Source, Future[None]], None],
        drain: Callable[[Source], None],
        settled: Callable[[Source], bool],
    ) -> None:
        if capacity < 1:
            raise ValueError("publication capacity must be positive")

        # The incarnation names this address space in every locator it
        # publishes, so a locator from an earlier life of the process is
        # refused rather than resolved against the wrong owner.
        self.name = f"uniserve-publications-{uuid.uuid4().hex}"
        self._capacity = capacity
        self._reclaim = reclaim
        self._drain = drain
        self._settled = settled
        self._publications: dict[bytes, _Publication[Source]] = {}
        self._lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._error: BaseException | None = None

    def publish(
        self, locator: Locator, source: Source, *, pending: bool = False
    ) -> None:
        """Register an immutable source.

        The producer may still be writing its bytes.
        """
        key = publication_key(locator)
        with self._lock:
            self._reap_locked()
            if self._error is not None:
                raise self._error
            if self._closing or len(self._publications) >= self._capacity:
                raise resource_error(
                    "transport publication capacity is unavailable"
                )
            if key in self._publications:
                raise invalid_descriptor(
                    "publication identity is already registered"
                )
            self._publications[key] = _Publication(
                locator, source, locator_digest(locator), pending
            )

    def source(self, locator: Locator, *, reading: bool = False) -> Source:
        """Borrow a registered source by its exact locator.

        The owner borrows its source through retirement; a reader borrows it
        only while the publication is live, and a failed producer refuses the
        reader as it would have refused a grant.
        """
        with self._lock:
            publication = self._publications.get(publication_key(locator))
            if (
                publication is None
                or publication.digest != locator_digest(locator)
                or (reading and publication.retired)
            ):
                raise invalid_descriptor(
                    "publication is retired, invalid, or belongs to another "
                    "view"
                )
            if reading and publication.error is not None:
                raise resource_error(
                    "publication producer failed before readiness"
                )
            return publication.source

    def complete(
        self,
        locator: Locator,
        *,
        error: BaseException | None = None,
        producer_completed: bool = True,
    ) -> None:
        """Record producer completion or failure.

        A failure with unknown physical completion keeps the source allocation
        registered even after the publication retires.
        """
        if not producer_completed and error is None:
            raise ValueError("unknown producer completion requires a failure")

        with self._lock:
            publication = self._publications[publication_key(locator)]
            publication.pending = not producer_completed
            publication.error = error
            if not producer_completed and not publication.retirement.done():
                publication.retirement.set_exception(error)
            self._reclaim_locked(publication)

    def release(self, locator: Locator) -> Future[None] | None:
        """Retire a publication; its source returns once its readers finish."""
        if locator.transport.endpoint != self.name:
            raise invalid_descriptor(
                "publication release belongs to another endpoint"
            )
        with self._lock:
            publication = self._publications.get(publication_key(locator))
            if publication is not None:
                if publication.locator != locator:
                    raise invalid_descriptor(
                        "publication release changed its registered view"
                    )
                publication.retired = True
                self._reclaim_locked(publication)
                return publication.retirement
        return None

    def retirement(self, locator: Locator) -> Future[None]:
        """Expose the registered publication's physical ownership completion."""
        if locator.transport.endpoint != self.name:
            raise invalid_descriptor("publication belongs to another endpoint")
        with self._lock:
            publication = self._publications.get(publication_key(locator))
            if publication is None or publication.locator != locator:
                raise invalid_descriptor(
                    "publication does not match its registered view"
                )
            return publication.retirement

    def awaiting_acknowledgment(self) -> bool:
        """Report whether a retired source still waits on a consumer's word."""
        with self._lock:
            return any(
                publication.retired
                and not publication.pending
                and not publication.reclaiming
                for publication in self._publications.values()
            )

    def reap(self) -> None:
        """Hand back every retired source no consumer is still reading.

        An acknowledgment is written into the published storage and reaches
        this process with no notification, so the producer sweeps here.
        """
        with self._lock:
            for publication in tuple(self._publications.values()):
                if publication.retired and not publication.reclaiming:
                    self._reclaim_locked(publication)
            self._reap_locked()

    def _reclaim_locked(self, publication: _Publication[Source]) -> None:
        """Hand a source back once retired, complete and no longer read."""
        if (
            publication.retired
            and not publication.pending
            and not publication.reclaiming
            and self._settled(publication.source)
        ):
            publication.reclaiming = True
            try:
                self._reclaim(publication.source, publication.retirement)
            except BaseException as error:
                if not publication.retirement.done():
                    publication.retirement.set_exception(error)
                raise

    def _reap_locked(self) -> None:
        """Forget retired publications whose owners took their storage back."""
        for key, publication in tuple(self._publications.items()):
            if (
                publication.reclaiming
                and publication.retirement.done()
                and publication.retirement.exception() is None
            ):
                del self._publications[key]

    def close(self) -> None:
        """Retire every publication and surface unresolved ownership."""
        if not self._closed:
            with self._lock:
                self._closing = True
                for publication in tuple(self._publications.values()):
                    publication.retired = True
                    self._reclaim_locked(publication)
            # Reclamation whose physical completion is still outstanding is
            # drained synchronously: the owner is closing.
            for publication in tuple(self._publications.values()):
                if publication.reclaiming and not publication.retirement.done():
                    self._drain(publication.source)
            with self._lock:
                self._reap_locked()
            self._closed = True

        if self._error is not None:
            raise self._error
        if self._publications:
            raise resource_error(
                "publication lost producer completion or acknowledgment; "
                "sources remain retained"
            )
