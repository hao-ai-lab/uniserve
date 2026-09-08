"""Process endpoints for bounded publication grants and physical read retirement."""

from __future__ import annotations

import hashlib
import json
import selectors
import socket
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..execution.batch import CudaIpcTransfer, Locator, PosixShmTransfer
from ..foundation.errors import invalid_descriptor, resource_error

Source = TypeVar("Source")


def locator_digest(locator: Locator) -> bytes:
    """Bind a grant to the complete registered view, including its producer fence."""

    encoded = json.dumps(
        locator.to_mapping(), sort_keys=True, separators=(",", ":"), default=bytes.hex
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def publication_key(locator: Locator) -> bytes:
    """Return the fixed-width source key carried by the physical reader protocol."""

    handle = locator.transport
    if isinstance(handle, CudaIpcTransfer):
        return handle.publication_id.encode("ascii")
    if isinstance(handle, PosixShmTransfer):
        return hashlib.sha256(handle.name.encode("utf-8")).digest()
    raise invalid_descriptor("process publication requires a shared transport")


def open_reader(locator: Locator) -> socket.socket:
    """Acquire source ownership before opening any shared allocation or readiness signal."""

    handle = locator.transport
    if not isinstance(handle, (CudaIpcTransfer, PosixShmTransfer)):
        raise invalid_descriptor("process read requires a shared transport")
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        connection.connect("\0" + handle.endpoint)
        connection.sendall(publication_key(locator) + locator_digest(locator))
        response = connection.recv(1)
        if not response:
            raise resource_error("publication endpoint was lost before readiness")
        if response == b"F":
            raise resource_error("publication producer failed before readiness")
        if response != b"G":
            raise invalid_descriptor("publication is retired, invalid, or has no reader capacity")
    except OSError as error:
        connection.close()
        raise resource_error("publication endpoint was lost before readiness") from error
    except BaseException:
        connection.close()
        raise
    return connection


def finish_reader(connection: socket.socket) -> None:
    """Acknowledge that the reader has stopped accessing the source allocation."""

    try:
        connection.sendall(b"A")
        response = connection.recv(1)
    except OSError as error:
        raise resource_error("publication endpoint lost reader acknowledgement") from error
    if response != b"D":
        raise resource_error("publication endpoint lost reader acknowledgement")


@dataclass(slots=True)
class _Publication(Generic[Source]):
    locator: Locator
    source: Source
    digest: bytes
    pending: bool
    readers: int = 0
    retired: bool = False
    error: BaseException | None = None
    retirement: Future[None] = field(default_factory=Future)
    reclaiming: bool = False


class PublicationEndpoint(Generic[Source]):
    """Own registered sources and reader grants for one address-space incarnation.

    A grant pins the exact source before the reader opens its allocation. A
    retirement rejects new grants and waits for both producer completion and
    every granted read. A disconnect after a grant preserves the source and
    reports failed retirement; it never proves device completion.
    """

    def __init__(
        self,
        *,
        reader_capacity: int,
        publication_capacity: int,
        reclaim: Callable[[Source, Future[None]], None],
        drain: Callable[[Source], None],
    ) -> None:
        if min(reader_capacity, publication_capacity) < 1:
            raise ValueError("publication and reader capacities must be positive")
        self.name = f"uniserve-read-{uuid.uuid4().hex}"
        self._reader_capacity = reader_capacity
        self._publication_capacity = publication_capacity
        self._reclaim = reclaim
        self._drain = drain
        self._publications: dict[bytes, _Publication[Source]] = {}
        self._lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._lost_readers = 0
        self._error: BaseException | None = None
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._listener.bind("\0" + self.name)
        self._listener.listen(reader_capacity)
        self._listener.setblocking(False)
        self._control_rx, self._control_tx = socket.socketpair()
        self._control_rx.setblocking(False)
        self._control_tx.setblocking(False)
        self._worker = threading.Thread(
            target=self._serve, name="uniserve-publication-readers", daemon=True
        )
        self._worker.start()

    def publish(self, locator: Locator, source: Source, *, pending: bool = False) -> None:
        """Register an immutable source whose producer may still be writing its bytes."""

        key = publication_key(locator)
        with self._lock:
            self._reap_locked()
            if self._error is not None:
                raise self._error
            if self._closing or len(self._publications) >= self._publication_capacity:
                raise resource_error("transport publication capacity is unavailable")
            if key in self._publications:
                raise invalid_descriptor("publication identity is already registered")
            self._publications[key] = _Publication(
                locator, source, locator_digest(locator), pending
            )

    def source(self, locator: Locator) -> Source:
        """Borrow a registered source after this endpoint granted the caller's read."""

        with self._lock:
            return self._publications[publication_key(locator)].source

    def complete(
        self,
        locator: Locator,
        *,
        error: BaseException | None = None,
        producer_completed: bool = True,
    ) -> None:
        """Expose producer readiness or failure to waiting readers.

        A failure with unknown physical completion keeps the source allocation
        registered even after every reader is rejected and publication retires.
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
        try:
            self._control_tx.send(b"R")
        except BlockingIOError:
            # An already queued notification also scans all pending grants.
            pass

    def release(self, locator: Locator) -> Future[None] | None:
        """Revoke publication while retaining already granted physical readers."""

        if locator.transport.endpoint != self.name:
            raise invalid_descriptor("publication release belongs to another endpoint")
        with self._lock:
            publication = self._publications.get(publication_key(locator))
            if publication is not None:
                if publication.locator != locator:
                    raise invalid_descriptor("publication release changed its registered view")
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
                raise invalid_descriptor("publication does not match its registered view")
            return publication.retirement

    def _reclaim_locked(self, publication: _Publication[Source]) -> None:
        if (
            publication.retired
            and not publication.pending
            and publication.readers == 0
            and not publication.reclaiming
        ):
            publication.reclaiming = True
            try:
                self._reclaim(publication.source, publication.retirement)
            except BaseException as error:
                if not publication.retirement.done():
                    publication.retirement.set_exception(error)
                raise

    def _reap_locked(self) -> None:
        """Remove retired metadata after its physical owner has returned the allocation."""

        for key, publication in tuple(self._publications.items()):
            if (
                publication.reclaiming
                and publication.retirement.done()
                and publication.retirement.exception() is None
            ):
                del self._publications[key]

    def close(self) -> None:
        """Drain granted readers, close the endpoint, and surface unresolved ownership."""

        if not self._closed:
            with self._lock:
                self._closing = True
                for publication in tuple(self._publications.values()):
                    publication.retired = True
                    self._reclaim_locked(publication)
            try:
                self._control_tx.send(b"C")
            except BlockingIOError:
                pass
            self._worker.join()
            for publication in tuple(self._publications.values()):
                if publication.reclaiming and not publication.retirement.done():
                    self._drain(publication.source)
            with self._lock:
                self._reap_locked()
            self._control_rx.close()
            self._control_tx.close()
            self._closed = True
        if self._error is not None:
            raise self._error
        if self._lost_readers:
            raise resource_error("reader disconnected without completion; sources remain retained")
        if self._publications:
            raise resource_error("publication lost producer completion; sources remain retained")

    def _serve(self) -> None:
        selector = selectors.DefaultSelector()
        selector.register(self._listener, selectors.EVENT_READ)
        selector.register(self._control_rx, selectors.EVENT_READ)
        clients: dict[socket.socket, tuple[_Publication[Source] | None, bool]] = {}
        closing = False

        def remove(connection: socket.socket) -> None:
            publication, granted = clients.pop(connection)
            if publication is not None:
                with self._lock:
                    if not granted:
                        publication.readers -= 1
                        self._reclaim_locked(publication)
                    else:
                        self._lost_readers += 1
                        if not publication.retirement.done():
                            publication.retirement.set_exception(
                                resource_error("reader disconnected without physical completion")
                            )
            selector.unregister(connection)
            connection.close()

        def grant(connection: socket.socket) -> None:
            publication, granted = clients[connection]
            if publication is None or granted:
                return
            with self._lock:
                if publication.error is not None or (closing and publication.pending):
                    response = b"F"
                elif publication.pending:
                    return
                else:
                    response = b"G"
            try:
                connection.send(response)
            except OSError:
                remove(connection)
            else:
                if response == b"G":
                    clients[connection] = publication, True
                else:
                    remove(connection)

        try:
            while not closing or clients:
                for key, _events in selector.select():
                    connection = key.fileobj
                    if connection is self._control_rx:
                        self._control_rx.recv(4096)
                        if self._closing and not closing:
                            closing = True
                            selector.unregister(self._listener)
                            self._listener.close()
                        for client in tuple(clients):
                            if closing and clients[client][0] is None:
                                remove(client)
                            else:
                                grant(client)
                        continue
                    if connection is self._listener:
                        if closing:
                            continue
                        client, _address = self._listener.accept()
                        client.setblocking(False)
                        if len(clients) >= self._reader_capacity:
                            try:
                                client.send(b"E")
                            except OSError:
                                pass
                            finally:
                                client.close()
                        else:
                            clients[client] = None, False
                            selector.register(client, selectors.EVENT_READ)
                        continue
                    assert isinstance(connection, socket.socket)
                    if connection not in clients:
                        continue
                    publication, granted = clients[connection]
                    try:
                        packet = connection.recv(65)
                    except BlockingIOError:
                        continue
                    except OSError:
                        packet = b""
                    if publication is None and len(packet) == 64 and not closing:
                        with self._lock:
                            publication = self._publications.get(packet[:32])
                            if (
                                publication is not None
                                and not publication.retired
                                and publication.digest == packet[32:]
                            ):
                                publication.readers += 1
                                clients[connection] = publication, False
                            else:
                                publication = None
                        if publication is not None:
                            grant(connection)
                            continue
                    elif publication is not None and granted and packet == b"A":
                        # Reclaim before replying so an observed D includes the
                        # producer's physical release and capacity accounting.
                        with self._lock:
                            publication.readers -= 1
                            self._reclaim_locked(publication)
                        clients[connection] = None, False
                        try:
                            connection.send(b"D")
                        except OSError:
                            pass
                        remove(connection)
                        continue
                    if publication is None and packet:
                        try:
                            connection.send(b"E")
                        except OSError:
                            pass
                    remove(connection)
        except BaseException as error:
            self._error = error
        finally:
            for connection in tuple(clients):
                remove(connection)
            self._listener.close()
            selector.close()
