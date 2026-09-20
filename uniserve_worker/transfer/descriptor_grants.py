"""Grant a producer's shareable descriptors to consumer processes on this host.

A device whose driver exports fabric handles publishes a handle that names the
allocation itself: any process that receives those bytes can import it. A
device that exports POSIX descriptors publishes a file descriptor, which names
an open file of the producing process and means nothing in another one. Such a
descriptor reaches a consumer only through `SCM_RIGHTS` on a Unix-domain
socket, which is what this module provides.

The producing transport serves one socket per address space, named after the
endpoint its locators already carry, so a consumer holding a locator can reach
the grant without any additional field. Every rank of a deployment whose
devices export descriptors runs on one host and is started by one engine, so
they agree on the directory below.
"""

from __future__ import annotations

import array
import os
import socket
import sys
import tempfile
import threading
from pathlib import Path

from ..foundation.errors import invalid_descriptor
from ..protocol.transfer import DESCRIPTOR_HANDLE_BYTES

#: A request names one publication; a reply is either one descriptor or the
#: refusal byte below.
_REQUEST_BYTES = 32
_REFUSED = b"\x00"
_GRANTED = b"\x01"
_TIMEOUT_SECONDS = 30.0


def _directory() -> Path:
    return Path(tempfile.gettempdir()) / "uniserve-descriptor-grants"


def address_of(endpoint: str) -> Path:
    """Return the socket a given publishing endpoint serves its grants on."""
    return _directory() / f"{endpoint}.sock"


class DescriptorGrants:
    """Serve one address space's publication descriptors to local consumers.

    Descriptors are registered as the producer publishes and withdrawn as the
    publications retire. The registered descriptor stays owned by whoever
    opened it: a direct export belongs to its publication, a pool handle to
    the pool, and this table only lends them out.
    """

    def __init__(self, endpoint: str) -> None:
        self._descriptors: dict[str, int] = {}
        self._lock = threading.Lock()
        self._closed = False
        self.address = address_of(endpoint)
        self.address.parent.mkdir(parents=True, exist_ok=True)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._listener.bind(str(self.address))
        self._listener.listen(64)
        self._thread = threading.Thread(
            target=self._serve,
            name="uniserve-descriptor-grants",
            daemon=True,
        )
        self._thread.start()

    def register(self, publication_id: str, descriptor: int) -> None:
        """Make one publication's descriptor grantable."""
        with self._lock:
            self._descriptors[publication_id] = descriptor

    def release(self, publication_id: str) -> None:
        """Withdraw a publication; later requests for it are refused."""
        with self._lock:
            self._descriptors.pop(publication_id, None)

    def close(self) -> None:
        """Stop serving and remove the socket."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._descriptors.clear()
        self._listener.close()
        self.address.unlink(missing_ok=True)

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return  # the listener closed
            with connection:
                try:
                    self._reply(connection)
                except OSError:
                    continue

    def _reply(self, connection: socket.socket) -> None:
        connection.settimeout(_TIMEOUT_SECONDS)
        request = connection.recv(_REQUEST_BYTES)
        if len(request) != _REQUEST_BYTES:
            connection.sendall(_REFUSED)
            return
        with self._lock:
            descriptor = self._descriptors.get(
                request.decode("ascii", "ignore")
            )
        if descriptor is None:
            connection.sendall(_REFUSED)
            return
        connection.sendmsg(
            [_GRANTED],
            [
                (
                    socket.SOL_SOCKET,
                    socket.SCM_RIGHTS,
                    bytes(_fd_array(descriptor)),
                )
            ],
        )


def _fd_array(descriptor: int) -> bytes:
    return array.array("i", [descriptor]).tobytes()


def fetch(endpoint: str, publication_id: str) -> int:
    """Receive one publication's descriptor from its producing rank.

    The returned descriptor belongs to this process; the caller closes it once
    the allocation has been imported, which holds its own reference.
    """
    address = address_of(endpoint)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        connection.settimeout(_TIMEOUT_SECONDS)
        try:
            connection.connect(str(address))
        except OSError as error:
            raise invalid_descriptor(
                "CUDA publication endpoint serves no descriptor grants"
            ) from error
        connection.sendall(publication_id.encode("ascii").ljust(_REQUEST_BYTES))
        payload, descriptors, _flags, _address = socket.recv_fds(
            connection, 1, 1
        )
        if payload != _GRANTED or len(descriptors) != 1:
            for descriptor in descriptors:
                os.close(descriptor)
            raise invalid_descriptor(
                "CUDA publication has no live descriptor to grant"
            )
        return descriptors[0]
    finally:
        connection.close()


def descriptor_bytes(descriptor: int) -> bytes:
    """Encode a descriptor the way an exported handle carries one."""
    return descriptor.to_bytes(DESCRIPTOR_HANDLE_BYTES, sys.byteorder)
