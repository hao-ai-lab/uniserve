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

The producer side, `DescriptorGrants`, serves from a native thread of the
`_uniserve_ipc` extension that never takes the GIL. A producer's device work
can wait on the consumers that ask it for descriptors, as a collective waits
for ranks that are still importing their inputs, and its main thread may
meanwhile hold the GIL in a CUDA launch that waits for queue space; a grant
that needed the interpreter would then deadlock the ranks. `fetch` runs on a
consumer's read thread, whose blocking socket calls release the GIL.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
from pathlib import Path

from uniserve_worker._uniserve_ipc import (
    GRANT_GRANTED,
    GRANT_REQUEST_BYTES,
    DescriptorGrants,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import DESCRIPTOR_HANDLE_BYTES

#: A request names one publication, `uuid.uuid4().hex`
#: (`CudaVmmTransport.publish`), whose 32 characters fill it exactly; a reply
#: is the granted byte with one descriptor or the refusal byte.
#: `SOCK_SEQPACKET` delivers each as a single record.
_GRANTED = bytes([GRANT_GRANTED])
_TIMEOUT_SECONDS = 30.0


def _directory() -> Path:
    return Path(tempfile.gettempdir()) / "uniserve-descriptor-grants"


def address_of(endpoint: str) -> Path:
    """Return the socket a given publishing endpoint serves its grants on."""
    return _directory() / f"{endpoint}.sock"


def serve(endpoint: str) -> DescriptorGrants:
    """Serve this address space's publication descriptors to local consumers.

    Descriptors are registered as the producer publishes and withdrawn as the
    publications retire. A registered descriptor stays owned by whoever opened
    it: a direct export belongs to its publication, a pool handle to the pool,
    and the grants only lend them out. `register`, `release` and `close` may
    be called from any thread; once `release` returns, no grant of that
    publication is still being sent, so its owner may close the descriptor.

    Raises:
        OSError: The socket cannot be bound at `address_of(endpoint)`.
    """
    return DescriptorGrants(str(address_of(endpoint)))


def fetch(endpoint: str, publication_id: str) -> int:
    """Receive one publication's descriptor from its producing rank.

    The returned descriptor belongs to this process; the caller closes it once
    the allocation has been imported, which holds its own reference.

    Raises:
        WorkerError: `invalid_descriptor` when the endpoint's socket cannot be
            reached, or when the reply carries no descriptor, as for a
            publication that is not, or no longer, registered.
        OSError: A socket failure after connecting, such as a timeout.
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
        connection.sendall(
            publication_id.encode("ascii").ljust(GRANT_REQUEST_BYTES)
        )
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
