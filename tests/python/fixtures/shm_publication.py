"""Controlled external shared-memory publisher for physical retirement tests."""

from __future__ import annotations

import socket
import uuid
from multiprocessing import shared_memory

from uniserve_worker.protocol.transfer import (
    Locator,
    PosixShmTransfer,
    WorkerEndpoint,
)


def serve_pending_publication(channel, shape=(1024,)) -> None:
    """Exit on command before permitting a registered reader to access bytes."""
    storage = shared_memory.SharedMemory(create=True, size=4096)
    endpoint = f"uniserve-test-pending-{uuid.uuid4().hex}"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
            listener.bind("\0" + endpoint)
            listener.listen(1)
            locator = Locator(
                source=WorkerEndpoint.local("publisher"),
                transport=PosixShmTransfer(
                    endpoint=endpoint, name=storage.name
                ),
                nbytes=4096,
                dtype="float32",
                shape=shape,
                offset=(0,) * len(shape),
                device="cpu",
            )
            channel.send(locator.to_mapping())
            connection, _address = listener.accept()
            with connection:
                assert len(connection.recv(128)) == 64
                channel.send("pending")
                assert channel.recv() == "exit"
    finally:
        storage.close()
        storage.unlink()
        channel.close()
