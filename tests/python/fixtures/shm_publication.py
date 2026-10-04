"""External shared-storage publisher for physical retirement tests."""

from __future__ import annotations

from multiprocessing import shared_memory

from uniserve_worker.protocol.transfer import (
    Locator,
    PosixShmTransfer,
    WorkerEndpoint,
)
from uniserve_worker.transport import segment


def serve_pending_publication(channel, shape=(1024,)) -> None:
    """Publish a segment that never becomes ready, then fail it on command.

    On "exit" the producer marks the segment failed and unlinks it, as a
    consumer of a failing producer observes.
    """
    nbytes = 4096
    storage = shared_memory.SharedMemory(
        create=True, size=segment.HEADER_BYTES + nbytes
    )
    try:
        locator = Locator(
            source=WorkerEndpoint.local("publisher"),
            transport=PosixShmTransfer(
                endpoint="uniserve-test-pending", name=storage.name
            ),
            nbytes=nbytes,
            dtype="float32",
            shape=shape,
            offset=(0,) * len(shape),
            device="cpu",
        )
        segment.initialize(storage.buf)
        channel.send(locator.to_mapping())
        while True:
            command = channel.recv()
            if command == "settled":
                channel.send(
                    segment.settled(
                        storage.buf, range(segment.MAX_ACKNOWLEDGMENT_SLOTS)
                    )
                )
            elif command == "exit":
                segment.set_state(storage.buf, segment.FAILED)
                break
            else:
                raise ValueError(f"unknown publisher command: {command}")
    finally:
        storage.close()
        storage.unlink()
        channel.close()
