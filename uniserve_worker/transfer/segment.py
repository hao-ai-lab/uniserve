"""Layout of a shared-memory publication's segment.

A host product published over shared memory carries everything a consumer
needs inside the segment itself, so that no connection to the producer is
required: the publication's identity, its readiness, and one acknowledgment
word per instance rank. The words are written and read with release and
acquire ordering, because the readiness word announces the payload written
before it and the two live in different processes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from .._uniserve_ipc import atomic_load_u32, atomic_store_u32
from ..foundation.errors import resource_error
from .vmm_pool import ACK_WORD_BYTES, MAX_ACKNOWLEDGMENT_SLOTS

#: Byte offset and length of the locator digest that identifies the segment.
DIGEST_OFFSET = 0
DIGEST_BYTES = 32
#: Byte offset of the readiness word.
STATE_OFFSET = 32
#: Readiness word values: the producer is still writing, the bytes are
#: readable, or the producer failed and they never will be.
PENDING = 0
READY = 1
FAILED = 2
#: Byte offset of the first acknowledgment word.
ACK_OFFSET = 64
#: Bytes of the header, after which the payload begins; a multiple of the
#: alignment every payload dtype needs.
HEADER_BYTES = 512
#: How long a consumer waits for a pending publication before failing.
READINESS_TIMEOUT_S = 120.0

assert ACK_OFFSET + ACK_WORD_BYTES * MAX_ACKNOWLEDGMENT_SLOTS <= HEADER_BYTES


def ack_offset(slot: int) -> int:
    """Return the byte offset of one rank's acknowledgment word."""
    if not 0 <= slot < MAX_ACKNOWLEDGMENT_SLOTS:
        raise ValueError(f"acknowledgment slot {slot} is out of range")
    return ACK_OFFSET + slot * ACK_WORD_BYTES


def initialize(buffer: memoryview, digest: bytes) -> None:
    """Write a fresh header: this digest, pending, no acknowledgments."""
    buffer[DIGEST_OFFSET : DIGEST_OFFSET + DIGEST_BYTES] = digest
    for slot in range(MAX_ACKNOWLEDGMENT_SLOTS):
        atomic_store_u32(buffer, ack_offset(slot), 0)
    atomic_store_u32(buffer, STATE_OFFSET, PENDING)


def digest(buffer: memoryview) -> bytes:
    """Return the digest the producer wrote."""
    return bytes(buffer[DIGEST_OFFSET : DIGEST_OFFSET + DIGEST_BYTES])


def set_state(buffer: memoryview, state: int) -> None:
    """Announce readiness or failure, after every payload byte."""
    atomic_store_u32(buffer, STATE_OFFSET, state)


def state(buffer: memoryview) -> int:
    """Read the readiness word."""
    return atomic_load_u32(buffer, STATE_OFFSET)


def await_ready(
    buffer: memoryview,
    *,
    timeout: float = READINESS_TIMEOUT_S,
    check: Callable[[], None] | None = None,
) -> None:
    """Wait until the producer announces the payload, failing on its failure.

    The wait spins briefly, because a device-to-host publication usually
    completes within microseconds of the consumer arriving, and then sleeps
    so a long wait costs no core. `check` runs on every turn so a cancelled
    read stops waiting as soon as it is cancelled.
    """
    deadline = time.monotonic() + timeout
    pause = 0.0
    while True:
        if check is not None:
            check()
        current = state(buffer)
        if current == READY:
            return
        if current == FAILED:
            raise resource_error("publication producer failed before readiness")
        if time.monotonic() >= deadline:
            raise resource_error(
                "publication endpoint was lost before readiness"
            )
        if pause:
            time.sleep(pause)
        pause = min(1e-3, pause + 5e-5)


def acknowledge(buffer: memoryview, slot: int) -> None:
    """Write this rank's acknowledgment word, after its reads of the payload."""
    atomic_store_u32(buffer, ack_offset(slot), 1)


def acknowledged(buffer: memoryview, slots: Sequence[int]) -> bool:
    """Report whether every named consumer has acknowledged the segment."""
    return all(atomic_load_u32(buffer, ack_offset(slot)) for slot in slots)
