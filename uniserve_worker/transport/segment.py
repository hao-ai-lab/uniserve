"""Layout of a shared storage export's segment.

A host product exported over shared storage carries everything a consumer
needs inside the segment itself, so that no connection to the producer is
required: its readiness and one acknowledgment
word per instance rank. A consumer claims its word when it begins reading
and acknowledges it once its reads are done, so a producer whose
export the engine has retired can tell a consumer still reading from
one that never began: the engine retires an export only after every
consumer's call has resolved or will never be submitted, so no consumer
begins reading after that. The words are written and read with release and
acquire ordering, because the readiness word announces the payload written
before it and the two live in different processes.

`ShmTransport` writes the header as producer and reads it as consumer. The
acknowledgment slot count and word size are shared with the device pool's
chunk header in `vmm_pool`, so one slot numbering serves both mechanisms.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from uniserve_worker._uniserve_ipc import atomic_load_u32, atomic_store_u32
from uniserve_worker.errors import resource_error
from uniserve_worker.transport.vmm_pool import (
    ACK_WORD_BYTES,
    MAX_ACKNOWLEDGMENT_SLOTS,
)

#: Byte offset of the readiness word.
STATE_OFFSET = 0
#: Readiness word values: the producer is still writing, the bytes are
#: readable, or the producer failed and they never will be.
PENDING = 0
READY = 1
FAILED = 2
#: Byte offset of the first acknowledgment word.
ACK_OFFSET = 64
#: Acknowledgment word values: the consumer never began reading, it is
#: reading, or its reads are done.
UNCLAIMED = 0
CLAIMED = 1
ACKNOWLEDGED = 2
#: Bytes of the header, after which the payload begins; a multiple of the
#: alignment every payload dtype needs.
HEADER_BYTES = 512
#: How long a consumer waits for a pending export before failing.
READINESS_TIMEOUT_S = 120.0

assert ACK_OFFSET + ACK_WORD_BYTES * MAX_ACKNOWLEDGMENT_SLOTS <= HEADER_BYTES


def ack_offset(slot: int) -> int:
    """Return the byte offset of one rank's acknowledgment word."""
    if not 0 <= slot < MAX_ACKNOWLEDGMENT_SLOTS:
        raise ValueError(f"acknowledgment slot {slot} is out of range")
    return ACK_OFFSET + slot * ACK_WORD_BYTES


def initialize(buffer: memoryview) -> None:
    """Initialize pending readiness and unclaimed reader slots.

    The producer calls this before the locator naming the segment leaves
    `ShmTransport.export`, so no consumer can observe a partial header.
    """
    for slot in range(MAX_ACKNOWLEDGMENT_SLOTS):
        atomic_store_u32(buffer, ack_offset(slot), 0)
    atomic_store_u32(buffer, STATE_OFFSET, PENDING)


def set_state(buffer: memoryview, state: int) -> None:
    """Announce readiness or failure, after every payload byte.

    The store has release ordering, so a consumer whose `state` load sees
    `READY` also sees the payload written before this call.
    """
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

    The first turn polls without sleeping; later turns sleep for a pause
    that grows linearly up to one millisecond, so a long wait does not
    occupy a core. `check` runs on every turn, so a cancelled read stops
    waiting within one turn of its cancellation.

    Raises:
        WorkerError: `resource_error` when the producer marks the segment
            `FAILED` or `timeout` seconds pass before `READY`. Whatever
            `check` raises propagates.
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
            raise resource_error("export producer failed before readiness")
        if time.monotonic() >= deadline:
            raise resource_error("export endpoint was lost before readiness")
        if pause:
            time.sleep(pause)
        pause = min(1e-3, pause + 5e-5)


def claim(buffer: memoryview, slot: int) -> None:
    """Mark this rank as reading the payload, before its first read of it."""
    atomic_store_u32(buffer, ack_offset(slot), CLAIMED)


def acknowledge(buffer: memoryview, slot: int) -> None:
    """Write this rank's acknowledgment word, after its reads of the payload."""
    atomic_store_u32(buffer, ack_offset(slot), ACKNOWLEDGED)


def settled(buffer: memoryview, slots: Sequence[int]) -> bool:
    """Report whether no named consumer is still reading the segment.

    Consulted once the engine has retired the export, after which a
    consumer that has not claimed its word never will: every named word is
    then either acknowledged or untouched.
    """
    return all(
        atomic_load_u32(buffer, ack_offset(slot)) != CLAIMED for slot in slots
    )
