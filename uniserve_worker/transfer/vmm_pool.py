"""One bounded VMM pool per device on a publishing rank.

A device product used to take its own allocation: every publication created
physical pages, exported a handle for them, and released them when the
publication retired. Section 5.7 reserves one bounded pool per device at
startup and exports one handle for it, so a publication is a chunk of that
pool addressed by offset and size.

Two things follow. A consumer imports one handle per producing device rather
than one per product, and it can keep that mapping for as long as the producer
lives. And a publishing rank no longer needs its PyTorch allocator segments to
be exportable, which is what the ``expandable_segments:False`` constraint
existed to guarantee.

A chunk carries its own acknowledgment header but does not track who owes an
acknowledgment: that belongs to the publication the chunk backs, which the
device transport owns and retires.

A product that does not fit the pool falls back to host transport for that
product, and the exhaustion is reported once rather than per publication.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import torch

_LOG = logging.getLogger(__name__)


#: Bytes of one rank's acknowledgment word.
ACK_WORD_BYTES = 4
#: Acknowledgment slots one chunk header carries.
#:
#: Every rank of an instance owns one slot, assigned by the head, and writes
#: that slot's word in each chunk it reads. Giving a rank its own word removes
#: any need for cross-process atomics, at one word per rank rather than a page.
MAX_ACKNOWLEDGMENT_SLOTS = 64
#: Bytes reserved at the head of every chunk for its acknowledgments.
HEADER_BYTES = ACK_WORD_BYTES * MAX_ACKNOWLEDGMENT_SLOTS


@dataclass(frozen=True, slots=True)
class PoolChunk:
    """One publication's span of its device's pool.

    The chunk begins with one acknowledgment word per instance rank. A consumer
    writes the word of its own slot when its reads retire, and the producing
    rank returns the chunk once every consumer named for its products has
    acknowledged. That is what lets a product retire without a reader-grant
    connection, which a Unix socket could not carry to another host.
    """

    #: Byte offset of the chunk's header within the pool's exported allocation.
    offset: int
    #: Byte length reserved for this chunk, header included.
    nbytes: int
    #: Flat byte view of the chunk's payload, for the producer to copy into.
    storage: torch.Tensor
    #: Byte offset of the payload within the pool, which a consumer reads from.
    payload_offset: int
    #: Acknowledgment words, indexed by the writing rank's slot.
    acknowledgments: torch.Tensor

    def acknowledged(self, slots: Sequence[int]) -> bool:
        """Report whether every named consumer slot has acknowledged this chunk.

        A product with no remote consumer is acknowledged on publication: no
        other rank reads it, so nothing can be waiting to retire it.
        """
        if not slots:
            return True
        return bool(self.acknowledgments[list(slots)].all().item())


class PoolExhaustedError(Exception):
    """Raised when a product does not fit its device's pool."""


class VmmPool:
    """A device's bounded pool of exportable pages.

    Chunks are handed out from a rising watermark and returned when their
    publication retires. The pool does not compact: a publication's chunk is
    live while a consumer may still read it, and moving it would invalidate the
    offset that consumer was given.
    """

    def __init__(self, device: torch.device, *, capacity_bytes: int) -> None:
        from uniserve_kernel.peer_memory import allocate, allocation_granularity

        if capacity_bytes <= 0:
            raise ValueError("a VMM pool needs a positive capacity")
        page = allocation_granularity(device)
        self._page = page
        self._capacity = ((capacity_bytes + page - 1) // page) * page
        self._allocation = allocate(
            (self._capacity,), dtype=torch.uint8, device=device
        )
        self._storage = self._allocation.map_local()
        self._handle = self._allocation.export_handle()
        self._device = device

        self._lock = threading.Lock()
        # Offsets handed out and not yet released, by offset.
        self._live: dict[int, int] = {}
        self._watermark = 0
        self._reported_exhaustion = False

    @property
    def handle(self) -> bytes:
        """Return the pool's shareable handle, exported once at reservation."""
        return self._handle

    @property
    def mapping(self) -> torch.Tensor:
        """Return this rank's flat byte view of the whole pool.

        A consumer maps the same allocation from the pool's handle, so offsets
        into this view are the offsets a publication carries.
        """
        return self._storage

    @property
    def capacity(self) -> int:
        """Return the pool's byte capacity, rounded up to CUDA pages."""
        return self._capacity

    def reserve(self, nbytes: int) -> PoolChunk:
        """Reserve one chunk, or report that the product does not fit.

        A chunk is page aligned so a consumer can map it without knowing how
        the producer packed the pool.
        """
        if nbytes <= 0:
            raise ValueError("a pool chunk needs a positive length")
        # The header precedes the payload inside one reservation, so a chunk
        # is one span a consumer maps once.
        needed = nbytes + HEADER_BYTES
        span = ((needed + self._page - 1) // self._page) * self._page
        with self._lock:
            offset = self._free_offset(span)
            if offset is None:
                # One report names the pool that filled; per-publication
                # reporting would say the same thing once per product.
                if not self._reported_exhaustion:
                    self._reported_exhaustion = True
                    _LOG.warning(
                        "device %s VMM pool of %d bytes cannot fit a %d byte "
                        "product; those products use host transport",
                        self._device,
                        self._capacity,
                        nbytes,
                    )
                raise PoolExhaustedError(
                    f"product of {nbytes} bytes does not fit the "
                    f"{self._capacity} byte pool on {self._device}"
                )
            self._live[offset] = span
            payload = offset + HEADER_BYTES
            acknowledgments = self._storage[offset:payload].view(torch.int32)
            # A chunk is reused, so its acknowledgments start clear.
            acknowledgments.zero_()
            return PoolChunk(
                offset=offset,
                nbytes=span,
                storage=self._storage[payload : payload + nbytes],
                payload_offset=payload,
                acknowledgments=acknowledgments,
            )

    def release(self, chunk: PoolChunk) -> None:
        """Return one chunk's span to the pool."""
        with self._lock:
            if self._live.pop(chunk.offset, None) is None:
                return
            if not self._live:
                # An empty pool restarts at the beginning, which is what keeps
                # a steady publish-and-retire cycle from walking the watermark
                # to the end of a pool it never actually fills.
                self._watermark = 0

    def _free_offset(self, span: int) -> int | None:
        """Find the lowest offset where this span fits, or None."""
        if self._watermark + span <= self._capacity:
            offset = self._watermark
            self._watermark += span
            return offset
        # The watermark is at the end; a gap below it may still fit.
        cursor = 0
        for offset in sorted(self._live):
            if offset - cursor >= span:
                return cursor
            cursor = max(cursor, offset + self._live[offset])
        if self._capacity - cursor >= span:
            return cursor
        return None
