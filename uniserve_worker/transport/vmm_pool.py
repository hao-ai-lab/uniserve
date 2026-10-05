"""One bounded VMM pool per device on a exporting rank.

`CudaVmmTransport` reserves one pool per device the first time that device
exports a product it cannot export where it lies, sized by the rank's
transfer byte budget, and exports one shareable handle for the pool's whole
allocation. Such an export is a chunk of that pool addressed by byte
offset and size, so a consumer imports one handle per producing device rather
than one per product.

Because the pool is reserved outside the PyTorch caching allocator,
export never needs exportable allocator segments, and the engine enables
expandable segments on every rank unless the head's environment already
configures the allocator, in which case every rank receives the head's
setting.

A chunk carries its own acknowledgment header but does not track who owes an
acknowledgment: that belongs to the export the chunk backs, which
`CudaVmmTransport` owns, retires and sweeps.

A product that does not fit the pool raises `PoolExhaustedError`, and
`exports.export_tensor` exports that product as host bytes instead,
over whichever host mechanisms the rank binds for its consumers. The pool
logs its exhaustion once rather than once per product.
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
#: Acknowledgment word values: the consumer never began reading the chunk, it
#: is reading, or its reads are done.
UNCLAIMED = 0
CLAIMED = 1
ACKNOWLEDGED = 2
#: Acknowledgment slots one chunk header carries.
#:
#: Every rank of an instance owns one slot, assigned by the head, and writes
#: that slot's word in each chunk it reads. Because no two ranks write the same
#: word, no cross-process read-modify-write is needed. `segment` lays out the
#: shared storage header's acknowledgment words with the same bound.
MAX_ACKNOWLEDGMENT_SLOTS = 64
#: Bytes reserved at the head of every chunk for its acknowledgments.
HEADER_BYTES = ACK_WORD_BYTES * MAX_ACKNOWLEDGMENT_SLOTS
#: Byte alignment of one chunk within its pool.
#:
#: A consumer imports the pool's whole allocation and addresses chunks by
#: offset, so a chunk does not have to begin on a CUDA page. What it does have
#: to satisfy is the alignment its own contents need: the acknowledgment words
#: are 32-bit, and payload spans, which begin `HEADER_BYTES` after the chunk,
#: are read as tensors of the product's dtype and copied in vector widths.
#: This bound covers both and keeps a pool's capacity proportional to what it
#: carries rather than to the number of products in it.
CHUNK_ALIGNMENT = 512


@dataclass(frozen=True, slots=True)
class PoolChunk:
    """One export's span of its device's pool.

    The chunk begins with one acknowledgment word per instance rank. A
    consumer claims the word of its own slot before its first read and
    acknowledges it when those reads retire, and the producing rank returns
    the chunk once no named consumer is still reading. Retirement therefore
    needs no connection to the producer, so a consumer on another host
    acknowledges a chunk the same way as one on this host.
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

    def settled(self, slots: Sequence[int]) -> bool:
        """Report whether no named consumer is still reading this chunk.

        Consulted once the producing export has been retired, after
        which no consumer begins reading: the engine retires an export
        only when every consuming call has resolved or will never be
        submitted. A named slot that never claimed its word therefore holds
        nothing, and one that claimed must acknowledge before its span is
        handed out again. A product with no remote consumer is settled on
        export: no other rank reads it.

        The words live in device memory, so the check runs on the current
        stream and blocks the host until that stream reaches it.
        """
        if not slots:
            return True
        words = self.acknowledgments[list(slots)]
        return bool((words != CLAIMED).all().item())


class PoolExhaustedError(Exception):
    """Raised when a product does not fit its device's pool."""


class VmmPool:
    """A device's bounded pool of exportable pages.

    Chunks are handed out from a rising watermark and returned when their
    export retires. The pool does not compact: an export's chunk is
    live while a consumer may still read it, and moving it would invalidate the
    offset that consumer was given.

    `reserve` and `release` are serialized by one lock and may be called from
    any thread.
    """

    def __init__(self, device: torch.device, *, capacity_bytes: int) -> None:
        """Reserve and export the pool's physical allocation on `device`.

        Raises:
            ValueError: When `capacity_bytes` is not positive. Errors from
                reserving, mapping or exporting the allocation propagate.
        """
        from uniserve_kernels.peer_storage import (
            allocate,
            allocation_granularity,
        )

        if capacity_bytes <= 0:
            raise ValueError("a VMM pool needs a positive capacity")
        # The exported allocation is a whole number of allocation-granularity
        # units; the chunks inside it are not.
        page = allocation_granularity(device)
        self._capacity = ((capacity_bytes + page - 1) // page) * page
        self._allocation = allocate(
            (self._capacity,), dtype=torch.uint8, device=device
        )
        self._storage = self._allocation.map_local()
        self._handle = self._allocation.export_handle()
        self._device = device

        self._lock = threading.Lock()
        # Byte span of each chunk handed out and not yet released, keyed by
        # the chunk's offset.
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
        into this view are the offsets an export carries.
        """
        return self._storage

    @property
    def capacity(self) -> int:
        """Return the pool's byte capacity, rounded up to the granularity."""
        return self._capacity

    def reserve(self, nbytes: int) -> PoolChunk:
        """Reserve one chunk for an `nbytes` payload.

        The chunk spans the acknowledgment header plus the payload, rounded up
        to `CHUNK_ALIGNMENT`, and its acknowledgment words are cleared.

        Raises:
            ValueError: When `nbytes` is not positive.
            PoolExhaustedError: When no free span of the pool fits the chunk.
                The first exhaustion of this pool is also logged.
        """
        if nbytes <= 0:
            raise ValueError("a pool chunk needs a positive length")
        # The header precedes the payload inside one reservation, so a chunk
        # is one span a consumer maps once.
        needed = nbytes + HEADER_BYTES
        span = (
            (needed + CHUNK_ALIGNMENT - 1) // CHUNK_ALIGNMENT
        ) * CHUNK_ALIGNMENT
        with self._lock:
            offset = self._free_offset(span)
            if offset is None:
                # One report names the pool that filled; per-export
                # reporting would say the same thing once per product.
                if not self._reported_exhaustion:
                    self._reported_exhaustion = True
                    _LOG.warning(
                        "device %s VMM pool of %d bytes cannot fit a %d byte "
                        "product; such products are exported as host bytes "
                        "where the rank binds a host mechanism for them",
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
            # A span is reused, and a stale CLAIMED word would keep the new
            # chunk from ever settling, so every word starts unclaimed.
            acknowledgments.zero_()
            return PoolChunk(
                offset=offset,
                nbytes=span,
                storage=self._storage[payload : payload + nbytes],
                payload_offset=payload,
                acknowledgments=acknowledgments,
            )

    def release(self, chunk: PoolChunk) -> None:
        """Return one chunk's span to the pool; a chunk not live is ignored."""
        with self._lock:
            if self._live.pop(chunk.offset, None) is None:
                return
            if not self._live:
                # An empty pool restarts at the beginning, which is what keeps
                # a steady export-and-retire cycle from walking the watermark
                # to the end of a pool it never actually fills.
                self._watermark = 0

    def _free_offset(self, span: int) -> int | None:
        """Return an offset where `span` bytes fit, or None.

        The watermark is used while the span fits below the capacity;
        otherwise the lowest free gap that fits, before, between or after the
        live chunks, is returned. The caller holds the lock and records the span
        as live.

        Every live chunk ends at or below the watermark, which is what lets
        the watermark path hand out the space above it without consulting the
        live chunks. A gap before or between live chunks ends at a live
        chunk's start, so it keeps that invariant; a span after the last live
        chunk can reach past the watermark, which then rises to its end.
        """
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
            self._watermark = max(self._watermark, cursor + span)
            return cursor
        return None
