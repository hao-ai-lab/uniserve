"""Chunk placement in a publishing rank's bounded VMM pool."""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest
import torch

from uniserve_kernels import peer_storage
from uniserve_worker.transport.vmm_pool import (
    CHUNK_ALIGNMENT,
    HEADER_BYTES,
    PoolChunk,
    PoolExhaustedError,
    VmmPool,
)

pytestmark = pytest.mark.unit


class _HostAllocation:
    """Host bytes standing in for the CUDA driver's exportable allocation."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        del device
        self._storage = torch.zeros(shape, dtype=dtype)

    def map_local(self) -> torch.Tensor:
        return self._storage

    def export_handle(self) -> bytes:
        return bytes(64)


@pytest.fixture
def make_pool(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], VmmPool]:
    """Build pools over host memory instead of a CUDA VMM allocation.

    Placement is bookkeeping over the pool's byte range, so replacing the
    driver allocation keeps every reservation observable on the CPU. The
    granularity equals the chunk alignment, so a pool of `units` holds
    exactly that many chunk units.
    """
    monkeypatch.setattr(
        peer_storage, "allocation_granularity", lambda device: CHUNK_ALIGNMENT
    )
    monkeypatch.setattr(peer_storage, "allocate", _HostAllocation)
    return lambda units: VmmPool(
        torch.device("cpu"), capacity_bytes=units * CHUNK_ALIGNMENT
    )


def _payload(units: int) -> int:
    """Return the payload length whose chunk spans exactly `units` units."""
    return units * CHUNK_ALIGNMENT - HEADER_BYTES


def _spans(chunks: list[PoolChunk]) -> list[tuple[int, int]]:
    return sorted(
        (chunk.offset, chunk.offset + chunk.nbytes) for chunk in chunks
    )


def _assert_disjoint_within(chunks: list[PoolChunk], capacity: int) -> None:
    spans = _spans(chunks)
    for start, end in spans:
        assert 0 <= start < end <= capacity, "a chunk lies outside the pool"
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert end <= start, f"live chunks overlap: {spans}"


def _largest_gap(chunks: list[PoolChunk], capacity: int) -> int:
    """Return the longest free byte span around the live chunks."""
    largest = cursor = 0
    for start, end in _spans(chunks):
        largest = max(largest, start - cursor)
        cursor = end
    return max(largest, capacity - cursor)


def test_a_chunk_placed_after_the_last_live_chunk_is_not_handed_out_again(
    make_pool: Callable[[int], VmmPool],
) -> None:
    """A span that reaches past the watermark stays owned by its chunk.

    With the watermark near the end of the pool, a product too large for the
    space above it may still fit after the last live chunk. The next product
    must then be placed outside that chunk, or its header clear and payload
    copy land in bytes a consumer of the first product is still reading.
    """
    pool = make_pool(8)
    low = pool.reserve(_payload(2))
    high = pool.reserve(_payload(4))
    pool.release(high)

    # Only the space after `low` fits five units, and it runs past the four
    # units `high` had occupied.
    tail = pool.reserve(_payload(5))
    tail.storage.fill_(0xAB)
    # One unit is still free, after `tail`.
    last = pool.reserve(_payload(1))
    last.storage.fill_(0xCD)

    _assert_disjoint_within([low, tail, last], pool.capacity)
    assert bool((tail.storage == 0xAB).all()), (
        "a later reservation overwrote a live chunk's payload"
    )


def test_live_chunks_stay_disjoint_and_within_capacity(
    make_pool: Callable[[int], VmmPool],
) -> None:
    """Every chunk handed out owns its span until it is released.

    Products of different sizes are published and retired in an arbitrary
    order against a pool that they keep full, which reaches every placement
    the pool makes: at the watermark, in a gap between live chunks, and after
    the last one. A product is refused only when no free span fits it.
    """
    pool = make_pool(16)
    generator = random.Random(0)
    live: list[PoolChunk] = []
    for _ in range(5000):
        if live and generator.random() < 0.5:
            pool.release(live.pop(generator.randrange(len(live))))
        else:
            units = generator.randint(1, 12)
            try:
                live.append(pool.reserve(_payload(units)))
            except PoolExhaustedError:
                assert _largest_gap(live, pool.capacity) < (
                    units * CHUNK_ALIGNMENT
                ), "a product was refused although a free span fits it"
        _assert_disjoint_within(live, pool.capacity)
