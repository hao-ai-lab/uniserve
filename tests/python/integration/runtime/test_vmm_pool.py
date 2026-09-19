"""A publishing rank's bounded VMM pool and its acknowledgment header."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.transfer.vmm_pool import (
    ACK_WORD_BYTES,
    ACKNOWLEDGED,
    CLAIMED,
    HEADER_BYTES,
    PoolExhaustedError,
    VmmPool,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
def pool() -> VmmPool:
    """Reserve a small pool on the first CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("a VMM pool requires a CUDA device")
    return VmmPool(torch.device("cuda:0"), capacity_bytes=8 << 20)


def test_a_pool_exports_one_handle_for_every_chunk_it_hands_out(
    pool: VmmPool,
) -> None:
    """Consumers import a producing device once, not once per product.

    The pool's handle is what a publication carries, so two products from one
    device name the same allocation at different offsets.
    """
    first = pool.reserve(4096)
    second = pool.reserve(4096)

    assert first.offset != second.offset
    assert first.payload_offset == first.offset + HEADER_BYTES
    assert first.storage.numel() == 4096
    # One handle addresses both chunks; only the offsets differ.
    assert len(pool.handle) in (4, 64)


def test_a_product_that_does_not_fit_is_refused_by_size(pool: VmmPool) -> None:
    """A product larger than the pool keeps its own allocation.

    The pool reports that it cannot hold the product rather than failing the
    publication, so the caller decides what to do instead.
    """
    with pytest.raises(PoolExhaustedError, match="does not fit"):
        pool.reserve(pool.capacity * 2)


def test_a_chunk_is_held_only_by_the_slots_reading_it(pool: VmmPool) -> None:
    """Each consuming rank owns one word, addressed by the slot the head gave.

    Owning a word rather than sharing a counter is what lets a consumer claim
    and acknowledge with plain stores into the chunk it already mapped, with
    no cross-process atomic and no connection back to the producer.
    """
    chunk = pool.reserve(4096)
    named = (3, 7)

    # A chunk whose consumers never began reading holds nothing: the producer
    # sweeps only after the engine retired the publication, and no consumer
    # claims it after that.
    assert chunk.settled(named), "an unclaimed chunk is free"

    chunk.acknowledgments[3] = CLAIMED
    assert not chunk.settled(named), "a chunk being read is held"
    chunk.acknowledgments[3] = ACKNOWLEDGED
    assert chunk.settled(named), "an acknowledged read releases the chunk"

    # A word outside the named slots is not this product's to wait on.
    chunk.acknowledgments[5] = CLAIMED
    assert chunk.settled(named), "an unnamed slot does not hold the chunk"

    chunk.acknowledgments[7] = CLAIMED
    assert not chunk.settled(named), "any named reader holds the chunk"
    chunk.acknowledgments[7] = ACKNOWLEDGED
    assert chunk.settled(named), "every named reader has finished"

    # A product no other rank reads has nothing to wait for.
    assert chunk.settled(())


def test_a_consumer_addresses_its_word_by_slot_from_the_chunk_offset(
    pool: VmmPool,
) -> None:
    """A consumer locates its word from the offset the publication carries.

    It maps the producer's allocation and knows only the chunk's offset and its
    own slot, so the header must be addressable by that arithmetic alone.
    """
    chunk = pool.reserve(4096)
    slot = 9

    # The same arithmetic a reading rank performs against its own mapping.
    word = pool.mapping.view(torch.uint8)[
        chunk.offset + slot * ACK_WORD_BYTES :
    ][:ACK_WORD_BYTES].view(torch.int32)
    word.fill_(CLAIMED)
    assert not chunk.settled((slot,))

    word.fill_(ACKNOWLEDGED)
    assert chunk.settled((slot,))


def test_a_reused_chunk_starts_unclaimed(pool: VmmPool) -> None:
    """A chunk's words are cleared when its span is handed out again.

    A stale acknowledgment would let the next publication's reader appear
    finished before it had read, so the pool clears the words at reservation
    rather than at release.
    """
    first = pool.reserve(4096)
    first.acknowledgments[:4] = ACKNOWLEDGED
    pool.release(first)

    reused = pool.reserve(4096)
    assert reused.offset == first.offset, (
        "the released span is handed out again"
    )
    assert all(word == 0 for word in reused.acknowledgments[:4].tolist()), (
        "a reused chunk starts unclaimed"
    )
