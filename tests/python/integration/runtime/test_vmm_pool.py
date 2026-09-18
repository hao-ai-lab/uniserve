"""A publishing rank's bounded VMM pool and its acknowledgment header."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.transfer.vmm_pool import (
    ACK_WORD_BYTES,
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


def test_a_chunk_is_acknowledged_only_by_the_slots_named_for_it(
    pool: VmmPool,
) -> None:
    """Each consuming rank owns one word, addressed by the slot the head gave.

    Owning a word rather than sharing a counter is what lets a consumer
    acknowledge with a plain store into the chunk it already mapped, with no
    cross-process atomic and no connection back to the producer.
    """
    chunk = pool.reserve(4096)
    named = (3, 7)

    assert not chunk.acknowledged(named), "an unacknowledged chunk is held"

    chunk.acknowledgments[3] = 1
    assert not chunk.acknowledged(named), "a partly acknowledged chunk is held"
    # A word outside the named slots is not this product's to acknowledge.
    chunk.acknowledgments[5] = 1
    assert not chunk.acknowledged(named), "an unnamed slot does not acknowledge"

    chunk.acknowledgments[7] = 1
    assert chunk.acknowledged(named), "every named slot has acknowledged"

    # A product no other rank reads has nothing to wait for.
    assert chunk.acknowledged(())


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
    word.fill_(1)

    assert chunk.acknowledged((slot,))


def test_a_reused_chunk_starts_unacknowledged(pool: VmmPool) -> None:
    """A chunk's acknowledgments are cleared when its span is handed out again.

    Stale acknowledgments would retire a product before its consumers had read
    it, so the pool clears them at reservation rather than at release.
    """
    first = pool.reserve(4096)
    first.acknowledgments[:4] = 1
    pool.release(first)

    reused = pool.reserve(4096)
    assert reused.offset == first.offset, (
        "the released span is handed out again"
    )
    assert not reused.acknowledged((0,)), "a reused chunk starts unacknowledged"
