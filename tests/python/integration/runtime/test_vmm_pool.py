"""A publishing rank's bounded VMM pool and its acknowledgment retirement."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.transfer.vmm_pool import (
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


def test_a_chunk_retires_when_every_named_consumer_acknowledges(
    pool: VmmPool,
) -> None:
    """Retirement is the consumers' acknowledgments, not a reader connection.

    A chunk is held while any consumer the component names has not written its
    word, and the producing rank's sweep returns it once they all have. This is
    what lets a consumer on another host retire a product the same way one on
    this host does.
    """
    chunk = pool.reserve(4096)
    pool.await_acknowledgment(chunk, consumers=2)

    assert pool.recycle() == 0, "an unacknowledged chunk is held"

    chunk.acknowledgments[0] = 1
    assert pool.recycle() == 0, "a partly acknowledged chunk is held"

    chunk.acknowledgments[1] = 1
    assert pool.recycle() == 1, "a fully acknowledged chunk returns"
    assert pool.recycle() == 0, "a returned chunk is not returned twice"


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
    assert not reused.acknowledged(1), "a reused chunk starts unacknowledged"
