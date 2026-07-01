"""Contracts for the paged text KV coordinator and physical KV pool.

These are pure CPU tensor tests (no GPU required): they exercise the public
behavior of ``PagedTextCache`` / ``BatchedPagedTextCache`` (transient growth,
per-row block tables, cross-block-boundary appends) and ``PagedKVPool`` range
and construction validation, asserting concrete tensor values and the typed
``WorkerError`` taxonomy through public interfaces only.
"""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import (
    BatchedPagedTextCache,
    PagedTextCache,
)

pytestmark = pytest.mark.unit


def _make_pool(
    *,
    num_layers: int = 1,
    num_blocks: int = 16,
    block_size: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 3,
) -> PagedKVPool:
    return PagedKVPool(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )


# --------------------------------------------------------------------------
# PagedTextCache.request_cache_for_update
# --------------------------------------------------------------------------


def test_update_view_reuses_request_cache_across_layers_for_one_span():
    pool = _make_pool(block_size=4, num_blocks=16, num_layers=2)
    cache = PagedTextCache(pool, [0], num_layers=2, length=0)

    first = cache.request_cache_for_update(0, 1)
    cache.finish_layer_update(0, 1)
    second = cache.request_cache_for_update(1, 1)

    assert second is first
    assert second.block_table(device="cpu") is first.block_table(device="cpu")
    assert second.cache_seqlens(device="cpu") is first.cache_seqlens(device="cpu")

    cache.finish_layer_update(1, 1)
    third = cache.request_cache_for_update(0, 1)

    assert third is not first
    assert third.base_len == 1


# --------------------------------------------------------------------------
# PagedTextCache.request_cache_for_transient
# --------------------------------------------------------------------------


def test_transient_at_capacity_without_allocator_raises_invalid_descriptor():
    # One block of capacity (block_size=4) cannot hold 5 transient tokens, and
    # no allocator is wired to grow the tail.
    pool = _make_pool(block_size=4)
    cache = PagedTextCache(pool, [0], num_layers=1, length=0)

    with pytest.raises(WorkerError) as exc:
        cache.request_cache_for_transient(0, 5)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_transient_within_existing_capacity_does_not_call_allocator():
    # 3 tokens fit in the single pre-allocated block (capacity 4), so the
    # allocator must not be consulted and the block table is unchanged.
    pool = _make_pool(block_size=4)
    calls: list[int] = []

    def allocate(n: int) -> list[int]:
        calls.append(n)
        return list(range(1, 1 + n))

    cache = PagedTextCache(pool, [0], num_layers=1, length=0, allocate_blocks=allocate)
    view = cache.request_cache_for_transient(0, 3)

    assert calls == []
    assert list(cache.block_ids) == [0]
    assert view.base_len == 0


def test_transient_view_reuses_metadata_cache_until_blocks_grow():
    pool = _make_pool(block_size=4, num_blocks=16)
    requested: list[int] = []

    def allocate(n: int) -> list[int]:
        requested.append(n)
        start = 1 + sum(requested[:-1])
        return list(range(start, start + n))

    cache = PagedTextCache(pool, [0], num_layers=1, length=0, allocate_blocks=allocate)
    first = cache.request_cache_for_transient(0, 2)
    second = cache.request_cache_for_transient(0, 3)

    assert first is second
    grown = cache.request_cache_for_transient(0, 9)
    assert grown is not first
    assert requested == [2]


def test_transient_overflow_grows_blocks_via_ceil_div_without_advancing_length():
    # Capacity is one block (4 tokens). Requesting 13 transient tokens leaves
    # 13 - 4 = 9 missing, which is ceil(9 / 4) = 3 additional blocks. The
    # persistent length must stay at its pre-transient value (0).
    pool = _make_pool(block_size=4, num_blocks=16)
    requested: list[int] = []

    def allocate(n: int) -> list[int]:
        requested.append(n)
        return list(range(1, 1 + n))

    cache = PagedTextCache(pool, [0], num_layers=1, length=0, allocate_blocks=allocate)
    view = cache.request_cache_for_transient(0, 13)

    assert requested == [3]
    assert list(cache.block_ids) == [0, 1, 2, 3]
    assert cache.length == 0
    # The transient view is anchored at the (unchanged) persistent length.
    assert view.base_len == 0


def test_transient_overflow_exact_block_multiple_grows_minimal_blocks():
    # Missing tokens that are an exact multiple of block_size grow exactly that
    # many blocks (no off-by-one over-allocation): need 8 tokens, have 4,
    # missing 4 -> ceil(4 / 4) = 1 block.
    pool = _make_pool(block_size=4, num_blocks=16)
    requested: list[int] = []

    def allocate(n: int) -> list[int]:
        requested.append(n)
        return list(range(1, 1 + n))

    cache = PagedTextCache(pool, [0], num_layers=1, length=0, allocate_blocks=allocate)
    cache.request_cache_for_transient(0, 8)

    assert requested == [1]
    assert list(cache.block_ids) == [0, 1]


def test_transient_allocator_short_return_raises_invalid_descriptor():
    # An allocator that returns fewer blocks than requested leaves the cache
    # under-provisioned; the cache must fail fast with INVALID_DESCRIPTOR.
    pool = _make_pool(block_size=4, num_blocks=16)

    def allocate_short(n: int) -> list[int]:
        del n
        return [1]  # always one block, regardless of the request

    cache = PagedTextCache(pool, [0], num_layers=1, length=0, allocate_blocks=allocate_short)

    with pytest.raises(WorkerError) as exc:
        cache.request_cache_for_transient(0, 13)  # needs 3 extra blocks, gets 1

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


# --------------------------------------------------------------------------
# BatchedPagedTextCache block tables / cache_seqlens
# --------------------------------------------------------------------------


def test_batched_block_table_is_per_row_and_zero_padded_to_max_width():
    # Row 0 owns two blocks, row 1 owns one; the block table is padded to the
    # widest row with zeros in the unused slots.
    pool = _make_pool(block_size=4, num_blocks=16)
    row0 = PagedTextCache(pool, [5, 6], num_layers=1, length=3)
    row1 = PagedTextCache(pool, [7], num_layers=1, length=2)
    batched = BatchedPagedTextCache([row0, row1])

    request = batched.request_cache_for_transient(0, 1)
    table = request.block_table(device="cpu")

    assert tuple(table.shape) == (2, 2)
    assert table.dtype == torch.int32
    assert table.tolist() == [[5, 6], [7, 0]]


def test_batched_cache_seqlens_reports_each_row_base_length():
    pool = _make_pool(block_size=4, num_blocks=16)
    row0 = PagedTextCache(pool, [0, 1], num_layers=1, length=3)
    row1 = PagedTextCache(pool, [2, 3], num_layers=1, length=5)
    batched = BatchedPagedTextCache([row0, row1])

    request = batched.request_cache_for_transient(0, 1)
    seqlens = request.cache_seqlens(device="cpu")

    assert seqlens.dtype == torch.int32
    assert seqlens.tolist() == [3, 5]


def test_batched_request_cache_reuses_metadata_tensors_per_device():
    pool = _make_pool(block_size=4, num_blocks=16)
    row0 = PagedTextCache(pool, [0, 1], num_layers=1, length=3)
    row1 = PagedTextCache(pool, [2, 3], num_layers=1, length=5)
    batched = BatchedPagedTextCache([row0, row1])

    request = batched.request_cache_for_transient(0, 1)

    assert request.block_table(device="cpu") is request.block_table(device="cpu")
    assert request.cache_seqlens(device="cpu") is request.cache_seqlens(device="cpu")


def test_batched_transient_view_reuses_request_cache_until_a_row_grows():
    pool = _make_pool(block_size=4, num_blocks=16)
    requested: list[int] = []

    def allocate(n: int) -> list[int]:
        requested.append(n)
        start = 4 + sum(requested[:-1])
        return list(range(start, start + n))

    row0 = PagedTextCache(pool, [0], num_layers=1, length=1, allocate_blocks=allocate)
    row1 = PagedTextCache(pool, [1], num_layers=1, length=2, allocate_blocks=allocate)
    batched = BatchedPagedTextCache([row0, row1])

    first = batched.request_cache_for_transient(0, 1)
    second = batched.request_cache_for_transient(0, 2)
    assert first is second

    grown = batched.request_cache_for_transient(0, 5)
    assert grown is not first
    assert requested == [1, 1]


def test_batched_get_seq_length_is_max_persistent_length():
    pool = _make_pool(block_size=4, num_blocks=16)
    row0 = PagedTextCache(pool, [0, 1], num_layers=1, length=3)
    row1 = PagedTextCache(pool, [2, 3], num_layers=1, length=7)
    batched = BatchedPagedTextCache([row0, row1])

    assert batched.get_seq_length() == 7


def test_batched_requires_positive_transient_token_count():
    pool = _make_pool(block_size=4, num_blocks=16)
    cache = PagedTextCache(pool, [0], num_layers=1, length=0)
    batched = BatchedPagedTextCache([cache])

    with pytest.raises(WorkerError) as exc:
        batched.request_cache_for_transient(0, 0)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_batched_rejects_caches_with_different_pools():
    pool_a = _make_pool(block_size=4, num_blocks=16)
    pool_b = _make_pool(block_size=4, num_blocks=16)
    cache_a = PagedTextCache(pool_a, [0], num_layers=1, length=0)
    cache_b = PagedTextCache(pool_b, [0], num_layers=1, length=0)

    with pytest.raises(WorkerError) as exc:
        BatchedPagedTextCache([cache_a, cache_b])

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


# --------------------------------------------------------------------------
# BatchedPagedTextCache transient append round-trips through pool.read
# --------------------------------------------------------------------------


def test_batched_transient_append_lands_at_each_row_base_length():
    # Two rows with distinct base lengths. The transient append writes each row
    # at its own base length, readable back via pool.read at that offset, while
    # the persistent lengths stay put.
    torch.manual_seed(0)
    pool = _make_pool(num_layers=1, block_size=4, num_blocks=16, num_kv_heads=2, head_dim=3)
    row0 = PagedTextCache(pool, [0, 1], num_layers=1, length=3)
    row1 = PagedTextCache(pool, [2, 3], num_layers=1, length=5)
    batched = BatchedPagedTextCache([row0, row1])

    request = batched.request_cache_for_transient(0, 2)
    k = torch.randn(2, 2, 2, 3)  # [batch, tokens, heads, dim]
    v = torch.randn(2, 2, 2, 3)
    request.append(0, k, v)

    row0_k, row0_v = pool.read(0, [0, 1], start=3, length=2)
    row1_k, row1_v = pool.read(0, [2, 3], start=5, length=2)

    torch.testing.assert_close(row0_k, k[0])
    torch.testing.assert_close(row0_v, v[0])
    torch.testing.assert_close(row1_k, k[1])
    torch.testing.assert_close(row1_v, v[1])
    # Transient writes never advance the persistent length.
    assert row0.length == 3
    assert row1.length == 5


def test_batched_transient_append_spans_block_boundary():
    # Row base length 3 with block_size 4: appending 5 tokens spills from the
    # first block (offset 3, one slot) into the second block (four slots). The
    # multi-block read must reassemble the contiguous token sequence.
    torch.manual_seed(1)
    pool = _make_pool(num_layers=1, block_size=4, num_blocks=16, num_kv_heads=2, head_dim=3)
    cache = PagedTextCache(pool, [4, 5], num_layers=1, length=3)
    batched = BatchedPagedTextCache([cache])

    request = batched.request_cache_for_transient(0, 5)
    k = torch.randn(1, 5, 2, 3)
    v = torch.randn(1, 5, 2, 3)
    request.append(0, k, v)

    read_k, read_v = pool.read(0, [4, 5], start=3, length=5)

    assert tuple(read_k.shape) == (5, 2, 3)
    torch.testing.assert_close(read_k, k[0])
    torch.testing.assert_close(read_v, v[0])
    assert cache.length == 3


def test_batched_transient_append_does_not_disturb_persistent_prefix():
    # Tokens already resident below the base length must be untouched by a
    # transient append at the tail.
    torch.manual_seed(2)
    pool = _make_pool(num_layers=1, block_size=4, num_blocks=16, num_kv_heads=2, head_dim=3)
    cache = PagedTextCache(pool, [0, 1], num_layers=1, length=0)

    # Seed a persistent prefix of 3 tokens directly via the pool.
    prefix_k = torch.randn(3, 2, 3)
    prefix_v = torch.randn(3, 2, 3)
    pool.write(0, [0, 1], start=0, k=prefix_k, v=prefix_v)
    cache.length = 3

    batched = BatchedPagedTextCache([cache])
    request = batched.request_cache_for_transient(0, 2)
    tail_k = torch.randn(1, 2, 2, 3)
    tail_v = torch.randn(1, 2, 2, 3)
    request.append(0, tail_k, tail_v)

    kept_k, kept_v = pool.read(0, [0, 1], start=0, length=3)
    torch.testing.assert_close(kept_k, prefix_k)
    torch.testing.assert_close(kept_v, prefix_v)


# --------------------------------------------------------------------------
# PagedKVPool.view range validation
# --------------------------------------------------------------------------


def test_view_rejects_base_len_overrunning_host_blocks():
    # One block of capacity (4 tokens); a base length of 5 overruns the host
    # blocks and must surface as INVALID_DESCRIPTOR at view construction.
    pool = _make_pool(block_size=4, num_blocks=16)

    with pytest.raises(WorkerError) as exc:
        pool.view([0], 5)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_view_accepts_base_len_filling_host_blocks_exactly():
    # A base length equal to the host capacity is valid (boundary case).
    pool = _make_pool(block_size=4, num_blocks=16)

    view = pool.view([0], 4)

    assert view.base_len == 4
    assert view.length() == 4
    assert view.block_table() is view.block_table()
    assert view.cache_seqlens() is view.cache_seqlens()


def test_single_request_cache_append_varlen_matches_append_without_advancing_length():
    torch.manual_seed(22)
    pool = _make_pool(num_layers=1, block_size=4, num_blocks=16, num_kv_heads=2, head_dim=3)
    cache = PagedTextCache(pool, [0, 1, 2], num_layers=1, length=3)
    view = cache.request_cache_for_transient(0, 5)
    k = torch.randn(5, 2, 3)
    v = torch.randn(5, 2, 3)

    view.append_varlen(
        0,
        k,
        v,
        (5,),
        block_table=view.block_table(device="cpu"),
        cache_seqlens=view.cache_seqlens(device="cpu"),
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
    )

    read_k, read_v = pool.read(0, [0, 1, 2], start=3, length=5)
    torch.testing.assert_close(read_k, k)
    torch.testing.assert_close(read_v, v)
    assert cache.length == 3
    assert view.base_lens == (3,)


def test_view_rejects_block_id_outside_pool_capacity():
    pool = _make_pool(block_size=4, num_blocks=8)

    with pytest.raises(WorkerError) as exc:
        pool.view([8], 0)  # valid ids are 0..7

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


# --------------------------------------------------------------------------
# PagedKVPool construction validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"num_layers": 0},
        {"num_layers": -1},
        {"num_blocks": 0},
        {"num_blocks": -2},
        {"block_size": 0},
        {"block_size": -1},
    ],
)
def test_construction_rejects_non_positive_dimensions(override):
    kwargs = {
        "num_layers": 2,
        "num_blocks": 8,
        "block_size": 4,
        "num_kv_heads": 2,
        "head_dim": 3,
        "device": "cpu",
        "dtype": torch.float32,
    }
    kwargs.update(override)

    with pytest.raises(WorkerError) as exc:
        PagedKVPool(**kwargs)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
