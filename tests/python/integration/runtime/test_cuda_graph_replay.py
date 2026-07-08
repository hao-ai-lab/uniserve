"""CUDA-graph capture/replay behavior for text decode.

Covers the observable contract of the decode CUDA-graph runner and shared graph
plumbing in ``uniserve_worker.execution.cuda_graph_base`` /
``decode_cuda_graph`` / ``text_graph_runner``:

* batch-size bucketing rounds a request up to the nearest configured warmup
  bucket (and prefill token bucketing likewise);
* the decode input copy zero-pads a short batch up to the captured bucket and
  extends the ``cache_seqlens_cpu`` / ``kv_seqlens_cpu`` host mirrors;
* ``_share_input_buffer`` pools by ``(name, dtype, device)`` and slices the
  largest captured buffer for smaller buckets;
* eager-vs-graph replay produces identical logits and the returned tensor is
  sliced back to the unpadded request count, with CUDA-graph stats accounting
  the unpadded vs padded token split.

The capture/replay tests need a real CUDA device (available here, so they run).
The bucketing tests are pure-Python and run anywhere.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.contracts.forward_batch import ForwardBatch
from uniserve_worker.contracts.forward_context import ForwardContext, TextAttentionMetadata
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.execution.cuda_graph_base import (
    _reset_for_testing,
    _share_decode_graph_input_buffer,
    _share_input_buffer,
)
from uniserve_worker.execution.decode_cuda_graph import (
    DecodeCudaGraphRunner,
    PrefillCudaGraphRunner,
    copy_text_decode_graph_inputs,
    copy_text_initial_prefill_graph_inputs,
    make_text_initial_prefill_graph_state,
    make_text_decode_graph_state,
    resolve_paged_decode_graph_backend,
)
from uniserve_worker.execution.text_graph_runner import TextGraphRunner, _padded_prefill_max_kv_tokens
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import BatchedPagedRequestCache

pytestmark = pytest.mark.integration

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="decode CUDA graph capture/replay requires a CUDA device"
)

requires_decode_graph_backend = pytest.mark.skipif(
    resolve_paged_decode_graph_backend(None) is None,
    reason="TextGraphRunner decode replay requires a graph-aware paged decode backend",
)

_VOCAB = 32
_HIDDEN = 16


def _kv_pool(
    device: torch.device,
    *,
    num_blocks: int = 16,
    block_size: int = 4,
    head_dim: int = 8,
) -> PagedKVPool:
    return PagedKVPool(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=head_dim,
        device=device,
        dtype=torch.bfloat16,
    )


def _decode_metadata(
    pool: PagedKVPool,
    *,
    block_ids_by_row: list[list[int]],
    cache_seqlens_cpu: tuple[int, ...],
    kv_seqlens_cpu: tuple[int, ...],
    device: torch.device,
) -> TextAttentionMetadata:
    batch = len(block_ids_by_row)
    cache = BatchedPagedRequestCache(pool, block_ids_by_row, list(cache_seqlens_cpu))
    return TextAttentionMetadata(
        cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=cache_seqlens_cpu,
        query_lens=torch.ones(batch, dtype=torch.int32, device=device),
        query_lens_cpu=tuple(1 for _ in range(batch)),
        kv_seqlens_cpu=kv_seqlens_cpu,
        mode=ForwardMode.DECODE,
    )


def _prefill_metadata(
    pool: PagedKVPool,
    *,
    block_ids_by_row: list[list[int]],
    cache_seqlens_cpu: tuple[int, ...],
    query_lens_cpu: tuple[int, ...],
    device: torch.device,
    max_context_len: int = 0,
) -> TextAttentionMetadata:
    cache = BatchedPagedRequestCache(pool, block_ids_by_row, list(cache_seqlens_cpu))
    query_lens = torch.tensor(query_lens_cpu, dtype=torch.int32, device=device)
    kv_lens = torch.tensor(
        [base + query for base, query in zip(cache_seqlens_cpu, query_lens_cpu, strict=True)],
        dtype=torch.int32,
        device=device,
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    return TextAttentionMetadata(
        cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=cache_seqlens_cpu,
        query_lens=query_lens,
        query_lens_cpu=query_lens_cpu,
        kv_seqlens=kv_lens,
        kv_seqlens_cpu=tuple(int(x.item()) for x in kv_lens),
        cu_seqlens_q=torch.cat([zero, torch.cumsum(query_lens, dim=0).to(torch.int32)]),
        cu_seqlens_k=torch.cat([zero, torch.cumsum(kv_lens, dim=0).to(torch.int32)]),
        max_seqlen_q=max(query_lens_cpu, default=0),
        max_seqlen_k=max((b + q for b, q in zip(cache_seqlens_cpu, query_lens_cpu, strict=True)), default=0),
        max_context_len=int(max_context_len),
        mode=ForwardMode.EXTEND,
    )


class _LinearLogitsModel:
    """A CUDA-graph-safe one-token model: embed by id, add position, project.

    Static shapes, no host syncs or data-dependent control flow, so it is safe to
    capture. The same closure computes the eager reference so the test can assert
    capture/replay output equality exactly.
    """

    def __init__(self, device: torch.device, *, seed: int) -> None:
        gen = torch.Generator(device=device).manual_seed(seed)
        self.emb = torch.randn(_VOCAB, _HIDDEN, device=device, generator=gen)
        self.proj = torch.randn(_VOCAB, _HIDDEN, device=device, generator=gen)

    def logits(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.emb[input_ids.reshape(-1)] + positions.reshape(-1, 1).float()
        return x @ self.proj.t()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, fb) -> torch.Tensor:
        del fb
        return self.logits(input_ids, positions)

    def text_decode_graph_query_geometry(self) -> tuple[int, float, torch.dtype]:
        # Plausible query geometry matching the test pool; the fake forward
        # never runs attention, but declaring it keeps the model eligible for
        # the decode graph and exercises the real FlashInfer re-plan hook.
        return 1, 1.0, torch.bfloat16


# --------------------------------------------------------------------------- #
# Pure-Python bucketing (no CUDA needed)
# --------------------------------------------------------------------------- #


def test_decode_bucket_rounds_up_to_nearest_configured_bucket():
    runner = DecodeCudaGraphRunner(name="t", default_warmup_batch_sizes=(1, 2, 4, 8, 16))

    assert runner.bucket_batch_size(3) == 4
    assert runner.bucket_batch_size(5) == 8


def test_decode_bucket_keeps_exact_batch_when_it_equals_a_bucket():
    runner = DecodeCudaGraphRunner(name="t", default_warmup_batch_sizes=(1, 2, 4, 8, 16))

    assert runner.bucket_batch_size(8) == 8


def test_decode_bucket_falls_back_to_raw_batch_above_largest_bucket():
    runner = DecodeCudaGraphRunner(name="t", default_warmup_batch_sizes=(1, 2, 4, 8, 16))

    # No configured bucket is >= 20, so the runner uses the request size verbatim.
    assert runner.bucket_batch_size(20) == 20


def test_decode_bucket_skips_disabled_bucket_and_picks_next_larger():
    runner = DecodeCudaGraphRunner(name="t", default_warmup_batch_sizes=(2, 4, 8))
    runner.disabled.add(4)

    # batch 3 would normally round to 4, but 4 is disabled -> next bucket is 8.
    assert runner.bucket_batch_size(3) == 8


def test_prefill_token_bucket_rounds_up_to_nearest_configured_bucket():
    runner = PrefillCudaGraphRunner(name="t", default_warmup_token_buckets=(4, 8, 16, 32))

    assert runner.bucket_num_tokens(5) == 8
    assert runner.bucket_num_tokens(16) == 16


def test_prefill_token_bucket_falls_back_to_raw_count_above_largest_bucket():
    runner = PrefillCudaGraphRunner(name="t", default_warmup_token_buckets=(4, 8, 16, 32))

    assert runner.bucket_num_tokens(33) == 33


def test_prefill_batch_bucket_rounds_up_to_nearest_configured_bucket():
    runner = PrefillCudaGraphRunner(name="t", default_warmup_batch_sizes=(1, 2, 4, 8))

    assert runner.bucket_batch_size(3) == 4
    assert runner.bucket_batch_size(6) == 8


def test_prefill_batch_bucket_uses_largest_viable_bucket_for_token_bucket():
    runner = PrefillCudaGraphRunner(name="t", default_warmup_batch_sizes=(1, 2, 4, 8))

    assert runner.bucket_batch_size(1, num_tokens=4) == 8
    assert runner.bucket_batch_size(3, num_tokens=12) == 8


def test_prefill_kv_bucket_uses_context_capacity_when_available():
    runner = PrefillCudaGraphRunner(name="t", default_warmup_token_buckets=(4, 8, 16, 32))

    assert runner.bucket_kv_tokens(24, max_context_len=128) == 128
    assert runner.bucket_kv_tokens(160, max_context_len=128) == 160


def test_prefill_graph_warmup_mode_does_not_live_capture_missing_shape():
    runner = PrefillCudaGraphRunner(name="t", default_enabled=True, default_warmup=True)
    stats = ForwardStats()

    out = runner.maybe_run(
        kv_pool=None,
        num_blocks=0,
        num_tokens=8,
        max_kv_tokens=8,
        batch_size=1,
        input_ids=torch.zeros(8, dtype=torch.long),
        positions=torch.arange(8, dtype=torch.long),
        attention_metadata=None,
        last_token_indices=torch.tensor([7], dtype=torch.long),
        raw_num_tokens=8,
        ctx=ForwardContext(stats=stats),
        forward_fn=lambda state: pytest.fail("missing warmup shape must not capture during serving"),
    )

    assert out is None
    assert stats.cuda_graph_misses == 1
    assert stats.cuda_graph_captures == 0


def test_text_graph_runner_routes_cached_prefix_prefill_to_prefill_runner():
    class PrefillStub:
        def enabled(self) -> bool:
            return True

        def bucket_kv_tokens(self, max_kv_tokens: int, *, max_context_len: int = 0) -> int:
            return int(max_context_len or max_kv_tokens)

        def can_use(self, *args, **kwargs) -> bool:
            return True

        def maybe_run(self, **kwargs):
            self.kwargs = kwargs
            return torch.ones(1, _VOCAB)

    prefill = PrefillStub()

    runner = TextGraphRunner.__new__(TextGraphRunner)
    runner._prefill = prefill
    runner.kv_pool = None
    runner.num_blocks = 0
    runner.max_context_len = 128
    metadata = TextAttentionMetadata(
        cache=None,
        block_table=None,
        cache_seqlens=None,
        cache_seqlens_cpu=(64,),
        query_lens=None,
        query_lens_cpu=(8,),
        kv_seqlens=None,
        kv_seqlens_cpu=(72,),
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=8,
        max_seqlen_k=72,
        mode=ForwardMode.EXTEND,
    )
    fb = SimpleNamespace(
        batch_size=1,
        last_token_indices=torch.tensor([7], dtype=torch.long),
        spec_token_ids=[],
        num_token_non_padded=8,
    )

    out = runner._maybe_prefill(
        None,
        torch.zeros(8, dtype=torch.long),
        torch.arange(8, dtype=torch.long),
        fb,
        metadata,
        ForwardContext(stats=ForwardStats()),
    )

    assert out is not None
    assert prefill.kwargs["max_kv_tokens"] == 128
    assert prefill.kwargs["batch_size"] == 1


@requires_cuda
def test_text_graph_runner_routes_mixed_extend_decode_to_prefill_runner():
    """A mixed extend+decode group replays through the prefill graph buckets."""

    runner = TextGraphRunner.__new__(TextGraphRunner)
    calls = {}
    runner._maybe_decode = lambda *a, **k: calls.setdefault("decode", True)
    runner._maybe_prefill = lambda *a, **k: calls.setdefault("prefill", True) or torch.ones(2, _VOCAB)
    metadata = SimpleNamespace(cache=object.__new__(BatchedPagedRequestCache))
    fb = SimpleNamespace(forward_mode=ForwardMode.MIXED, attn_metadata=metadata)
    out = runner.maybe_run(
        None,
        torch.zeros(9, dtype=torch.long, device="cuda"),
        torch.arange(9, dtype=torch.long, device="cuda"),
        fb,
        ForwardContext(stats=ForwardStats()),
    )
    assert out is not None
    assert "prefill" in calls and "decode" not in calls


def test_reorder_mixed_puts_roomy_row_last_for_bucket_padding():
    """A block-aligned final row swaps with one whose block tail can absorb pad."""

    from uniserve_worker.contracts.batches import TextBatch

    class PrefillStub:
        def enabled(self) -> bool:
            return True

        def bucket_num_tokens(self, num_tokens: int) -> int:
            return 80

    runner = TextGraphRunner.__new__(TextGraphRunner)
    runner._prefill = PrefillStub()
    runner.block_size = 64
    ops = (
        {"req_id": 1, "kind": "decode_und", "token_ids": [7], "pos_range": [10, 11]},
        {"req_id": 2, "kind": "prefill_und", "token_ids": list(range(64)), "pos_range": [0, 64]},
    )
    text = TextBatch.from_ops(
        ForwardMode.MIXED,
        ops,
        op_modes=(ForwardMode.DECODE, ForwardMode.EXTEND),
        allow_mixed_text=True,
    )
    out = runner.reorder_mixed_for_padding(text)
    # pad = 80 - 65 = 15: the aligned extend row (end=64) cannot absorb it, the
    # decode row (end=11, 53 tokens of tail room) can - it must move last.
    assert out.req_ids == (2, 1)
    assert out.pos_ranges[-1] == (10, 11)
    assert out.token_ids[0] == tuple(range(64))
    # already-roomy last row stays untouched
    assert runner.reorder_mixed_for_padding(out) is out


def test_padded_prefill_tokens_accepts_mixed_mode():
    """Token-bucket padding applies to mixed groups so they hit captured buckets."""

    class PrefillStub:
        def enabled(self) -> bool:
            return True

        def bucket_num_tokens(self, num_tokens: int) -> int:
            return 8

    runner = TextGraphRunner.__new__(TextGraphRunner)
    runner._prefill = PrefillStub()
    runner.block_size = 64
    text = SimpleNamespace(
        mode=ForwardMode.MIXED,
        spec_token_ids=[(), ()],
        token_ids=[(5,), (1, 2, 3)],
        pos_ranges=[(64, 65), (0, 3)],
    )
    assert runner.padded_num_tokens(text, attention_backend_name=None) == 8


@requires_cuda
def test_prefill_graph_inputs_preserve_cached_prefix_lengths():
    device = torch.device("cuda")
    pool = _kv_pool(device, num_blocks=16, block_size=4)
    cache = BatchedPagedRequestCache(pool, [[0, 1, 2, 3], [4, 5, 6, 7]], [5, 7])
    query_lens = torch.tensor([2, 3], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([7, 10], dtype=torch.int32, device=device)
    metadata = TextAttentionMetadata(
        cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=(5, 7),
        query_lens=query_lens,
        query_lens_cpu=(2, 3),
        kv_seqlens=kv_lens,
        kv_seqlens_cpu=(7, 10),
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, 7, 17], dtype=torch.int32, device=device),
        max_seqlen_q=3,
        max_seqlen_k=10,
        mode=ForwardMode.EXTEND,
    )
    state = make_text_initial_prefill_graph_state(
        kv_pool=pool,
        num_blocks=16,
        num_tokens=8,
        max_kv_tokens=16,
        batch_size=2,
        device=device,
    )

    copy_text_initial_prefill_graph_inputs(
        state,
        input_ids=torch.arange(5, dtype=torch.long, device=device),
        positions=torch.arange(5, dtype=torch.long, device=device),
        attention_metadata=metadata,
        raw_num_tokens=5,
        last_token_indices=torch.tensor([1, 4], dtype=torch.long, device=device),
    )

    assert state.cache.base_lens == [5, 7]
    assert state.metadata.cache_seqlens_cpu == (5, 7)
    assert state.metadata.query_lens_cpu == (2, 6)
    assert state.metadata.kv_seqlens_cpu == (7, 13)
    assert state.metadata.max_seqlen_k == 16
    assert state.cache_seqlens.cpu().tolist() == [5, 7]
    assert state.query_lens.cpu().tolist() == [2, 6]
    assert state.kv_seqlens.cpu().tolist() == [7, 13]
    assert state.cu_seqlens_q.cpu().tolist() == [0, 2, 8]
    assert state.cu_seqlens_k.cpu().tolist() == [0, 7, 20]


@requires_cuda
def test_prefill_graph_inputs_pad_short_batch_to_graph_bucket():
    device = torch.device("cuda")
    pool = _kv_pool(device, num_blocks=32, block_size=4)
    metadata = _prefill_metadata(
        pool,
        block_ids_by_row=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        cache_seqlens_cpu=(5, 6, 7),
        query_lens_cpu=(2, 3, 1),
        device=device,
        max_context_len=32,
    )
    state = make_text_initial_prefill_graph_state(
        kv_pool=pool,
        num_blocks=32,
        num_tokens=8,
        max_kv_tokens=32,
        batch_size=4,
        device=device,
        max_context_len=32,
    )

    copy_text_initial_prefill_graph_inputs(
        state,
        input_ids=torch.arange(6, dtype=torch.long, device=device),
        positions=torch.arange(6, dtype=torch.long, device=device),
        attention_metadata=metadata,
        raw_num_tokens=6,
        last_token_indices=torch.tensor([1, 4, 5], dtype=torch.long, device=device),
    )
    torch.cuda.synchronize()

    assert state.metadata.cache_seqlens_cpu == (5, 6, 7, 0)
    assert state.metadata.query_lens_cpu == (2, 3, 3, 0)
    assert state.metadata.kv_seqlens_cpu == (7, 9, 10, 0)
    assert state.cache_seqlens.cpu().tolist() == [5, 6, 7, 0]
    assert state.query_lens.cpu().tolist() == [2, 3, 3, 0]
    assert state.cu_seqlens_q.cpu().tolist() == [0, 2, 5, 8, 8]
    assert state.last_token_indices.cpu().tolist() == [1, 4, 5, 0]


def test_padded_prefill_max_kv_tokens_accounts_for_query_padding():
    metadata = TextAttentionMetadata(
        cache=None,
        block_table=None,
        cache_seqlens=None,
        cache_seqlens_cpu=(1597,),
        query_lens=None,
        query_lens_cpu=(29,),
        kv_seqlens=None,
        kv_seqlens_cpu=(1626,),
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=29,
        max_seqlen_k=1626,
        mode=ForwardMode.EXTEND,
    )

    assert (
        _padded_prefill_max_kv_tokens(
            metadata,
            padded_tokens=64,
            raw_tokens=29,
            batch_size=1,
        )
        == 1661
    )


@requires_cuda
@pytest.mark.gpu
def test_prefill_graph_pads_short_batch_to_bucket_and_returns_unpadded_rows():
    device = torch.device("cuda")
    torch.manual_seed(7)
    model = _LinearLogitsModel(device, seed=7)
    pool = _kv_pool(device, num_blocks=32, block_size=4)
    runner = PrefillCudaGraphRunner(
        name="t",
        default_enabled=True,
        default_warmup=False,
        default_warmup_batch_sizes=(4,),
    )
    input_ids = torch.tensor([7, 9, 11, 5, 6, 13], dtype=torch.long, device=device)
    positions = torch.arange(6, dtype=torch.long, device=device)
    metadata = _prefill_metadata(
        pool,
        block_ids_by_row=[[0, 1], [2, 3], [4, 5]],
        cache_seqlens_cpu=(0, 0, 0),
        query_lens_cpu=(2, 2, 2),
        device=device,
        max_context_len=32,
    )
    last_token_indices = torch.tensor([1, 3, 5], dtype=torch.long, device=device)
    reference = model.logits(input_ids, positions).index_select(0, last_token_indices)

    stats = ForwardStats()
    out = runner.maybe_run(
        kv_pool=pool,
        num_blocks=32,
        num_tokens=8,
        max_kv_tokens=32,
        batch_size=3,
        input_ids=input_ids,
        positions=positions,
        attention_metadata=metadata,
        last_token_indices=last_token_indices,
        raw_num_tokens=6,
        ctx=ForwardContext(stats=stats),
        forward_fn=lambda state: model.logits(state.input_ids, state.positions).index_select(0, state.last_token_indices),
    )
    torch.cuda.synchronize()

    assert out is not None
    assert tuple(out.shape) == (3, _VOCAB)
    torch.testing.assert_close(out, reference)
    assert (8, 4, 32) in runner.states
    assert stats.cuda_graph_captures == 1
    assert stats.cuda_graph_replays == 1
    assert stats.cuda_graph_unpadded_tokens == 6
    assert stats.cuda_graph_padded_tokens == 2


# --------------------------------------------------------------------------- #
# Shared graph input buffer pool (CUDA tensors -> needs a device)
# --------------------------------------------------------------------------- #


@requires_cuda
def test_share_input_buffer_slices_largest_captured_buffer_for_smaller_bucket():
    pool: dict = {}
    device = torch.device("cuda")
    largest = torch.empty((8, 4), dtype=torch.int32, device=device)

    shared_largest = _share_input_buffer(pool, "block_table", largest, strict=True)
    smaller = torch.empty((3, 4), dtype=torch.int32, device=device)
    shared_smaller = _share_input_buffer(pool, "block_table", smaller, strict=True)

    # The first (largest) allocation is kept as-is; the smaller bucket is a view
    # that aliases the largest buffer's storage rather than a fresh allocation.
    assert shared_largest is largest
    assert tuple(shared_smaller.shape) == (3, 4)
    assert shared_smaller.data_ptr() == largest.data_ptr()


@requires_cuda
def test_share_input_buffer_strict_grows_for_larger_late_bucket():
    pool: dict = {}
    device = torch.device("cuda")
    small = torch.empty(4, dtype=torch.int32, device=device)
    shared_small = _share_input_buffer(pool, "cache_seqlens", small, strict=True)

    large = torch.empty(8, dtype=torch.int32, device=device)
    shared_large = _share_input_buffer(pool, "cache_seqlens", large, strict=True)

    # Late larger lazy buckets get a new resident buffer. Existing graph states
    # keep references to their original smaller tensors.
    assert shared_small is small
    assert shared_large is large
    assert shared_large.data_ptr() != shared_small.data_ptr()

    smaller_again = torch.empty(2, dtype=torch.int32, device=device)
    shared_smaller_again = _share_input_buffer(
        pool, "cache_seqlens", smaller_again, strict=True
    )
    assert tuple(shared_smaller_again.shape) == (2,)
    assert shared_smaller_again.data_ptr() == shared_large.data_ptr()


@requires_cuda
def test_share_input_buffer_pools_separately_by_dtype():
    pool: dict = {}
    device = torch.device("cuda")
    int_buf = torch.empty((4, 2), dtype=torch.int32, device=device)
    long_buf = torch.empty((2, 2), dtype=torch.long, device=device)

    shared_int = _share_input_buffer(pool, "shared", int_buf, strict=True)
    shared_long = _share_input_buffer(pool, "shared", long_buf, strict=True)

    # Same name but different dtype -> distinct pool entries, each its own buffer.
    assert shared_int is int_buf
    assert shared_long is long_buf
    assert len(pool) == 2


@requires_cuda
def test_module_level_decode_buffer_pool_slices_then_resets():
    _reset_for_testing()
    device = torch.device("cuda")
    try:
        largest = torch.empty((8, 2), dtype=torch.int32, device=device)
        first = _share_decode_graph_input_buffer("text_decode.block_table", largest)
        smaller = torch.empty((3, 2), dtype=torch.int32, device=device)
        sliced = _share_decode_graph_input_buffer("text_decode.block_table", smaller)

        assert first is largest
        assert sliced.data_ptr() == largest.data_ptr()

        _reset_for_testing()

        # After reset the pool is empty, so a fresh smaller tensor becomes the
        # new resident buffer (returned as itself, not a stale slice).
        fresh = torch.empty((2, 2), dtype=torch.int32, device=device)
        assert _share_decode_graph_input_buffer("text_decode.block_table", fresh) is fresh
    finally:
        _reset_for_testing()


# --------------------------------------------------------------------------- #
# Decode input copy padding (CUDA tensors)
# --------------------------------------------------------------------------- #


@requires_cuda
def test_decode_input_copy_pads_short_batch_to_bucket_with_zeros():
    device = torch.device("cuda")
    pool = _kv_pool(device)
    state = make_text_decode_graph_state(kv_pool=pool, num_blocks=16, batch_size=4, device=device)
    metadata = _decode_metadata(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(3, 5),
        kv_seqlens_cpu=(4, 6),
        device=device,
    )
    input_ids = torch.tensor([[11], [22]], dtype=torch.long, device=device)
    positions = torch.tensor([[3], [5]], dtype=torch.long, device=device)

    copy_text_decode_graph_inputs(
        state, input_ids=input_ids, positions=positions, attention_metadata=metadata
    )
    torch.cuda.synchronize()

    # The two real rows are copied; rows [2, 4) of the bucket are zero-padded.
    assert state.input_ids.flatten().tolist() == [11, 22, 0, 0]
    assert state.positions.flatten().tolist() == [3, 5, 0, 0]
    assert state.cache_seqlens.tolist() == [3, 5, 0, 0]


@requires_cuda
def test_decode_input_copy_extends_cpu_seqlen_mirrors():
    device = torch.device("cuda")
    pool = _kv_pool(device)
    state = make_text_decode_graph_state(kv_pool=pool, num_blocks=16, batch_size=4, device=device)
    metadata = _decode_metadata(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(3, 5),
        kv_seqlens_cpu=(4, 6),
        device=device,
    )

    copy_text_decode_graph_inputs(
        state,
        input_ids=torch.tensor([[11], [22]], dtype=torch.long, device=device),
        positions=torch.tensor([[3], [5]], dtype=torch.long, device=device),
        attention_metadata=metadata,
    )

    # cache_seqlens_cpu pads with 0 (no resident KV); kv_seqlens_cpu pads with 1
    # (the padded rows still attend over a single synthetic token).
    assert state.metadata.cache_seqlens_cpu == (3, 5, 0, 0)
    assert state.metadata.kv_seqlens_cpu == (4, 6, 1, 1)


# --------------------------------------------------------------------------- #
# Eager-vs-graph replay (full capture + replay on CUDA)
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.gpu
def test_decode_graph_replay_matches_eager_for_exact_bucket_batch():
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _LinearLogitsModel(device, seed=0)
    pool = _kv_pool(device)
    runner = DecodeCudaGraphRunner(name="t", default_warmup_batch_sizes=(1, 2, 4))

    input_ids = torch.tensor([[7], [9]], dtype=torch.long, device=device)
    positions = torch.tensor([[3], [5]], dtype=torch.long, device=device)
    metadata = _decode_metadata(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(3, 5),
        kv_seqlens_cpu=(4, 6),
        device=device,
    )
    reference = model.logits(input_ids, positions)

    stats = ForwardStats()
    out = runner.maybe_run(
        kv_pool=pool,
        num_blocks=16,
        batch_size=2,
        input_ids=input_ids,
        positions=positions,
        attention_metadata=metadata,
        ctx=ForwardContext(stats=stats),
        forward_fn=lambda state: model.logits(state.input_ids, state.positions),
        prepare_backend=None,
    )
    torch.cuda.synchronize()

    assert out is not None
    # batch == an exact bucket, so the returned tensor has exactly the batch rows.
    assert tuple(out.shape) == (2, _VOCAB)
    torch.testing.assert_close(out, reference)
    # First call is a capture+replay: one capture, one replay, no padding.
    assert stats.cuda_graph_captures == 1
    assert stats.cuda_graph_replays == 1
    assert stats.cuda_graph_unpadded_tokens == 2
    assert stats.cuda_graph_padded_tokens == 0


@requires_cuda
@pytest.mark.gpu
def test_decode_graph_second_call_replays_without_recapture():
    device = torch.device("cuda")
    torch.manual_seed(1)
    model = _LinearLogitsModel(device, seed=1)
    pool = _kv_pool(device)
    runner = DecodeCudaGraphRunner(name="t", default_warmup_batch_sizes=(1, 2, 4))

    def run(ids, pos, seqlens, stats):
        metadata = _decode_metadata(
            pool,
            block_ids_by_row=[[0], [1]],
            cache_seqlens_cpu=seqlens,
            kv_seqlens_cpu=tuple(s + 1 for s in seqlens),
            device=device,
        )
        return runner.maybe_run(
            kv_pool=pool,
            num_blocks=16,
            batch_size=2,
            input_ids=ids,
            positions=pos,
            attention_metadata=metadata,
            ctx=ForwardContext(stats=stats),
            forward_fn=lambda state: model.logits(state.input_ids, state.positions),
            prepare_backend=None,
        )

    first_stats = ForwardStats()
    run(
        torch.tensor([[7], [9]], dtype=torch.long, device=device),
        torch.tensor([[3], [5]], dtype=torch.long, device=device),
        (3, 5),
        first_stats,
    )

    ids2 = torch.tensor([[5], [6]], dtype=torch.long, device=device)
    pos2 = torch.tensor([[1], [2]], dtype=torch.long, device=device)
    reference2 = model.logits(ids2, pos2)
    second_stats = ForwardStats()
    out2 = run(ids2, pos2, (1, 2), second_stats)
    torch.cuda.synchronize()

    # The bucket is already captured: the second call only replays.
    assert second_stats.cuda_graph_captures == 0
    assert second_stats.cuda_graph_replays == 1
    torch.testing.assert_close(out2, reference2)


@requires_cuda
@pytest.mark.gpu
def test_decode_graph_pads_short_batch_to_bucket_and_returns_unpadded_rows():
    device = torch.device("cuda")
    torch.manual_seed(2)
    model = _LinearLogitsModel(device, seed=2)
    pool = _kv_pool(device)
    # Only bucket 4 is configured, so a batch of 3 is padded up to 4.
    runner = DecodeCudaGraphRunner(name="t", default_warmup_batch_sizes=(4,))

    input_ids = torch.tensor([[7], [9], [11]], dtype=torch.long, device=device)
    positions = torch.tensor([[2], [3], [1]], dtype=torch.long, device=device)
    metadata = _decode_metadata(
        pool,
        block_ids_by_row=[[0], [1], [2]],
        cache_seqlens_cpu=(2, 3, 1),
        kv_seqlens_cpu=(3, 4, 2),
        device=device,
    )
    reference = model.logits(input_ids, positions)

    stats = ForwardStats()
    out = runner.maybe_run(
        kv_pool=pool,
        num_blocks=16,
        batch_size=3,
        input_ids=input_ids,
        positions=positions,
        attention_metadata=metadata,
        ctx=ForwardContext(stats=stats),
        forward_fn=lambda state: model.logits(state.input_ids, state.positions),
        prepare_backend=None,
    )
    torch.cuda.synchronize()

    assert out is not None
    # batch 3 rounds up to the captured bucket 4...
    assert runner.resolve_bucket(3) == 4
    # ...but the returned tensor is sliced back to the unpadded request count.
    assert tuple(out.shape) == (3, _VOCAB)
    torch.testing.assert_close(out, reference)
    # The padded row contributes to the padded-token counter, not the unpadded.
    assert stats.cuda_graph_unpadded_tokens == 3
    assert stats.cuda_graph_padded_tokens == 1


@requires_cuda
@pytest.mark.gpu
def test_decode_graph_state_buffers_are_bucket_sized_and_zero_padded():
    device = torch.device("cuda")
    torch.manual_seed(3)
    model = _LinearLogitsModel(device, seed=3)
    pool = _kv_pool(device)
    runner = DecodeCudaGraphRunner(name="t", default_warmup_batch_sizes=(4,))

    runner.maybe_run(
        kv_pool=pool,
        num_blocks=16,
        batch_size=3,
        input_ids=torch.tensor([[7], [9], [11]], dtype=torch.long, device=device),
        positions=torch.tensor([[2], [3], [1]], dtype=torch.long, device=device),
        attention_metadata=_decode_metadata(
            pool,
            block_ids_by_row=[[0], [1], [2]],
            cache_seqlens_cpu=(2, 3, 1),
            kv_seqlens_cpu=(3, 4, 2),
            device=device,
        ),
        ctx=ForwardContext(stats=ForwardStats()),
        forward_fn=lambda state: model.logits(state.input_ids, state.positions),
        prepare_backend=None,
    )
    torch.cuda.synchronize()

    state = runner.states[4]
    # The captured buffers are sized to the bucket (4), with the tail zero-padded
    # and the host seqlen mirrors extended (cache pads 0, kv pads 1).
    assert state.batch_size == 4
    assert state.input_ids.flatten().tolist() == [7, 9, 11, 0]
    assert state.cache_seqlens.tolist() == [2, 3, 1, 0]
    assert state.metadata.cache_seqlens_cpu == (2, 3, 1, 0)
    assert state.metadata.kv_seqlens_cpu == (3, 4, 2, 1)


# --------------------------------------------------------------------------- #
# TextGraphRunner end-to-end dispatch (CUDA)
# --------------------------------------------------------------------------- #


@requires_cuda
@requires_decode_graph_backend
@pytest.mark.gpu
def test_text_graph_runner_decode_replay_matches_eager_forward():
    device = torch.device("cuda")
    torch.manual_seed(4)
    model = _LinearLogitsModel(device, seed=4)
    # FlashInfer's decode kernels require a real head_dim; the graph runner now
    # re-plans the backend before every replay, so the pool must be plannable.
    pool = _kv_pool(device, head_dim=64)
    runner = TextGraphRunner(kv_pool=pool, num_blocks=16, block_size=4, device=device)

    input_ids = torch.tensor([[7], [9]], dtype=torch.long, device=device)
    positions = torch.tensor([[2], [3]], dtype=torch.long, device=device)
    metadata = _decode_metadata(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(2, 3),
        kv_seqlens_cpu=(3, 4),
        device=device,
    )
    fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        req_ids=(0, 1),
        input_ids=input_ids,
        positions=positions,
        attn_metadata=metadata,
    )
    reference = model.forward(input_ids, positions, fb)

    stats = ForwardStats()
    out = runner.maybe_run(model, input_ids, positions, fb, ForwardContext(stats=stats))
    torch.cuda.synchronize()

    assert out is not None
    assert tuple(out.shape) == (2, _VOCAB)
    torch.testing.assert_close(out, reference)
    assert stats.cuda_graph_captures == 1
    assert stats.cuda_graph_replays == 1


@requires_cuda
@pytest.mark.gpu
def test_text_graph_runner_skips_non_decode_extend_mode_without_prefill_graph():
    device = torch.device("cuda")
    torch.manual_seed(5)
    model = _LinearLogitsModel(device, seed=5)
    pool = _kv_pool(device)
    # Default worker config leaves prefill CUDA graphs disabled, so an ENCODE
    # (neither DECODE nor EXTEND) forward is never graphed -> the runner returns
    # None so the caller falls back to eager.
    runner = TextGraphRunner(kv_pool=pool, num_blocks=16, block_size=4, device=device)

    input_ids = torch.tensor([[7], [9]], dtype=torch.long, device=device)
    positions = torch.tensor([[2], [3]], dtype=torch.long, device=device)
    metadata = _decode_metadata(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(2, 3),
        kv_seqlens_cpu=(3, 4),
        device=device,
    )
    fb = ForwardBatch(
        forward_mode=ForwardMode.ENCODE,
        req_ids=(0, 1),
        input_ids=input_ids,
        positions=positions,
        attn_metadata=metadata,
    )

    assert runner.maybe_run(model, input_ids, positions, fb, ForwardContext(stats=ForwardStats())) is None


@requires_cuda
@pytest.mark.gpu
def test_text_graph_runner_skips_cpu_input_tensors():
    device = torch.device("cuda")
    torch.manual_seed(6)
    model = _LinearLogitsModel(device, seed=6)
    pool = _kv_pool(device)
    runner = TextGraphRunner(kv_pool=pool, num_blocks=16, block_size=4, device=device)

    metadata = _decode_metadata(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(2, 3),
        kv_seqlens_cpu=(3, 4),
        device=device,
    )
    input_ids_cpu = torch.tensor([[7], [9]], dtype=torch.long)
    positions_cpu = torch.tensor([[2], [3]], dtype=torch.long)
    fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        req_ids=(0, 1),
        input_ids=input_ids_cpu,
        positions=positions_cpu,
        attn_metadata=metadata,
    )

    # CUDA graphs require device inputs; a CPU forward is not graphed.
    assert (
        runner.maybe_run(model, input_ids_cpu, positions_cpu, fb, ForwardContext(stats=ForwardStats()))
        is None
    )
