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
    make_text_decode_graph_state,
)
from uniserve_worker.execution.text_graph_runner import TextGraphRunner
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import BatchedPagedRequestCache

pytestmark = pytest.mark.integration

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="decode CUDA graph capture/replay requires a CUDA device"
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
def test_share_input_buffer_strict_rejects_a_larger_bucket_than_captured():
    pool: dict = {}
    device = torch.device("cuda")
    _share_input_buffer(pool, "cache_seqlens", torch.empty(4, dtype=torch.int32, device=device), strict=True)

    with pytest.raises(AssertionError):
        _share_input_buffer(pool, "cache_seqlens", torch.empty(8, dtype=torch.int32, device=device), strict=True)


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
