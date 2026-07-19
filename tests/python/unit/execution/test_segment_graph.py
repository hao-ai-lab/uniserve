"""Unit coverage for general packed-forward graph lowering."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.contracts.forward_context import ForwardContext
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.execution import segment as packed_runtime
from uniserve_worker.execution import segment as pmg
from uniserve_worker.execution.segment import (
    SegmentGraphRunner,
    packed_graph_promotions_supported,
)
from uniserve_worker.runtime.forward_stream import (
    ForwardPagedKVSegment,
    ForwardPagedKVView,
    ForwardStreamBuilder,
)
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import (
    PagedTextCache,
    PagedTextCacheSpanCopy,
    copy_paged_text_cache_span,
)

pytestmark = pytest.mark.unit


def test_segment_graph_runner_is_enabled_by_default():
    assert SegmentGraphRunner().enabled()


def test_packed_forward_graph_reclaims_lru_capacity_after_releasing_state(monkeypatch):
    events: list[str] = []

    class Graph:
        def reset(self) -> None:
            events.append("reset")

    class State:
        def __init__(self, index: int) -> None:
            self.graph = Graph()
            self.release_backend = lambda: events.append(f"release_backend_{index}")

    runner = SegmentGraphRunner()
    victim_ref = None
    for index in range(pmg._MAX_RESIDENT_CAPACITIES):
        state = State(index)
        if index == 0:
            victim_ref = weakref.ref(state)
        key = (index,)
        runner.states[key] = state
        runner._touch_capacity(key)
    del state
    assert victim_ref is not None

    def empty_cache() -> None:
        assert victim_ref() is None
        events.append("empty_cache")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    runner._admit_capacity(
        (pmg._MAX_RESIDENT_CAPACITIES,),
        device=torch.device("cuda:0"),
    )

    assert (0,) not in runner.states
    assert len(runner.states) == pmg._MAX_RESIDENT_CAPACITIES - 1
    assert events == ["synchronize", "reset", "release_backend_0", "empty_cache"]


def _decode_stream(prefix_len: int):
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=int(prefix_len),
        visible_policy="causal",
    )
    return builder.build(device="cpu")


def _prefill_stream(prefix_len: int):
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="prefill_und",
        mode=ForwardMode.EXTEND,
        modality="und",
        segment_class="extend",
        q_len=2,
        prefix_len=int(prefix_len),
        visible_policy="causal",
    )
    return builder.build(device="cpu")


def _ragged_stream(q_lens: tuple[int, ...]):
    builder = ForwardStreamBuilder()
    for row, q_len in enumerate(q_lens):
        builder.add_segment(
            op_index=row,
            req_id=row + 1,
            kind="prefill_und",
            mode=ForwardMode.EXTEND,
            modality="und",
            segment_class="extend",
            q_len=q_len,
            prefix_len=row * 4,
            visible_policy="causal",
        )
    return builder.build(device="cpu")


def _resolver_inputs():
    pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    attention = SimpleNamespace(
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        scale=0.5,
    )
    owner = SimpleNamespace(
        segment_graph_attention=lambda: attention,
    )
    embeds = torch.zeros((1, 4), dtype=torch.float32)
    indicators = torch.zeros((1,), dtype=torch.bool)
    kv_view = ForwardPagedKVView(
        pool,
        [ForwardPagedKVSegment(block_ids=(0,), base_len=0, q_len=1)],
    )
    return owner, embeds, indicators, _decode_stream(prefix_len=0), kv_view


class _FakeAttentionBackend:
    def __init__(self, name: str, *, graph_capable: bool) -> None:
        self.name = name
        self.graph_capable = bool(graph_capable)

    def capabilities(self):
        return SimpleNamespace(visible_end_cuda_graph=self.graph_capable)


class _FakeAttentionProvider:
    def __init__(self, backend: _FakeAttentionBackend) -> None:
        self.name = backend.name
        self.backend = backend

    def can_run(self, req) -> bool:
        del req
        return True


class _FakeAttentionDispatcher:
    def __init__(self, providers) -> None:
        self.providers = tuple(providers)

    def ordered(self, override=None):
        del override
        return self.providers


def test_packed_forward_graph_backend_resolver_skips_non_graph_auto_provider(monkeypatch):
    non_graph = _FakeAttentionBackend("fa4_cute", graph_capable=False)
    graph = _FakeAttentionBackend("flashinfer", graph_capable=True)
    monkeypatch.setattr(
        pmg.ops,
        "attention_dispatcher",
        lambda: _FakeAttentionDispatcher(
            (_FakeAttentionProvider(non_graph), _FakeAttentionProvider(graph))
        ),
    )
    owner, embeds, _indicators, stream, kv_view = _resolver_inputs()

    backend = SegmentGraphRunner()._resolve_graph_backend(
        ForwardContext(attention_preference="auto"),
        owner,
        embeds,
        stream,
        kv_view,
    )

    assert backend is graph


def test_packed_forward_graph_backend_resolver_respects_explicit_non_graph_provider(
    monkeypatch,
):
    non_graph = _FakeAttentionBackend("fa4_cute", graph_capable=False)
    graph = _FakeAttentionBackend("flashinfer", graph_capable=True)
    monkeypatch.setattr(
        pmg.ops,
        "attention_dispatcher",
        lambda: _FakeAttentionDispatcher(
            (_FakeAttentionProvider(non_graph), _FakeAttentionProvider(graph))
        ),
    )
    owner, embeds, _indicators, stream, kv_view = _resolver_inputs()

    backend = SegmentGraphRunner()._resolve_graph_backend(
        ForwardContext(attention_preference="fa4_cute"),
        owner,
        embeds,
        stream,
        kv_view,
    )

    assert backend is None


def test_packed_forward_graph_backend_resolver_accepts_causal_visible_end_graph(monkeypatch):
    graph = _FakeAttentionBackend("fa4_cute", graph_capable=True)

    class Provider(_FakeAttentionProvider):
        def can_run(self, req) -> bool:
            assert req.regime is pmg.ops.AttentionRegime.VISIBLE_END
            assert req.visible_end is not None
            assert req.fully_visible is False
            return True

    monkeypatch.setattr(
        pmg.ops,
        "attention_dispatcher",
        lambda: _FakeAttentionDispatcher((Provider(graph),)),
    )
    owner, embeds, _indicators, _stream, kv_view = _resolver_inputs()

    backend = SegmentGraphRunner()._resolve_graph_backend(
        ForwardContext(attention_preference="auto"),
        owner,
        embeds,
        _prefill_stream(prefix_len=0),
        kv_view,
    )

    assert backend is graph


def _write_cache_span(cache: PagedTextCache, *, start: int, length: int) -> None:
    shape = (int(length), cache.pool.n_kv, cache.pool.head_dim)
    numel = int(length) * int(cache.pool.n_kv) * int(cache.pool.head_dim)
    values = torch.arange(numel, dtype=torch.float32).reshape(shape)
    for layer_idx in range(cache.pool.num_layers):
        k = values + float(layer_idx * 100)
        v = values.mul(-1) - float(layer_idx * 100)
        cache.pool.write(layer_idx, cache.block_ids, start=int(start), k=k, v=v)


def test_packed_forward_graph_key_reuses_paged_kv_capacity_bucket():
    pool = PagedKVPool(1, 6, 4, 1, 2, device="cpu", dtype=torch.float32)
    owner = SimpleNamespace()
    backend = SimpleNamespace(name="graph_backend")
    embeds = torch.zeros((1, 4), dtype=torch.float32)
    indicators = torch.zeros((1,), dtype=torch.bool)

    key_a = SegmentGraphRunner._graph_key(
        owner,
        embeds,
        indicators,
        _decode_stream(prefix_len=1),
        ForwardPagedKVView(
            pool,
            [ForwardPagedKVSegment(block_ids=(0, 1, 2), base_len=1, q_len=1)],
        ),
        backend,
    )
    key_b = SegmentGraphRunner._graph_key(
        owner,
        embeds,
        indicators,
        _decode_stream(prefix_len=2),
        ForwardPagedKVView(
            pool,
            [ForwardPagedKVSegment(block_ids=(0, 1, 2, 3), base_len=2, q_len=1)],
        ),
        backend,
    )
    key_c = SegmentGraphRunner._graph_key(
        owner,
        embeds,
        indicators,
        _decode_stream(prefix_len=2),
        ForwardPagedKVView(
            pool,
            [ForwardPagedKVSegment(block_ids=(0, 1, 2, 3, 4), base_len=2, q_len=1)],
        ),
        backend,
    )

    assert key_a == key_b
    assert key_c != key_a


def test_packed_forward_graph_key_excludes_ragged_query_distribution():
    pool = PagedKVPool(1, 8, 4, 1, 2, device="cpu", dtype=torch.float32)
    owner = SimpleNamespace()
    backend = SimpleNamespace(name="graph_backend")
    embeds = torch.zeros((3, 4), dtype=torch.float32)
    indicators = torch.zeros((3,), dtype=torch.bool)

    key_a = SegmentGraphRunner._graph_key(
        owner,
        embeds,
        indicators,
        _ragged_stream((1, 2)),
        ForwardPagedKVView(
            pool,
            [
                ForwardPagedKVSegment(block_ids=(0,), base_len=0, q_len=1),
                ForwardPagedKVSegment(block_ids=(1, 2), base_len=3, q_len=2),
            ],
        ),
        backend,
    )
    key_b = SegmentGraphRunner._graph_key(
        owner,
        embeds,
        indicators,
        _ragged_stream((2, 1)),
        ForwardPagedKVView(
            pool,
            [
                ForwardPagedKVSegment(block_ids=(3,), base_len=0, q_len=2),
                ForwardPagedKVSegment(block_ids=(4, 5), base_len=3, q_len=1),
            ],
        ),
        backend,
    )

    assert key_a == key_b


def test_packed_forward_stream_capacity_does_not_read_device_side_tables():
    class DeviceTable:
        shape = (3,)

        def __getitem__(self, index):
            raise AssertionError(f"capacity lookup synchronized a device side table at {index}")

    stream = SimpleNamespace(
        segments=(SimpleNamespace(q_len=1), SimpleNamespace(q_len=2)),
        cu_seqlens_q=DeviceTable(),
        visible_end=SimpleNamespace(shape=(2, 2)),
        indexes=SimpleNamespace(shape=(3,)),
        fully_visible=False,
    )

    assert pmg._stream_capacity(stream) == (
        2,
        3,
        (3,),
        (2, 2),
        (3,),
        False,
    )


def test_packed_forward_graph_promotes_only_single_direct_pool_pair():
    source_pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    target_pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    other_target_pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    layer_mismatch_pool = PagedKVPool(2, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    source = PagedTextCache(source_pool, [0], num_layers=1)
    target = PagedTextCache(target_pool, [0], num_layers=1)
    other_target = PagedTextCache(other_target_pool, [0], num_layers=1)
    layer_mismatch_target = PagedTextCache(layer_mismatch_pool, [0], num_layers=2)

    assert packed_graph_promotions_supported(
        (PagedTextCacheSpanCopy(source, target, start=0, length=1),)
    )
    assert not packed_graph_promotions_supported(
        (
            PagedTextCacheSpanCopy(source, target, start=0, length=1),
            PagedTextCacheSpanCopy(source, other_target, start=0, length=1),
        )
    )
    assert not packed_graph_promotions_supported(
        (PagedTextCacheSpanCopy(source, layer_mismatch_target, start=0, length=1),)
    )


def test_packed_forward_graph_promotion_copy_matches_span_copy_across_layers():
    source_pool = PagedKVPool(2, 6, 3, 1, 2, device="cpu", dtype=torch.float32)
    graph_target_pool = PagedKVPool(2, 6, 3, 1, 2, device="cpu", dtype=torch.float32)
    expected_target_pool = PagedKVPool(2, 6, 3, 1, 2, device="cpu", dtype=torch.float32)
    source = PagedTextCache(source_pool, [0, 1], num_layers=2)
    graph_target = PagedTextCache(graph_target_pool, [2, 3], num_layers=2)
    expected_target = PagedTextCache(expected_target_pool, [2, 3], num_layers=2)
    _write_cache_span(source, start=1, length=4)

    state = SimpleNamespace(
        promotion_source_pool=source_pool,
        promotion_target_pool=graph_target_pool,
        promotion_source_index=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        promotion_target_index=torch.tensor([7, 8, 9, 10], dtype=torch.long),
    )
    SegmentGraphRunner._copy_promotions_in_graph(state)
    copy_paged_text_cache_span(source, expected_target, start=1, length=4, num_layers=2)

    assert torch.equal(graph_target.pool.k, expected_target.pool.k)
    assert torch.equal(graph_target.pool.v, expected_target.pool.v)


def test_packed_forward_graph_promotion_indices_fill_static_capacity():
    source_pool = PagedKVPool(1, 2, 4, 1, 2, device="cpu", dtype=torch.float32)
    target_pool = PagedKVPool(1, 2, 4, 1, 2, device="cpu", dtype=torch.float32)
    source = PagedTextCache(source_pool, [0], num_layers=1)
    target = PagedTextCache(target_pool, [1], num_layers=1)
    promotion = PagedTextCacheSpanCopy(source, target, start=0, length=1)
    next_promotion = PagedTextCacheSpanCopy(source, target, start=1, length=1)

    _, _, source_positions, target_positions = pmg._promotion_index_values(
        (promotion,),
        capacity=4,
    )

    assert source_positions == [0, 0, 0, 0]
    assert target_positions == [4, 4, 4, 4]
    assert pmg._promotion_geometry((promotion,), capacity=4) == pmg._promotion_geometry(
        (promotion, next_promotion),
        capacity=4,
    )


def test_packed_decode_graph_padding_uses_reserved_kv_blocks():
    pool = PagedKVPool(
        1,
        4,
        64,
        1,
        2,
        device="cpu",
        dtype=torch.float32,
        reserved_tail_blocks=2,
    )
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=4,
        visible_policy="causal",
    )
    kv_segments = [ForwardPagedKVSegment(block_ids=(0,), base_len=4, q_len=1)]
    embed_chunks = [torch.ones((1, 4), dtype=torch.float32)]
    indicator_chunks: list[tuple[int, bool] | torch.Tensor] = [(1, False)]

    capacity = packed_runtime._append_decode_graph_padding(
        builder=builder,
        kv_segments=kv_segments,
        embed_chunks=embed_chunks,
        indicator_chunks=indicator_chunks,
        padding_pool=pool,
        decode_rows=1,
        hidden_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    stream = builder.build(device="cpu")
    kv_view = ForwardPagedKVView(pool, kv_segments)

    assert capacity == 128
    assert len(stream.segments) == 128
    assert sum(int(segment.q_len) for segment in stream.segments) == 128
    assert all(segment.mode is ForwardMode.DECODE for segment in stream.segments)
    assert all(segment.write_kv and segment.persist_kv for segment in kv_view.segments)
    assert [segment.base_len for segment in kv_view.segments[1:]] == list(range(127))
    assert all(segment.block_ids == (2, 3) for segment in kv_view.segments[1:])
    assert torch.cat(embed_chunks, dim=0).shape == (128, 4)
    assert sum(int(chunk[0]) for chunk in indicator_chunks) == 128
