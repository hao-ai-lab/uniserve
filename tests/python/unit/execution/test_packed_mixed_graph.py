from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.contracts.forward_context import ForwardContext
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.execution.forward.graph import packed_visible as pmg
from uniserve_worker.execution.forward.graph.packed_visible import (
    PACKED_MIXED_GRAPH_ENV,
    PackedMixedGraphRunner,
    packed_mixed_graph_promotions_supported,
)
from uniserve_worker.execution.forward.stream import (
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


def test_packed_mixed_graph_runner_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv(PACKED_MIXED_GRAPH_ENV, raising=False)

    assert PackedMixedGraphRunner().enabled()


def test_packed_mixed_graph_runner_env_can_disable(monkeypatch):
    monkeypatch.setenv(PACKED_MIXED_GRAPH_ENV, "0")

    assert not PackedMixedGraphRunner().enabled()


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


def _resolver_inputs():
    pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    owner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            self_attn=SimpleNamespace(
                                num_heads=1,
                                num_kv_heads=1,
                                head_dim=2,
                                scaling=0.5,
                            )
                        )
                    ]
                )
            )
        )
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
        return SimpleNamespace(paged_varlen_cuda_graph=self.graph_capable)


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


def test_packed_mixed_graph_backend_resolver_skips_non_graph_auto_provider(monkeypatch):
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

    backend = PackedMixedGraphRunner()._resolve_graph_backend(
        ForwardContext(attention_backend_name="auto"),
        owner,
        embeds,
        stream,
        kv_view,
    )

    assert backend is graph


def test_packed_mixed_graph_backend_resolver_respects_explicit_non_graph_provider(
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

    backend = PackedMixedGraphRunner()._resolve_graph_backend(
        ForwardContext(attention_backend_name="fa4_cute"),
        owner,
        embeds,
        stream,
        kv_view,
    )

    assert backend is None


def _write_cache_span(cache: PagedTextCache, *, start: int, length: int) -> None:
    shape = (int(length), cache.pool.n_kv, cache.pool.head_dim)
    numel = int(length) * int(cache.pool.n_kv) * int(cache.pool.head_dim)
    values = torch.arange(numel, dtype=torch.float32).reshape(shape)
    for layer_idx in range(cache.pool.num_layers):
        k = values + float(layer_idx * 100)
        v = values.mul(-1) - float(layer_idx * 100)
        cache.pool.write(layer_idx, cache.block_ids, start=int(start), k=k, v=v)


def test_packed_mixed_graph_key_reuses_block_capacity_across_base_lengths():
    pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    owner = SimpleNamespace()
    backend = SimpleNamespace(name="graph_backend")
    embeds = torch.zeros((1, 4), dtype=torch.float32)
    indicators = torch.zeros((1,), dtype=torch.bool)

    key_a = PackedMixedGraphRunner._graph_key(
        owner,
        embeds,
        indicators,
        _decode_stream(prefix_len=1),
        ForwardPagedKVView(
            pool,
            [ForwardPagedKVSegment(block_ids=(0, 1), base_len=1, q_len=1)],
        ),
        backend,
    )
    key_b = PackedMixedGraphRunner._graph_key(
        owner,
        embeds,
        indicators,
        _decode_stream(prefix_len=2),
        ForwardPagedKVView(
            pool,
            [ForwardPagedKVSegment(block_ids=(0, 1), base_len=2, q_len=1)],
        ),
        backend,
    )
    key_c = PackedMixedGraphRunner._graph_key(
        owner,
        embeds,
        indicators,
        _decode_stream(prefix_len=2),
        ForwardPagedKVView(
            pool,
            [ForwardPagedKVSegment(block_ids=(0, 1, 2), base_len=2, q_len=1)],
        ),
        backend,
    )

    assert key_a == key_b
    assert key_c != key_a


def test_packed_mixed_graph_promotes_only_single_direct_pool_pair():
    source_pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    target_pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    other_target_pool = PagedKVPool(1, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    layer_mismatch_pool = PagedKVPool(2, 4, 4, 1, 2, device="cpu", dtype=torch.float32)
    source = PagedTextCache(source_pool, [0], num_layers=1)
    target = PagedTextCache(target_pool, [0], num_layers=1)
    other_target = PagedTextCache(other_target_pool, [0], num_layers=1)
    layer_mismatch_target = PagedTextCache(layer_mismatch_pool, [0], num_layers=2)

    assert packed_mixed_graph_promotions_supported(
        (PagedTextCacheSpanCopy(source, target, start=0, length=1),)
    )
    assert not packed_mixed_graph_promotions_supported(
        (
            PagedTextCacheSpanCopy(source, target, start=0, length=1),
            PagedTextCacheSpanCopy(source, other_target, start=0, length=1),
        )
    )
    assert not packed_mixed_graph_promotions_supported(
        (PagedTextCacheSpanCopy(source, layer_mismatch_target, start=0, length=1),)
    )


def test_packed_mixed_graph_promotion_copy_matches_span_copy_across_layers():
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
    PackedMixedGraphRunner._copy_promotions_in_graph(state)
    copy_paged_text_cache_span(source, expected_target, start=1, length=4, num_layers=2)

    assert torch.equal(graph_target.pool.k, expected_target.pool.k)
    assert torch.equal(graph_target.pool.v, expected_target.pool.v)
