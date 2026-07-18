from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from uniserve_kernel import mm_attn_varlen
from uniserve_worker.backends.attention import get_attention_backend, has_attention_backend
from uniserve_worker.backends.attention.fa4_cute import _PREFIX_BOUNDS_CACHE, _cached_prefix_bounds
from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.models.sensenova import model as sensenova_u1
from uniserve_worker.runtime.forward_stream import ForwardPagedKVSegment, ForwardPagedKVView
from uniserve_worker.runtime.kv_pool import PagedKVPool

# Conformance tolerances for comparing the bf16 FA4 CUTE kernel against an fp32-promoted
# reference computed via SDPA. These are shared across the GPU tests so the bound is set in
# exactly one place rather than re-typed as a bare literal per assertion.
#
# KERNEL_*: a single attention op (one forward_paged / forward_visible_end call) vs SDPA.
#   The only error source is the bf16 matmul/softmax inside one kernel invocation, so a tight
#   1e-2 bound is appropriate.
# MODEL_*: a full one-layer _SenseNovaDecoderModel forward vs the eager reference. Error accumulates
#   through QKV/o projections, RoPE, MLP and the residual add (all bf16) on top of the attention
#   op, so the looser 4e-2 bound accounts for that extra accumulation.
KERNEL_CONFORMANCE_ATOL = 0.01
KERNEL_CONFORMANCE_RTOL = 0.01
MODEL_CONFORMANCE_ATOL = 0.04
MODEL_CONFORMANCE_RTOL = 0.04


def test_fa4_cute_prefix_bounds_use_packed_gqa_tile_space() -> None:
    _PREFIX_BOUNDS_CACHE.clear()
    visible_end = torch.tensor([[10, 20, 30]], dtype=torch.int32)
    cu_q = torch.tensor([0, 3], dtype=torch.int32)

    bounds = _cached_prefix_bounds(
        visible_end,
        cu_seqlens_q=cu_q,
        max_seqlen_q=3,
        qhead_per_kvhead=4,
        q_tile_size=4,
    )

    expected = torch.tensor([[[10, 10], [20, 20], [30, 30]]], dtype=torch.int32)
    torch.testing.assert_close(bounds, expected)


def test_fa4_cute_forward_paged_uses_plan_context_len_without_scalar_sync(monkeypatch) -> None:
    from uniserve_worker.backends.attention import fa4_cute

    backend = fa4_cute.Fa4CuteAttentionBackend()
    calls: list[dict[str, object]] = []

    def fake_flash_attn_fwd(q, k, v, **kwargs):
        del k, v
        calls.append(dict(kwargs))
        return q

    def forbid_item(self):  # pragma: no cover - only runs on regression.
        raise AssertionError("forward_paged must not read max sequence length through Tensor.item()")

    monkeypatch.setattr(fa4_cute, "_fa4_flash_attn_fwd", fake_flash_attn_fwd)
    monkeypatch.setattr(fa4_cute, "_write_paged_kv_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.Tensor, "item", forbid_item)

    q = torch.zeros((1, 1, 1, 128), dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    k_cache = torch.zeros((2, 64, 1, 128), dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cache_seqlens = torch.tensor([3], dtype=torch.int32)
    plan = SimpleNamespace(max_context_len=128)

    with use_forward_context(ForwardContext(attention_plan=plan)):
        out = backend.forward_paged(
            q,
            k_cache,
            v_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            k=k,
            v=v,
            causal=True,
            scale=128**-0.5,
        )

    assert out.shape == q.shape
    assert calls
    assert calls[0]["max_seqlen_k"] == 128


def _fa4_cute_unavailable_reason() -> str | None:
    """Return why the real FA4 CUTE kernel cannot run here, or None if it can.

    The real-kernel conformance tests need a CUDA device, importable FA4 CUTE and
    hybrid-mask provider packages, and a registered ``fa4_cute`` backend. Any of
    these missing means we cannot exercise the kernel.
    """
    if not torch.cuda.is_available():
        return "CUDA device not available"
    if not mm_attn_varlen.available():
        detail = mm_attn_varlen.import_error()
        return f"uniserve_kernel.mm_attn_varlen unavailable: {detail}"
    if not has_attention_backend("fa4_cute"):
        return "fa4_cute attention backend not registered"
    return None


def _require_fa4_cute() -> None:
    """Skip the test if the FA4 CUTE kernel is unavailable."""

    reason = _fa4_cute_unavailable_reason()
    if reason is None:
        return
    pytest.skip(f"requires CUDA and importable FA4 CUTE provider packages: {reason}")


@torch.inference_mode()
def test_fa4_cute_forward_paged_updates_cache_and_matches_dense_sdpa() -> None:
    _require_fa4_cute()
    backend = get_attention_backend("fa4_cute")
    batch = 2
    query_len = 3
    base_len = 5
    heads = 4
    kv_heads = 2
    head_dim = 128
    page_size = 8
    pages = 4
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(777)

    q = torch.randn(batch, heads, query_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, query_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, query_len, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(pages, page_size, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    cache_seqlens = torch.full((batch,), base_len, dtype=torch.int32, device=device)

    out = backend.forward_paged(
        q,
        k_cache,
        v_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        k=k,
        v=v,
        causal=False,
        scale=head_dim**-0.5,
    )
    torch.cuda.synchronize()

    refs = []
    for b in range(batch):
        full_k = torch.cat([k_cache[int(page)] for page in block_table[b]], dim=0)[:base_len + query_len]
        full_v = torch.cat([v_cache[int(page)] for page in block_table[b]], dim=0)[:base_len + query_len]
        full_k = full_k.repeat_interleave(heads // kv_heads, dim=1)
        full_v = full_v.repeat_interleave(heads // kv_heads, dim=1)
        refs.append(
            F.scaled_dot_product_attention(
                q[b].unsqueeze(0),
                full_k.transpose(0, 1).unsqueeze(0),
                full_v.transpose(0, 1).unsqueeze(0),
                scale=head_dim**-0.5,
            ).squeeze(0)
        )
    expected = torch.stack(refs, dim=0)
    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=KERNEL_CONFORMANCE_ATOL,
        rtol=KERNEL_CONFORMANCE_RTOL,
    )


@torch.inference_mode()
def test_fa4_cute_forward_visible_end_matches_dense_sdpa() -> None:
    _require_fa4_cute()
    backend = get_attention_backend("fa4_cute")
    batch = 2
    seqlen = 4
    heads = 4
    kv_heads = 2
    head_dim = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(778)

    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, kv_heads, head_dim, device=device, dtype=dtype)
    visible_end = torch.tensor([[1, 2, 4, 4], [4, 4, 4, 4]], dtype=torch.int32, device=device)

    out = backend.forward_visible_end(
        q,
        k,
        v,
        visible_end=visible_end,
        scale=head_dim**-0.5,
        use_prefix_bounds=False,
    )
    torch.cuda.synchronize()

    refs = []
    kv_index = torch.arange(seqlen, device=device)[None, :]
    for b in range(batch):
        mask = kv_index >= visible_end[b, :, None]
        additive = torch.zeros(seqlen, seqlen, device=device, dtype=dtype)
        additive.masked_fill_(mask, float("-inf"))
        refs.append(
            F.scaled_dot_product_attention(
                q[b].transpose(0, 1).unsqueeze(0),
                k[b].repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
                v[b].repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
                attn_mask=additive[None, None],
                scale=head_dim**-0.5,
            ).squeeze(0).transpose(0, 1)
        )
    expected = torch.stack(refs, dim=0)
    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=KERNEL_CONFORMANCE_ATOL,
        rtol=KERNEL_CONFORMANCE_RTOL,
    )


@pytest.mark.parametrize("use_prefix_bounds", [False, True])
@torch.inference_mode()
def test_fa4_cute_varlen_paged_visible_end_with_forward_kv_view_matches_sdpa(
    use_prefix_bounds: bool,
) -> None:
    _require_fa4_cute()
    backend = get_attention_backend("fa4_cute")
    heads = 4
    kv_heads = 2
    head_dim = 128
    block_size = 8
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(779)

    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    segments = [
        ForwardPagedKVSegment(block_ids=(0, 1), base_len=3, q_len=2),
        ForwardPagedKVSegment(block_ids=(2, 3), base_len=5, q_len=3),
    ]
    view = ForwardPagedKVView(pool, segments)
    for seg in segments:
        prefix_k = torch.randn(seg.base_len, kv_heads, head_dim, device=device, dtype=dtype)
        prefix_v = torch.randn_like(prefix_k)
        pool.write(0, list(seg.block_ids), start=0, k=prefix_k, v=prefix_v)

    q = torch.randn(sum(seg.q_len for seg in segments), heads, head_dim, device=device, dtype=dtype)
    k_current = torch.randn(sum(seg.q_len for seg in segments), kv_heads, head_dim, device=device, dtype=dtype)
    v_current = torch.randn_like(k_current)
    view.append_packed(0, k_current, v_current)

    visible_end = torch.zeros(len(segments), max(seg.q_len for seg in segments), dtype=torch.int32, device=device)
    visible_end[0, :2] = torch.tensor([4, 5], dtype=torch.int32, device=device)
    visible_end[1, :3] = 8
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    out = backend.forward_visible_end(
        q,
        pool.k[0],
        pool.v[0],
        visible_end=visible_end,
        cu_seqlens_q=cu_q,
        page_table=view.block_table(device=device),
        seqused_k=view.cache_seqlens_after(device=device),
        max_seqlen_q=3,
        max_seqlen_k=8,
        scale=head_dim**-0.5,
        use_prefix_bounds=use_prefix_bounds,
    )
    torch.cuda.synchronize()

    refs = []
    q_offset = 0
    for row, seg in enumerate(segments):
        q_row = q[q_offset : q_offset + seg.q_len]
        full_k, full_v = pool.read(0, list(seg.block_ids), start=0, length=seg.base_len + seg.q_len)
        assert full_k is not None and full_v is not None
        kv_index = torch.arange(full_k.shape[0], device=device)[None, :]
        mask = kv_index >= visible_end[row, : seg.q_len, None]
        additive = torch.zeros(seg.q_len, full_k.shape[0], device=device, dtype=dtype)
        additive.masked_fill_(mask, float("-inf"))
        refs.append(
            F.scaled_dot_product_attention(
                q_row.transpose(0, 1).unsqueeze(0),
                full_k.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
                full_v.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
                attn_mask=additive[None, None],
                scale=head_dim**-0.5,
            ).squeeze(0).transpose(0, 1)
        )
        q_offset += seg.q_len
    expected = torch.cat(refs, dim=0)
    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=KERNEL_CONFORMANCE_ATOL,
        rtol=KERNEL_CONFORMANCE_RTOL,
    )


@torch.inference_mode()
def test_fa4_cute_varlen_paged_fully_visible_skips_mask_and_matches_sdpa() -> None:
    _require_fa4_cute()
    backend = get_attention_backend("fa4_cute")
    heads = 4
    kv_heads = 2
    head_dim = 128
    block_size = 8
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(781)

    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    segments = [
        ForwardPagedKVSegment(block_ids=(0,), base_len=3, q_len=1),
        ForwardPagedKVSegment(block_ids=(1, 2), base_len=5, q_len=3),
    ]
    view = ForwardPagedKVView(pool, segments)
    for seg in segments:
        prefix_k = torch.randn(seg.base_len, kv_heads, head_dim, device=device, dtype=dtype)
        prefix_v = torch.randn_like(prefix_k)
        pool.write(0, list(seg.block_ids), start=0, k=prefix_k, v=prefix_v)

    q = torch.randn(sum(seg.q_len for seg in segments), heads, head_dim, device=device, dtype=dtype)
    k_current = torch.randn(sum(seg.q_len for seg in segments), kv_heads, head_dim, device=device, dtype=dtype)
    v_current = torch.randn_like(k_current)
    view.append_packed(0, k_current, v_current)

    visible_end = torch.zeros(len(segments), max(seg.q_len for seg in segments), dtype=torch.int32, device=device)
    for row, seg in enumerate(segments):
        visible_end[row, : seg.q_len] = seg.base_len + seg.q_len
    cu_q = torch.tensor([0, 1, 4], dtype=torch.int32, device=device)
    out = backend.forward_visible_end(
        q,
        pool.k[0],
        pool.v[0],
        visible_end=visible_end,
        cu_seqlens_q=cu_q,
        page_table=view.block_table(device=device),
        seqused_k=view.cache_seqlens_after(device=device),
        max_seqlen_q=3,
        max_seqlen_k=8,
        scale=head_dim**-0.5,
        fully_visible=True,
    )
    torch.cuda.synchronize()

    refs = []
    q_offset = 0
    for seg in segments:
        q_row = q[q_offset : q_offset + seg.q_len]
        full_k, full_v = pool.read(0, list(seg.block_ids), start=0, length=seg.base_len + seg.q_len)
        assert full_k is not None and full_v is not None
        refs.append(
            F.scaled_dot_product_attention(
                q_row.transpose(0, 1).unsqueeze(0),
                full_k.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
                full_v.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
                scale=head_dim**-0.5,
            ).squeeze(0).transpose(0, 1)
        )
        q_offset += seg.q_len
    expected = torch.cat(refs, dim=0)
    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=KERNEL_CONFORMANCE_ATOL,
        rtol=KERNEL_CONFORMANCE_RTOL,
    )


@torch.inference_mode()
def test_sensenova_packed_visible_path_uses_real_fa4_and_matches_dense() -> None:
    _require_fa4_cute()
    torch.manual_seed(780)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = Qwen3Config(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 128
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    cfg._attn_implementation = "eager"
    model = sensenova_u1._SenseNovaDecoderModel(cfg).to(device=device, dtype=dtype).eval()
    hidden = torch.randn(5, cfg.hidden_size, device=device, dtype=dtype)
    indicators = torch.tensor([False, False, True, True, True], device=device)
    indexes = torch.stack(
        [
            torch.tensor([0, 1, 2, 2, 2], device=device),
            torch.tensor([0, 0, 0, 0, 1], device=device),
            torch.tensor([0, 0, 0, 1, 0], device=device),
        ],
        dim=0,
    )
    dense_mask = torch.full((1, 1, 5, 5), float("-inf"), device=device, dtype=dtype)
    dense_mask[0, 0, 0, 0] = 0
    dense_mask[0, 0, 1, :2] = 0
    dense_mask[0, 0, 2:5, 2:5] = 0
    dense = model(
        inputs_embeds=hidden.unsqueeze(0),
        image_gen_indicators=indicators.unsqueeze(0),
        indexes=indexes,
        attention_mask={"full_attention": dense_mask},
    ).last_hidden_state.squeeze(0)

    from uniserve_worker.runtime.forward_stream import ForwardStreamBuilder

    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="extend",
        q_len=2,
        prefix_len=0,
        visible_policy="causal",
        indexes=indexes[:, :2],
    )
    builder.add_segment(
        op_index=1,
        req_id=2,
        kind="denoise_gen",
        mode=ForwardMode.DENOISE,
        modality="gen",
        segment_class="denoise",
        q_len=3,
        prefix_len=0,
        visible_policy="bidirectional",
        indexes=indexes[:, 2:],
    )
    stream = builder.build(device=device)
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=256,
        num_kv_heads=2,
        head_dim=128,
        device=device,
        dtype=dtype,
    )
    view = ForwardPagedKVView(
        pool,
        [
            ForwardPagedKVSegment(block_ids=(0,), base_len=0, q_len=2),
            ForwardPagedKVSegment(block_ids=(1,), base_len=0, q_len=3),
        ],
    )
    with use_forward_context(ForwardContext(attention_backend=get_attention_backend("fa4_cute"))):
        packed = model.forward_packed_visible(
            hidden,
            image_gen_indicators=indicators,
            indexes=stream.indexes,
            forward_stream=stream,
            kv_view=view,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        packed.float(),
        dense.float(),
        atol=MODEL_CONFORMANCE_ATOL,
        rtol=MODEL_CONFORMANCE_RTOL,
        equal_nan=True,
    )
