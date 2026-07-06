"""Parity tests for the shared layer library."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import uniserve_worker.foundation.runtime_config as runtime_config
from uniserve_worker.backends.attention import (
    AttentionCapabilities,
    get_attention_backend,
    has_attention_backend,
)
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.foundation.runtime_config import TorchCompileRuntimeConfig
from uniserve_worker.foundation.triton_compat import triton_device_supported
from uniserve_worker.nn import (
    DeviceMesh,
    HFRotaryEmbedding,
    LinearBase,
    MergedColumnParallelLinear,
    ParallelLMHead,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RotaryEmbedding,
    VocabParallelEmbedding,
    apply_rotary_emb,
    apply_rotary_pos_emb,
    get_rope,
    rotate_half,
    use_mesh,
)
from uniserve_worker.nn.diffusion import (
    CfgParams,
    FlowMatchSchedule,
    RenormKind,
    ScheduleDirection,
    ScheduleShiftDomain,
    combine_text_image_cfg,
)

pytestmark = pytest.mark.integration


def _set_worker_runtime(monkeypatch, **kwargs):
    monkeypatch.setattr(
        runtime_config,
        "_CURRENT_CONFIG",
        replace(runtime_config.get_worker_config(), **kwargs),
    )


def test_rmsnorm_matches_manual_fp32_reference():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 7, dtype=torch.float32)
    shared = RMSNorm(7, eps=1e-6)
    with torch.no_grad():
        weight = torch.randn(7)
        shared.weight.copy_(weight)

    got = shared(x)
    expected = weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    torch.testing.assert_close(got, expected)


def test_rmsnorm_uses_fp32_accumulation_for_low_precision_inputs():
    torch.manual_seed(1)
    x = (torch.randn(4, 9) * 10).to(torch.bfloat16)
    norm = RMSNorm(9, eps=1e-6)
    out = norm(x)
    manual = norm.weight * (
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps)
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, manual)


def test_rmsnorm_forward_with_residual_matches_separate_add_then_norm():
    torch.manual_seed(11)
    hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    norm = RMSNorm(8, eps=1e-6).to(dtype=torch.bfloat16)

    got_normed, got_residual = norm.forward_with_residual(hidden, residual)

    expected_residual = hidden + residual
    expected_normed = norm(expected_residual)
    torch.testing.assert_close(got_residual, expected_residual)
    torch.testing.assert_close(got_normed, expected_normed)


def test_packed_rotary_apply_matches_declared_formula():
    torch.manual_seed(2)
    x = torch.randn(6, 3, 8)
    pos = torch.arange(6)
    shared_rope = RotaryEmbedding(8, theta=10000.0)
    assert isinstance(get_rope(8, theta=10000.0), RotaryEmbedding)

    cos, sin = shared_rope.cos_sin_1d(pos)
    expected_freqs = pos.float()[:, None] * shared_rope.inv_freq[None, :]
    torch.testing.assert_close(cos, expected_freqs.cos())
    torch.testing.assert_close(sin, expected_freqs.sin())
    out = apply_rotary_emb(x, cos, sin)
    expected = (
        x * torch.cat([cos, cos], dim=-1).unsqueeze(1)
        + rotate_half(x) * torch.cat([sin, sin], dim=-1).unsqueeze(1)
    )
    torch.testing.assert_close(out, expected)
    assert out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_packed_rotary_matches_eager_formula_cuda(monkeypatch):
    pytest.importorskip("triton")
    if not triton_device_supported(torch.device("cuda")):
        pytest.skip("Triton cannot compile kernels for this CUDA device in this environment")
    torch.manual_seed(203)
    x = torch.randn(7, 5, 16, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(7, 8, device="cuda", dtype=torch.float32)
    sin = torch.randn(7, 8, device="cuda", dtype=torch.float32)
    got = apply_rotary_emb(x, cos, sin)
    x1 = x[..., :8].float()
    x2 = x[..., 8:].float()
    expected = torch.empty_like(x)
    expected[..., :8] = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
    expected[..., 8:] = x2 * cos.unsqueeze(1) + x1 * sin.unsqueeze(1)
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)


def test_qwen_rotary_apply_matches_declared_formula():
    torch.manual_seed(3)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    cos = torch.randn(2, 5, 8)
    sin = torch.randn(2, 5, 8)
    shared = apply_rotary_pos_emb(q, k, cos, sin)
    expected_q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
    expected_k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
    torch.testing.assert_close(shared[0], expected_q)
    torch.testing.assert_close(shared[1], expected_k)
    half = q.shape[-1] // 2
    torch.testing.assert_close(
        rotate_half(q), torch.cat([-q[..., half:], q[..., :half]], dim=-1)
    )


def test_rotary_embedding_matches_qwen3_frequency_range_config():
    from transformers import Qwen3Config

    cfg = Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    cfg.head_dim = 16
    ref = get_rope(config=cfg, keep_freq_range=True)
    assert isinstance(ref, HFRotaryEmbedding)
    shared = RotaryEmbedding(16, theta=10000.0, max_position_embeddings=128, keep_freq_range=True)
    x = torch.zeros(1, 5, 64)
    position_ids = torch.arange(5).unsqueeze(0)
    ref_cos, ref_sin = ref(x, position_ids)
    got_cos, got_sin = shared(x, position_ids)
    torch.testing.assert_close(got_cos, ref_cos)
    torch.testing.assert_close(got_sin, ref_sin)


def test_linear_weight_loader_and_qkv_shards():
    layer = LinearBase(3, 4, bias=False)
    loaded = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    layer.weight.weight_loader(layer.weight, loaded)
    torch.testing.assert_close(layer.weight, loaded)

    from uniserve_worker.nn import QKVParallelLinear

    qkv = QKVParallelLinear(hidden_size=3, head_size=2, total_num_heads=2, total_num_kv_heads=1, bias=True)
    q = torch.ones(4, 3)
    k = torch.ones(2, 3) * 2
    v = torch.ones(2, 3) * 3
    qkv.weight.weight_loader(qkv.weight, q, shard_id="q")
    qkv.weight.weight_loader(qkv.weight, k, shard_id="k")
    qkv.weight.weight_loader(qkv.weight, v, shard_id="v")
    qkv.bias.weight_loader(qkv.bias, torch.arange(4, dtype=torch.float32), shard_id="q")
    qkv.bias.weight_loader(qkv.bias, torch.arange(2, dtype=torch.float32) + 10, shard_id="k")
    qkv.bias.weight_loader(qkv.bias, torch.arange(2, dtype=torch.float32) + 20, shard_id="v")
    torch.testing.assert_close(qkv.weight[:4], q)
    torch.testing.assert_close(qkv.weight[4:6], k)
    torch.testing.assert_close(qkv.weight[6:8], v)
    torch.testing.assert_close(qkv.bias[:4], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(qkv.bias[4:6], torch.arange(2, dtype=torch.float32) + 10)
    torch.testing.assert_close(qkv.bias[6:8], torch.arange(2, dtype=torch.float32) + 20)


def test_linear_quant_method_owns_weight_creation():
    from uniserve_worker.nn.quant import QuantizeMethodBase

    class OwnedMethod(QuantizeMethodBase):
        def __init__(self):
            self.created = False

        def create_weights(self, module, *, input_size, output_size, bias, **kwargs):
            del kwargs
            self.created = True
            module.register_parameter("weight", nn.Parameter(torch.ones(output_size, input_size)))
            module.register_parameter(
                "bias",
                nn.Parameter(torch.zeros(output_size)) if bias else None,
            )

        def apply(self, module, x):
            return torch.nn.functional.linear(x, module.weight, module.bias)

    method = OwnedMethod()
    layer = LinearBase(3, 2, bias=True, quant_method=method)

    assert method.created
    assert layer.weight.shape == (2, 3)
    assert layer.bias.shape == (2,)
    assert hasattr(layer.weight, "weight_loader")
    assert hasattr(layer.bias, "weight_loader")


def test_quantization_config_context_selects_linear_method():
    from uniserve_worker.nn.quant import QuantizationConfig, use_quantization_config

    cfg = QuantizationConfig(method="unquantized", raw={"quant_method": "unquantized"})
    with use_quantization_config(cfg):
        layer = LinearBase(2, 3, bias=False, prefix="model.layers.0.mlp.down_proj")

    assert type(layer.quant_method).__name__ == "UnquantizedLinearMethod"
    assert layer.weight.shape == (3, 2)


def test_quantization_config_extracts_kv_cache_dtype():
    from uniserve_worker.nn.quant import (
        QuantizationConfig,
        get_current_kv_cache_dtype,
        kv_cache_dtype_from_model_config,
        use_quantization_config,
    )

    cfg = QuantizationConfig.from_model_config(
        {"quantization_config": {"quant_method": "fp8", "kv_cache_dtype": "fp8_e4m3"}}
    )
    assert cfg is not None
    assert cfg.kv_cache_dtype == "fp8_e4m3"
    assert kv_cache_dtype_from_model_config({"kv_cache_dtype": "fp8_e4m3"}) == "fp8_e4m3"
    with use_quantization_config(cfg):
        assert get_current_kv_cache_dtype({}) == "fp8_e4m3"

    with pytest.raises(ValueError, match="unsupported KV cache store dtype"):
        QuantizationConfig.from_model_config(
            {"quantization_config": {"quant_method": "fp8", "kv_cache_dtype": "int4"}}
        )


def test_fp8_linear_online_quantizes_and_uses_dequantized_correctness_floor():
    from uniserve_worker.nn.quant import QuantizationConfig, use_quantization_config
    from uniserve_worker.nn.quant.base import process_quantized_modules

    cfg = QuantizationConfig.from_model_config({"quantization_config": {"quant_method": "fp8"}})
    assert cfg is not None
    with use_quantization_config(cfg):
        layer = LinearBase(16, 8, bias=True, prefix="model.layers.0.mlp.down_proj")
    assert type(layer.quant_method).__name__ == "W8A8Fp8LinearMethod"
    dense_weight = torch.linspace(-1.5, 1.5, 128, dtype=torch.float32).reshape(8, 16)
    dense_bias = torch.linspace(-0.2, 0.2, 8, dtype=torch.float32)
    layer.weight.weight_loader(layer.weight, dense_weight)
    layer.bias.weight_loader(layer.bias, dense_bias)

    process_quantized_modules([layer])

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (8, 1)
    assert bool(getattr(layer.weight_scale, "_uniserve_skip_serving_cast", False))
    x = torch.linspace(-0.5, 0.5, 32, dtype=torch.float32).reshape(2, 16)
    got = layer(x)
    dequant_weight = layer.weight.float() * layer.weight_scale
    expected = F.linear(x, dequant_weight, layer.bias)
    torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_linear_cuda_scaled_mm_path_runs_finite():
    from uniserve_worker.nn.quant import QuantizationConfig, use_quantization_config
    from uniserve_worker.nn.quant.base import process_quantized_modules

    cfg = QuantizationConfig(method="fp8", raw={"quant_method": "fp8"})
    with use_quantization_config(cfg):
        layer = LinearBase(16, 32, bias=False).cuda()
    torch.manual_seed(1207)
    dense_weight = (torch.randn(32, 16) * 0.2).cuda()
    layer.weight.weight_loader(layer.weight, dense_weight)
    process_quantized_modules([layer])
    x = (torch.randn(4, 16, device="cuda", dtype=torch.bfloat16) * 0.2)

    out = layer(x)

    assert out.shape == (4, 32)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all().item()


def test_quantization_config_rejects_unsupported_checkpoint_method():
    from uniserve_worker.nn.quant import QuantizationConfig

    cfg = {"quantization_config": {"quant_method": "gptq"}}
    with pytest.raises(NotImplementedError, match="gptq"):
        QuantizationConfig.from_model_config(cfg)


def test_flashinfer_decode_write_uses_precomputed_locations():
    from uniserve_worker.backends.attention.flashinfer import _write_decode_token
    from uniserve_worker.contracts.forward_context import (
        ForwardContext,
        TextAttentionMetadata,
        use_forward_context,
    )

    k_cache = torch.zeros((4, 4, 1, 1), dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    metadata = TextAttentionMetadata(
        cache=None,
        block_table=None,
        cache_seqlens=None,
        decode_page_ids=torch.tensor([2, 3], dtype=torch.long),
        decode_page_offsets=torch.tensor([1, 2], dtype=torch.long),
    )

    with use_forward_context(ForwardContext(attention_metadata=metadata)):
        _write_decode_token(
            k_cache,
            v_cache,
            block_table=torch.zeros((2, 1), dtype=torch.int32),
            cache_seqlens=torch.zeros(2, dtype=torch.int32),
            k_current=torch.tensor([[[11.0]], [[12.0]]]),
            v_current=torch.tensor([[[21.0]], [[22.0]]]),
        )

    assert k_cache[2, 1, 0, 0].item() == 11.0
    assert k_cache[3, 2, 0, 0].item() == 12.0
    assert v_cache[2, 1, 0, 0].item() == 21.0
    assert v_cache[3, 2, 0, 0].item() == 22.0
    assert k_cache[0, 0, 0, 0].item() == 0.0


def test_flashinfer_decode_plan_workspace_reuses_static_buffers(monkeypatch):
    import uniserve_worker.backends.attention.flashinfer as flashinfer_mod

    backend = flashinfer_mod.FlashInferAttentionBackend()
    block_table = torch.tensor([[5, 0, 0], [7, 8, 9]], dtype=torch.int32)
    seq_lens = torch.tensor([4, 33], dtype=torch.int32)
    calls = []

    def fake_fill(block_table, seq_lens, page_size, workspace):
        del block_table, seq_lens, page_size
        calls.append(workspace)
        workspace.indptr[:3].copy_(torch.tensor([0, 1, 4], dtype=torch.int32))
        workspace.indices[:4].copy_(torch.tensor([5, 7, 8, 9], dtype=torch.int32))
        workspace.last_page_len[:2].copy_(torch.tensor([4, 1], dtype=torch.int32))
        return True

    monkeypatch.setattr(flashinfer_mod, "_fill_paged_decode_plan_tensors", fake_fill)

    wrapper_key = ("cpu", "fa2", False)
    plan = backend._decode_plan_tensors(wrapper_key, block_table, seq_lens, 16, index_count=4)
    again = backend._decode_plan_tensors(wrapper_key, block_table, seq_lens, 16, index_count=4)

    assert calls[0] is calls[1]
    assert plan.indptr.data_ptr() == again.indptr.data_ptr()
    assert plan.indices.data_ptr() == again.indices.data_ptr()
    torch.testing.assert_close(plan.indptr, torch.tensor([0, 1, 4], dtype=torch.int32))
    torch.testing.assert_close(plan.indices, torch.tensor([5, 7, 8, 9], dtype=torch.int32))
    torch.testing.assert_close(plan.last_page_len, torch.tensor([4, 1], dtype=torch.int32))

    graph_key = ("cuda_graph", "cpu", "fa2", False, 2, 6)
    graph_indptr = torch.empty(3, dtype=torch.int32)
    graph_indices = torch.empty(6, dtype=torch.int32)
    graph_last_page_len = torch.empty(2, dtype=torch.int32)
    backend._decode_graph_buffers[graph_key] = (
        graph_indptr,
        graph_indices,
        graph_last_page_len,
    )
    graph_plan = backend._decode_plan_tensors(
        graph_key,
        block_table,
        seq_lens,
        16,
        index_count=4,
    )

    assert graph_plan.indptr.data_ptr() == graph_indptr.data_ptr()
    assert graph_plan.indices.data_ptr() == graph_indices.data_ptr()
    assert graph_plan.last_page_len.data_ptr() == graph_last_page_len.data_ptr()
    torch.testing.assert_close(graph_plan.indices, torch.tensor([5, 7, 8, 9], dtype=torch.int32))


def test_flashinfer_decode_plan_workspace_is_mutable_after_inference_allocation(monkeypatch):
    import uniserve_worker.backends.attention.flashinfer as flashinfer_mod

    backend = flashinfer_mod.FlashInferAttentionBackend()
    block_table = torch.tensor([[5, 0, 0], [7, 8, 9]], dtype=torch.int32)
    seq_lens = torch.tensor([4, 33], dtype=torch.int32)
    seen = []

    def fake_fill(block_table, seq_lens, page_size, workspace):
        del block_table, seq_lens, page_size
        seen.append(workspace)
        workspace.indptr[:3].copy_(torch.tensor([0, 1, 4], dtype=torch.int32))
        workspace.indices[:4].copy_(torch.tensor([5, 7, 8, 9], dtype=torch.int32))
        workspace.last_page_len[:2].copy_(torch.tensor([4, 1], dtype=torch.int32))
        return True

    monkeypatch.setattr(flashinfer_mod, "_fill_paged_decode_plan_tensors", fake_fill)

    wrapper_key = flashinfer_mod.WrapperKey("decode", "cpu", "fa2", False)
    with torch.inference_mode():
        backend._decode_plan_tensors(wrapper_key, block_table, seq_lens, 16, index_count=4)

    workspace = seen[0]
    assert not workspace.indptr.is_inference()
    assert not workspace.indices.is_inference()
    assert not workspace.last_page_len.is_inference()
    assert not workspace.page_counts.is_inference()
    workspace.page_counts[:2].copy_(seq_lens)


def test_flashinfer_prefill_plan_workspace_reuses_static_buffers(monkeypatch):
    import uniserve_worker.backends.attention.flashinfer as flashinfer_mod

    backend = flashinfer_mod.FlashInferAttentionBackend()
    block_table = torch.tensor([[5, 0, 0], [7, 8, 9]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    kv_seqlens = torch.tensor([4, 33], dtype=torch.int32)
    calls = []

    def fake_fill(block_table, cu_seqlens_q, kv_seqlens, page_size, workspace):
        del block_table, cu_seqlens_q, kv_seqlens, page_size
        calls.append(workspace)
        workspace.qo_indptr[:3].copy_(torch.tensor([0, 2, 5], dtype=torch.int32))
        workspace.kv_indptr[:3].copy_(torch.tensor([0, 1, 4], dtype=torch.int32))
        workspace.indices[:4].copy_(torch.tensor([5, 7, 8, 9], dtype=torch.int32))
        workspace.last_page_len[:2].copy_(torch.tensor([4, 1], dtype=torch.int32))
        return True

    monkeypatch.setattr(flashinfer_mod, "_fill_paged_prefill_plan_tensors", fake_fill)

    wrapper_key = ("cpu", "auto")
    plan = backend._prefill_plan_tensors(wrapper_key, block_table, cu_seqlens_q, kv_seqlens, 16)
    again = backend._prefill_plan_tensors(wrapper_key, block_table, cu_seqlens_q, kv_seqlens, 16)

    assert calls[0] is calls[1]
    assert plan.qo_indptr.data_ptr() == again.qo_indptr.data_ptr()
    assert plan.kv_indptr.data_ptr() == again.kv_indptr.data_ptr()
    assert plan.indices.data_ptr() == again.indices.data_ptr()
    torch.testing.assert_close(plan.qo_indptr, torch.tensor([0, 2, 5], dtype=torch.int32))
    torch.testing.assert_close(plan.kv_indptr, torch.tensor([0, 1, 4], dtype=torch.int32))
    torch.testing.assert_close(plan.indices, torch.tensor([5, 7, 8, 9], dtype=torch.int32))
    torch.testing.assert_close(plan.last_page_len, torch.tensor([4, 1], dtype=torch.int32))


def test_flashinfer_prefill_plan_workspace_is_mutable_after_inference_allocation(monkeypatch):
    import uniserve_worker.backends.attention.flashinfer as flashinfer_mod

    backend = flashinfer_mod.FlashInferAttentionBackend()
    block_table = torch.tensor([[5, 0, 0], [7, 8, 9]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    kv_seqlens = torch.tensor([4, 33], dtype=torch.int32)
    seen = []

    def fake_fill(block_table, cu_seqlens_q, kv_seqlens, page_size, workspace):
        del block_table, cu_seqlens_q, kv_seqlens, page_size
        seen.append(workspace)
        workspace.qo_indptr[:3].copy_(torch.tensor([0, 2, 5], dtype=torch.int32))
        workspace.kv_indptr[:3].copy_(torch.tensor([0, 1, 4], dtype=torch.int32))
        workspace.indices[:4].copy_(torch.tensor([5, 7, 8, 9], dtype=torch.int32))
        workspace.last_page_len[:2].copy_(torch.tensor([4, 1], dtype=torch.int32))
        return True

    monkeypatch.setattr(flashinfer_mod, "_fill_paged_prefill_plan_tensors", fake_fill)

    wrapper_key = flashinfer_mod.WrapperKey("prefill", "cpu", "auto")
    with torch.inference_mode():
        backend._prefill_plan_tensors(wrapper_key, block_table, cu_seqlens_q, kv_seqlens, 16)

    workspace = seen[0]
    assert not workspace.qo_indptr.is_inference()
    assert not workspace.kv_indptr.is_inference()
    assert not workspace.indices.is_inference()
    assert not workspace.last_page_len.is_inference()
    assert not workspace.page_counts.is_inference()
    workspace.page_counts[:2].copy_(kv_seqlens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_paged_varlen_prefill_matches_causal_reference_cuda():
    from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context

    pytest.importorskip("flashinfer")
    backend = get_attention_backend("flashinfer")
    caps = backend.capabilities()
    if not bool(getattr(caps, "varlen_paged_kv", False)):
        pytest.skip("flashinfer paged prefill wrapper is unavailable")

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.float16
    num_heads = 2
    head_dim = 64
    page_size = 4
    query_lens = [2, 1]
    kv_lens = [5, 3]
    q = torch.randn(sum(query_lens), num_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(3, page_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(3, page_size, num_heads, head_dim, device=device, dtype=dtype)
    block_table = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32, device=device)
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 5, 8], dtype=torch.int32, device=device)
    stats = ForwardStats()

    with use_forward_context(ForwardContext(stats=stats)):
        out = backend.forward_varlen(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=2,
            max_seqlen_k=5,
            causal=True,
            scale=head_dim**-0.5,
            block_table=block_table,
        )
        reused = backend.forward_varlen(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=2,
            max_seqlen_k=5,
            causal=True,
            scale=head_dim**-0.5,
            block_table=block_table,
        )

    expected_rows = []
    q_offset = 0
    for row, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        page_count = (kv_len + page_size - 1) // page_size
        pages = block_table[row, :page_count].tolist()
        k_full = torch.cat([k_cache[int(page)] for page in pages], dim=0)[:kv_len].float()
        v_full = torch.cat([v_cache[int(page)] for page in pages], dim=0)[:kv_len].float()
        q_row = q[q_offset:q_offset + query_len].float()
        scores = torch.einsum("qhd,khd->hqk", q_row, k_full) * (head_dim**-0.5)
        prefix = kv_len - query_len
        keep = torch.zeros((query_len, kv_len), dtype=torch.bool, device=device)
        for query_idx in range(query_len):
            keep[query_idx, : prefix + query_idx + 1] = True
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        expected_rows.append(torch.einsum("hqk,khd->qhd", probs, v_full))
        q_offset += query_len
    expected = torch.cat(expected_rows, dim=0).to(dtype)

    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(reused, out)
    assert stats.flashinfer_prefill_plan_calls == 1
    assert stats.flashinfer_prefill_plan_reuses == 1


def test_torch_compile_helper_is_default_off_and_config_driven(monkeypatch):
    from uniserve_worker.runtime.compile import (
        TorchCompileConfig,
        compile_targets,
        maybe_compile_module,
        named_child_compile_targets,
    )

    module = nn.Linear(2, 2)
    assert maybe_compile_module(module, label="unit") is module

    calls = []

    def fake_compile(target, **kwargs):
        calls.append((target, kwargs))
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)
    _set_worker_runtime(
        monkeypatch,
        torch_compile=TorchCompileRuntimeConfig(
            enabled=True,
            backend="eager",
            mode=None,
            fullgraph=True,
            dynamic=False,
        ),
    )

    cfg = TorchCompileConfig.from_runtime_config()
    assert cfg.enabled
    assert cfg.backend == "eager"
    assert cfg.mode is None
    assert cfg.fullgraph is True
    assert cfg.dynamic is False

    assert maybe_compile_module(module, label="unit", config=cfg) is module
    assert len(calls) == 1

    root = nn.Module()
    root.block = nn.Module()
    root.block.mlp = nn.Linear(2, 2)

    def fake_compile_wrapper(target, **kwargs):
        calls.append((target, kwargs))
        wrapped = nn.Sequential(target)
        setattr(wrapped, "_uniserve_torch_compiled", True)
        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile_wrapper)
    report = compile_targets(
        named_child_compile_targets(
            root,
            predicate=lambda name, child: name.endswith("mlp") and isinstance(child, nn.Linear),
            label_prefix="unit",
        ),
        config=cfg,
    )

    assert report.attempted == 1
    assert report.compiled == 1
    assert report.labels == ("unit.block.mlp",)
    assert isinstance(root.block.mlp, nn.Sequential)
    assert calls[0][1] == {"backend": "eager", "fullgraph": True, "dynamic": False}
    assert maybe_compile_module(module, label="unit", config=cfg) is module
    assert len(calls) == 2


def test_torch_compile_helper_prepares_blackwell_ptxas(monkeypatch):
    import uniserve_worker.runtime.compile as compile_mod
    from uniserve_worker.runtime.compile import TorchCompileConfig

    module = nn.Linear(2, 2)
    calls = []

    monkeypatch.setattr(compile_mod, "_first_module_device", lambda _module: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (10, 0))
    monkeypatch.setattr(compile_mod, "ensure_blackwell_ptxas", lambda: calls.append("ptxas") or True)
    monkeypatch.setattr(torch, "compile", lambda target, **_kwargs: target)

    cfg = TorchCompileConfig(
        enabled=True,
        backend="eager",
        mode=None,
        fullgraph=False,
        dynamic=None,
    )
    assert compile_mod.maybe_compile_module(module, label="unit.blackwell", config=cfg) is module
    assert calls == ["ptxas"]


def test_column_and_row_parallel_loaders_narrow_full_rank_weights():
    from uniserve_worker.nn import ColumnParallelLinear, RowParallelLinear

    with use_mesh(DeviceMesh.tp(1, 2)):
        col = ColumnParallelLinear(4, 6, bias=False)
        row = RowParallelLinear(6, 4, bias=False)

    full_col = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    full_row = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    col.weight.weight_loader(col.weight, full_col)
    row.weight.weight_loader(row.weight, full_row)

    assert col.weight.shape == (3, 4)
    assert row.weight.shape == (4, 3)
    torch.testing.assert_close(col.weight, full_col[3:6])
    torch.testing.assert_close(row.weight, full_row[:, 3:6])


def test_vocab_parallel_embedding_and_lm_head_tp1_match_dense_with_padding():
    emb = VocabParallelEmbedding(5, 3, pad_vocab_size_to=8)
    head = ParallelLMHead(3, 5, bias=False, pad_vocab_size_to=8)
    table = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    emb.weight.weight_loader(emb.weight, table)
    head.weight.weight_loader(head.weight, table)

    ids = torch.tensor([[0, 4, 2]], dtype=torch.long)
    hidden = torch.randn(2, 3)

    assert emb.weight.shape == (8, 3)
    torch.testing.assert_close(emb.weight[:5], table)
    torch.testing.assert_close(emb.weight[5:], torch.zeros(3, 3))
    torch.testing.assert_close(emb(ids), F.embedding(ids, table))
    torch.testing.assert_close(head(hidden), F.linear(hidden, table))


def test_vocab_parallel_loaders_narrow_full_rank_weights_and_fail_loud_without_collectives():
    with use_mesh(DeviceMesh.tp(1, 2)):
        emb = VocabParallelEmbedding(6, 2, pad_vocab_size_to=1)
        head = ParallelLMHead(2, 6, bias=True, pad_vocab_size_to=1)

    table = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    bias = torch.arange(6, dtype=torch.float32)
    emb.weight.weight_loader(emb.weight, table)
    head.weight.weight_loader(head.weight, table)
    head.bias.weight_loader(head.bias, bias)

    assert emb.weight.shape == (3, 2)
    assert head.weight.shape == (3, 2)
    torch.testing.assert_close(emb.weight, table[3:6])
    torch.testing.assert_close(head.weight, table[3:6])
    torch.testing.assert_close(head.bias, bias[3:6])

    with pytest.raises(RuntimeError, match="transport"):
        emb(torch.tensor([3, 4], dtype=torch.long))
    with pytest.raises(RuntimeError, match="transport"):
        head(torch.randn(1, 2))


def test_row_parallel_forward_requires_or_uses_tp_collective():
    from uniserve_worker.nn import DeviceMesh, RowParallelLinear, use_mesh

    # tp>1 axis with no transport: forward must fail loudly via reshard rather
    # than silently skip the reduce.
    with use_mesh(DeviceMesh.tp(0, 2)):
        row = RowParallelLinear(4, 3, bias=False)
    torch.nn.init.normal_(row.weight)
    x = torch.randn(2, 2)
    with pytest.raises(RuntimeError, match="transport"):
        row(x)

    # With a transport bound to the tp axis, forward issues the all-reduce
    # through reshard (Partial -> Replicate).
    calls = []

    class _FakeTransport:
        size = 2
        coord = 0

        def all_reduce(self, tensor, op="sum"):
            calls.append(op)
            return tensor + 1

    with use_mesh(DeviceMesh.tp(0, 2, transport=_FakeTransport())):
        row2 = RowParallelLinear(4, 3, bias=False)
    torch.nn.init.normal_(row2.weight)
    out = row2(x)
    expected = F.linear(x, row2.weight, None) + 1
    torch.testing.assert_close(out, expected)
    assert calls == ["sum"]


def test_qkv_parallel_loader_shards_q_and_replicates_small_kv_heads():
    from uniserve_worker.nn import QKVParallelLinear

    with use_mesh(DeviceMesh.tp(1, 2)):
        qkv = QKVParallelLinear(hidden_size=3, head_size=2, total_num_heads=4, total_num_kv_heads=1, bias=True)

    q = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    k = torch.ones(2, 3) * 2
    v = torch.ones(2, 3) * 3
    qkv.weight.weight_loader(qkv.weight, q, shard_id="q")
    qkv.weight.weight_loader(qkv.weight, k, shard_id="k")
    qkv.weight.weight_loader(qkv.weight, v, shard_id="v")

    assert qkv.weight.shape == (8, 3)
    torch.testing.assert_close(qkv.weight[:4], q[4:8])
    torch.testing.assert_close(qkv.weight[4:6], k)
    torch.testing.assert_close(qkv.weight[6:8], v)


def test_bagel_model_uses_shared_linear_and_vision_seams():
    from uniserve_worker.models.bagel import (
        BagelConfig,
        BagelForUnifiedGeneration,
        LLMConfig,
        _BagelGraph,
    )
    from uniserve_worker.nn import QKVParallelLinear
    from uniserve_worker.nn.decoder import KVCache, MoTDecoderLayer, MoTModel, Segment
    from uniserve_worker.nn.vision import SiglipNavitEncoder

    cfg = LLMConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
    )
    layer = MoTDecoderLayer(cfg)
    assert isinstance(layer.qkv_proj, QKVParallelLinear)
    assert isinstance(layer.attn, RadixAttention)
    assert isinstance(layer.o_proj, LinearBase)
    assert isinstance(layer.mlp.gate_up_proj, MergedColumnParallelLinear)

    bcfg = BagelConfig(
        llm=cfg,
        vit_hidden_size=16,
        vit_intermediate_size=32,
        vit_num_hidden_layers=1,
        vit_num_attention_heads=4,
    )
    model = _BagelGraph(bcfg).to(dtype=torch.bfloat16)
    assert isinstance(model.lm, MoTModel)
    assert isinstance(model.lm_head, LinearBase)
    assert isinstance(model.lm_head, ParallelLMHead)
    assert isinstance(model.lm.embed_tokens, VocabParallelEmbedding)
    assert isinstance(model.vae2llm, LinearBase)
    assert isinstance(model.vit_model, SiglipNavitEncoder)
    assert isinstance(model.vit_model.encoder.layers[0].self_attn.attn, RadixAttention)
    assert isinstance(model.vit_model.encoder.layers[0].self_attn.q_proj, LinearBase)
    assert isinstance(model.vit_model.encoder.layers[0].mlp[0], LinearBase)

    seg = Segment(
        embeds=torch.randn(3, cfg.hidden_size, dtype=torch.bfloat16),
        positions=torch.arange(3),
        is_gen=torch.zeros(3, dtype=torch.bool),
        cache=KVCache(cfg.num_hidden_layers),
        causal=True,
        update_cache=True,
    )
    assert model.lm.forward_segments([seg])[0].shape == (3, cfg.hidden_size)

    bcfg.max_latent_size = 8
    wrapper = BagelForUnifiedGeneration(config=bcfg, kv_token_capacity=256)
    assert wrapper.caps().max_latent_size == 64


def test_sensenova_dense_decoder_uses_shared_linear_seams():
    from transformers import Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova_u1

    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 4
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    mlp = sensenova_u1._SenseNovaMLP(cfg)
    attn = sensenova_u1._SenseNovaAttention(cfg, layer_idx=0)
    lm = sensenova_u1._SenseNovaLanguageModel(cfg)
    assert isinstance(mlp.gate_up_proj, MergedColumnParallelLinear)
    assert isinstance(attn.attn, RadixAttention)
    assert isinstance(attn.qkv_proj, QKVParallelLinear)
    assert isinstance(attn.o_proj_mot_gen, LinearBase)
    assert isinstance(lm.model.embed_tokens, VocabParallelEmbedding)
    assert isinstance(lm.lm_head, ParallelLMHead)
    assert lm.get_input_embeddings() is lm.model.embed_tokens
    assert lm.get_output_embeddings() is lm.lm_head


def test_sensenova_text_single_token_uses_fused_qkv_projection(monkeypatch):
    from transformers import Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova_u1

    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 4
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    attn = sensenova_u1._SenseNovaAttention(cfg, layer_idx=0)

    def fake_qk_norm_rope(query_states, key_states, _indexes, **_kwargs):
        return query_states.split([2, 1, 1], dim=-1), key_states.split([2, 1, 1], dim=-1)

    monkeypatch.setattr(attn, "_qk_norm_rope_3d", fake_qk_norm_rope)
    text_calls = 0
    gen_calls = 0
    orig_text_forward = attn.qkv_proj.forward
    orig_gen_forward = attn.qkv_proj_mot_gen.forward

    def text_forward(x):
        nonlocal text_calls
        text_calls += 1
        return orig_text_forward(x)

    def gen_forward(x):
        nonlocal gen_calls
        gen_calls += 1
        return orig_gen_forward(x)

    monkeypatch.setattr(attn.qkv_proj, "forward", text_forward)
    monkeypatch.setattr(attn.qkv_proj_mot_gen, "forward", gen_forward)

    hidden = torch.randn(1, 1, cfg.hidden_size)
    indexes = torch.zeros(3, 1, dtype=torch.long)
    attn._project_qkv(hidden, indexes, gen_branch=False)
    attn._project_qkv(hidden, indexes, gen_branch=True)

    assert text_calls == 1
    assert gen_calls == 0


def test_sensenova_projection_loaders_match_checkpoint_linears():
    from transformers import Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova_u1
    from uniserve_worker.nn.placement import get_shard_plan

    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 4
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    attn = sensenova_u1._SenseNovaAttention(cfg, layer_idx=0)
    mlp = sensenova_u1._SenseNovaMLP(cfg)

    q = torch.randn(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    k = torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    v = torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    gate = torch.randn(cfg.intermediate_size, cfg.hidden_size)
    up = torch.randn(cfg.intermediate_size, cfg.hidden_size)

    for shard, tensor in (("q", q), ("k", k), ("v", v)):
        attn.qkv_proj.weight.weight_loader(attn.qkv_proj.weight, tensor, shard_id=shard)
    for shard, tensor in (("gate", gate), ("up", up)):
        mlp.gate_up_proj.weight.weight_loader(mlp.gate_up_proj.weight, tensor, shard_id=shard)

    torch.testing.assert_close(attn.qkv_proj.weight, torch.cat([q, k, v], dim=0))
    torch.testing.assert_close(mlp.gate_up_proj.weight, torch.cat([gate, up], dim=0))
    assert get_shard_plan(attn.qkv_proj.weight).mode.shard_keys == ("q", "k", "v")
    assert get_shard_plan(mlp.gate_up_proj.weight).mode.shard_keys == ("gate", "up")

    x = torch.randn(3, cfg.hidden_size)
    qkv_ref = torch.cat([F.linear(x, q), F.linear(x, k), F.linear(x, v)], dim=-1)
    gate_up_ref = torch.cat([F.linear(x, gate), F.linear(x, up)], dim=-1)
    torch.testing.assert_close(attn.qkv_proj(x), qkv_ref)
    torch.testing.assert_close(mlp.gate_up_proj(x), gate_up_ref)


def test_sensenova_qk_norm_rope_3d_matches_eager_formula(monkeypatch):
    transformers = pytest.importorskip("transformers")
    Qwen3Config = transformers.Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova_u1
    from uniserve_worker.nn import apply_rotary_emb, apply_rotary_pos_emb

    monkeypatch.setenv("UNISERVE_QK_NORM_ROPE_PROVIDER", "eager")
    monkeypatch.setenv("UNISERVE_QK_NORM_PROVIDER", "eager")
    torch.manual_seed(123)
    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 4
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    attn = sensenova_u1._SenseNovaAttention(cfg, layer_idx=0)
    query_states = torch.randn(1, 5, attn.num_heads, attn.head_dim)
    key_states = torch.randn(1, 5, attn.num_kv_heads, attn.head_dim)
    indexes = torch.stack([torch.arange(5), torch.arange(5) % 3, torch.arange(5) % 2], dim=0)

    got_q, got_k = attn._qk_norm_rope_3d(
        query_states,
        key_states,
        indexes,
        q_norm=attn.q_norm,
        k_norm=attn.k_norm,
        q_norm_hw=attn.q_norm_hw,
        k_norm_hw=attn.k_norm_hw,
    )

    q_t, q_hw = query_states.chunk(2, dim=-1)
    k_t, k_hw = key_states.chunk(2, dim=-1)
    q_t = attn.q_norm(q_t).transpose(1, 2)
    k_t = attn.k_norm(k_t).transpose(1, 2)
    q_hw = attn.q_norm_hw(q_hw).transpose(1, 2)
    k_hw = attn.k_norm_hw(k_hw).transpose(1, 2)
    q_h, q_w = q_hw.chunk(2, dim=-1)
    k_h, k_w = k_hw.chunk(2, dim=-1)

    def apply_axis(q_axis, k_axis, cos_axis, sin_axis):
        if int(cos_axis.shape[-1]) >= int(q_axis.shape[-1]):
            return apply_rotary_pos_emb(q_axis, k_axis, cos_axis, sin_axis, None, 1)
        cos_axis = cos_axis.unsqueeze(0) if q_axis.ndim == 4 and cos_axis.ndim == 2 else cos_axis
        sin_axis = sin_axis.unsqueeze(0) if k_axis.ndim == 4 and sin_axis.ndim == 2 else sin_axis
        return apply_rotary_emb(q_axis, cos_axis, sin_axis), apply_rotary_emb(k_axis, cos_axis, sin_axis)

    cos_t, sin_t = (x.unsqueeze(0) for x in attn.rotary_emb.cos_sin_1d(indexes[0]))
    q_t, k_t = apply_axis(q_t, k_t, cos_t, sin_t)
    cos_h, sin_h = attn.rotary_emb_hw.cos_sin_1d(indexes[1])
    q_h, k_h = apply_axis(q_h, k_h, cos_h, sin_h)
    cos_w, sin_w = attn.rotary_emb_hw.cos_sin_1d(indexes[2])
    q_w, k_w = apply_axis(q_w, k_w, cos_w, sin_w)

    got_q_parts = got_q.split([attn.head_dim // 2, attn.head_dim // 4, attn.head_dim // 4], dim=-1)
    got_k_parts = got_k.split([attn.head_dim // 2, attn.head_dim // 4, attn.head_dim // 4], dim=-1)
    for got, ref in zip(got_q_parts, (q_t, q_h, q_w), strict=True):
        torch.testing.assert_close(got, ref)
    for got, ref in zip(got_k_parts, (k_t, k_h, k_w), strict=True):
        torch.testing.assert_close(got, ref)


def test_sensenova_dense_mixed_mot_layer_runs_finite():
    from transformers import Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova_u1

    torch.manual_seed(53)
    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 4
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    cfg._attn_implementation = "eager"
    layer = sensenova_u1._SenseNovaDecoderLayer(cfg, layer_idx=0)
    # Deterministic local-generator init (load-time-init MoT params are torch.empty);
    # removes the uninitialized-memory dependence so the xfail below is stable.
    _g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _p in layer.parameters():
            _p.normal_(0.0, 0.02, generator=_g)
    hidden = torch.randn(1, 4, 16)
    indicators = torch.tensor([[False, True, False, True]])
    indexes = torch.stack(
        [
            torch.arange(4),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
        ],
        dim=0,
    )
    out = layer(
        hidden,
        image_gen_indicators=indicators,
        exist_non_image_gen_tokens=True,
        exist_image_gen_tokens=True,
        indexes=indexes,
        attention_mask=torch.zeros(1, 1, 4, 4),
    )
    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()


def test_sensenova_fm_modules_use_shared_linear_seams():
    from uniserve_worker.nn.diffusion import fm_modules

    block = fm_modules.ResBlock(channels=8)
    final = fm_modules.FinalLayer(model_channels=8, out_channels=3)
    head = fm_modules.FlowMatchingHead(input_dim=8, out_dim=4, dim=16, layers=2)
    assert isinstance(block.mlp[0], LinearBase)
    assert isinstance(block.adaLN_modulation[-1], LinearBase)
    assert isinstance(final.linear, LinearBase)
    assert isinstance(head.net.input_proj, LinearBase)
    out = head(torch.randn(3, 8), torch.linspace(0, 1, 3))
    assert out.shape == (3, 4)


def test_torch_sdpa_matches_bagel_sdpa_causal_gqa():
    torch.manual_seed(4)
    qi = torch.randn(3, 4, 8)
    fk = torch.randn(7, 2, 8)
    fv = torch.randn(7, 2, 8)
    fk_rep = fk.repeat_interleave(2, dim=1)
    fv_rep = fv.repeat_interleave(2, dim=1)
    q4 = qi.permute(1, 0, 2).unsqueeze(0)
    k4 = fk_rep.permute(1, 0, 2).unsqueeze(0)
    v4 = fv_rep.permute(1, 0, 2).unsqueeze(0)
    iq = torch.arange(qi.shape[0]).unsqueeze(1)
    ik = torch.arange(fk.shape[0]).unsqueeze(0)
    bad = ik > ((fk.shape[0] - qi.shape[0]) + iq)
    mask = torch.zeros(qi.shape[0], fk.shape[0], dtype=qi.dtype)
    mask.masked_fill_(bad, float("-inf"))
    ref = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask[None, None], scale=8**-0.5)
    ref = ref.squeeze(0).permute(1, 0, 2)
    got = get_attention_backend("torch_sdpa").forward(qi, fk, fv, causal=True, scale=8**-0.5)
    torch.testing.assert_close(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not has_attention_backend("sgl_kernel"), reason="sgl_kernel attention backend unavailable")
def test_sgl_kernel_attention_matches_torch_sdpa_causal_gqa_cuda():
    torch.manual_seed(407)
    q = torch.randn(2, 16, 9, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 9, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 4, 9, 128, device="cuda", dtype=torch.bfloat16)

    got = get_attention_backend("sgl_kernel").forward(q, k, v, causal=True, scale=128**-0.5)
    ref = get_attention_backend("torch_sdpa").forward(q, k, v, causal=True, scale=128**-0.5)

    torch.testing.assert_close(got, ref, atol=2.5e-3, rtol=2.5e-3)


def test_qwen_attention_helper_matches_torch_sdpa_layout():
    from transformers import Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova_u1

    torch.manual_seed(5)
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 3, 8)
    value = torch.randn(2, 2, 3, 8)
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 8
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    cfg._attn_implementation = "eager"
    attn = sensenova_u1._SenseNovaAttention(cfg, layer_idx=0)
    ref = get_attention_backend("torch_sdpa").forward(query, key, value, causal=False, scale=8**-0.5)
    got, _ = attn._attend_bhld(query, key, value, None)
    torch.testing.assert_close(got, ref.transpose(1, 2).contiguous())


def test_uni_attention_records_forward_context_stats():
    from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context

    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    attn = RadixAttention(num_heads=2, num_kv_heads=2, head_dim=4)
    stats = ForwardStats()

    with use_forward_context(ForwardContext(attention_backend_name="torch_sdpa", stats=stats)):
        out = attn(q, k, v, causal=False)

    assert out.shape == q.shape
    assert stats.attention_launches == 1
    assert stats.attention_ns > 0
    assert stats.attention_backend_counts == {"torch_sdpa": 1}


def test_ops_dispatcher_records_generic_operator_stats():
    import uniserve_worker.ops as ops
    from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context

    hidden = torch.randn(3, 8)
    weight = torch.ones(8)
    stats = ForwardStats()

    with use_forward_context(ForwardContext(stats=stats)):
        out = ops.rms_norm(hidden, weight, 1e-6, override="eager")

    assert out.shape == hidden.shape
    assert stats.operators.launches == 1
    assert stats.operators.ns > 0
    assert stats.operators.counts == {"rms_norm:eager": 1}
    wire = stats.to_wire()
    assert wire["operator_launches"] == 1
    assert wire["operator_counts"] == {"rms_norm:eager": 1}


def test_uni_attention_reuses_context_attention_metadata_for_paged_update(monkeypatch):
    from uniserve_worker.contracts.forward_context import (
        ForwardContext,
        TextAttentionMetadata,
        use_forward_context,
    )

    class FakePagedBackend:
        name = "fake_paged"

        def __init__(self):
            self.calls = 0
            self.kwargs = None

        def capabilities(self):
            return AttentionCapabilities(paged_kv=True, paged_block_size_multiple=1)

        def forward_paged(self, q, k_cache, v_cache, **kwargs):
            del k_cache, v_cache
            self.calls += 1
            self.kwargs = kwargs
            return torch.zeros_like(q)

    class FakePool:
        block_size = 1

        def layer_cache(self, layer):
            assert layer == 0
            empty = torch.empty(2, 1, 2, 4)
            return empty, empty

    class FakeCache:
        pool = FakePool()
        base_len = 0

        def block_table(self, *, device=None):  # pragma: no cover - must not be called
            del device
            raise AssertionError("block_table should be reused from ForwardContext metadata")

        def cache_seqlens(self, *, device=None):  # pragma: no cover - must not be called
            del device
            raise AssertionError("cache_seqlens should be reused from ForwardContext metadata")

    attn = RadixAttention(2, 2, 4, layer_id=0)
    backend = FakePagedBackend()
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 2, 1, 4)
    v = torch.randn(1, 2, 1, 4)
    cache = FakeCache()
    block_table = torch.tensor([[0]], dtype=torch.int32)
    cache_seqlens = torch.tensor([0], dtype=torch.int32)
    metadata = TextAttentionMetadata(
        cache=cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
    )
    stats = ForwardStats()

    with use_forward_context(ForwardContext(attention_backend=backend, attention_metadata=metadata, stats=stats)):
        out = attn(q, k, v, kv_cache=cache, update_cache=True, causal=True)

    assert out.shape == q.shape
    assert backend.calls == 1
    assert backend.kwargs["block_table"] is block_table
    assert backend.kwargs["cache_seqlens"] is cache_seqlens
    assert stats.attention_metadata_hits == 1
    assert stats.attention_metadata_misses == 0


def test_uni_attention_empty_batched_paged_prefill_appends_current_kv(monkeypatch):
    from uniserve_worker.contracts.forward_context import (
        ForwardContext,
        TextAttentionMetadata,
        use_forward_context,
    )

    class FakePlainBackend:
        name = "fake_plain"

        def __init__(self):
            self.calls = 0
            self.kwargs = None

        def capabilities(self):
            return AttentionCapabilities()

        def forward(self, q, k, v, **kwargs):
            del k, v
            self.calls += 1
            self.kwargs = kwargs
            return torch.zeros_like(q)

    class FakeCache:
        pool = object()

        def __init__(self):
            self.base_lens = [0, 0]
            self.append_calls = []

        def append(self, layer, k, v):
            self.append_calls.append((layer, k, v))

        def get(self, layer):  # pragma: no cover - branch must avoid dense fallback.
            del layer
            raise AssertionError("empty batched paged prefill should append directly")

    attn = RadixAttention(2, 2, 4, layer_id=0)
    backend = FakePlainBackend()
    q = torch.randn(2, 2, 3, 4)
    k = torch.randn(2, 2, 3, 4)
    v = torch.randn(2, 2, 3, 4)
    cache = FakeCache()
    metadata = TextAttentionMetadata(
        cache=cache,
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        cache_seqlens=torch.tensor([0, 0], dtype=torch.int32),
        mode="extend",
    )
    stats = ForwardStats()

    with use_forward_context(ForwardContext(attention_backend=backend, attention_metadata=metadata, stats=stats)):
        out = attn(q, k, v, kv_cache=cache, update_cache=True, causal=True)

    assert out.shape == q.shape
    assert backend.calls == 1
    assert backend.kwargs["causal"] is True
    assert len(cache.append_calls) == 1
    layer, appended_k, appended_v = cache.append_calls[0]
    assert layer == 0
    torch.testing.assert_close(appended_k, k.transpose(1, 2).contiguous())
    torch.testing.assert_close(appended_v, v.transpose(1, 2).contiguous())
    assert stats.attention_backend_counts == {"fake_plain": 1}


def test_uni_attention_runs_paged_varlen_prefill_with_context_metadata(monkeypatch):
    from uniserve_worker.contracts.forward_context import (
        ForwardContext,
        TextAttentionMetadata,
        use_forward_context,
    )

    class FakeVarlenBackend:
        name = "fake_varlen"

        def __init__(self):
            self.calls = 0
            self.args = None
            self.kwargs = None

        def capabilities(self):
            return AttentionCapabilities(varlen_attention=True, varlen_paged_kv=True)

        def forward_varlen(self, q, k_cache, v_cache, **kwargs):
            self.calls += 1
            self.args = (q, k_cache, v_cache)
            self.kwargs = kwargs
            return torch.zeros_like(q)

    class FakePool:
        block_size = 2
        supports_paged_attention_storage = True

        def __init__(self):
            self.k_cache = torch.empty(4, 2, 2, 4)
            self.v_cache = torch.empty(4, 2, 2, 4)

        def layer_cache(self, layer):
            assert layer == 0
            return self.k_cache, self.v_cache

    class FakeCache:
        def __init__(self):
            self.pool = FakePool()
            self.base_lens = [2, 1]
            self.append_calls = []

        def append_varlen(self, layer, k, v, query_lens, **kwargs):
            del kwargs
            self.append_calls.append((layer, k, v, tuple(query_lens)))

    attn = RadixAttention(2, 2, 4, layer_id=0)
    backend = FakeVarlenBackend()
    q = torch.randn(3, 2, 4)
    k = torch.randn(3, 2, 4)
    v = torch.randn(3, 2, 4)
    cache = FakeCache()
    block_table = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
    metadata = TextAttentionMetadata(
        cache=cache,
        block_table=block_table,
        cache_seqlens=torch.tensor([2, 1], dtype=torch.int32),
        query_lens=torch.tensor([2, 1], dtype=torch.int32),
        query_lens_cpu=(2, 1),
        kv_seqlens=torch.tensor([4, 2], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2, 3], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 4, 6], dtype=torch.int32),
        max_seqlen_q=2,
        max_seqlen_k=4,
        mode="extend",
    )
    stats = ForwardStats()

    with use_forward_context(ForwardContext(attention_backend=backend, attention_metadata=metadata, stats=stats)):
        out = attn(q, k, v, kv_cache=cache, update_cache=True, causal=True)

    assert out.shape == q.shape
    assert cache.append_calls == [(0, k, v, (2, 1))]
    assert backend.calls == 1
    assert backend.args == (q, cache.pool.k_cache, cache.pool.v_cache)
    assert backend.kwargs["block_table"] is block_table
    assert backend.kwargs["cu_seqlens_q"] is metadata.cu_seqlens_q
    assert backend.kwargs["cu_seqlens_k"] is metadata.cu_seqlens_k
    assert backend.kwargs["max_seqlen_q"] == 2
    assert backend.kwargs["max_seqlen_k"] == 4
    assert stats.attention_backend_counts == {"fake_varlen_paged_varlen": 1}


def test_uni_attention_runs_transient_paged_varlen_without_context_metadata():
    from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context

    class FakeVarlenBackend:
        name = "fake_varlen"

        def __init__(self):
            self.calls = 0
            self.args = None
            self.kwargs = None

        def capabilities(self):
            return AttentionCapabilities(varlen_attention=True, varlen_paged_kv=True)

        def forward_varlen(self, q, k_cache, v_cache, **kwargs):
            self.calls += 1
            self.args = (q, k_cache, v_cache)
            self.kwargs = kwargs
            return torch.zeros_like(q)

    class FakePool:
        block_size = 4
        supports_paged_attention_storage = True

        def __init__(self):
            self.k_cache = torch.empty(8, 4, 2, 4)
            self.v_cache = torch.empty(8, 4, 2, 4)

        def layer_cache(self, layer):
            assert layer == 0
            return self.k_cache, self.v_cache

    class FakeCache:
        def __init__(self):
            self.pool = FakePool()
            self.base_lens = (3, 5)
            self.append_calls = []
            self._block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
            self._cache_seqlens = torch.tensor([3, 5], dtype=torch.int32)

        def block_table(self, *, device=None):
            return self._block_table.to(device=device)

        def cache_seqlens(self, *, device=None):
            return self._cache_seqlens.to(device=device)

        def append_varlen(self, layer, k, v, query_lens, **kwargs):
            self.append_calls.append((layer, k, v, tuple(query_lens), kwargs))

    attn = RadixAttention(2, 2, 4, layer_id=0)
    backend = FakeVarlenBackend()
    cache = FakeCache()
    q = torch.randn(2, 2, 3, 4)
    k = torch.randn(2, 2, 3, 4)
    v = torch.randn(2, 2, 3, 4)

    with use_forward_context(ForwardContext(attention_backend=backend)):
        out = attn(q, k, v, kv_cache=cache, update_cache=True, causal=False)

    assert out.shape == q.shape
    assert backend.calls == 1
    q_run, k_cache, v_cache = backend.args
    torch.testing.assert_close(q_run, q.transpose(1, 2).reshape(6, 2, 4).contiguous())
    assert k_cache is cache.pool.k_cache
    assert v_cache is cache.pool.v_cache
    assert backend.kwargs["causal"] is False
    assert backend.kwargs["block_table"].tolist() == [[0, 1], [2, 3]]
    assert backend.kwargs["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert backend.kwargs["cu_seqlens_k"].tolist() == [0, 6, 14]
    assert backend.kwargs["max_seqlen_q"] == 3
    assert backend.kwargs["max_seqlen_k"] == 8
    assert len(cache.append_calls) == 1
    layer, appended_k, appended_v, query_lens, append_kwargs = cache.append_calls[0]
    assert layer == 0
    torch.testing.assert_close(appended_k, k.transpose(1, 2).reshape(6, 2, 4).contiguous())
    torch.testing.assert_close(appended_v, v.transpose(1, 2).reshape(6, 2, 4).contiguous())
    assert query_lens == (3, 3)
    assert append_kwargs["block_table"].tolist() == [[0, 1], [2, 3]]
    assert append_kwargs["cache_seqlens"].tolist() == [3, 5]
    assert append_kwargs["cu_seqlens_q"].tolist() == [0, 3, 6]


def test_uni_attention_falls_back_for_non_trunk_fa4_geometry():
    from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context

    class FakeFa4:
        name = "fa4_cute"

        def capabilities(self):
            return AttentionCapabilities(paged_kv=True)

        def forward(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("unsupported geometry should not reach fa4_cute")

    q = torch.randn(1, 2, 3, 72)
    k = torch.randn(1, 2, 3, 72)
    v = torch.randn(1, 2, 3, 72)
    attn = RadixAttention(num_heads=2, num_kv_heads=2, head_dim=72)
    stats = ForwardStats()

    with use_forward_context(ForwardContext(attention_backend=FakeFa4(), stats=stats)):
        out = attn(q, k, v, causal=False)

    assert out.shape == q.shape
    assert stats.attention_backend_counts == {"torch_sdpa": 1}


def test_qwen_attention_mixed_mot_path_uses_correct_branch_projections():
    from transformers import Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova_u1

    torch.manual_seed(52)
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 8
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    cfg._attn_implementation = "eager"
    attn = sensenova_u1._SenseNovaAttention(cfg, layer_idx=0)
    # The MoT gen-branch projections are load-time-initialized (torch.empty), so the
    # forward otherwise depends on uninitialized memory (order/allocation-flaky).
    # Seed every param from a LOCAL generator: fully deterministic, distinct per
    # param (branch projections stay distinguishable), and no global-RNG draw.
    _g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _p in attn.parameters():
            _p.normal_(0.0, 0.02, generator=_g)

    hidden = torch.randn(1, 4, 32)
    indicators = torch.tensor([[False, True, False, True]])
    indexes = torch.stack(
        [
            torch.arange(4),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
        ],
        dim=0,
    )
    mask = torch.zeros(1, 1, 4, 4)

    got, _ = attn(
        hidden,
        image_gen_indicators=indicators,
        exist_non_image_gen_tokens=True,
        exist_image_gen_tokens=True,
        indexes=indexes,
        attention_mask=mask,
    )

    und_q, und_k, und_v = attn._project_qkv(hidden, indexes, gen_branch=False)
    gen_q, gen_k, gen_v = attn._project_qkv(hidden, indexes, gen_branch=True)
    branch_mask = indicators[:, None, :, None]
    q = torch.where(branch_mask, gen_q, und_q)
    k = torch.where(branch_mask, gen_k, und_k)
    v = torch.where(branch_mask, gen_v, und_v)
    ref_attn, _ = attn._attend_bhld(q, k, v, mask)
    ref_attn = ref_attn.reshape(1, 4, -1).contiguous()
    ref = torch.zeros_like(hidden)
    ref[~indicators] = attn.o_proj(ref_attn[~indicators])
    ref[indicators] = attn.o_proj_mot_gen(ref_attn[indicators])
    torch.testing.assert_close(got, ref)


def test_sensenova_packed_visible_path_matches_dense_block_diagonal_mot():
    from transformers import Qwen3Config

    from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context
    from uniserve_worker.contracts.forward_mode import ForwardMode
    from uniserve_worker.execution.forward_stream import (
        ForwardPagedKVSegment,
        ForwardPagedKVView,
        ForwardStreamBuilder,
    )
    from uniserve_worker.models.sensenova import model as sensenova_u1
    from uniserve_worker.runtime.kv_pool import PagedKVPool

    class FakeVisibleBackend:
        name = "fake_visible"

        def capabilities(self):
            return AttentionCapabilities(visible_end=True, paged_kv=True)

        def forward_visible_end(
            self,
            q,
            k_cache,
            v_cache,
            *,
            visible_end,
            cu_seqlens_q,
            page_table,
            seqused_k,
            scale,
            **_kwargs,
        ):
            refs = []
            for row in range(page_table.shape[0]):
                q_start = int(cu_seqlens_q[row].item())
                q_end = int(cu_seqlens_q[row + 1].item())
                q_row = q[q_start:q_end]
                pages = [int(page.item()) for page in page_table[row]]
                full_k = torch.cat([k_cache[page] for page in pages], dim=0)[: int(seqused_k[row].item())]
                full_v = torch.cat([v_cache[page] for page in pages], dim=0)[: int(seqused_k[row].item())]
                full_k = full_k.repeat_interleave(q_row.shape[1] // full_k.shape[1], dim=1)
                full_v = full_v.repeat_interleave(q_row.shape[1] // full_v.shape[1], dim=1)
                kv_index = torch.arange(full_k.shape[0], device=q.device)[None, :]
                bad = kv_index >= visible_end[row, : q_row.shape[0], None]
                mask = torch.zeros(q_row.shape[0], full_k.shape[0], dtype=q.dtype, device=q.device)
                mask.masked_fill_(bad, float("-inf"))
                refs.append(
                    F.scaled_dot_product_attention(
                        q_row.transpose(0, 1).unsqueeze(0),
                        full_k.transpose(0, 1).unsqueeze(0),
                        full_v.transpose(0, 1).unsqueeze(0),
                        attn_mask=mask[None, None],
                        scale=scale,
                    ).squeeze(0).transpose(0, 1)
                )
            return torch.cat(refs, dim=0)

    torch.manual_seed(91)
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 8
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    cfg._attn_implementation = "eager"
    model = sensenova_u1._SenseNovaDecoderModel(cfg)
    # Deterministic local-generator init (load-time-init MoT params are torch.empty);
    # removes the uninitialized-memory dependence so the xfail below is stable.
    _g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _p in model.parameters():
            _p.normal_(0.0, 0.02, generator=_g)
    hidden = torch.randn(5, 32)
    indicators = torch.tensor([False, False, True, True, True])
    indexes = torch.stack(
        [
            torch.tensor([0, 1, 2, 2, 2]),
            torch.tensor([0, 0, 0, 0, 1]),
            torch.tensor([0, 0, 0, 1, 0]),
        ],
        dim=0,
    )

    dense_mask = torch.full((1, 1, 5, 5), float("-inf"))
    dense_mask[0, 0, 0, 0] = 0
    dense_mask[0, 0, 1, :2] = 0
    dense_mask[0, 0, 2:5, 2:5] = 0
    dense = model(
        inputs_embeds=hidden.unsqueeze(0),
        image_gen_indicators=indicators.unsqueeze(0),
        indexes=indexes,
        attention_mask={"full_attention": dense_mask},
    ).last_hidden_state.squeeze(0)

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
    stream = builder.build(device=hidden.device)
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=256,
        num_kv_heads=2,
        head_dim=8,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    view = ForwardPagedKVView(
        pool,
        [
            ForwardPagedKVSegment(block_ids=(0,), base_len=0, q_len=2),
            ForwardPagedKVSegment(block_ids=(1,), base_len=0, q_len=3),
        ],
    )
    with use_forward_context(ForwardContext(attention_backend=FakeVisibleBackend())):
        packed = model.forward_packed_visible(
            hidden,
            image_gen_indicators=indicators,
            indexes=stream.indexes,
            forward_stream=stream,
            kv_view=view,
        )

    torch.testing.assert_close(packed, dense, atol=1e-5, rtol=1e-5)


def test_sensenova_packed_visible_all_gen_uses_single_modality_qkv(monkeypatch):
    from transformers import Qwen3Config

    from uniserve_worker.execution.forward_stream import ForwardStream
    from uniserve_worker.models.sensenova import model as sensenova_u1

    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 8
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    cfg._attn_implementation = "eager"
    attn = sensenova_u1._SenseNovaAttention(cfg, layer_idx=0)
    attn.o_proj_mot_gen = nn.Identity()

    hidden = torch.randn(5, 32)
    indexes = torch.stack(
        [
            torch.full((5,), 7, dtype=torch.long),
            torch.arange(5, dtype=torch.long),
            torch.zeros(5, dtype=torch.long),
        ],
        dim=0,
    )
    stream = ForwardStream(
        segments=(),
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
        visible_end=torch.full((1, 5), 5, dtype=torch.int32),
        indexes=indexes,
    )
    calls: list[bool] = []

    def fail_routed(*_args, **_kwargs):
        raise AssertionError("all-gen packed visible should not use routed QKV")

    def fake_project(hidden_states, got_indexes, *, gen_branch, packed_rope=None):
        del packed_rope
        calls.append(bool(gen_branch))
        assert hidden_states.shape == (1, 5, 32)
        torch.testing.assert_close(got_indexes, indexes)
        q = torch.arange(1 * 4 * 5 * 8, dtype=hidden.dtype).reshape(1, 4, 5, 8)
        k = torch.zeros(1, 2, 5, 8, dtype=hidden.dtype)
        v = torch.zeros(1, 2, 5, 8, dtype=hidden.dtype)
        return q, k, v

    def fake_attend(q, k, v, *, forward_stream, kv_view):
        del k, v, kv_view
        assert forward_stream is stream
        return q

    monkeypatch.setattr(attn, "_project_qkv_flat_routed", fail_routed)
    monkeypatch.setattr(attn, "_project_qkv", fake_project)
    monkeypatch.setattr(attn, "_attend_packed_visible", fake_attend)

    out = attn.forward_packed_visible(
        hidden,
        image_gen_indicators=torch.ones(5, dtype=torch.bool),
        indexes=indexes,
        exist_non_image_gen_tokens=False,
        exist_image_gen_tokens=True,
        und=torch.zeros(5, dtype=torch.bool),
        forward_stream=stream,
        kv_view=None,
    )

    expected = (
        torch.arange(4 * 5 * 8, dtype=hidden.dtype)
        .reshape(4, 5, 8)
        .transpose(0, 1)
        .contiguous()
        .reshape(5, 32)
    )
    torch.testing.assert_close(out, expected)
    assert calls == [True]


def test_sensenova_packed_visible_fully_visible_uses_paged_varlen(monkeypatch):
    from uniserve_worker.contracts.forward_mode import ForwardMode
    from uniserve_worker.execution.forward_stream import (
        ForwardPagedKVSegment,
        ForwardPagedKVView,
        ForwardStreamBuilder,
    )
    from uniserve_worker.models.sensenova import model as sensenova_u1
    from uniserve_worker.runtime.kv_pool import PagedKVPool

    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=1,
        visible_policy="causal",
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
        indexes=torch.tensor([[2, 2, 2], [0, 0, 1], [0, 1, 0]]),
    )
    stream = builder.build()
    assert stream.fully_visible

    pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    view = ForwardPagedKVView(
        pool,
        [
            ForwardPagedKVSegment(block_ids=(0,), base_len=1, q_len=1),
            ForwardPagedKVSegment(block_ids=(1,), base_len=0, q_len=3),
        ],
    )
    q = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    k = q + 100
    v = q + 200
    calls = []

    def fake_attention(got_q, got_k_cache, got_v_cache, **kwargs):
        calls.append(kwargs)
        assert got_q is q
        assert got_k_cache.data_ptr() == pool.k[0].data_ptr()
        assert got_v_cache.data_ptr() == pool.v[0].data_ptr()
        assert kwargs["regime"] is sensenova_u1.ops.AttentionRegime.EXTEND
        assert kwargs["causal"] is False
        assert kwargs["max_seqlen_q"] == 3
        assert kwargs["max_seqlen_k"] == 3
        assert kwargs["block_table"].tolist() == [[0], [1]]
        assert kwargs["cu_seqlens_q"].tolist() == [0, 1, 4]
        assert kwargs["cu_seqlens_k"].tolist() == [0, 2, 5]
        return got_q + 1

    monkeypatch.setattr(sensenova_u1.ops, "attention", fake_attention)
    owner = SimpleNamespace(layer_idx=0, scaling=0.5)

    out = sensenova_u1._SenseNovaAttention._attend_packed_visible(
        owner,
        q,
        k,
        v,
        forward_stream=stream,
        kv_view=view,
    )

    assert len(calls) == 1
    torch.testing.assert_close(out, q + 1)
    torch.testing.assert_close(pool.k[0, 0, 1], k[0])
    torch.testing.assert_close(pool.k[0, 1, :3], k[1:])


def test_sensenova_admitted_forward_does_not_split_fallback(monkeypatch):
    from uniserve_worker.contracts.batches import UniForwardBatch
    from uniserve_worker.models.sensenova import model as sensenova_u1

    wrapper = sensenova_u1.SenseNovaU1ForUnifiedGeneration(
        config={"llm_config": {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 4}}
    )
    batch = UniForwardBatch.from_ops(
        [
            {"req_id": 1, "kind": "decode_und", "token_ids": [11], "pos_range": [0, 1]},
            {"req_id": 2, "kind": "denoise_gen", "cfg": {"branch_count": 1}},
        ]
    )

    class RequestStates:
        def get(self, req_id):
            return {"req_id": req_id}

    monkeypatch.setattr(wrapper, "prepare_denoise", lambda _state, _op: object())
    monkeypatch.setattr(sensenova_u1, "run_packed_mixed_forward", lambda *_args: False)

    def split_fallback_called(_op):
        raise AssertionError("admitted mixed batch must not use text split fallback")

    monkeypatch.setattr(wrapper, "run_text_logits", split_fallback_called)
    with pytest.raises(WorkerError, match="split-mode fallback is disabled"):
        wrapper.run_forward(
            batch,
            request_states=RequestStates(),
            group=list(enumerate(batch.ops)),
        )


def test_sensenova_forward_text_input_ids_consumes_last_sampled_relay():
    from uniserve_worker.execution import packed_mixed_forward

    relay_tensor = torch.tensor([7], dtype=torch.long)
    state = SimpleNamespace(decode_relay=SimpleNamespace(token_tensor=relay_tensor))

    class RequestStates:
        def get(self, req_id):
            assert req_id == 1
            return state

    ids = packed_mixed_forward._forward_text_input_ids(
        {
            "req_id": 1,
            "kind": "decode_und",
            "token_ids": [0],
            "token_source": "last_sampled",
        },
        req_id=1,
        tokens=[0],
        request_states=RequestStates(),
        device=torch.device("cpu"),
    )

    assert int(ids.data_ptr()) == int(relay_tensor.data_ptr())
    torch.testing.assert_close(ids, torch.tensor([7], dtype=torch.long))


def test_sensenova_forward_text_input_ids_requires_last_sampled_relay():
    from uniserve_worker.execution import packed_mixed_forward

    state = SimpleNamespace(decode_relay=SimpleNamespace(token_tensor=None))

    class RequestStates:
        def get(self, req_id):
            assert req_id == 1
            return state

    with pytest.raises(WorkerError, match="last_sampled"):
        packed_mixed_forward._forward_text_input_ids(
            {
                "req_id": 1,
                "kind": "decode_und",
                "token_ids": [0],
                "token_source": "last_sampled",
            },
            req_id=1,
            tokens=[0],
            request_states=RequestStates(),
            device=torch.device("cpu"),
        )


def test_sensenova_forward_sampling_updates_decode_relay():
    from uniserve_worker.execution import packed_mixed_forward
    from uniserve_worker.nn.sampler import apply_sampling_batched_with_device_tokens

    state = SimpleNamespace(
        sampling={"temperature": 0.0},
        decode_relay=SimpleNamespace(
            token_id=None,
            token_tensor=None,
            position_id=None,
            position_tensor=None,
        ),
    )

    class RequestStates:
        def get(self, req_id):
            assert req_id == 1
            return state

    logits = torch.tensor([[[-10.0, -5.0, 8.0, -2.0]]])
    sampled = apply_sampling_batched_with_device_tokens(
        logits.reshape(1, -1),
        [dict(state.sampling)],
        [[]],
        [None],
        [None],
    )
    sample = sampled.samples[0]

    assert RequestStates().get(1) is state
    packed_mixed_forward._store_forward_sampled_token_relay(
        state,
        token_id=int(sample.token_id),
        device=torch.device("cpu"),
        position_id=9,
        token_tensor=sampled.device_tokens[0:1],
    )

    assert int(sample.token_id) == 2
    assert state.decode_relay.token_id == 2
    torch.testing.assert_close(state.decode_relay.token_tensor, torch.tensor([2], dtype=torch.long))
    assert state.decode_relay.position_id == 9
    torch.testing.assert_close(state.decode_relay.position_tensor, torch.tensor([9], dtype=torch.long))


def test_sensenova_text_batch_delegates_to_scalar_text_stepper(monkeypatch):
    from uniserve_worker.models.sensenova import model as sensenova_u1
    from uniserve_worker.runtime.kv_pool import PagedKVPool

    class FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward_packed_visible(
            self,
            inputs_embeds,
            *,
            image_gen_indicators,
            indexes,
            forward_stream,
            kv_view,
        ):
            self.calls.append(
                {
                    "inputs": inputs_embeds.detach().clone(),
                    "indicators": image_gen_indicators.detach().clone(),
                    "indexes": indexes.detach().clone(),
                    "visible_end": forward_stream.visible_end.detach().clone(),
                    "segments": kv_view.segments,
                }
            )
            return inputs_embeds + indexes[0].to(inputs_embeds.dtype).unsqueeze(1)

    class FakeLanguage(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeDecoder()
            self.embed = nn.Embedding(16, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)

        def get_input_embeddings(self):
            return self.embed

    wrapper = sensenova_u1.SenseNovaU1ForUnifiedGeneration(
        config={"llm_config": {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 2}}
    )
    language = FakeLanguage()
    wrapper.model = SimpleNamespace(language_model=language)
    wrapper.device = "cpu"
    wrapper.kv_pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=8,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )

    scalar_calls = []

    def scalar_fallback(op):
        scalar_calls.append(dict(op))
        return torch.full((1, 8), float(len(scalar_calls)))

    # Batching + the one-token decode graph now live in the system text driver;
    # the model's run_text_logits_batch delegates to it, and the driver falls back
    # to its per-op scalar stepper for ineligible batches (here: multi-token prefill).
    driver = wrapper._text_driver()
    monkeypatch.setattr(driver, "_run_text_logits_one", scalar_fallback)
    logits = wrapper.run_text_logits_batch(
        [
            {
                "req_id": 7,
                "kind": "prefill_und",
                "token_ids": [3, 4],
                "pos_range": [0, 2],
                "new_block_ids": [0],
            }
        ]
    )

    assert len(logits) == 1
    assert tuple(logits[0].shape) == (1, 8)
    torch.testing.assert_close(logits[0], torch.ones((1, 8)))
    assert scalar_calls == [
        {
            "req_id": 7,
            "kind": "prefill_und",
            "token_ids": [3, 4],
            "pos_range": [0, 2],
            "new_block_ids": [0],
        }
    ]
    assert language.model.calls == []


def test_sensenova_text_decode_batch_delegates_to_scalar_text_stepper(monkeypatch):
    from uniserve_worker.models.sensenova import model as sensenova_u1
    from uniserve_worker.runtime.kv_pool import PagedKVPool

    class FakeLanguage(nn.Module):
        def forward(self, **_kwargs):
            raise AssertionError("CPU decode must fall back to the scalar stepper, not the neural forward")

    wrapper = sensenova_u1.SenseNovaU1ForUnifiedGeneration(
        config={"llm_config": {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 2}}
    )
    wrapper.model = SimpleNamespace(language_model=FakeLanguage())
    wrapper.device = "cpu"
    wrapper.kv_pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=8,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    for req_id, block_id, length in ((11, 0, 3), (12, 1, 5)):
        state = wrapper.interleaved_image_state(req_id)
        state.cond.block_ids = [block_id]
        wrapper._ensure_host_cache(state.cond)
        state.cond.past.length = length
        state.cond.t_index = length - 1
        state.cond.last_token_id = 100 + req_id

    scalar_calls = []

    def scalar_fallback(op):
        scalar_calls.append(dict(op))
        row = len(scalar_calls) - 1
        return torch.tensor([[float(row * 3), float(row * 3 + 1), float(row * 3 + 2)]])

    driver = wrapper._text_driver()
    monkeypatch.setattr(driver, "_run_text_logits_one", scalar_fallback)

    logits = wrapper.run_text_logits_batch(
        [
            {"req_id": 11, "kind": "decode_und", "token_ids": [7], "pos_range": [3, 4]},
            {"req_id": 12, "kind": "decode_und", "token_ids": [9], "pos_range": [5, 6]},
        ]
    )

    assert len(logits) == 2
    torch.testing.assert_close(logits[0], torch.tensor([[0.0, 1.0, 2.0]]))
    torch.testing.assert_close(logits[1], torch.tensor([[3.0, 4.0, 5.0]]))
    assert scalar_calls == [
        {"req_id": 11, "kind": "decode_und", "token_ids": [7], "pos_range": [3, 4]},
        {"req_id": 12, "kind": "decode_und", "token_ids": [9], "pos_range": [5, 6]},
    ]


def test_sensenova_denoise_forward_segment_is_transient_not_persistent():
    from uniserve_worker.execution.forward_stream import ForwardStreamBuilder
    from uniserve_worker.models.sensenova import model as sensenova_u1

    wrapper = sensenova_u1.SenseNovaU1ForUnifiedGeneration(
        config={"llm_config": {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 4}}
    )
    builder = ForwardStreamBuilder()
    kv_segments = []
    indexes = torch.tensor([[5, 5], [0, 0], [0, 1]], dtype=torch.long)

    wrapper._add_denoise_forward_segment(
        builder=builder,
        kv_segments=kv_segments,
        row_index=0,
        req_id=3,
        op={"kind": "denoise_gen"},
        cache=SimpleNamespace(block_ids=[4], length=6),
        indexes=indexes,
        q_len=2,
        branch_index=1,
        device=torch.device("cpu"),
    )
    stream = builder.build(device=torch.device("cpu"))

    assert stream.visible_end.tolist() == [[8, 8]]
    assert len(kv_segments) == 1
    assert kv_segments[0].write_kv is True
    assert kv_segments[0].persist_kv is False
    assert kv_segments[0].branch_id == 1


def test_sensenova_duplicate_new_request_preserves_live_interleaved_cache():
    from uniserve_worker.models.sensenova import model as sensenova_u1
    from uniserve_worker.runtime.request_state import RequestStateTable

    wrapper = sensenova_u1.SenseNovaU1ForUnifiedGeneration(
        config={"llm_config": {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 4}}
    )
    states = RequestStateTable()
    state = states.create_or_update(
        6,
        {"req_id": 6, "block_ids": [10, 11], "sampling": {"temperature": 0.0}},
    )
    wrapper.on_new_request(6, state)
    image_state = wrapper.interleaved_image_state(6)
    sentinel_past = object()
    image_state.cond.block_ids = [10, 11, 12, 13, 14]
    image_state.cond.past = sentinel_past
    image_state.cond.t_index = 1085

    duplicate = states.create_or_update(
        6,
        {"req_id": 6, "block_ids": [], "sampling": {"temperature": 0.0}},
    )
    wrapper.on_new_request(6, duplicate)

    assert wrapper.interleaved_image_state(6) is image_state
    assert states.get(6).block_ids == [10, 11]
    assert image_state.cond.block_ids == [10, 11, 12, 13, 14]
    assert image_state.cond.past is sentinel_past
    assert image_state.cond.t_index == 1085


def test_packed_mixed_syncs_host_cache_blocks_from_runner_state():
    from uniserve_worker.execution.packed_mixed_forward import _sync_host_cache_blocks
    from uniserve_worker.runtime.kv_pool import PagedKVPool
    from uniserve_worker.runtime.paged_text_cache import PagedTextCache
    from uniserve_worker.runtime.request_state import RequestStateTable

    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    cache = SimpleNamespace(
        block_ids=[0],
        past=PagedTextCache(pool, [0], num_layers=1, length=2),
    )
    states = RequestStateTable()
    states.create_or_update(17, {"req_id": 17, "block_ids": [0, 3]})

    with pytest.raises(WorkerError, match="does not have enough logical blocks"):
        cache.past.ensure_capacity(3)

    _sync_host_cache_blocks(SimpleNamespace(kv_pool=pool), cache, states, 17)

    assert cache.block_ids == [0, 3]
    assert cache.past.block_ids == [0, 3]
    cache.past.ensure_capacity(3)

    stale = SimpleNamespace(
        block_ids=[1],
        past=PagedTextCache(pool, [1], num_layers=1, length=2),
    )
    states.create_or_update(18, {"req_id": 18, "block_ids": [1]})

    with pytest.raises(WorkerError, match="does not have enough logical blocks"):
        stale.past.ensure_capacity(3)

    _sync_host_cache_blocks(
        SimpleNamespace(kv_pool=pool),
        stale,
        states,
        18,
        {"req_id": 18, "new_block_ids": [2]},
    )

    assert states.get(18).block_ids == [1, 2]
    assert stale.block_ids == [1, 2]
    assert stale.past.block_ids == [1, 2]
    stale.past.ensure_capacity(3)


def test_packed_mixed_hydrates_cached_prefix_length_from_pos_range():
    from uniserve_worker.contracts.batches import UniForwardBatch
    from uniserve_worker.contracts.forward_mode import ForwardMode
    from uniserve_worker.execution.denoise_driver import TextImageDenoiseStep
    from uniserve_worker.execution.forward_stream import (
        ForwardPagedKVSegment,
        ForwardStreamBuilder,
    )
    from uniserve_worker.execution.packed_mixed_forward import run_packed_mixed_forward
    from uniserve_worker.runtime.kv_pool import PagedKVPool
    from uniserve_worker.runtime.paged_text_cache import PagedTextCache
    from uniserve_worker.runtime.request_state import RequestStateTable

    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    cond_cache = SimpleNamespace(
        block_ids=[0, 1],
        past=PagedTextCache(pool, [0, 1], num_layers=1, length=0),
        t_index=-1,
        last_logits=None,
        last_token_id=None,
    )
    image_cache = PagedTextCache(pool, [2], num_layers=1, length=0)
    image_embeds = torch.randn(1, 1, 4)
    img = SimpleNamespace(token_h=1, token_w=1, width=16, height=16)
    indexes = torch.tensor([[0], [0], [0]], dtype=torch.long)
    step = TextImageDenoiseStep(
        req_id=8,
        state=SimpleNamespace(),
        op={"req_id": 8, "kind": "denoise_gen", "cfg": {"branch_count": 1}},
        latent=torch.zeros(1, 3),
        t=torch.tensor(0.0),
        t_next=torch.tensor(1.0),
        step_index=0,
        total_steps=1,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.0, 1.0),
        cfg_renorm_type="none",
        cfg_renorm_min=0.0,
        extra={"img": img, "image_embeds": image_embeds},
    )

    class Owner:
        model = object()
        device = torch.device("cpu")
        num_layers = 1
        eos_id = 0
        kv_pool = pool
        residency = SimpleNamespace(release_scratch_cache=lambda _cache: None)

        def __init__(self) -> None:
            self.state = SimpleNamespace(cond=cond_cache)
            self.text_base_len = None
            self.kv_view = None
            self.updated = None

        def _forward_target_pool(self, _denoise_steps):
            return pool

        def interleaved_image_state(self, req_id):
            assert int(req_id) == 7
            return self.state

        def _extend_cache_blocks(self, _cache, _op):
            return None

        def _ensure_host_cache(self, _cache):
            return None

        def _same_kv_pool(self, candidate, first):
            return first is None or candidate is first

        def _stage_text_cache_for_forward(self, *_args, **_kwargs):
            raise AssertionError("text cache should already target the packed forward pool")

        def packed_text_embeddings(self, ids):
            return torch.ones((int(ids.numel()), 4), dtype=torch.float32)

        def _text_indexes(self, start, seq_len, *, device):
            return torch.stack(
                [
                    torch.arange(int(start), int(start) + int(seq_len), device=device),
                    torch.zeros(int(seq_len), dtype=torch.long, device=device),
                    torch.zeros(int(seq_len), dtype=torch.long, device=device),
                ],
                dim=0,
            )

        def _add_text_forward_segment(
            self,
            *,
            builder: ForwardStreamBuilder,
            kv_segments: list[ForwardPagedKVSegment],
            row_index: int,
            req_id: int,
            op: dict,
            mode: ForwardMode,
            cache,
            q_len: int,
            start_pos: int,
            device: torch.device,
        ) -> None:
            del start_pos
            self.text_base_len = int(cache.past.length)
            builder.add_segment(
                op_index=row_index,
                req_id=req_id,
                kind=str(op["kind"]),
                mode=mode,
                modality="und",
                segment_class="decode",
                q_len=q_len,
                prefix_len=int(cache.past.length),
                indexes=self._text_indexes(int(cache.past.length), q_len, device=device),
            )
            kv_segments.append(
                ForwardPagedKVSegment(
                    block_ids=tuple(cache.past.block_ids),
                    base_len=int(cache.past.length),
                    q_len=q_len,
                    write_kv=True,
                )
            )

        def _denoise_branch_inputs(self, _img, _branch):
            return indexes, image_cache

        def _wait_gen_cache_ready(self, _cache):
            return None

        def _add_denoise_forward_segment(
            self,
            *,
            builder: ForwardStreamBuilder,
            kv_segments: list[ForwardPagedKVSegment],
            row_index: int,
            req_id: int,
            op: dict,
            cache,
            indexes: torch.Tensor,
            q_len: int,
            branch_index: int,
            device: torch.device,
        ) -> None:
            builder.add_segment(
                op_index=row_index,
                req_id=req_id,
                kind=str(op["kind"]),
                mode=ForwardMode.DENOISE,
                modality="gen",
                segment_class="denoise",
                q_len=q_len,
                prefix_len=int(cache.length),
                branch_id=branch_index,
                visible_policy="bidirectional",
                indexes=indexes.to(device=device),
            )
            kv_segments.append(
                ForwardPagedKVSegment(
                    block_ids=tuple(cache.block_ids),
                    base_len=int(cache.length),
                    q_len=q_len,
                    write_kv=True,
                    persist_kv=False,
                    branch_id=branch_index,
                )
            )

        def packed_decoder_forward(self, input_embeds, **kwargs):
            self.kv_view = kwargs["kv_view"]
            return input_embeds

        def packed_text_logits(self, hidden):
            logits = torch.zeros((hidden.shape[0], hidden.shape[1], 5), dtype=torch.float32)
            logits[..., 3] = 1.0
            return logits

        def packed_hidden_to_velocity(self, _hidden_states, _t, latent, **_kwargs):
            return torch.zeros_like(latent)

        def accept_denoise_update(self, _step, updated):
            self.updated = updated

    states = RequestStateTable()
    states.create_or_update(7, {"req_id": 7, "block_ids": [0, 1], "sampling": {"temperature": 0.0}})
    states.create_or_update(8, {"req_id": 8, "block_ids": [2]})
    batch = UniForwardBatch.from_ops(
        [
            {"req_id": 7, "kind": "decode_und", "token_ids": [4], "pos_range": [4, 5]},
            step.op,
        ]
    )
    results = [None, None]
    owner = Owner()

    assert run_packed_mixed_forward(owner, batch, states, [(1, step)], results)
    assert owner.text_base_len == 4
    assert cond_cache.past.length == 5
    assert cond_cache.t_index == 4
    assert owner.kv_view is not None
    assert owner.kv_view.cache_seqlens_after().tolist()[0] == 5
    assert results[0].sampled_token_id == 3


def test_sensenova_packed_mixed_runs_text_only_forward_batch():
    from uniserve_worker.contracts.batches import UniForwardBatch
    from uniserve_worker.contracts.forward_mode import ForwardMode
    from uniserve_worker.execution.forward_stream import (
        ForwardPagedKVSegment,
        ForwardStreamBuilder,
    )
    from uniserve_worker.execution.packed_mixed_forward import run_packed_mixed_forward
    from uniserve_worker.runtime.kv_pool import PagedKVPool
    from uniserve_worker.runtime.paged_text_cache import PagedTextCache
    from uniserve_worker.runtime.request_state import RequestStateTable

    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    decode_cache = SimpleNamespace(
        block_ids=[0],
        past=PagedTextCache(pool, [0], num_layers=1, length=2),
        t_index=1,
        last_logits=None,
        last_token_id=None,
    )
    extend_cache = SimpleNamespace(
        block_ids=[1],
        past=PagedTextCache(pool, [1], num_layers=1, length=0),
        t_index=-1,
        last_logits=None,
        last_token_id=None,
    )

    class Owner:
        model = object()
        device = torch.device("cpu")
        num_layers = 1
        eos_id = 0
        kv_pool = pool
        residency = SimpleNamespace(release_scratch_cache=lambda _cache: None)

        def __init__(self) -> None:
            self.states = {
                7: SimpleNamespace(cond=decode_cache),
                8: SimpleNamespace(cond=extend_cache),
            }
            self.text_base_lens: list[int] = []
            self.kv_view = None

        def _forward_target_pool(self, denoise_steps):
            assert denoise_steps == []
            return None

        def interleaved_image_state(self, req_id):
            return self.states[int(req_id)]

        def _extend_cache_blocks(self, _cache, _op):
            return None

        def _ensure_host_cache(self, _cache):
            return None

        def _same_kv_pool(self, candidate, first):
            return first is None or candidate is first

        def _stage_text_cache_for_forward(self, *_args, **_kwargs):
            raise AssertionError("text-only forward should use the host KV pool directly")

        def packed_text_embeddings(self, ids):
            return torch.ones((int(ids.numel()), 4), dtype=torch.float32)

        def _text_indexes(self, start, seq_len, *, device):
            return torch.stack(
                [
                    torch.arange(int(start), int(start) + int(seq_len), device=device),
                    torch.zeros(int(seq_len), dtype=torch.long, device=device),
                    torch.zeros(int(seq_len), dtype=torch.long, device=device),
                ],
                dim=0,
            )

        def _add_text_forward_segment(
            self,
            *,
            builder: ForwardStreamBuilder,
            kv_segments: list[ForwardPagedKVSegment],
            row_index: int,
            req_id: int,
            op: dict,
            mode: ForwardMode,
            cache,
            q_len: int,
            start_pos: int,
            device: torch.device,
        ) -> None:
            self.text_base_lens.append(int(cache.past.length))
            builder.add_segment(
                op_index=row_index,
                req_id=req_id,
                kind=str(op["kind"]),
                mode=mode,
                modality="und",
                segment_class="decode" if mode is ForwardMode.DECODE else "extend",
                q_len=q_len,
                prefix_len=int(cache.past.length),
                indexes=self._text_indexes(int(start_pos), q_len, device=device),
            )
            kv_segments.append(
                ForwardPagedKVSegment(
                    block_ids=tuple(cache.past.block_ids),
                    base_len=int(cache.past.length),
                    q_len=q_len,
                    write_kv=True,
                )
            )

        def packed_decoder_forward(self, input_embeds, **kwargs):
            self.kv_view = kwargs["kv_view"]
            return input_embeds

        def packed_text_logits(self, hidden):
            logits = torch.zeros((hidden.shape[0], hidden.shape[1], 6), dtype=torch.float32)
            logits[..., 4] = 1.0
            return logits

    states = RequestStateTable()
    states.create_or_update(7, {"req_id": 7, "block_ids": [0], "sampling": {"temperature": 0.0}})
    states.create_or_update(8, {"req_id": 8, "block_ids": [1], "sampling": {"temperature": 0.0}})
    batch = UniForwardBatch.from_ops(
        [
            {"req_id": 7, "kind": "decode_und", "token_ids": [13], "pos_range": [2, 3]},
            {"req_id": 8, "kind": "prefill_und", "token_ids": [21, 22], "pos_range": [0, 2]},
        ]
    )
    results = [None, None]
    owner = Owner()

    assert run_packed_mixed_forward(owner, batch, states, [], results)
    assert owner.text_base_lens == [2, 0]
    assert owner.kv_view is not None
    assert owner.kv_view.cache_seqlens_after().tolist() == [3, 2]
    assert decode_cache.past.length == 3
    assert decode_cache.t_index == 2
    assert extend_cache.past.length == 2
    assert extend_cache.t_index == 1
    assert [result.sampled_token_id for result in results] == [4, 4]


def test_sensenova_packed_mixed_reserves_transient_denoise_cache_capacity():
    from uniserve_worker.contracts.batches import UniForwardBatch
    from uniserve_worker.contracts.forward_mode import ForwardMode
    from uniserve_worker.execution.denoise_driver import TextImageDenoiseStep
    from uniserve_worker.execution.forward_stream import (
        ForwardPagedKVSegment,
        ForwardStreamBuilder,
    )
    from uniserve_worker.execution.packed_mixed_forward import run_packed_mixed_forward
    from uniserve_worker.runtime.kv_pool import PagedKVPool
    from uniserve_worker.runtime.paged_text_cache import PagedTextCache

    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    allocated: list[int] = []

    def allocate_blocks(n: int) -> list[int]:
        start = 1 + len(allocated)
        blocks = list(range(start, start + int(n)))
        allocated.extend(blocks)
        return blocks

    cache = PagedTextCache(
        pool,
        [0],
        num_layers=1,
        length=4,
        allocate_blocks=allocate_blocks,
    )
    image_embeds = torch.randn(1, 2, 4)
    img = SimpleNamespace(token_h=1, token_w=2, width=32, height=16)
    indexes = torch.tensor([[4, 4], [0, 0], [0, 1]], dtype=torch.long)
    step = TextImageDenoiseStep(
        req_id=9,
        state=SimpleNamespace(),
        op={"req_id": 9, "kind": "denoise_gen", "cfg": {"branch_count": 1}},
        latent=torch.zeros(1, 3),
        t=torch.tensor(0.0),
        t_next=torch.tensor(1.0),
        step_index=0,
        total_steps=1,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.0, 1.0),
        cfg_renorm_type="none",
        cfg_renorm_min=0.0,
        extra={"img": img, "image_embeds": image_embeds},
    )

    class Owner:
        model = object()
        device = torch.device("cpu")
        num_layers = 1
        residency = SimpleNamespace(release_scratch_cache=lambda _cache: None)

        def __init__(self) -> None:
            self.kv_view = None
            self.updated = None

        def _forward_target_pool(self, _denoise_steps):
            return pool

        def _denoise_branch_inputs(self, _img, _branch):
            return indexes, cache

        def _wait_gen_cache_ready(self, _cache):
            return None

        def _same_kv_pool(self, candidate, first):
            return first is None or candidate is first

        def _add_denoise_forward_segment(
            self,
            *,
            builder: ForwardStreamBuilder,
            kv_segments: list[ForwardPagedKVSegment],
            row_index: int,
            req_id: int,
            op: dict,
            cache,
            indexes: torch.Tensor,
            q_len: int,
            branch_index: int,
            device: torch.device,
        ) -> None:
            builder.add_segment(
                op_index=row_index,
                req_id=req_id,
                kind=str(op["kind"]),
                mode=ForwardMode.DENOISE,
                modality="gen",
                segment_class="denoise",
                q_len=q_len,
                prefix_len=int(cache.length),
                branch_id=branch_index,
                visible_policy="bidirectional",
                indexes=indexes.to(device=device),
            )
            kv_segments.append(
                ForwardPagedKVSegment(
                    block_ids=tuple(cache.block_ids),
                    base_len=int(cache.length),
                    q_len=q_len,
                    write_kv=True,
                    persist_kv=False,
                    branch_id=branch_index,
                )
            )

        def packed_decoder_forward(self, input_embeds, **kwargs):
            self.kv_view = kwargs["kv_view"]
            return input_embeds

        def packed_hidden_to_velocity(self, _hidden_states, _t, latent, **_kwargs):
            return torch.zeros_like(latent)

        def accept_denoise_update(self, _step, updated):
            self.updated = updated

    owner = Owner()
    batch = UniForwardBatch.from_ops([step.op])

    assert run_packed_mixed_forward(owner, batch, SimpleNamespace(), [(0, step)], [None])
    assert allocated == [1]
    assert cache.length == 4
    assert cache.block_ids == [0, 1]
    assert owner.kv_view is not None
    assert owner.kv_view.cache_seqlens_after().tolist() == [6]
    assert owner.kv_view.persistent_cache_seqlens_after().tolist() == [4]
    torch.testing.assert_close(owner.updated, torch.zeros(1, 3))


def test_qwen_attention_paged_update_is_not_env_gated(monkeypatch):
    from transformers import Qwen3Config

    from uniserve_worker.models.sensenova import model as sensenova_u1

    class FakePagedBackend:
        name = "fake_paged"

        def __init__(self):
            self.calls = 0

        def capabilities(self):
            return AttentionCapabilities(paged_kv=True)

        def forward_paged(self, q, k_cache, v_cache, **kwargs):
            self.calls += 1
            assert kwargs["k"] is not None
            assert kwargs["v"] is not None
            return torch.zeros_like(q)

    class FakePool:
        block_size = 256

        def layer_cache(self, layer):
            assert layer == 0
            return torch.empty(1, 256, 2, 8), torch.empty(1, 256, 2, 8)

    class FakeView:
        pool = FakePool()
        base_len = 0

        def block_table(self, *, device=None):
            return torch.zeros(1, 1, dtype=torch.int32, device=device)

        def cache_seqlens(self, *, device=None):
            return torch.zeros(1, dtype=torch.int32, device=device)

    class FakeCache:
        def __init__(self):
            self.finished = False
            self.cancelled = False

        def request_cache_for_update(self, layer_idx, n_tokens):
            assert layer_idx == 0
            assert n_tokens == 2
            return FakeView()

        def finish_layer_update(self, layer_idx, n_tokens):
            self.finished = (layer_idx, n_tokens) == (0, 2)

        def cancel_layer_update(self, layer_idx):
            self.cancelled = True

    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        attention_bias=False,
    )
    cfg.head_dim = 8
    cfg.layer_types = ["full_attention"]
    cfg.rope_theta_hw = 10000.0
    cfg.max_position_embeddings_hw = 128
    cfg._attn_implementation = "eager"
    attn = sensenova_u1._SenseNovaAttention(cfg, layer_idx=0)
    backend = FakePagedBackend()

    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 2, 2, 8)
    value = torch.randn(1, 2, 2, 8)
    cache = FakeCache()
    from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context

    with use_forward_context(ForwardContext(attention_backend=backend)):
        got = attn._attend_paged_update(
            query,
            key,
            value,
            cache,
            attention_mask=None,
            causal=False,
        )

    assert got is not None
    assert backend.calls == 1
    assert cache.finished
    assert not cache.cancelled


def test_uni_attention_quantized_batched_paged_cache_requires_scale_aware_backend():
    from uniserve_worker.runtime.kv_pool import PagedKVPool
    from uniserve_worker.runtime.paged_text_cache import BatchedPagedRequestCache

    pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=2,
        num_kv_heads=1,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        store_dtype="fp8_e4m3",
    )
    cache = BatchedPagedRequestCache(pool, [[0], [1]], [0, 0])
    attn = RadixAttention(1, 1, 4, layer_id=0)

    q = torch.randn(2, 1, 4)
    k = torch.randn(2, 1, 4)
    v = torch.randn(2, 1, 4)
    with pytest.raises(WorkerError, match="scale-aware paged attention backend"):
        attn(q, k, v, kv_cache=cache, update_cache=True, causal=True)


def test_torch_sdpa_casts_float_masks_to_query_dtype():
    torch.manual_seed(90)
    q = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    mask[:, :, :, -1] = float("-inf")
    out = get_attention_backend("torch_sdpa").forward(q, k, v, causal=False, attn_mask=mask)
    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape


def test_flowmatch_schedule_bagel_descending_shift_contract():
    full = FlowMatchSchedule(
        5,
        shift=1.7,
        direction=ScheduleDirection.DESCENDING,
        shift_domain=ScheduleShiftDomain.TIME,
    ).timesteps(device="cpu")
    base = torch.linspace(1.0, 0.0, 6)
    expected = 1.7 * base / (1 + (1.7 - 1) * base)
    torch.testing.assert_close(full, expected)
    torch.testing.assert_close(full[:-1] - full[1:], torch.diff(full).neg())


def test_flowmatch_schedule_matches_sensenova_shifted_sigma():
    base = torch.linspace(0.0, 1.0, 6)
    sigma = 1 - base
    ref = 1 - (3.0 * sigma / (1 + (3.0 - 1) * sigma))
    got = FlowMatchSchedule(
        5,
        shift=3.0,
        direction=ScheduleDirection.ASCENDING,
        shift_domain=ScheduleShiftDomain.SIGMA,
    ).timesteps()
    torch.testing.assert_close(got, ref)


def test_cfg_params_accepts_rust_wire_shape():
    params = CfgParams.from_mapping(
        {
            "branch_count": 3,
            "text_scale": 4.0,
            "img_scale": 2.0,
            "renorm_type": "global",
            "renorm_min": 0.1,
            "interval": [0.2, 0.8],
        }
    )
    assert params.branch_count == 3
    assert params.scales == (4.0, 2.0)
    assert params.renorm is RenormKind.GLOBAL
    assert params.renorm_min == 0.1
    # cfg-interval gating is owned by the per-step ``cfg_interval`` field, not by
    # CfgParams; from_mapping tolerates the extra wire key but does not mirror it.
    assert not hasattr(params, "interval")


def test_shared_cfg_matches_sensenova_text_and_mixed_formulas():
    out_cond = torch.tensor([[[3.0, 5.0], [7.0, 11.0]]])
    out_img_cond = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    out_uncond = torch.tensor([[[1.0, 1.5], [2.0, 2.5]]])

    text_guided = combine_text_image_cfg(
        out_cond,
        out_img_cond,
        None,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        renorm="none",
        image_scale_applies_to_text=False,
    )
    torch.testing.assert_close(text_guided, out_img_cond + 4.0 * (out_cond - out_img_cond))

    mixed_guided = combine_text_image_cfg(
        out_cond,
        out_img_cond,
        out_uncond,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        renorm="none",
        image_scale_applies_to_text=False,
    )
    expected = out_uncond + 4.0 * (out_cond - out_img_cond) + 2.0 * (out_img_cond - out_uncond)
    torch.testing.assert_close(mixed_guided, expected)


def test_shared_cfg_channel_renorm_is_channel_last_like_sensenova():
    out_cond = torch.ones(1, 2, 3)
    out_img_cond = torch.zeros(1, 2, 3)
    guided = combine_text_image_cfg(
        out_cond,
        out_img_cond,
        None,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        renorm="channel",
        image_scale_applies_to_text=False,
    )
    norm_cond = torch.norm(out_cond, dim=-1, keepdim=True)
    norm_guided = torch.norm(4.0 * out_cond, dim=-1, keepdim=True)
    expected = 4.0 * out_cond * (norm_cond / (norm_guided + 1e-8)).clamp(min=0, max=1.0)
    torch.testing.assert_close(guided, expected)


def test_shared_text_image_cfg_matches_bagel_global_and_channel_formulas():
    out_cond = torch.tensor([[3.0, 5.0], [7.0, 11.0]])
    out_text_uncond = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    out_img_uncond = torch.tensor([[1.0, 1.5], [2.0, 2.5]])
    text_scale = 4.0
    img_scale = 2.0
    text_guided = out_text_uncond + text_scale * (out_cond - out_text_uncond)
    unrenormed = out_img_uncond + img_scale * (text_guided - out_img_uncond)

    guided_global = combine_text_image_cfg(
        out_cond,
        out_text_uncond,
        out_img_uncond,
        cfg_text_scale=text_scale,
        cfg_img_scale=img_scale,
        renorm="global",
        image_scale_applies_to_text=True,
    )
    global_scale = (torch.norm(out_cond) / (torch.norm(unrenormed) + 1e-8)).clamp(0.0, 1.0)
    torch.testing.assert_close(guided_global, unrenormed * global_scale)

    guided_channel = combine_text_image_cfg(
        out_cond,
        out_text_uncond,
        out_img_uncond,
        cfg_text_scale=text_scale,
        cfg_img_scale=img_scale,
        renorm="channel",
        image_scale_applies_to_text=True,
    )
    channel_scale = (
        torch.norm(out_cond, dim=-1, keepdim=True)
        / (torch.norm(unrenormed, dim=-1, keepdim=True) + 1e-8)
    ).clamp(0.0, 1.0)
    torch.testing.assert_close(guided_channel, unrenormed * channel_scale)


def test_shared_text_image_cfg_matches_bagel_text_channel_formula():
    out_cond = torch.tensor([[3.0, 5.0], [7.0, 11.0]])
    out_text_uncond = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    out_img_uncond = torch.tensor([[1.0, 1.5], [2.0, 2.5]])
    text_scale = 4.0
    img_scale = 2.0
    text_guided_raw = out_text_uncond + text_scale * (out_cond - out_text_uncond)
    text_scale_tensor = (
        torch.norm(out_cond, dim=-1, keepdim=True)
        / (torch.norm(text_guided_raw, dim=-1, keepdim=True) + 1e-8)
    ).clamp(0.0, 1.0)
    text_guided = text_guided_raw * text_scale_tensor
    expected = out_img_uncond + img_scale * (text_guided - out_img_uncond)

    guided = combine_text_image_cfg(
        out_cond,
        out_text_uncond,
        out_img_uncond,
        cfg_text_scale=text_scale,
        cfg_img_scale=img_scale,
        renorm="text_channel",
        image_scale_applies_to_text=True,
    )
    torch.testing.assert_close(guided, expected)
