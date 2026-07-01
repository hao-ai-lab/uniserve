"""Conformance for shared MoE, sampler, and logits helpers."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

import uniserve_worker.foundation.runtime_config as runtime_config
import uniserve_worker.nn.sampler as sampler_mod
from uniserve_worker.nn import (
    DeviceMesh,
    FusedMoE,
    LogitsProcessor,
    Sampler,
    use_mesh,
)
from uniserve_worker.nn.mesh import CollectiveTransport
from uniserve_worker.nn.sampler import (
    _apply_min_p_top_k_top_p_in_place,
    apply_sampling_batched,
    sample_one_from_logits,
)

pytestmark = pytest.mark.unit


def _set_worker_runtime(monkeypatch, **kwargs):
    monkeypatch.setattr(
        runtime_config,
        "_CURRENT_CONFIG",
        replace(runtime_config.get_worker_config(), **kwargs),
    )


def _tp_group_mesh(group, *, rank: int = 1, size: int = 2) -> DeviceMesh:
    """A tp=size mesh whose collective transport carries ``group`` (tests
    monkeypatch ``torch.distributed`` directly)."""
    transport = CollectiveTransport(axis="tp", _size=size, _coord=rank, group=group)
    return DeviceMesh.tp(rank, size, transport=transport)


def _reference_moe(x, logits, experts, top_k, *, norm_topk_prob: bool = True):
    probs = torch.softmax(logits, dim=-1)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(top_k):
            expert_idx = int(ids[token, slot])
            out[token] += experts[expert_idx](x[token : token + 1]).squeeze(0) * weights[token, slot]
    return out


def _reference_min_p_top_k_top_p(logits, sp):
    logits = logits.clone()
    min_p = float(sp.get("min_p", 0.0) or 0.0)
    if min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        logits.masked_fill_(probs < min_p * probs.max(), float("-inf"))

    top_k = int(sp.get("top_k", 0) or 0)
    if 0 < top_k < logits.numel():
        values, indices = torch.topk(logits, top_k, sorted=True)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(0, indices, values)
        logits = masked

    top_p = float(sp.get("top_p", 1.0) or 1.0)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        logits[sorted_idx[cutoff]] = float("-inf")
    return logits


def _reference_rms_norm(x, weight, eps):
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(variance + eps)
    return weight * out.to(x.dtype)


def test_fused_moe_matches_per_token_expert_loop():
    torch.manual_seed(6)
    experts = [nn.Linear(4, 4, bias=False) for _ in range(3)]
    moe = FusedMoE(experts, top_k=2)
    x = torch.randn(7, 4)
    logits = torch.randn(7, 3)
    torch.testing.assert_close(moe(x, logits), _reference_moe(x, logits, experts, 2))


def test_fused_moe_can_match_unnormalized_topk_routing():
    torch.manual_seed(7)
    experts = [nn.Linear(4, 4, bias=False) for _ in range(4)]
    moe = FusedMoE(experts, top_k=2, norm_topk_prob=False)
    x = torch.randn(5, 4)
    logits = torch.randn(5, 4)
    torch.testing.assert_close(
        moe(x, logits),
        _reference_moe(x, logits, experts, 2, norm_topk_prob=False),
    )


def test_sampler_wraps_shared_sampling_pipeline():
    logits = torch.tensor([0.1, 2.0, -1.0, 0.5])
    sp = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "logit_bias": [(3, 5.0)]}

    # Independently-derived expectation: the +5.0 bias on index 3 makes it the
    # greedy argmax, and the top-2 log-softmax over the biased logits ranks
    # index 3 then index 1. Asserting against this (rather than re-running
    # sample_one_from_logits, which Sampler.sample directly wraps) keeps the test
    # from being tautological against the implementation it exercises.
    biased = logits.clone()
    biased[3] += 5.0
    ref_logprobs = torch.log_softmax(biased, dim=-1)

    token, logprob, top = Sampler().sample(logits, sp, n_logprobs=2)

    assert token == 3
    assert logprob == pytest.approx(float(ref_logprobs[3]))
    assert [entry[0] for entry in top] == [3, 1]
    assert [entry[1] for entry in top] == pytest.approx(
        [float(ref_logprobs[3]), float(ref_logprobs[1])]
    )


def test_batched_sampler_matches_scalar_greedy_rows():
    logits = torch.tensor(
        [
            [0.1, 2.0, -1.0, 0.5],
            [1.0, 0.4, 3.0, -2.0],
        ]
    )
    params = [
        {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "logit_bias": [(3, 5.0)], "n_logprobs": 2},
        {
            "temperature": 0.0,
            "top_k": 2,
            "top_p": 1.0,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
            "n_logprobs": 1,
        },
    ]
    recent = [[], [2, 2, 0]]
    allowed = [None, [0, 2, 3]]
    suppress = [[1], None]

    got = apply_sampling_batched(logits, params, recent, allowed, suppress)
    expected = [
        sample_one_from_logits(
            logits[0], params[0], recent=recent[0], allowed=allowed[0], suppress=suppress[0], n_logprobs=2
        ),
        sample_one_from_logits(
            logits[1], params[1], recent=recent[1], allowed=allowed[1], suppress=suppress[1], n_logprobs=1
        ),
    ]

    assert got == expected


def test_batched_sampler_omits_unrequested_logprobs_per_row():
    logits = torch.tensor(
        [
            [0.1, 2.0, -1.0],
            [1.0, 0.4, 3.0],
        ]
    )
    params = [
        {"temperature": 0.0},
        {"temperature": 0.0, "n_logprobs": 1},
    ]

    got = apply_sampling_batched(logits, params, [[], []], [None, None], [None, None])

    assert got[0][0] == 1
    assert got[0][1] is None
    assert got[0][2] is None
    assert got[1][0] == 2
    assert got[1][1] is not None
    assert got[1][2] == [[2, got[1][1]]]


def test_tp_sampler_sync_broadcasts_rank0_tokens(monkeypatch):
    calls = []
    group = object()

    monkeypatch.setattr(sampler_mod.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(sampler_mod.torch.distributed, "is_initialized", lambda: True)

    def fake_broadcast(tensor, *, src, group):
        calls.append((src, group))
        tensor.copy_(torch.tensor([7, 8], dtype=tensor.dtype))

    monkeypatch.setattr(sampler_mod.torch.distributed, "broadcast", fake_broadcast)

    with use_mesh(_tp_group_mesh(group)):
        got = sampler_mod.sync_tp_sampled_tokens(torch.tensor([3, 4], dtype=torch.long))

    assert calls == [(0, group)]
    torch.testing.assert_close(got, torch.tensor([7, 8], dtype=torch.long))


def test_tp_sampler_sync_uses_group_zero_global_rank(monkeypatch):
    calls = []
    group = object()

    monkeypatch.setattr(sampler_mod.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(sampler_mod.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        sampler_mod.torch.distributed,
        "get_global_rank",
        lambda seen_group, group_rank: 4 if seen_group is group and group_rank == 0 else -1,
        raising=False,
    )

    def fake_broadcast(tensor, *, src, group):
        calls.append((src, group))
        tensor.copy_(torch.tensor([9, 10], dtype=tensor.dtype))

    monkeypatch.setattr(sampler_mod.torch.distributed, "broadcast", fake_broadcast)

    with use_mesh(_tp_group_mesh(group)):
        got = sampler_mod.sync_tp_sampled_tokens(torch.tensor([3, 4], dtype=torch.long))

    assert calls == [(4, group)]
    torch.testing.assert_close(got, torch.tensor([9, 10], dtype=torch.long))


def test_plain_greedy_batched_sampler_syncs_tp_fast_path(monkeypatch):
    calls = []
    group = object()

    monkeypatch.setattr(sampler_mod.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(sampler_mod.torch.distributed, "is_initialized", lambda: True)

    def fake_broadcast(tensor, *, src, group):
        calls.append((src, group))
        tensor.copy_(torch.tensor([2, 0], dtype=tensor.dtype))

    monkeypatch.setattr(sampler_mod.torch.distributed, "broadcast", fake_broadcast)

    logits = torch.tensor(
        [
            [0.0, 4.0, 1.0],
            [3.0, 0.5, 2.0],
        ]
    )
    with use_mesh(_tp_group_mesh(group)):
        got = sampler_mod.apply_sampling_batched_with_device_tokens(
            logits,
            [{"temperature": 0.0}, {"temperature": 0.0}],
            [[], []],
            [None, None],
            [None, None],
        )

    assert calls == [(0, group)]
    assert [sample[0] for sample in got.samples] == [2, 0]
    torch.testing.assert_close(got.device_tokens, torch.tensor([2, 0], dtype=torch.long))


def test_greedy_sampler_fast_path_applies_argmax_processors(monkeypatch):
    def fail_fallback(*args, **kwargs):
        raise AssertionError("greedy masks and bias should stay on the device fast path")

    monkeypatch.setattr(sampler_mod, "_apply_common_min_p_top_k_top_p_in_place", fail_fallback)
    logits = torch.tensor(
        [
            [0.0, 4.0, 3.0],
            [2.0, 3.0, 0.0],
            [4.0, 3.9, 0.0],
        ]
    )

    got = sampler_mod.apply_sampling_batched_with_device_tokens(
        logits,
        [
            {"temperature": 0.0, "top_p": 0.7},
            {"temperature": 0.0, "logit_bias": [(0, 4.0)]},
            {"temperature": 0.0, "frequency_penalty": 1.0},
        ],
        [[], [], [0]],
        [None, [0, 1], None],
        [[1], None, None],
    )

    assert [sample[0] for sample in got.samples] == [2, 0, 1]
    torch.testing.assert_close(got.device_tokens, torch.tensor([2, 0, 1], dtype=torch.long))


def test_sampler_top_p_filters_only_finite_candidates_after_top_k():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, -1.0])
    sp = {"top_k": 2, "top_p": 0.7, "min_p": 0.0}

    got = logits.clone()
    _apply_min_p_top_k_top_p_in_place(got, sp, got.numel())

    expected = _reference_min_p_top_k_top_p(logits, sp)
    torch.testing.assert_close(got, expected)


def test_logits_processor_prunes_positions_before_lm_head():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    lm_head = nn.Linear(4, 5, bias=False)
    with torch.no_grad():
        lm_head.weight.fill_(1.0)
    positions = torch.tensor([1, 4])
    got = LogitsProcessor()(hidden, lm_head, positions=positions)
    expected = lm_head(hidden.reshape(-1, 4).index_select(0, positions))
    torch.testing.assert_close(got, expected)


def test_logits_processor_can_chunk_large_lm_head_projection(monkeypatch):
    class RecordingHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 3, bias=False)
            self.sizes: list[int] = []

        def forward(self, hidden):
            self.sizes.append(int(hidden.shape[0]))
            return self.proj(hidden)

    _set_worker_runtime(monkeypatch, logits_processor_chunk_size=2)
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    lm_head = RecordingHead()

    got = LogitsProcessor()(hidden, lm_head)
    expected = lm_head.proj(hidden)

    torch.testing.assert_close(got, expected)
    assert lm_head.sizes == [2, 2, 1]


def test_ops_qk_norm_rope_multi_axis_matches_reference():
    import uniserve_worker.ops as ops
    from uniserve_worker.nn.rope import apply_rotary_emb

    torch.manual_seed(123)
    tokens, q_heads, k_heads = 5, 3, 2
    axis_dims = (4, 2, 2)
    q = torch.randn(tokens, q_heads, sum(axis_dims), dtype=torch.float32)
    k = torch.randn(tokens, k_heads, sum(axis_dims), dtype=torch.float32)
    q_weights = tuple(torch.randn(dim) for dim in axis_dims)
    k_weights = tuple(torch.randn(dim) for dim in axis_dims)
    cos = tuple(torch.randn(tokens, dim // 2).cos().contiguous() for dim in axis_dims)
    sin = tuple(torch.randn(tokens, dim // 2).sin().contiguous() for dim in axis_dims)

    got_q, got_k = ops.qk_norm_rope(
        q,
        k,
        q_weights,
        k_weights,
        cos,
        sin,
        1e-6,
        axis_dims=axis_dims,
        override="eager",
    )

    q_parts = q.split(axis_dims, dim=-1)
    k_parts = k.split(axis_dims, dim=-1)
    ref_q = torch.cat(
        [apply_rotary_emb(_reference_rms_norm(part, weight, 1e-6), c, s) for part, weight, c, s in zip(q_parts, q_weights, cos, sin, strict=True)],
        dim=-1,
    )
    ref_k = torch.cat(
        [apply_rotary_emb(_reference_rms_norm(part, weight, 1e-6), c, s) for part, weight, c, s in zip(k_parts, k_weights, cos, sin, strict=True)],
        dim=-1,
    )
    torch.testing.assert_close(got_q, ref_q)
    torch.testing.assert_close(got_k, ref_k)


def test_ops_qk_norm_rope_batched_multi_axis_preserves_batch_tables():
    import uniserve_worker.ops as ops
    from uniserve_worker.nn.rope import apply_rotary_emb

    torch.manual_seed(125)
    batch, seq_len, q_heads, k_heads = 2, 3, 3, 2
    axis_dims = (4, 2, 2)
    q = torch.randn(batch, q_heads, seq_len, sum(axis_dims), dtype=torch.float32)
    k = torch.randn(batch, k_heads, seq_len, sum(axis_dims), dtype=torch.float32)
    q_t = torch.randn(axis_dims[0])
    k_t = torch.randn(axis_dims[0])
    q_hw = torch.randn(axis_dims[1] + axis_dims[2])
    k_hw = torch.randn(axis_dims[1] + axis_dims[2])
    q_weights = (q_t, q_hw, q_hw)
    k_weights = (k_t, k_hw, k_hw)
    # Flattened [batch * seq] tables must map back to their originating batch.
    table_positions = torch.arange(batch * seq_len, dtype=torch.float32)
    cos = (
        torch.stack([torch.cos(table_positions), torch.cos(table_positions + 0.25)], dim=-1),
        torch.cos(table_positions[:, None] + 0.5),
        torch.cos(table_positions[:, None] + 1.0),
    )
    sin = (
        torch.stack([torch.sin(table_positions), torch.sin(table_positions + 0.25)], dim=-1),
        torch.sin(table_positions[:, None] + 0.5),
        torch.sin(table_positions[:, None] + 1.0),
    )

    got_q, got_k = ops.qk_norm_rope(
        q,
        k,
        q_weights,
        k_weights,
        cos,
        sin,
        1e-6,
        axis_dims=axis_dims,
        override="eager",
    )

    q_t_part, q_h_part, q_w_part = q.split(axis_dims, dim=-1)
    k_t_part, k_h_part, k_w_part = k.split(axis_dims, dim=-1)
    q_t_ref = _reference_rms_norm(q_t_part, q_t, 1e-6)
    k_t_ref = _reference_rms_norm(k_t_part, k_t, 1e-6)
    q_hw_ref = _reference_rms_norm(torch.cat([q_h_part, q_w_part], dim=-1), q_hw, 1e-6)
    k_hw_ref = _reference_rms_norm(torch.cat([k_h_part, k_w_part], dim=-1), k_hw, 1e-6)
    q_h_ref, q_w_ref = q_hw_ref.split(axis_dims[1:], dim=-1)
    k_h_ref, k_w_ref = k_hw_ref.split(axis_dims[1:], dim=-1)

    def rotate(part, table_cos, table_sin):
        flat = part.permute(0, 2, 1, 3).reshape(batch * seq_len, part.shape[1], part.shape[-1])
        rotated = apply_rotary_emb(flat, table_cos, table_sin)
        return rotated.reshape(batch, seq_len, part.shape[1], part.shape[-1]).permute(0, 2, 1, 3)

    ref_q = torch.cat(
        [rotate(part, c, s) for part, c, s in zip((q_t_ref, q_h_ref, q_w_ref), cos, sin, strict=True)],
        dim=-1,
    )
    ref_k = torch.cat(
        [rotate(part, c, s) for part, c, s in zip((k_t_ref, k_h_ref, k_w_ref), cos, sin, strict=True)],
        dim=-1,
    )
    torch.testing.assert_close(got_q, ref_q)
    torch.testing.assert_close(got_k, ref_k)


def test_triton_qk_norm_rope_provider_dispatches_multi_axis(monkeypatch):
    from uniserve_worker.ops.providers import _TritonQKNormRopeProvider
    from uniserve_worker.ops.requests import QKNormRopeReq

    q = torch.empty(1, 1, 1, 8)
    k = torch.empty(1, 1, 1, 8)
    axis_dims = (4, 2, 2)
    q_weights = (torch.empty(4), torch.empty(4), torch.empty(4))
    k_weights = (torch.empty(4), torch.empty(4), torch.empty(4))
    cos = (torch.empty(1, 2), torch.empty(1, 1), torch.empty(1, 1))
    sin = (torch.empty(1, 2), torch.empty(1, 1), torch.empty(1, 1))
    req = QKNormRopeReq(q, k, q_weights, k_weights, cos, sin, 1e-6, axis_dims=axis_dims)
    marker = (object(), object())
    provider = _TritonQKNormRopeProvider()

    monkeypatch.setattr(provider, "_can_run_multi_axis", lambda seen: seen is req)
    monkeypatch.setattr(provider, "_run_multi_axis", lambda seen: marker if seen is req else None)

    assert provider.can_run(req)
    assert provider.run(req) is marker


def test_triton_qk_norm_rope_override_falls_back_when_multi_axis_ineligible():
    import uniserve_worker.ops as ops
    from uniserve_worker.ops.providers import _TritonQKNormRopeProvider
    from uniserve_worker.ops.requests import QKNormRopeReq

    torch.manual_seed(126)
    batch, seq_len, q_heads, k_heads = 1, 3, 4, 2
    axis_dims = (4, 2, 2)
    q = torch.randn(batch, q_heads, seq_len, sum(axis_dims), dtype=torch.float32)
    k = torch.randn(batch, k_heads, seq_len, sum(axis_dims), dtype=torch.float32)
    q_t = torch.randn(axis_dims[0])
    k_t = torch.randn(axis_dims[0])
    q_hw = torch.randn(axis_dims[1] + axis_dims[2])
    k_hw = torch.randn(axis_dims[1] + axis_dims[2])
    q_weights = (q_t, q_hw, q_hw)
    k_weights = (k_t, k_hw, k_hw)
    table_positions = torch.arange(batch * seq_len, dtype=torch.float32)
    cos = (
        torch.stack([torch.cos(table_positions), torch.cos(table_positions + 0.25)], dim=-1),
        torch.cos(table_positions[:, None] + 0.5),
        torch.cos(table_positions[:, None] + 1.0),
    )
    sin = (
        torch.stack([torch.sin(table_positions), torch.sin(table_positions + 0.25)], dim=-1),
        torch.sin(table_positions[:, None] + 0.5),
        torch.sin(table_positions[:, None] + 1.0),
    )
    req = QKNormRopeReq(q, k, q_weights, k_weights, cos, sin, 1e-6, axis_dims=axis_dims)

    provider = _TritonQKNormRopeProvider()
    assert not provider.can_run(req)
    with pytest.raises(RuntimeError, match="ineligible"):
        provider.run(req)

    got_q, got_k = ops.qk_norm_rope(
        q,
        k,
        q_weights,
        k_weights,
        cos,
        sin,
        1e-6,
        axis_dims=axis_dims,
        override="triton",
    )
    ref_q, ref_k = ops.qk_norm_rope(
        q,
        k,
        q_weights,
        k_weights,
        cos,
        sin,
        1e-6,
        axis_dims=axis_dims,
        override="eager",
    )
    torch.testing.assert_close(got_q, ref_q)
    torch.testing.assert_close(got_k, ref_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_qk_norm_rope_grouped_multi_axis_matches_eager_cuda():
    import uniserve_worker.ops as ops
    from uniserve_worker.foundation.triton_compat import triton_device_supported

    if not triton_device_supported(torch.device("cuda")):
        pytest.skip("Triton fused layers are not supported on this CUDA device")

    torch.manual_seed(127)
    batch, seq_len, q_heads, k_heads = 1, 1, 16, 4
    axis_dims = (64, 32, 32)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    q = torch.randn(batch, q_heads, seq_len, sum(axis_dims), dtype=dtype, device=device)
    k = torch.randn(batch, k_heads, seq_len, sum(axis_dims), dtype=dtype, device=device)
    q_t = torch.randn(axis_dims[0], dtype=dtype, device=device)
    k_t = torch.randn(axis_dims[0], dtype=dtype, device=device)
    q_hw = torch.randn(axis_dims[1] + axis_dims[2], dtype=dtype, device=device)
    k_hw = torch.randn(axis_dims[1] + axis_dims[2], dtype=dtype, device=device)
    table_positions = torch.arange(batch * seq_len, dtype=torch.float32, device=device)
    cos = (
        torch.cos(table_positions[:, None] + torch.arange(axis_dims[0] // 2, device=device)).to(dtype),
        torch.cos(table_positions[:, None] + torch.arange(axis_dims[1] // 2, device=device)).to(dtype),
        torch.cos(table_positions[:, None] + torch.arange(axis_dims[2] // 2, device=device)).to(dtype),
    )
    sin = (
        torch.sin(table_positions[:, None] + torch.arange(axis_dims[0] // 2, device=device)).to(dtype),
        torch.sin(table_positions[:, None] + torch.arange(axis_dims[1] // 2, device=device)).to(dtype),
        torch.sin(table_positions[:, None] + torch.arange(axis_dims[2] // 2, device=device)).to(dtype),
    )

    with torch.inference_mode():
        got_q, got_k = ops.qk_norm_rope(
            q,
            k,
            (q_t, q_hw, q_hw),
            (k_t, k_hw, k_hw),
            cos,
            sin,
            1e-6,
            axis_dims=axis_dims,
            override="triton",
        )
        ref_q, ref_k = ops.qk_norm_rope(
            q,
            k,
            (q_t, q_hw, q_hw),
            (k_t, k_hw, k_hw),
            cos,
            sin,
            1e-6,
            axis_dims=axis_dims,
            override="eager",
        )
    torch.testing.assert_close(got_q, ref_q, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(got_k, ref_k, atol=8e-2, rtol=8e-2)


def test_ops_qk_norm_grouped_multi_axis_matches_reference():
    import uniserve_worker.ops as ops

    torch.manual_seed(124)
    tokens, q_heads, k_heads = 7, 4, 2
    axis_dims = (4, 2, 2)
    q = torch.randn(tokens, q_heads, sum(axis_dims), dtype=torch.float32)
    k = torch.randn(tokens, k_heads, sum(axis_dims), dtype=torch.float32)
    q_t = torch.randn(axis_dims[0])
    k_t = torch.randn(axis_dims[0])
    q_hw = torch.randn(axis_dims[1] + axis_dims[2])
    k_hw = torch.randn(axis_dims[1] + axis_dims[2])
    got_q, got_k = ops.qk_norm(
        q,
        k,
        (q_t, q_hw, q_hw),
        (k_t, k_hw, k_hw),
        1e-6,
        axis_dims=axis_dims,
        override="eager",
    )

    q_t_part, q_h_part, q_w_part = q.split(axis_dims, dim=-1)
    k_t_part, k_h_part, k_w_part = k.split(axis_dims, dim=-1)
    ref_q = torch.cat(
        [
            _reference_rms_norm(q_t_part, q_t, 1e-6),
            *_reference_rms_norm(torch.cat([q_h_part, q_w_part], dim=-1), q_hw, 1e-6).split(axis_dims[1:], dim=-1),
        ],
        dim=-1,
    )
    ref_k = torch.cat(
        [
            _reference_rms_norm(k_t_part, k_t, 1e-6),
            *_reference_rms_norm(torch.cat([k_h_part, k_w_part], dim=-1), k_hw, 1e-6).split(axis_dims[1:], dim=-1),
        ],
        dim=-1,
    )
    torch.testing.assert_close(got_q, ref_q)
    torch.testing.assert_close(got_k, ref_k)
