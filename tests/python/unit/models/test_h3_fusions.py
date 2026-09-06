from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from uniserve_worker import ops
from uniserve_worker.models.minimax_h3.fusions import (
    attention_residual_modulated_rmsnorm,
    attention_residual_modulated_rmsnorm_fp8,
    gated_residual,
    row_modulated_rmsnorm,
    value_first_swiglu,
    value_first_swiglu_fp8,
)
from uniserve_worker.nn.quant.kv_cache import fp8_quantize, fp8_scale_from

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

_HIDDEN_SIZE = 5376
_FFN_SIZE = 14336


def _rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(value.dtype)


def test_h3_block_edges_match_strided_bf16_reference() -> None:
    torch.manual_seed(23)
    rows = 8
    states = 6
    eps = 1e-5
    hidden = torch.randn(rows, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    attention = torch.randn_like(hidden)
    feed_forward = torch.randn_like(hidden)
    weight = torch.randn(_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    parameters = torch.randn(
        states,
        6 * _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = parameters.chunk(
        6,
        dim=-1,
    )
    row_indices = torch.randint(states, (rows,), device="cuda", dtype=torch.long)

    expected_attn_norm = _rmsnorm(hidden, weight, eps)
    expected_attn_norm = expected_attn_norm * (
        1.0 + scale_attn.index_select(0, row_indices)
    ) + shift_attn.index_select(0, row_indices)
    actual_attn_norm = row_modulated_rmsnorm(
        hidden,
        weight,
        shift_attn,
        scale_attn,
        row_indices,
        eps=eps,
    )

    expected_residual = hidden + gate_attn.index_select(0, row_indices) * attention
    expected_ffn_norm = _rmsnorm(expected_residual, weight, eps)
    expected_ffn_norm = expected_ffn_norm * (
        1.0 + scale_ffn.index_select(0, row_indices)
    ) + shift_ffn.index_select(0, row_indices)
    actual_residual, actual_ffn_norm = attention_residual_modulated_rmsnorm(
        hidden,
        attention.clone(),
        gate_attn,
        weight,
        shift_ffn,
        scale_ffn,
        row_indices,
        eps=eps,
    )
    expected_output = expected_residual + gate_ffn.index_select(0, row_indices) * feed_forward
    actual_output = gated_residual(
        actual_residual,
        feed_forward.clone(),
        gate_ffn,
        row_indices,
    )

    assert shift_attn.stride(0) == 6 * _HIDDEN_SIZE
    assert torch.equal(actual_attn_norm, expected_attn_norm)
    assert torch.equal(actual_residual, expected_residual)
    assert torch.equal(actual_ffn_norm, expected_ffn_norm)
    assert torch.equal(actual_output, expected_output)


def test_h3_swiglu_matches_bf16_reference() -> None:
    torch.manual_seed(29)
    value_gate = torch.randn(
        8,
        2 * _FFN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value, gate = value_gate.chunk(2, dim=-1)
    expected = value * F.silu(gate.float()).to(gate.dtype)

    assert torch.equal(value_first_swiglu(value_gate), expected)


def test_h3_fp8_boundaries_match_bf16_reference() -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 execution requires compute capability 9 or newer")
    torch.manual_seed(31)
    rows = 8
    states = 6
    eps = 1e-5
    hidden = torch.randn(rows, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    attention = torch.randn_like(hidden)
    weight = torch.randn(_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    parameters = torch.randn(
        states,
        6 * _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    _, _, gate_attn, shift_ffn, scale_ffn, _ = parameters.chunk(6, dim=-1)
    row_indices = torch.randint(states, (rows,), device="cuda", dtype=torch.long)

    expected_residual = hidden + gate_attn.index_select(0, row_indices) * attention
    expected_normalized = _rmsnorm(expected_residual, weight, eps)
    expected_normalized = expected_normalized * (
        1.0 + scale_ffn.index_select(0, row_indices)
    ) + shift_ffn.index_select(0, row_indices)
    expected_scale = fp8_scale_from(expected_normalized.float(), dim=1)
    expected_fp8 = fp8_quantize(expected_normalized.float(), expected_scale)
    actual_residual, actual_fp8, actual_scale = attention_residual_modulated_rmsnorm_fp8(
        hidden,
        attention.clone(),
        gate_attn,
        weight,
        shift_ffn,
        scale_ffn,
        row_indices,
        eps=eps,
    )

    value_gate = torch.randn(
        rows,
        2 * _FFN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value, gate = value_gate.chunk(2, dim=-1)
    expected_swiglu = value * F.silu(gate.float()).to(gate.dtype)
    expected_swiglu_scale = fp8_scale_from(expected_swiglu.float(), dim=1)
    expected_swiglu_fp8 = fp8_quantize(expected_swiglu.float(), expected_swiglu_scale)
    actual_swiglu_fp8, actual_swiglu_scale = value_first_swiglu_fp8(value_gate)

    assert torch.equal(actual_residual, expected_residual)
    assert torch.equal(actual_scale, expected_scale)
    assert torch.equal(actual_fp8, expected_fp8)
    assert torch.equal(actual_swiglu_scale, expected_swiglu_scale)
    assert torch.equal(actual_swiglu_fp8, expected_swiglu_fp8)


def test_h3_text_fp8_swiglu_matches_unfused_boundary() -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 execution requires compute capability 9 or newer")
    torch.manual_seed(37)
    rows = 8
    intermediate_size = 25_600
    gate_up = torch.randn(
        rows,
        2 * intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected_activated = ops.silu_and_mul(gate_up)
    expected_activated_scale = fp8_scale_from(expected_activated.float(), dim=1)
    expected_activated_fp8 = fp8_quantize(
        expected_activated.float(),
        expected_activated_scale,
    )
    actual_activated_fp8, actual_activated_scale = ops.silu_and_mul_fp8(gate_up)

    assert torch.equal(actual_activated_scale, expected_activated_scale)
    assert torch.equal(actual_activated_fp8, expected_activated_fp8)
