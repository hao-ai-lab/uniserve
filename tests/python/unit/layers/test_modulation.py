from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from uniserve_worker import ops
from uniserve_worker.nn.quant.kv_cache import fp8_quantize, fp8_scale_from
from uniserve_worker.ops import (
    gated_residual,
    gated_residual_rms_norm,
    gated_residual_rms_norm_fp8,
    modulated_rms_norm,
    value_first_swiglu,
    value_first_swiglu_fp8,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(value.dtype)


@pytest.mark.parametrize("width", [128, 5376])
def test_block_edges_match_strided_bf16_reference(width) -> None:
    torch.manual_seed(23)
    rows = 8
    states = 6
    eps = 1e-5
    hidden = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    attention = torch.randn_like(hidden)
    feed_forward = torch.randn_like(hidden)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    parameters = torch.randn(
        states,
        6 * width,
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
    actual_attn_norm = modulated_rms_norm(
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
    actual_residual, actual_ffn_norm = gated_residual_rms_norm(
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

    assert shift_attn.stride(0) == 6 * width
    assert torch.equal(actual_attn_norm, expected_attn_norm)
    assert torch.equal(actual_residual, expected_residual)
    assert torch.equal(actual_ffn_norm, expected_ffn_norm)
    assert torch.equal(actual_output, expected_output)


@pytest.mark.parametrize("expanded", [128, 14336])
def test_swiglu_matches_bf16_reference(expanded) -> None:
    torch.manual_seed(29)
    value_gate = torch.randn(
        8,
        2 * expanded,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value, gate = value_gate.chunk(2, dim=-1)
    expected = value * F.silu(gate.float()).to(gate.dtype)

    assert torch.equal(value_first_swiglu(value_gate, activation_dtype=torch.bfloat16), expected)


@pytest.mark.parametrize(("width", "expanded"), [(128, 256), (5376, 14336)])
def test_fp8_boundaries_match_bf16_reference(width, expanded) -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 execution requires compute capability 9 or newer")
    torch.manual_seed(31)
    rows = 8
    states = 6
    eps = 1e-5
    hidden = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    attention = torch.randn_like(hidden)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    parameters = torch.randn(
        states,
        6 * width,
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
    actual_residual, actual_fp8, actual_scale = gated_residual_rms_norm_fp8(
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
        2 * expanded,
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


def test_text_fp8_swiglu_matches_unfused_boundary() -> None:
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
