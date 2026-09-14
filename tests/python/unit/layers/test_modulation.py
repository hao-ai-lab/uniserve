from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from uniserve import ops
from uniserve.ops import (
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
    normalized = value.double() * torch.rsqrt(value.double().pow(2).mean(-1, keepdim=True) + eps)
    return normalized * weight.double()


def _assert_e4m3_error(values, scale, reference):
    # E4M3 nearest rounding has unit roundoff 2^-4 and half a subnormal
    # spacing of 2^-10. The absolute term retains the BF16 composition bound.
    actual = values.float() * scale
    error = (actual.double() - reference.double()).abs()
    bound = reference.double().abs() / 16 + scale.double() / 1024 + 2e-2
    assert torch.isfinite(actual).all()
    assert (scale > 0).all()
    assert (error <= bound).all()


@pytest.mark.parametrize(
    "width,rows,weight_dtype",
    [
        (128, 8, torch.bfloat16),
        (5376, 8, torch.bfloat16),
        (5376, 9344, torch.bfloat16),
        (5376, 21888, torch.float32),
        (16384, 16, torch.float32),
        (132, 3, torch.float32),
    ],
)
def test_block_edges_match_strided_bf16_reference(width, rows, weight_dtype) -> None:
    torch.manual_seed(23)
    states = 6
    eps = 1e-5
    hidden = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    attention = torch.randn_like(hidden)
    feed_forward = torch.randn_like(hidden)
    weight = torch.randn(width, device="cuda", dtype=weight_dtype)
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
    expected_attn_norm = (
        expected_attn_norm * (1.0 + scale_attn.index_select(0, row_indices).double())
        + shift_attn.index_select(0, row_indices).double()
    )
    actual_attn_norm = modulated_rms_norm(
        hidden,
        weight,
        shift_attn,
        scale_attn,
        row_indices,
        eps=eps,
    )

    expected_residual = (
        hidden.double() + gate_attn.index_select(0, row_indices).double() * attention.double()
    )
    expected_ffn_norm = _rmsnorm(expected_residual, weight, eps)
    expected_ffn_norm = (
        expected_ffn_norm * (1.0 + scale_ffn.index_select(0, row_indices).double())
        + shift_ffn.index_select(0, row_indices).double()
    )
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
    expected_output = (
        expected_residual + gate_ffn.index_select(0, row_indices).double() * feed_forward.double()
    )
    actual_output = gated_residual(
        actual_residual,
        feed_forward.clone(),
        gate_ffn,
        row_indices,
    )

    torch.testing.assert_close(
        actual_attn_norm.double(), expected_attn_norm.double(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_residual.double(), expected_residual.double(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_ffn_norm.double(), expected_ffn_norm.double(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_output.double(), expected_output.double(), rtol=2e-2, atol=2e-2
    )


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
    expected = value.double() * F.silu(gate.double())

    torch.testing.assert_close(
        value_first_swiglu(value_gate).double(),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


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

    expected_residual = (
        hidden.double() + gate_attn.index_select(0, row_indices).double() * attention.double()
    )
    expected_normalized = _rmsnorm(expected_residual, weight, eps)
    expected_normalized = (
        expected_normalized * (1.0 + scale_ffn.index_select(0, row_indices).double())
        + shift_ffn.index_select(0, row_indices).double()
    )
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
    expected_swiglu = value.double() * F.silu(gate.double())
    actual_swiglu_fp8, actual_swiglu_scale = value_first_swiglu_fp8(value_gate)

    torch.testing.assert_close(
        actual_residual.double(), expected_residual.double(), rtol=2e-2, atol=2e-2
    )
    _assert_e4m3_error(actual_fp8, actual_scale, expected_normalized)
    _assert_e4m3_error(actual_swiglu_fp8, actual_swiglu_scale, expected_swiglu)


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
    actual_activated_fp8, actual_activated_scale = ops.silu_and_mul_fp8(gate_up)

    _assert_e4m3_error(actual_activated_fp8, actual_activated_scale, expected_activated)
