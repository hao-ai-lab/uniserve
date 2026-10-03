from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from uniserve.nn.functional import (
    Rounding,
    gated_residual,
    gated_residual_rms_norm,
    gated_residual_rms_norm_fp8,
    modulated_rms_norm,
    silu_and_mul,
    value_first_swiglu,
    value_first_swiglu_fp8,
)
from uniserve.quantization import Quantizer

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required"
    ),
]


def _rmsnorm(
    value: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = value.double() * torch.rsqrt(
        value.double().pow(2).mean(-1, keepdim=True) + eps
    )
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
def test_block_edges_match_strided_bf16_reference(
    width, rows, weight_dtype
) -> None:
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
    shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
        parameters.chunk(
            6,
            dim=-1,
        )
    )
    row_indices = torch.randint(
        states, (rows,), device="cuda", dtype=torch.long
    )

    expected_attn_norm = _rmsnorm(hidden, weight, eps)
    expected_attn_norm = (
        expected_attn_norm
        * (1.0 + scale_attn.index_select(0, row_indices).double())
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
        hidden.double()
        + gate_attn.index_select(0, row_indices).double() * attention.double()
    )
    expected_ffn_norm = _rmsnorm(expected_residual, weight, eps)
    expected_ffn_norm = (
        expected_ffn_norm
        * (1.0 + scale_ffn.index_select(0, row_indices).double())
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
        expected_residual
        + gate_ffn.index_select(0, row_indices).double() * feed_forward.double()
    )
    actual_output = gated_residual(
        actual_residual,
        feed_forward.clone(),
        gate_ffn,
        row_indices,
    )

    torch.testing.assert_close(
        actual_attn_norm.double(),
        expected_attn_norm.double(),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        actual_residual.double(),
        expected_residual.double(),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        actual_ffn_norm.double(),
        expected_ffn_norm.double(),
        rtol=2e-2,
        atol=2e-2,
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
    row_indices = torch.randint(
        states, (rows,), device="cuda", dtype=torch.long
    )

    expected_residual = (
        hidden.double()
        + gate_attn.index_select(0, row_indices).double() * attention.double()
    )
    expected_normalized = _rmsnorm(expected_residual, weight, eps)
    expected_normalized = (
        expected_normalized
        * (1.0 + scale_ffn.index_select(0, row_indices).double())
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
        actual_residual.double(),
        expected_residual.double(),
        rtol=2e-2,
        atol=2e-2,
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
    expected_activated = silu_and_mul(gate_up)
    encoded = Quantizer("fp8", axis=0).quantize(
        torch.zeros_like(expected_activated)
    )
    silu_and_mul(gate_up, out=encoded)
    buffers = encoded.buffers()
    actual_activated_fp8, actual_activated_scale = (
        buffers["values"],
        buffers["scale"],
    )

    _assert_e4m3_error(
        actual_activated_fp8, actual_activated_scale, expected_activated
    )


_FP8_SWIGLU_WIDTH = 512


@pytest.mark.parametrize(
    "rows",
    (
        8,
        # Eight rows past 2**31 packed input elements: the last rows start
        # beyond the signed 32-bit offset range.
        pytest.param(
            2**31 // (2 * _FP8_SWIGLU_WIDTH) + 8, marks=pytest.mark.slow
        ),
    ),
)
@torch.inference_mode()
def test_text_fp8_swiglu_kernel_matches_reference(rows) -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 execution requires compute capability 9 or newer")
    torch.manual_seed(53)
    width = _FP8_SWIGLU_WIDTH
    gate_up = torch.empty(
        rows, 2 * width, device="cuda", dtype=torch.bfloat16
    ).normal_()
    quantizer = Quantizer("fp8", axis=0)
    encoded = quantizer.from_tensors(
        {
            "values": torch.empty(
                rows, width, device="cuda", dtype=torch.float8_e4m3fn
            ),
            "scale": torch.empty(rows, 1, device="cuda", dtype=torch.float32),
        },
        shape=(rows, width),
        dtype=torch.bfloat16,
    )

    # Inference mode admits the fused kernel; the first and last rows bound
    # every row offset the launch computes.
    silu_and_mul(gate_up, out=encoded)
    buffers = encoded.buffers()
    for checked in (slice(0, 8), slice(rows - 8, rows)):
        gate, value = gate_up[checked].double().chunk(2, dim=-1)
        _assert_e4m3_error(
            buffers["values"][checked],
            buffers["scale"][checked],
            F.silu(gate) * value,
        )


@pytest.mark.gpu
def test_modulated_rms_norm_retains_its_source_rows():
    """The retained rows are the normalized rows' source, byte for byte."""
    torch.manual_seed(7)
    hidden = torch.randn(96, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.rand(256, device="cuda", dtype=torch.bfloat16) + 0.5
    shift = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    row_indices = torch.randint(0, 3, (96,), device="cuda")
    retained = torch.zeros_like(hidden)

    expected = modulated_rms_norm(
        hidden, weight, shift, scale, row_indices, eps=1e-6
    )
    actual = modulated_rms_norm(
        hidden, weight, shift, scale, row_indices, eps=1e-6, retain=retained
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(retained, hidden, rtol=0, atol=0)
    with pytest.raises(ValueError, match="retained rows"):
        modulated_rms_norm(
            hidden,
            weight,
            shift,
            scale,
            row_indices,
            eps=1e-6,
            retain=retained[:, :128],
        )


def _unit_rows(rows: int, width: int, device: str) -> torch.Tensor:
    """BF16 rows of +-1, whose RMS normalization is exact in any order.

    A row's mean square is exactly 1, so its weighted normalization
    ``w * rsqrt(1 + eps)`` rounds to ``w`` in BF16 for every reduction order
    and reciprocal square root, and stepwise results are exact functions of
    the operands.
    """
    signs = torch.randint(0, 2, (rows, width), device=device) * 2 - 1
    return signs.to(torch.bfloat16)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_stepwise_modulation_rounds_each_eager_operation(device) -> None:
    """Stepwise modulation equals eager BF16 PyTorch bit for bit."""
    torch.manual_seed(11)
    rows, width, eps = 64, 5376, 1e-6
    hidden = _unit_rows(rows, width, device)
    # The gated update stays below half a BF16 spacing at 1, so the rounded
    # residual keeps every row's exact unit mean square.
    update = torch.rand(rows, width, device=device).to(torch.bfloat16)
    gate = torch.full((3, width), 2.0**-12, device=device).to(torch.bfloat16)
    weight = (torch.rand(width, device=device) + 0.5).to(torch.bfloat16)
    shift = torch.randn(3, width, device=device).to(torch.bfloat16)
    scale = (torch.randn(3, width, device=device) * 0.05).to(torch.bfloat16)
    row_indices = torch.randint(0, 3, (rows,), device=device)

    def eager(value):
        normalized = F.rms_norm(value, (width,), weight, eps)
        return normalized * (
            1.0 + scale.index_select(0, row_indices)
        ) + shift.index_select(0, row_indices)

    residual = hidden + gate.index_select(0, row_indices) * update
    modulated = modulated_rms_norm(
        hidden,
        weight,
        shift,
        scale,
        row_indices,
        eps=eps,
        rounding=Rounding.STEPWISE,
    )
    summed, normalized = gated_residual_rms_norm(
        hidden,
        update.clone(),
        gate,
        weight,
        shift,
        scale,
        row_indices,
        eps=eps,
        rounding=Rounding.STEPWISE,
    )

    torch.testing.assert_close(modulated, eager(hidden), rtol=0, atol=0)
    torch.testing.assert_close(summed, residual, rtol=0, atol=0)
    torch.testing.assert_close(normalized, eager(residual), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_stepwise_gated_residual_rounds_the_gated_update(device) -> None:
    """Stepwise gated residuals equal eager BF16 PyTorch bit for bit."""
    torch.manual_seed(12)
    hidden = torch.randn(96, 384, device=device).to(torch.bfloat16)
    update = torch.randn(96, 384, device=device).to(torch.bfloat16)
    gate = torch.randn(3, 384, device=device).to(torch.bfloat16)
    row_indices = torch.randint(0, 3, (96,), device=device)

    expected = hidden + gate.index_select(0, row_indices) * update
    actual = gated_residual(
        hidden,
        update.clone(),
        gate,
        row_indices,
        rounding=Rounding.STEPWISE,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_stepwise_swiglu_rounds_the_activated_gate(device) -> None:
    """Stepwise SwiGLU multiplies the BF16-rounded SiLU by the value.

    The activated gate is the call's own SiLU rounded once, read back by
    gating with unit values, so the expectation does not depend on how
    either implementation evaluates the exponential.
    """
    torch.manual_seed(13)
    gate = torch.randn(32, 2048, device=device).to(torch.bfloat16)
    value = torch.randn(32, 2048, device=device).to(torch.bfloat16)

    activated = silu_and_mul(torch.cat((gate, torch.ones_like(value)), -1))
    actual = silu_and_mul(
        torch.cat((gate, value), -1), rounding=Rounding.STEPWISE
    )

    torch.testing.assert_close(actual, activated * value, rtol=0, atol=0)
