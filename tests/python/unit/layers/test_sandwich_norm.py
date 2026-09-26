"""Sandwich RMS normalization of a residual stream.

``sandwich_rms_norm`` adds post-normalized sublayer outputs to a residual
stream, optionally scales it and returns further normalizations of the
stream, rounding every stored intermediate to the row dtype. The tests
compare it with a float64 evaluation of the same expression that rounds at
those points.
"""

from __future__ import annotations

import pytest
import torch

from uniserve.nn.functional import rms_norm, sandwich_rms_norm

pytestmark = pytest.mark.unit

EPS = 1e-6


@pytest.fixture(params=("cpu", pytest.param("cuda", marks=pytest.mark.gpu)))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return request.param


def _rms(values, weight=None):
    values = values.double()
    normalized = values * torch.rsqrt(
        values.square().mean(-1, keepdim=True) + EPS
    )
    return normalized if weight is None else normalized * weight.double()


def _round(values, dtype):
    return values.to(dtype).double()


@pytest.mark.parametrize("width", (64, 2816))
def test_attention_sandwich_adds_the_normalized_update_and_normalizes(
    device, width
):
    """One unnormalized update; three norms, one unweighted with factors."""
    torch.manual_seed(3)
    dtype = torch.bfloat16
    residual = torch.randn((37, width), dtype=dtype, device=device) * 3
    update = torch.randn_like(residual) * 0.2
    weights = [
        torch.randn((width,), dtype=dtype, device=device) * 0.1 + 1
        for _ in range(3)
    ]
    factor = torch.rand((width,), dtype=dtype, device=device) + 0.5
    scalar = width**-0.5

    stream, (first, second, routed) = sandwich_rms_norm(
        residual,
        ((update, None),),
        weights[0],
        eps=EPS,
        norms=((weights[1],), (weights[2],), (None, factor, scalar)),
    )

    expected = _round(
        residual.double() + _round(_rms(update, weights[0]), dtype), dtype
    )
    torch.testing.assert_close(stream.double(), expected, rtol=2e-2, atol=2e-2)
    for actual, weight in ((first, weights[1]), (second, weights[2])):
        torch.testing.assert_close(
            actual.double(),
            _round(_rms(expected, weight), dtype),
            rtol=2e-2,
            atol=2e-2,
        )
    unweighted = _round(_rms(expected), dtype)
    torch.testing.assert_close(
        routed.double(),
        _round(_round(unweighted * factor.double(), dtype) * scalar, dtype),
        rtol=2e-2,
        atol=2e-2,
    )
    assert all(
        value.dtype == dtype and value.shape == residual.shape
        for value in (stream, first, second, routed)
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_feedforward_sandwich_sums_normalized_branches_and_scales(
    device, dtype
):
    """Two normalized branches, a final norm, the residual and a scale."""
    torch.manual_seed(5)
    width = 1024
    residual = torch.randn((19, width), dtype=dtype, device=device)
    dense = torch.randn_like(residual) * 4
    experts = torch.randn_like(residual) * 0.01
    dense_weight, expert_weight, weight = (
        torch.randn((width,), dtype=dtype, device=device) for _ in range(3)
    )
    scale = torch.tensor([0.37], dtype=dtype, device=device)

    stream, normalized = sandwich_rms_norm(
        residual,
        ((dense, dense_weight), (experts, expert_weight)),
        weight,
        eps=EPS,
        scale=scale,
    )

    summed = _round(
        _round(_rms(dense, dense_weight), dtype)
        + _round(_rms(experts, expert_weight), dtype),
        dtype,
    )
    expected = _round(
        _round(residual.double() + _round(_rms(summed, weight), dtype), dtype)
        * scale.double(),
        dtype,
    )
    torch.testing.assert_close(stream.double(), expected, rtol=2e-2, atol=2e-2)
    assert normalized == ()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("width", (64, 2816))
def test_cuda_sandwich_is_bit_identical_to_separate_normalizations(width):
    """One launch equals rms_norm launches and tensor operations exactly."""
    torch.manual_seed(7)
    device, dtype = "cuda", torch.bfloat16
    residual = torch.randn((300, width), dtype=dtype, device=device) * 3
    update, dense, experts = (torch.randn_like(residual) for _ in range(3))
    weights = torch.randn((6, width), dtype=dtype, device=device)
    factor = torch.rand((width,), dtype=dtype, device=device) + 0.5
    scale = torch.tensor([0.37], dtype=dtype, device=device)
    unit = torch.ones(width, device=device)

    stream, (first, routed) = sandwich_rms_norm(
        residual,
        ((update, None),),
        weights[0],
        eps=EPS,
        norms=((weights[1],), (None, factor, 0.125)),
    )
    expected = residual + rms_norm(update, weights[0], EPS)
    assert torch.equal(stream, expected)
    assert torch.equal(first, rms_norm(expected, weights[1], EPS))
    assert torch.equal(routed, rms_norm(expected, unit, EPS) * factor * 0.125)

    stream, _ = sandwich_rms_norm(
        residual,
        ((dense, weights[2]), (experts, weights[3])),
        weights[4],
        eps=EPS,
        scale=scale,
    )
    summed = rms_norm(dense, weights[2], EPS) + rms_norm(
        experts, weights[3], EPS
    )
    expected = (residual + rms_norm(summed, weights[4], EPS)) * scale
    assert torch.equal(stream, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_sandwich_of_strided_rows_raises():
    rows = torch.randn((8, 2 * 64), dtype=torch.bfloat16, device="cuda")
    strided = rows[:, :64]
    weight = torch.ones(64, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(
        ValueError, match="sandwich_rms_norm has no CUDA kernel"
    ):
        sandwich_rms_norm(strided, ((strided, None),), weight, eps=EPS)
