"""CFG post-guidance renorm and RMSNorm against an explicit fp32 reference."""

from __future__ import annotations

import pytest
import torch

from uniserve.diffusion import AdditiveGuidance, Branch, LinearGrid, Renorm
from uniserve.nn import functional
from uniserve.nn.norm import RMSNorm

pytestmark = pytest.mark.unit


def _combine(base, conditioned, scale, renorm, minimum=0.0):
    guidance = AdditiveGuidance(scale, 1.0, (0.0, 1.0), renorm, minimum)
    schedule = LinearGrid(
        1.0, direction="descending", shift_domain="time"
    ).schedule(steps=1, device=base.device)
    return guidance.combine(
        {Branch.TEXT_UNCONDITIONAL: base, Branch.CONDITIONED: conditioned},
        schedule,
        0,
    )


# --- shared references ------------------------------------------------------


def _rms_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """FP32 statistics and affine transform with an activation output cast."""
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return (weight.float() * xf).to(in_dtype)


def _guided_two_branch(
    base: torch.Tensor, cond: torch.Tensor, scale: float
) -> torch.Tensor:
    """Pre-renorm guided velocity for the two-branch form.

    The two-branch form is ``base + scale*(cond-base)``.
    """
    return base + scale * (cond - base)


def _match_norm_reference(
    guided: torch.Tensor,
    ref: torch.Tensor,
    *,
    dims: tuple[int, ...],
    minimum: float,
    eps: float,
) -> torch.Tensor:
    guided_norm = guided.float().norm(dim=dims, keepdim=True).clamp_min(eps)
    ref_norm = ref.float().norm(dim=dims, keepdim=True).clamp_min(minimum)
    scale = (ref_norm / guided_norm).clamp(max=1.0)
    return guided * scale.to(dtype=guided.dtype)


# --- renorm: NONE is identity ----------------------------------------------


def test_renorm_none_returns_unmodified_guided_velocity():
    torch.manual_seed(0)
    base = torch.randn(2, 3, 8)
    cond = torch.randn(2, 3, 8)
    scale = 2.5

    out = _combine(base, cond, scale, Renorm.NONE)

    torch.testing.assert_close(out, _guided_two_branch(base, cond, scale))


# --- renorm: RESCALE blend --------------------------------------------------


def test_renorm_rescale_blends_channel_matched_with_raw_guided():
    torch.manual_seed(1)
    base = torch.randn(2, 4, 16)
    cond = torch.randn(2, 4, 16)
    scale = 3.0
    guided = _guided_two_branch(base, cond, scale)
    eps = torch.finfo(guided.dtype).eps
    matched = _match_norm_reference(
        guided, cond, dims=(guided.ndim - 1,), minimum=0.0, eps=eps
    )
    expected = 0.7 * matched + 0.3 * guided

    out = _combine(base, cond, scale, Renorm.RESCALE)

    torch.testing.assert_close(out, expected)


# --- renorm: CFG_ZERO_STAR --------------------------------------------------


def test_renorm_cfg_zero_star_subtracts_per_sample_mean():
    torch.manual_seed(2)
    base = torch.randn(2, 5, 8)
    cond = torch.randn(2, 5, 8)
    scale = 4.0
    guided = _guided_two_branch(base, cond, scale)
    expected = guided - guided.mean(dim=(1, 2), keepdim=True)

    out = _combine(base, cond, scale, Renorm.CFG_ZERO_STAR)

    torch.testing.assert_close(out, expected)


# --- renorm_min clamp -------------------------------------------------------


def test_renorm_min_clamps_reference_norm_up_disabling_downscale():
    # Guided velocity has a much larger norm than the reference branch, so with
    # the default minimum the channel match downscales toward the ref norm.
    # A large renorm_min clamps the reference norm above the guided norm, making
    # the scale clamp to 1.0, i.e. the guided velocity passes through unchanged.
    torch.manual_seed(4)
    base = torch.randn(2, 4, 16) * 0.1
    cond = torch.randn(2, 4, 16) * 5.0
    scale = 4.0
    guided = _guided_two_branch(base, cond, scale)

    out_default = _combine(base, cond, scale, Renorm.CHANNEL, 0.0)
    out_clamped = _combine(base, cond, scale, Renorm.CHANNEL, 1e6)

    # The default match downscales (strictly different from raw guided)...
    assert not torch.allclose(out_default, guided)
    # ...while the large floor disables the downscale, returning guided
    # unchanged.
    torch.testing.assert_close(out_clamped, guided)


# --- unknown renorm kind ----------------------------------------------------


def test_guidance_rejects_an_unrecognized_renormalization():
    with pytest.raises(TypeError, match="renormalization must use Renorm"):
        AdditiveGuidance(2.0, 1.0, (0.0, 1.0), "channel", 0.0)


# --- RMSNorm vs fp32 reciprocal-rms reference -------------------------------


def test_rmsnorm_module_forward_matches_reference_with_unit_weight_cpu():
    """The default unit-weight RMSNorm module.

    The module preserves normalized CPU values.
    """
    torch.manual_seed(8)
    hidden = 64
    module = RMSNorm(hidden)  # default weight is all-ones
    x = torch.randn(3, 5, hidden)

    with torch.no_grad():
        out = module(x)
    expected = _rms_reference(x, module.weight, module.eps)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA RMSNorm needs a device"
)
def test_rmsnorm_matches_fp32_reference_on_cuda():
    torch.manual_seed(10)
    hidden = 512
    weight = torch.randn(
        hidden, device="cuda", dtype=torch.float32
    ).contiguous()
    x = torch.randn(16, hidden, device="cuda", dtype=torch.float32).contiguous()

    with torch.no_grad():
        actual = functional.rms_norm(x, weight, 1e-6)
    expected = _rms_reference(x, weight, 1e-6)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rmsnorm_preserves_normalization_and_residual_values(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA RMSNorm needs a device")
    generator = torch.Generator(device=device).manual_seed(181)
    module = RMSNorm(128, 1e-5, device=device).to(dtype)
    with torch.no_grad():
        module.weight.copy_(
            torch.randn((128,), generator=generator, device=device)
        )
        values = torch.randn(
            (3, 5, 128), generator=generator, device=device
        ).to(dtype)
        residual = torch.randn(
            values.shape, generator=generator, device=device
        ).to(dtype)
        expected = torch.nn.functional.rms_norm(
            values.float(), (128,), weight=module.weight.float(), eps=1e-5
        ).to(dtype)
        torch.testing.assert_close(
            module(values), expected, rtol=2e-2, atol=2e-2
        )
        normalized, combined = functional.add_rms_norm(
            values, residual, module.weight, module.eps
        )
        expected_sum = values + residual
        expected_normalized = torch.nn.functional.rms_norm(
            expected_sum.float(), (128,), weight=module.weight.float(), eps=1e-5
        ).to(dtype)
        torch.testing.assert_close(combined, expected_sum, rtol=0, atol=0)
        torch.testing.assert_close(
            normalized, expected_normalized, rtol=2e-2, atol=2e-2
        )
