"""CFG post-guidance renorm and RMSNorm against an explicit fp32 reference."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker import ops
from uniserve_worker.nn.diffusion.cfg import (
    CfgParams,
    RenormKind,
    combine_cfg,
)
from uniserve_worker.nn.norm import RMSNorm

pytestmark = pytest.mark.unit


# --- shared references ------------------------------------------------------


def _rms_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 reciprocal-rms reference for RMSNorm.

    Variance is accumulated in fp32, normalization applied in fp32, the result
    is cast back to the input dtype, and only then scaled by ``weight``.
    """

    in_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return weight * xf.to(in_dtype)


def _guided_two_branch(base: torch.Tensor, cond: torch.Tensor, scale: float) -> torch.Tensor:
    """Pre-renorm guided velocity for the two-branch ``base + scale*(cond-base)`` form."""

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
    branches = torch.stack([base, cond], dim=0)
    scale = 2.5

    out = combine_cfg(
        branches,
        CfgParams(branch_count=2, scales=(scale,), renorm=RenormKind.NONE),
    )

    torch.testing.assert_close(out, _guided_two_branch(base, cond, scale))


# --- renorm: RESCALE blend --------------------------------------------------


def test_renorm_rescale_blends_channel_matched_with_raw_guided():
    torch.manual_seed(1)
    base = torch.randn(2, 4, 16)
    cond = torch.randn(2, 4, 16)
    branches = torch.stack([base, cond], dim=0)
    scale = 3.0
    guided = _guided_two_branch(base, cond, scale)
    eps = torch.finfo(guided.dtype).eps
    matched = _match_norm_reference(
        guided, branches[-1], dims=(guided.ndim - 1,), minimum=0.0, eps=eps
    )
    expected = 0.7 * matched + 0.3 * guided

    out = combine_cfg(
        branches,
        CfgParams(branch_count=2, scales=(scale,), renorm=RenormKind.RESCALE),
    )

    torch.testing.assert_close(out, expected)


# --- renorm: CFG_ZERO_STAR --------------------------------------------------


def test_renorm_cfg_zero_star_subtracts_per_sample_mean():
    torch.manual_seed(2)
    base = torch.randn(2, 5, 8)
    cond = torch.randn(2, 5, 8)
    branches = torch.stack([base, cond], dim=0)
    scale = 4.0
    guided = _guided_two_branch(base, cond, scale)
    expected = guided - guided.mean(dim=(1, 2), keepdim=True)

    out = combine_cfg(
        branches,
        CfgParams(branch_count=2, scales=(scale,), renorm=RenormKind.CFG_ZERO_STAR),
    )

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
    branches = torch.stack([base, cond], dim=0)
    scale = 4.0
    guided = _guided_two_branch(base, cond, scale)

    out_default = combine_cfg(
        branches,
        CfgParams(branch_count=2, scales=(scale,), renorm=RenormKind.CHANNEL, renorm_min=0.0),
    )
    out_clamped = combine_cfg(
        branches,
        CfgParams(branch_count=2, scales=(scale,), renorm=RenormKind.CHANNEL, renorm_min=1e6),
    )

    # The default match downscales (strictly different from raw guided)...
    assert not torch.allclose(out_default, guided)
    # ...while the large floor disables the downscale, returning guided unchanged.
    torch.testing.assert_close(out_clamped, guided)


# --- unknown renorm kind ----------------------------------------------------


def test_unknown_renorm_kind_raises_value_error():
    torch.manual_seed(5)
    branches = torch.stack([torch.randn(1, 4, 8), torch.randn(1, 4, 8)], dim=0)

    class _UnregisteredKind:
        pass

    params = CfgParams(branch_count=2, scales=(2.0,), renorm=_UnregisteredKind())

    with pytest.raises(ValueError, match="unknown renorm kind"):
        combine_cfg(branches, params)


# --- RMSNorm vs fp32 reciprocal-rms reference -------------------------------


def test_rmsnorm_module_forward_matches_reference_with_unit_weight_cpu():
    """The default unit-weight RMSNorm module reproduces the reference exactly on CPU."""

    torch.manual_seed(8)
    hidden = 64
    module = RMSNorm(hidden)  # default weight is all-ones
    x = torch.randn(3, 5, hidden)

    with torch.no_grad():
        out = module(x)
    expected = _rms_reference(x, module.weight, module.eps)

    assert torch.equal(out, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RMSNorm needs a device")
def test_rmsnorm_matches_fp32_reference_exactly_on_cuda():
    torch.manual_seed(10)
    hidden = 512
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32).contiguous()
    x = torch.randn(16, hidden, device="cuda", dtype=torch.float32).contiguous()

    with torch.no_grad():
        actual = ops.rms_norm(x, weight, 1e-6)
    expected = _rms_reference(x, weight, 1e-6)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rmsnorm_fp32_affine_rounds_only_the_weighted_result(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA RMSNorm needs a device")
    generator = torch.Generator(device=device).manual_seed(181)
    module = RMSNorm(128, 1e-5, affine_in_fp32=True, device=device).to(dtype)
    with torch.no_grad():
        module.weight.copy_(torch.randn((128,), generator=generator, device=device))
        values = torch.randn((3, 5, 128), generator=generator, device=device).to(dtype)
        residual = torch.randn(values.shape, generator=generator, device=device).to(dtype)
        expected = torch.nn.functional.rms_norm(
            values.float(), (128,), weight=module.weight.float(), eps=1e-5
        ).to(dtype)
        torch.testing.assert_close(module(values), expected)
        normalized, combined = module.forward_with_residual(values, residual)
        expected_sum = values + residual
        expected_normalized = torch.nn.functional.rms_norm(
            expected_sum.float(), (128,), weight=module.weight.float(), eps=1e-5
        ).to(dtype)
        torch.testing.assert_close(combined, expected_sum, rtol=0, atol=0)
        torch.testing.assert_close(normalized, expected_normalized)
