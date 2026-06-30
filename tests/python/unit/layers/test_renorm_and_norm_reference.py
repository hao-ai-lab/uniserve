"""Behavioral tests for CFG post-guidance renorm and the RMSNorm reference.

Covers ``uniserve_worker.nn.diffusion.cfg`` renorm handlers (none/identity,
RESCALE, CFG_ZERO_STAR, the ``renorm_min`` clamp, and the unknown-kind
``ValueError``) and ``uniserve_worker.nn.norm.RMSNorm`` against an explicit
fp32 reciprocal-rms reference. The eager reference path is exercised
unconditionally (CPU); the fused/triton dispatch path is compared to the eager
reference only when a usable CUDA device makes it eligible.
"""
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
    """fp32 reciprocal-rms reference matching the eager kernel exactly.

    Variance is accumulated in fp32, normalization applied in fp32, the result
    is cast back to the input dtype, and only then scaled by ``weight`` --
    mirroring ``weight * x.to(in_dtype)`` in the production eager path.
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


def test_renorm_cfg_zero_star_output_is_zero_mean_per_sample():
    torch.manual_seed(3)
    base = torch.randn(3, 6, 8)
    cond = torch.randn(3, 6, 8)
    branches = torch.stack([base, cond], dim=0)

    out = combine_cfg(
        branches,
        CfgParams(branch_count=2, scales=(2.0,), renorm=RenormKind.CFG_ZERO_STAR),
    )

    per_sample_mean = out.mean(dim=(1, 2))
    torch.testing.assert_close(per_sample_mean, torch.zeros_like(per_sample_mean), atol=1e-6, rtol=0)


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


_NORM_TOLERANCE = {
    torch.float32: dict(atol=0.0, rtol=0.0),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=6e-2, rtol=6e-2),
}


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
def test_rmsnorm_eager_matches_fp32_reference(dtype):
    """The eager RMSNorm path always matches the fp32-accumulation reference.

    fp32 is exact; fp16/bf16 agree within low-precision tolerance because both
    the kernel and the reference accumulate the variance in fp32 and round only
    on the final cast back to ``dtype``. The eager path is the terminal fallback
    and runs on CPU without any optional kernel package.
    """

    torch.manual_seed(7)
    hidden = 128
    weight = torch.randn(hidden, dtype=dtype)
    x = torch.randn(4, 6, hidden, dtype=dtype)

    out = ops.rms_norm(x, weight, 1e-6, override="eager")
    expected = _rms_reference(x, weight, 1e-6)

    if dtype is torch.float32:
        assert torch.equal(out, expected)
    else:
        torch.testing.assert_close(out, expected, **_NORM_TOLERANCE[dtype])


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


# --- fused/triton dispatch path agrees with the eager reference -------------


def _fused_rmsnorm_eligible(dtype: torch.dtype) -> bool:
    if not torch.cuda.is_available():
        return False
    from uniserve_worker.nn.norm import _TritonRmsNorm

    weight = torch.randn(128, device="cuda", dtype=dtype).contiguous()
    x = torch.randn(8, 128, device="cuda", dtype=dtype).contiguous()
    with torch.no_grad():
        return bool(_TritonRmsNorm().is_eligible(x, weight, 1e-6))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused RMSNorm path needs CUDA")
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
def test_rmsnorm_fused_path_matches_eager_reference_on_cuda(dtype):
    """When the fused (triton) provider is eligible it matches the eager reference.

    The fused kernel is gated by hardware/grad-mode eligibility; when it cannot
    run here we skip rather than silently testing eager-vs-eager. The eager
    reference itself is verified unconditionally by the CPU tests above.
    """

    if not _fused_rmsnorm_eligible(dtype):
        pytest.skip("fused triton RMSNorm not eligible on this device/grad mode")

    torch.manual_seed(9)
    hidden = 256
    weight = torch.randn(hidden, device="cuda", dtype=dtype).contiguous()
    x = torch.randn(8, hidden, device="cuda", dtype=dtype).contiguous()

    with torch.no_grad():
        fused = ops.rms_norm(x, weight, 1e-6)  # auto-dispatch selects the fused kernel
        eager = ops.rms_norm(x, weight, 1e-6, override="eager")

    # fp32 differs from eager only by fp32 reduction-order noise; fp16/bf16 by
    # their wider low-precision rounding tolerance.
    tolerance = dict(atol=1e-5, rtol=1e-5) if dtype is torch.float32 else _NORM_TOLERANCE[dtype]
    torch.testing.assert_close(fused, eager, **tolerance)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused RMSNorm path needs CUDA")
def test_rmsnorm_fused_path_matches_fp32_reference_exactly_on_cuda():
    """The fused fp32 kernel reproduces the fp32 reciprocal-rms reference to fp32 precision."""

    if not _fused_rmsnorm_eligible(torch.float32):
        pytest.skip("fused triton RMSNorm not eligible on this device/grad mode")

    torch.manual_seed(10)
    hidden = 512
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32).contiguous()
    x = torch.randn(16, hidden, device="cuda", dtype=torch.float32).contiguous()

    with torch.no_grad():
        fused = ops.rms_norm(x, weight, 1e-6)
    expected = _rms_reference(x, weight, 1e-6)

    torch.testing.assert_close(fused, expected, atol=1e-5, rtol=1e-5)
