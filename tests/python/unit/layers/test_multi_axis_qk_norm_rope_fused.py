"""Equivalence of the rotated-tail fused multi-axis qk_norm_rope path.

The 3-axis multi-axis norm+RoPE call used by SenseNova spatial (image) tokens
has axis 0 as its own norm+RoPE group and axes 1..2 sharing one norm weight
with per-axis rotation tables. The fused single-launch kernel must be
bit-exact (including zero signs and reduction order) against the general
multi-axis pipeline it replaces: the unfused group kernels plus the
contiguous/packed-rope/cat composition.
"""
from __future__ import annotations

import pytest
import torch

from uniserve_worker import ops

EPS = 1e-6
Q_HEADS = 8
K_HEADS = 2


def _inputs(dtype: torch.dtype, tokens: int, d0: int, d1: int, *, strided: bool, seed: int = 11):
    dim = d0 + 2 * d1
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, dt=dtype):
        return torch.randn(*shape, device="cuda", dtype=dt, generator=generator)

    if strided:
        # Mimic the production layout: q/k are last-dim slices of a fused QKV
        # projection output (unit stride rows, non-contiguous stride 0/1).
        qkv = rand(tokens, (Q_HEADS + 2 * K_HEADS) * dim)
        q = qkv[:, : Q_HEADS * dim].view(tokens, Q_HEADS, dim)
        k = qkv[:, Q_HEADS * dim : (Q_HEADS + K_HEADS) * dim].view(tokens, K_HEADS, dim)
    else:
        q = rand(tokens, Q_HEADS, dim)
        k = rand(tokens, K_HEADS, dim)
    w0_q, w0_k = rand(d0).abs() + 0.5, rand(d0).abs() + 0.5
    whw_q, whw_k = rand(2 * d1).abs() + 0.5, rand(2 * d1).abs() + 0.5

    def tables(half):
        angles = torch.rand(tokens, half, device="cuda", dtype=torch.float32, generator=generator) * 6.28
        return angles.cos(), angles.sin()

    cos_t, sin_t = tables(d0 // 2)
    cos_h, sin_h = tables(d1 // 2)
    cos_w, sin_w = tables(d1 // 2)
    return (
        q,
        k,
        (w0_q, whw_q, whw_q),
        (w0_k, whw_k, whw_k),
        (cos_t, cos_h, cos_w),
        (sin_t, sin_h, sin_w),
    )


def _reference_general_pipeline(q, k, wq, wk, cos, sin, d0, d1):
    """The exact pre-fusion provider composition, built from the unchanged
    primitives: group-0 norm+rope kernel, shared-norm kernel on the tail,
    per-axis contiguous + packed-rope, final cat."""
    from uniserve_worker.nn import norm as norm_mod
    from uniserve_worker.nn import rope as rope_mod

    packed_rope = rope_mod._TritonPackedRope()
    g0 = rope_mod.try_triton_qk_rms_norm_rope(
        q[..., :d0], k[..., :d0], wq[0], wk[0], cos[0], sin[0], EPS, EPS
    )
    assert g0 is not None
    g1 = norm_mod.try_triton_qk_rms_norm(q[..., d0:], k[..., d0:], wq[1], wk[1], EPS, EPS)
    assert g1 is not None
    qn, kn = g1
    out_q, out_k = [g0[0]], [g0[1]]
    for local in range(2):
        sl = slice(local * d1, (local + 1) * d1)
        out_q.append(packed_rope.run(qn[..., sl].contiguous(), cos[1 + local], sin[1 + local]))
        out_k.append(packed_rope.run(kn[..., sl].contiguous(), cos[1 + local], sin[1 + local]))
    return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 5, 64])
@pytest.mark.parametrize("dims", [(64, 32), (48, 24), (32, 16)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strided", [True, False])
def test_rotated_tail_fused_path_is_bit_exact_on_cuda(tokens, dims, dtype, strided):
    d0, d1 = dims
    q, k, wq, wk, cos, sin = _inputs(dtype, tokens, d0, d1, strided=strided)
    with torch.inference_mode():
        ref_q, ref_k = _reference_general_pipeline(q, k, wq, wk, cos, sin, d0, d1)
        got_q, got_k = ops.qk_norm_rope(
            q, k, wq, wk, cos, sin, EPS, axis_dims=(d0, d1, d1)
        )
    assert got_q.shape == ref_q.shape and got_k.shape == ref_k.shape
    assert _bitwise_equal(got_q, ref_q)
    assert _bitwise_equal(got_k, ref_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_kernel_matches_general_pipeline_directly():
    from uniserve_worker.nn import rope as rope_mod

    q, k, wq, wk, cos, sin = _inputs(torch.bfloat16, 33, 64, 32, strided=True)
    with torch.inference_mode():
        ref_q, ref_k = _reference_general_pipeline(q, k, wq, wk, cos, sin, 64, 32)
        got = rope_mod.try_triton_qk_multi_axis_rms_norm_rope(
            q, k, wq[0], wq[1], wk[0], wk[1], cos, sin, EPS, EPS, axis_dims=(64, 32, 32)
        )
    assert got is not None
    assert _bitwise_equal(got[0], ref_q)
    assert _bitwise_equal(got[1], ref_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_kernel_declines_unverified_shapes():
    from uniserve_worker.nn import rope as rope_mod

    q, k, wq, wk, cos, sin = _inputs(torch.bfloat16, 4, 64, 32, strided=False)
    with torch.inference_mode():
        # Mismatched tail axes are outside the fused kernel's contract.
        assert not rope_mod.can_run_triton_qk_multi_axis_rms_norm_rope(
            q, k, wq[0], wq[1], wk[0], wk[1], cos, sin, axis_dims=(64, 32, 16)
        )
        # Tile widths above the bitwise-verified {32, 64} family must decline
        # (the reduction tree is only proven for those tile shapes).
        assert not rope_mod.can_run_triton_qk_multi_axis_rms_norm_rope(
            q, k, wq[0], wq[1], wk[0], wk[1], cos, sin, axis_dims=(130, 32, 32)
        )
        # The verified family itself is eligible.
        assert rope_mod.can_run_triton_qk_multi_axis_rms_norm_rope(
            q, k, wq[0], wq[1], wk[0], wk[1], cos, sin, axis_dims=(64, 32, 32)
        )
