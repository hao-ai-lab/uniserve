"""Shared-axis QK normalization with an unrotated spatial tail."""

from __future__ import annotations

import pytest
import torch

from uniserve import ops

DIM = 128
ROPE_DIM = 64
Q_HEADS = 8
K_HEADS = 2
EPS = 1e-6


def _inputs(device: str, dtype: torch.dtype, tokens: int):
    torch.manual_seed(7)
    q = torch.randn(tokens, Q_HEADS, DIM, device=device, dtype=dtype)
    k = torch.randn(tokens, K_HEADS, DIM, device=device, dtype=dtype)
    w_t_q = torch.randn(ROPE_DIM, device=device, dtype=dtype).abs() + 0.5
    w_hw_q = torch.randn(DIM - ROPE_DIM, device=device, dtype=dtype).abs() + 0.5
    w_t_k = torch.randn(ROPE_DIM, device=device, dtype=dtype).abs() + 0.5
    w_hw_k = torch.randn(DIM - ROPE_DIM, device=device, dtype=dtype).abs() + 0.5
    angles = (
        torch.rand(tokens, ROPE_DIM // 2, device=device, dtype=torch.float32)
        * 6.0
    )
    cos_t, sin_t = torch.cos(angles), torch.sin(angles)
    # Zero spatial positions: identity rotation tables for the h/w axes.
    cos_hw = torch.ones(
        tokens, (DIM - ROPE_DIM) // 4, device=device, dtype=torch.float32
    )
    sin_hw = torch.zeros_like(cos_hw)
    weights_q = (w_t_q, w_hw_q, w_hw_q)
    weights_k = (w_t_k, w_hw_k, w_hw_k)
    cos = (cos_t, cos_hw, cos_hw)
    sin = (sin_t, sin_hw, sin_hw)
    return q, k, weights_q, weights_k, cos, sin


def _run(q, k, wq, wk, cos, sin, *, identity_axes):
    # Serving always evaluates under inference mode; the fused kernels
    # require it.
    with torch.inference_mode():
        return ops.qk_norm_rope(
            q,
            k,
            wq,
            wk,
            cos,
            sin,
            EPS,
            axis_dims=(ROPE_DIM, (DIM - ROPE_DIM) // 2, (DIM - ROPE_DIM) // 2),
            identity_axes=identity_axes,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_identity_axes_match_full_multi_axis_on_cuda(tokens, dtype):
    q, k, wq, wk, cos, sin = _inputs("cuda", dtype, tokens)
    base_q, base_k = _run(q, k, wq, wk, cos, sin, identity_axes=None)
    fast_q, fast_k = _run(q, k, wq, wk, cos, sin, identity_axes=(1, 2))
    tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-3
    for source, weights, base, actual in (
        (q, wq, base_q, fast_q),
        (k, wk, base_k, fast_k),
    ):
        head, tail = source.double().split((ROPE_DIM, DIM - ROPE_DIM), dim=-1)
        head = (
            head
            * torch.rsqrt(head.square().mean(-1, keepdim=True) + EPS)
            * weights[0].double()
        )
        tail = (
            tail
            * torch.rsqrt(tail.square().mean(-1, keepdim=True) + EPS)
            * weights[1].double()
        )
        left, right = head.chunk(2, dim=-1)
        cosine, sine = (
            cos[0].double().unsqueeze(1),
            sin[0].double().unsqueeze(1),
        )
        expected = torch.cat(
            (left * cosine - right * sine, right * cosine + left * sine, tail),
            -1,
        )
        torch.testing.assert_close(
            base.double(), expected, rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            actual.double(), expected, rtol=tolerance, atol=tolerance
        )
