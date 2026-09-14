"""Model-facing functions over typed operator requests.

This module is the stable boundary between neural-network layers and provider
dispatch. Each function normalizes convenient model inputs into the request
shape shared by eager, Triton, and extension-backed implementations.
"""

from __future__ import annotations

from uniserve.ops.attention import can_run_attention as _can_run_attention
from uniserve.ops.attention import run_attention
from uniserve.ops.requests import (
    AddRmsNormReq,
    AttentionReq,
    MultiAxisQKNormReq,
    MultiAxisQKNormRopeReq,
    PackedRopeReq,
    QKNormReq,
    QKNormRopeReq,
    RmsNormReq,
    SiluAndMulReq,
)
from uniserve.ops.rms import add_rms_norm_dispatcher, qk_norm_dispatcher, rms_norm_dispatcher
from uniserve.ops.rope import qk_norm_rope_dispatcher, rope_dispatcher
from uniserve.ops.silu import silu_and_mul_dispatcher


def rms_norm(hidden_states, weight, eps: float, *, override: str | None = None):
    """Normalize the final hidden dimension with a selected RMSNorm provider."""

    return rms_norm_dispatcher().run(
        RmsNormReq(hidden_states, weight, float(eps)), override=override
    )


def add_rms_norm(
    hidden_states,
    residual,
    weight,
    eps: float,
    *,
    in_place: bool = False,
    override: str | None = None,
):
    """Add a residual and return its RMS-normalized value plus the residual sum.

    ``in_place`` allows an eligible provider to reuse caller-owned storage for
    the combined residual.
    """

    return add_rms_norm_dispatcher().run(
        AddRmsNormReq(hidden_states, residual, weight, float(eps), bool(in_place)),
        override=override,
    )


def silu_and_mul(x, y=None, *, override: str | None = None):
    """Apply SiLU gating to packed halves or to explicit gate and value tensors."""

    if y is not None:
        import torch

        # Providers consume one tensor whose final dimension is laid out as
        # ``[gate, value]``; concatenate explicit operands into that contract.
        x = torch.cat((x, y), dim=-1)

    return silu_and_mul_dispatcher().run(SiluAndMulReq(x), override=override)


def qk_norm(
    q,
    k,
    q_weight,
    k_weight,
    eps: float,
    *,
    axis_dims=None,
    override: str | None = None,
):
    """RMS-normalize query and key tensors as one vector or independent axes."""

    req: QKNormReq | MultiAxisQKNormReq
    if axis_dims is None:
        req = QKNormReq(q, k, q_weight, k_weight, float(eps))
    else:
        # Multi-axis requests preserve independent normalization domains and
        # therefore carry one weight tensor for each declared axis width.
        req = MultiAxisQKNormReq(
            q,
            k,
            tuple(int(v) for v in axis_dims),
            tuple(q_weight),
            tuple(k_weight),
            float(eps),
        )

    return qk_norm_dispatcher().run(req, override=override)


def qk_norm_rope(
    q,
    k,
    q_weight,
    k_weight,
    cos,
    sin,
    eps: float,
    *,
    position_ids=None,
    unsqueeze_dim: int = 1,
    axis_dims=None,
    identity_axes=None,
    in_place: bool = False,
    override: str | None = None,
):
    """Fuse query/key RMS normalization with rotary-position application.

    ``axis_dims`` partitions the final dimension into independently normalized
    rotary axes. ``identity_axes`` names partitions that remain unrotated in a
    multi-axis request. In-place execution is available for the single-axis
    request when supported by the selected provider.
    """

    req: QKNormRopeReq | MultiAxisQKNormRopeReq
    if axis_dims is None:
        # Single-axis requests carry ordinary weight and factor tensors.
        req = QKNormRopeReq(
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            float(eps),
            position_ids,
            int(unsqueeze_dim),
            bool(in_place),
        )
    else:
        # Multi-axis requests preserve each feature partition's weights and
        # factor tables for group-aware provider planning.
        if in_place:
            raise ValueError("in-place qk_norm_rope does not support multi-axis requests")

        req = MultiAxisQKNormRopeReq(
            q,
            k,
            tuple(int(v) for v in axis_dims),
            tuple(q_weight),
            tuple(k_weight),
            tuple(cos),
            tuple(sin),
            float(eps),
            () if identity_axes is None else tuple(int(v) for v in identity_axes),
            position_ids,
            int(unsqueeze_dim),
        )

    return qk_norm_rope_dispatcher().run(req, override=override)


def rope(x, cos, sin, *, override: str | None = None):
    """Apply packed rotary factors through the selected RoPE provider."""

    return rope_dispatcher().run(PackedRopeReq(x, cos, sin), override=override)


def attention(req: AttentionReq, *, provider):
    """Execute an attention request through its bound backend provider."""

    return run_attention(provider, req)


def can_run_attention(req: AttentionReq, *, provider) -> bool:
    """Report whether a bound attention provider accepts ``req``."""

    return _can_run_attention(provider, req)
