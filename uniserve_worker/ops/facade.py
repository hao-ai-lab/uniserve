"""Thin model-facing operator functions."""

from __future__ import annotations

from .attention import can_run_attention as _can_run_attention
from .attention import run_attention
from .requests import (
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
from .rms import add_rms_norm_dispatcher, qk_norm_dispatcher, rms_norm_dispatcher
from .rope import qk_norm_rope_dispatcher, rope_dispatcher
from .silu import silu_and_mul_dispatcher


def rms_norm(hidden_states, weight, eps: float, *, override: str | None = None):
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
    return add_rms_norm_dispatcher().run(
        AddRmsNormReq(hidden_states, residual, weight, float(eps), bool(in_place)),
        override=override,
    )


def silu_and_mul(x, y=None, *, override: str | None = None):
    if y is not None:
        import torch

        x = torch.cat((x, y), dim=-1)
    return silu_and_mul_dispatcher().run(SiluAndMulReq(x), override=override)


def qk_norm(q, k, q_weight, k_weight, eps: float, *, axis_dims=None, override: str | None = None):
    if axis_dims is None:
        req = QKNormReq(q, k, q_weight, k_weight, float(eps))
    else:
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
    if axis_dims is None:
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
    return rope_dispatcher().run(PackedRopeReq(x, cos, sin), override=override)


def attention(req: AttentionReq, *, provider):
    return run_attention(provider, req)


def can_run_attention(req: AttentionReq, *, provider) -> bool:
    return _can_run_attention(provider, req)
