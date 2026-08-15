"""Thin model-facing operator functions."""

from __future__ import annotations

from ..execution.forward_batch import AttentionSelection
from .requests import (
    AddRmsNormReq,
    AttentionRegime,
    AttentionReq,
    PackedRopeReq,
    QKNormReq,
    QKNormRopeReq,
    RmsNormReq,
    SiluAndMulReq,
)


def rms_norm(hidden_states, weight, eps: float, *, override: str | None = None):
    from .providers import rms_norm_dispatcher

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
    from .providers import add_rms_norm_dispatcher

    return add_rms_norm_dispatcher().run(
        AddRmsNormReq(hidden_states, residual, weight, float(eps), bool(in_place)),
        override=override,
    )


def silu_and_mul(x, y=None, *, override: str | None = None):
    from .providers import silu_and_mul_dispatcher

    if y is not None:
        import torch

        x = torch.cat((x, y), dim=-1)
    return silu_and_mul_dispatcher().run(SiluAndMulReq(x), override=override)


def qk_norm(q, k, q_weight, k_weight, eps: float, *, axis_dims=None, override: str | None = None):
    from .providers import qk_norm_dispatcher

    return qk_norm_dispatcher().run(
        QKNormReq(
            q,
            k,
            q_weight,
            k_weight,
            float(eps),
            None if axis_dims is None else tuple(int(v) for v in axis_dims),
        ),
        override=override,
    )


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
    quant=None,
    override: str | None = None,
):
    from .providers import qk_norm_rope_dispatcher

    return qk_norm_rope_dispatcher().run(
        QKNormRopeReq(
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            float(eps),
            position_ids,
            int(unsqueeze_dim),
            None if axis_dims is None else tuple(int(v) for v in axis_dims),
            None if identity_axes is None else tuple(int(v) for v in identity_axes),
            quant,
        ),
        override=override,
    )


def qk_norm_packed_rope(
    q, k, q_weight, k_weight, cos, sin, eps: float, *, override: str | None = None
):
    q, k = qk_norm(q, k, q_weight, k_weight, eps, override=override)
    return rope(q, cos, sin, override=override), rope(k, cos, sin, override=override)


def rope(x, cos, sin, *, override: str | None = None):
    from .providers import rope_dispatcher

    return rope_dispatcher().run(PackedRopeReq(x, cos, sin), override=override)


def _attention_req(
    q,
    k,
    v,
    *,
    regime: AttentionRegime | str = AttentionRegime.DENSE,
    causal: bool,
    scale: float,
    attn_mask=None,
    kv_cache=None,
    metadata=None,
    ctx=None,
    block_table=None,
    cache_seqlens=None,
    current_k=None,
    current_v=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    visible_end=None,
    page_table=None,
    seqused_k=None,
    use_prefix_bounds: bool = False,
    fully_visible: bool = False,
):
    return AttentionReq(
        q=q,
        k=k,
        v=v,
        regime=AttentionRegime(regime),
        causal=bool(causal),
        scale=float(scale),
        attn_mask=attn_mask,
        kv_cache=kv_cache,
        metadata=metadata,
        ctx=ctx,
        stats=getattr(ctx, "stats", None),
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        current_k=current_k,
        current_v=current_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        visible_end=visible_end,
        page_table=page_table,
        seqused_k=seqused_k,
        use_prefix_bounds=use_prefix_bounds,
        fully_visible=fully_visible,
    )


def attention(
    q,
    k,
    v,
    *,
    regime: AttentionRegime | str = AttentionRegime.DENSE,
    causal: bool,
    scale: float,
    attn_mask=None,
    kv_cache=None,
    metadata=None,
    ctx=None,
    selection: AttentionSelection,
    block_table=None,
    cache_seqlens=None,
    current_k=None,
    current_v=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    visible_end=None,
    page_table=None,
    seqused_k=None,
    use_prefix_bounds: bool = False,
    fully_visible: bool = False,
):
    from .providers import run_attention

    req = _attention_req(
        q,
        k,
        v,
        regime=regime,
        causal=causal,
        scale=scale,
        attn_mask=attn_mask,
        kv_cache=kv_cache,
        metadata=metadata,
        ctx=ctx,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        current_k=current_k,
        current_v=current_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        visible_end=visible_end,
        page_table=page_table,
        seqused_k=seqused_k,
        use_prefix_bounds=use_prefix_bounds,
        fully_visible=fully_visible,
    )
    return run_attention(selection, req)


def can_run_attention(
    q,
    k,
    v,
    *,
    regime: AttentionRegime | str = AttentionRegime.DENSE,
    causal: bool,
    scale: float,
    attn_mask=None,
    kv_cache=None,
    metadata=None,
    ctx=None,
    selection: AttentionSelection,
    block_table=None,
    cache_seqlens=None,
    current_k=None,
    current_v=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    visible_end=None,
    page_table=None,
    seqused_k=None,
    use_prefix_bounds: bool = False,
    fully_visible: bool = False,
) -> bool:
    from .providers import can_run_attention as can_run

    req = _attention_req(
        q,
        k,
        v,
        regime=regime,
        causal=causal,
        scale=scale,
        attn_mask=attn_mask,
        kv_cache=kv_cache,
        metadata=metadata,
        ctx=ctx,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        current_k=current_k,
        current_v=current_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        visible_end=visible_end,
        page_table=page_table,
        seqused_k=seqused_k,
        use_prefix_bounds=use_prefix_bounds,
        fully_visible=fully_visible,
    )
    return can_run(selection, req)
