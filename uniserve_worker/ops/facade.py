"""Thin model-facing operator functions."""
from __future__ import annotations

from .requests import (
    AddRmsNormReq,
    AttentionRegime,
    AttentionReq,
    QKNormReq,
    QKNormRopeReq,
    PackedRopeReq,
    RmsNormReq,
    SiluAndMulReq,
    TpAllReduceReq,
)


def rms_norm(hidden_states, weight, eps: float, *, override: str | None = None):
    from .providers import rms_norm_dispatcher

    return rms_norm_dispatcher().run(RmsNormReq(hidden_states, weight, float(eps)), override=override)


def add_rms_norm(hidden_states, residual, weight, eps: float, *, in_place: bool = False, override: str | None = None):
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
        QKNormReq(q, k, q_weight, k_weight, float(eps), None if axis_dims is None else tuple(int(v) for v in axis_dims)),
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
    quant=None,
    adapters=None,
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
            quant,
            adapters,
        ),
        override=override,
    )


def qk_norm_packed_rope(q, k, q_weight, k_weight, cos, sin, eps: float, *, override: str | None = None):
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
    backend=None,
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
        backend=backend,
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
    backend=None,
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
    override: str | None = None,
):
    from .providers import attention_dispatcher

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
        backend=backend,
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
    )
    return attention_dispatcher().run(req, override=override)


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
    backend=None,
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
    override: str | None = None,
) -> bool:
    from .providers import attention_dispatcher

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
        backend=backend,
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
    )
    return any(provider.can_run(req) for provider in attention_dispatcher().ordered(override))


def tp_all_reduce(tensor, op: str = "sum", *, axis, override: str | None = None):
    from .providers import tp_all_reduce_dispatcher

    dispatcher = tp_all_reduce_dispatcher()
    handoff = dispatcher.dispatch(
        TpAllReduceReq(tensor, str(op), axis),
        override=override,
        mesh=axis,
    )
    return dispatcher.combine(handoff, override=override, mesh=axis)


def attention_dispatcher():
    from .providers import attention_dispatcher as get_dispatcher

    return get_dispatcher()
