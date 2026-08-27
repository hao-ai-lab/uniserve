"""Capability-driven attention providers."""

from __future__ import annotations

from typing import cast

import torch

from ..backends.attention.base import (
    AttentionBackend,
    AttentionCapabilities,
    PagedAttentionBackend,
    SegmentedAttentionBackend,
    VarlenAttentionBackend,
    VisibleEndAttentionBackend,
)
from ..execution.forward_batch import AttentionMode
from .requests import (
    AttentionReq,
    PagedDecodeAttention,
    VarlenAttention,
    VisibleEndAttention,
)


def _head_dim(req: AttentionReq) -> int:
    return int(req.q.shape[-1])


def _kv_dims(req: AttentionReq) -> tuple[int, int]:
    if (
        isinstance(req, VisibleEndAttention)
        and req.prefix_k is not None
        and req.prefix_v is not None
    ):
        return int(req.prefix_k.shape[-1]), int(req.prefix_v.shape[-1])
    if isinstance(req, PagedDecodeAttention):
        k = req.current_k if req.current_k is not None else req.k
        v = req.current_v if req.current_v is not None else req.v
        return int(k.shape[-1]), int(v.shape[-1])
    return int(req.k.shape[-1]), int(req.v.shape[-1])


def _device_supported(caps: AttentionCapabilities, tensor: torch.Tensor) -> bool:
    if caps.cuda_only and tensor.device.type != "cuda":
        return False
    minimum = caps.min_cuda_capability
    if minimum is None:
        return True
    if tensor.device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(tensor.device)
    return (int(major), int(minor)) >= (int(minimum[0]), int(minimum[1]))


def _paged_storage_supported(caps: AttentionCapabilities, req: AttentionReq) -> bool:
    kv_cache = getattr(req, "kv_cache", None)
    view_block_size = getattr(kv_cache, "block_size", None)
    block_table = getattr(req, "block_table", None)
    if view_block_size is not None:
        if not bool(getattr(kv_cache, "supports_paged_attention_storage", True)):
            return False
        block_size = int(view_block_size or 0)
    elif block_table is not None:
        paged_k = (
            req.prefix_k
            if isinstance(req, VisibleEndAttention) and req.prefix_k is not None
            else req.k
        )
        if not isinstance(paged_k, torch.Tensor) or paged_k.ndim != 4:
            return False
        block_size = int(paged_k.shape[1])
    else:
        return True
    if block_size <= 0:
        return False
    multiple = int(caps.paged_block_size_multiple or 1)
    return block_size % max(1, multiple) == 0


def _is_one_token_decode(req: PagedDecodeAttention) -> bool:
    if req.q.ndim == 3:
        plan = req.ctx
        query_lens = getattr(plan, "query_lens_cpu", ()) or ()
        if getattr(plan, "forward_mode", None) is AttentionMode.PAGED_DECODE and len(
            query_lens
        ) == int(req.q.shape[0]):
            return all(int(length) == 1 for length in query_lens)
        return int(req.q.shape[0]) == 1
    if req.q.ndim == 4:
        return int(req.q.shape[2]) == 1
    return False


def _can_run(backend: AttentionBackend, req: AttentionReq) -> bool:
    caps = backend.capabilities()
    if not caps.available or not _device_supported(caps, req.q):
        return False
    if _head_dim(req) < int(caps.min_head_dim or 1):
        return False
    k_dim, v_dim = _kv_dims(req)
    if not caps.supports_trunk_geometry(_head_dim(req), k_dim, v_dim):
        return False
    if isinstance(req, VisibleEndAttention):
        if req.prefix_k is not None:
            return (
                bool(caps.segmented_attention)
                and (
                    not bool(getattr(req.ctx, "cuda_graph_capture", False))
                    or bool(caps.segmented_attention_cuda_graph)
                )
                and _paged_storage_supported(caps, req)
            )
        return bool(caps.visible_end)
    if isinstance(req, VarlenAttention):
        if req.block_table is not None:
            return (
                bool(caps.varlen_attention)
                and bool(caps.varlen_paged_kv)
                and _paged_storage_supported(caps, req)
            )
        return not caps.requires_paged_varlen and bool(caps.varlen_attention)
    if isinstance(req, PagedDecodeAttention):
        if caps.paged_decode_only and not _is_one_token_decode(req):
            return False
        return bool(caps.paged_kv) and _paged_storage_supported(caps, req)
    if caps.paged_decode_only or (req.attn_mask is not None and not caps.accepts_dense_mask):
        return False
    if req.q.ndim != req.k.ndim or req.q.ndim != req.v.ndim:
        return False
    return int(req.q.ndim) in caps.dense_ranks


def _run(backend: AttentionBackend, req: AttentionReq) -> torch.Tensor:
    if isinstance(req, VisibleEndAttention) and req.prefix_k is not None:
        if req.prefix_v is None or req.prefix_lens is None or req.cu_seqlens_q is None:
            raise ValueError("segmented attention metadata is incomplete")
        return cast(SegmentedAttentionBackend, backend).forward_segmented(
            req.q,
            req.k,
            req.v,
            req.prefix_k,
            req.prefix_v,
            page_table=req.page_table,
            prefix_lens=req.prefix_lens,
            cu_seqlens_q=req.cu_seqlens_q,
            visible_current_end=req.visible_end,
            scale=req.scale,
            fully_visible_current=req.fully_visible,
            context=req.ctx,
        )
    if isinstance(req, VisibleEndAttention):
        return cast(VisibleEndAttentionBackend, backend).forward_visible_end(
            req.q,
            req.k,
            req.v,
            visible_end=req.visible_end,
            cu_seqlens_q=req.cu_seqlens_q,
            cu_seqlens_k=req.cu_seqlens_k,
            page_table=req.page_table,
            seqused_k=req.seqused_k,
            max_seqlen_q=req.max_seqlen_q,
            max_seqlen_k=req.max_seqlen_k,
            scale=req.scale,
            use_prefix_bounds=req.use_prefix_bounds,
            fully_visible=req.fully_visible,
            context=req.ctx,
        )
    if isinstance(req, VarlenAttention):
        return cast(VarlenAttentionBackend, backend).forward_varlen(
            req.q,
            req.k,
            req.v,
            cu_seqlens_q=req.cu_seqlens_q,
            cu_seqlens_k=req.cu_seqlens_k,
            max_seqlen_q=int(req.max_seqlen_q),
            max_seqlen_k=int(req.max_seqlen_k),
            causal=req.causal,
            scale=req.scale,
            block_table=req.block_table,
            context=req.ctx,
        )
    if isinstance(req, PagedDecodeAttention):
        return cast(PagedAttentionBackend, backend).forward_paged(
            req.q,
            req.k,
            req.v,
            block_table=req.block_table,
            cache_seqlens=req.cache_seqlens,
            k=req.current_k,
            v=req.current_v,
            causal=req.causal,
            scale=req.scale,
            context=req.ctx,
        )
    return backend.forward(
        req.q,
        req.k,
        req.v,
        causal=req.causal,
        scale=req.scale,
        attn_mask=req.attn_mask,
        context=req.ctx,
    )


def can_run_attention(provider: AttentionBackend, req: AttentionReq) -> bool:
    return _can_run(provider, req)


def run_attention(provider: AttentionBackend, req: AttentionReq) -> torch.Tensor:
    if not _can_run(provider, req):
        raise RuntimeError(
            f"bound attention provider {provider.name!r} rejects the request geometry"
        )
    return _run(provider, req)
