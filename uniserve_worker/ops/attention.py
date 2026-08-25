"""Capability-driven attention providers."""
from __future__ import annotations

import torch

from ..backends.attention.base import AttentionCapabilities
from ..execution.forward_batch import AttentionSelection, ForwardMode
from .core import Dispatcher, Operator
from .requests import (
    AttentionReq,
    PagedDecodeAttention,
    VarlenAttention,
    VisibleEndAttention,
)


def _capabilities(backend) -> AttentionCapabilities:
    raw = getattr(backend, "capabilities", None)
    if not callable(raw):
        return AttentionCapabilities()
    caps = raw()
    if isinstance(caps, AttentionCapabilities):
        return caps
    return AttentionCapabilities(
        available=bool(getattr(caps, "available", True)),
        paged_kv=bool(getattr(caps, "paged_kv", False)),
        varlen_attention=bool(getattr(caps, "varlen_attention", False)),
        varlen_paged_kv=bool(getattr(caps, "varlen_paged_kv", False)),
        requires_paged_varlen=bool(getattr(caps, "requires_paged_varlen", False)),
        visible_end=bool(getattr(caps, "visible_end", False)),
        segmented_attention=bool(getattr(caps, "segmented_attention", False)),
        segmented_attention_cuda_graph=bool(
            getattr(caps, "segmented_attention_cuda_graph", False)
        ),
        paged_block_size_multiple=int(getattr(caps, "paged_block_size_multiple", 1) or 1),
        min_head_dim=int(getattr(caps, "min_head_dim", 1) or 1),
        paged_decode_only=bool(getattr(caps, "paged_decode_only", False)),
        paged_varlen_cuda_graph=bool(getattr(caps, "paged_varlen_cuda_graph", False)),
        visible_end_cuda_graph=bool(getattr(caps, "visible_end_cuda_graph", False)),
        trunk_geometries=frozenset(getattr(caps, "trunk_geometries", ()) or ()),
        cuda_only=bool(getattr(caps, "cuda_only", False)),
        min_cuda_capability=getattr(caps, "min_cuda_capability", None),
        dense_ranks=frozenset(getattr(caps, "dense_ranks", (3, 4)) or (3, 4)),
        accepts_dense_mask=bool(getattr(caps, "accepts_dense_mask", False)),
    )


def _head_dim(req: AttentionReq) -> int:
    return int(req.q.shape[-1])


def _kv_dims(req: AttentionReq) -> tuple[int, int]:
    if isinstance(req, VisibleEndAttention) and req.prefix_k is not None and req.prefix_v is not None:
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
        paged_k = req.prefix_k if isinstance(req, VisibleEndAttention) and req.prefix_k is not None else req.k
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
        if getattr(plan, "forward_mode", None) is ForwardMode.PAGED_DECODE and len(query_lens) == int(req.q.shape[0]):
            return all(int(length) == 1 for length in query_lens)
        return int(req.q.shape[0]) == 1
    if req.q.ndim == 4:
        return int(req.q.shape[2]) == 1
    return False


class AttentionProvider(Operator):
    def __init__(self, backend) -> None:
        super().__init__(backend.name, "attention")
        self.backend = backend

    def launch_name(self, req: AttentionReq) -> str:
        if isinstance(req, VarlenAttention) and req.block_table is not None:
            return f"{self.name}_paged_varlen"
        return self.name

    def can_run(self, req: AttentionReq) -> bool:
        caps = _capabilities(self.backend)
        if not caps.available:
            return False
        if not _device_supported(caps, req.q):
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
            if caps.requires_paged_varlen:
                return False
            return bool(caps.varlen_attention)
        if isinstance(req, PagedDecodeAttention):
            if caps.paged_decode_only and not _is_one_token_decode(req):
                return False
            return bool(caps.paged_kv) and _paged_storage_supported(caps, req)
        if caps.paged_decode_only:
            return False
        if req.attn_mask is not None and not caps.accepts_dense_mask:
            return False
        if req.q.ndim != req.k.ndim or req.q.ndim != req.v.ndim:
            return False
        return int(req.q.ndim) in caps.dense_ranks

    def run(self, req: AttentionReq) -> torch.Tensor:
        if isinstance(req, VisibleEndAttention) and req.prefix_k is not None:
            if req.prefix_v is None or req.prefix_lens is None or req.cu_seqlens_q is None:
                raise ValueError("segmented attention metadata is incomplete")
            return self.backend.forward_segmented(
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
            return self.backend.forward_visible_end(
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
            return self.backend.forward_varlen(
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
            return self.backend.forward_paged(
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
        return self.backend.forward(
            req.q,
            req.k,
            req.v,
            causal=req.causal,
            scale=req.scale,
            attn_mask=req.attn_mask,
            context=req.ctx,
        )


def attention_dispatcher(selection: AttentionSelection) -> Dispatcher[AttentionReq, torch.Tensor]:
    return Dispatcher(
        "attention",
        [AttentionProvider(backend) for backend in selection.providers],
    )


def can_run_attention(selection: AttentionSelection, req: AttentionReq) -> bool:
    return any(provider.can_run(req) for provider in attention_dispatcher(selection).ordered())


def _segmented_graph_provider(selection: AttentionSelection) -> str | None:
    for backend in selection.providers:
        caps = _capabilities(backend)
        if caps.available and caps.segmented_attention and caps.segmented_attention_cuda_graph:
            return str(backend.name)
    return None


def run_attention(selection: AttentionSelection, req: AttentionReq) -> torch.Tensor:
    override = None
    if isinstance(req, VisibleEndAttention) and req.prefix_k is not None:
        override = _segmented_graph_provider(selection)
    return attention_dispatcher(selection).run(req, override=override)
