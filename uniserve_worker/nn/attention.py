"""Attention over one row-aligned forward tensor contract."""

from __future__ import annotations

import torch
from torch import nn

import uniserve_worker.ops as ops

from ..execution.forward_batch import AttentionSelection, ForwardBatch, ForwardMode
from ..runtime.cache_pool import CachePool


class RadixAttention(nn.Module):
    """Execute dense or paged attention using startup-owned resources."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        if min(num_heads, num_kv_heads, head_dim) < 1:
            raise ValueError("attention geometry must be positive")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.layer_id = int(layer_id)
        self.scale = self.head_dim**-0.5
        self._cache_pool: CachePool | None = None
        self._selection: AttentionSelection | None = None

    def bind(self, cache_pool: CachePool, selection: AttentionSelection) -> None:
        self._cache_pool = cache_pool
        self._selection = selection

    @property
    def selection(self) -> AttentionSelection:
        if self._selection is None:
            raise RuntimeError("attention module has not been bound to a startup backend")
        return self._selection

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        *,
        causal: bool,
        scale: float | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        effective_scale = self.scale if scale is None else float(scale)
        selection = self._selection
        if selection is None:
            raise RuntimeError("attention module has not been bound to a startup backend")
        if context.forward_mode is ForwardMode.DENSE:
            return ops.attention(
                ops.DenseAttention(
                    q=q,
                    k=k,
                    v=v,
                    causal=causal,
                    scale=effective_scale,
                    attn_mask=attn_mask,
                    ctx=context,
                ),
                selection=selection,
            )
        if attn_mask is not None:
            raise ValueError("paged attention does not accept a dense attention mask")
        pool = self._cache_pool
        if pool is None or context.block_table is None:
            raise RuntimeError("paged attention has no bound physical cache")
        if context.forward_mode is ForwardMode.PAGED_DECODE:
            return self._decode(q, k, v, context, causal, effective_scale, selection, pool)
        if context.forward_mode is ForwardMode.PAGED_VARLEN:
            return self._varlen(q, k, v, context, causal, effective_scale, selection, pool)
        if context.forward_mode is ForwardMode.REQUEST_INDEXED_DECODE:
            raise ValueError("request-indexed decode metadata was not staged")
        return self._packed(q, k, v, context, effective_scale, selection, pool)

    def _decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        causal: bool,
        scale: float,
        selection: AttentionSelection,
        pool: CachePool,
    ) -> torch.Tensor:
        if q.ndim != 3 or int(q.shape[0]) != int(context.block_table.shape[0]):
            raise ValueError("paged decode query rows do not match its page table")
        if context.kv_lens is None:
            raise ValueError("paged decode requires resulting KV lengths")
        if context.has_cache_writes:
            pool.write_locations(self.layer_id, context.out_cache_loc, k, v)
        k_cache, v_cache = pool.layer_cache(self.layer_id, context.group_id)
        out = ops.attention(
            ops.PagedDecodeAttention(
                q=q.unsqueeze(2),
                k=k_cache,
                v=v_cache,
                block_table=context.block_table,
                cache_seqlens=context.kv_lens,
                causal=causal,
                scale=scale,
                ctx=context,
            ),
            selection=selection,
        )
        return out.squeeze(2) if out.ndim == 4 else out

    def _varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        causal: bool,
        scale: float,
        selection: AttentionSelection,
        pool: CachePool,
    ) -> torch.Tensor:
        raw_tokens = sum(context.query_lens_cpu)
        if (
            q.ndim != 3
            or raw_tokens < 1
            or raw_tokens > int(q.shape[0])
            or context.cu_seqlens_q is None
            or context.cu_seqlens_k is None
        ):
            raise ValueError("paged varlen query geometry is invalid")
        q_run, k_run, v_run = q[:raw_tokens], k[:raw_tokens], v[:raw_tokens]
        if context.has_cache_writes:
            pool.write_locations(self.layer_id, context.out_cache_loc[:raw_tokens], k_run, v_run)
        k_cache, v_cache = pool.layer_cache(self.layer_id, context.group_id)
        out = ops.attention(
            ops.VarlenAttention(
                q=q_run,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                causal=causal,
                scale=scale,
                block_table=context.block_table,
                ctx=context,
            ),
            selection=selection,
        )
        if raw_tokens == int(q.shape[0]):
            return out
        padded = q.new_zeros(q.shape)
        padded[:raw_tokens].copy_(out)
        return padded

    def _packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        scale: float,
        selection: AttentionSelection,
        pool: CachePool,
    ) -> torch.Tensor:
        if (
            q.ndim != 3
            or context.cu_seqlens_q is None
            or context.visible_end is None
        ):
            raise ValueError("packed attention geometry is invalid")
        if context.has_cache_writes:
            pool.write_locations(self.layer_id, context.out_cache_loc, k, v)
        k_cache, v_cache = pool.layer_cache(self.layer_id, context.group_id)
        return ops.attention(
            ops.VisibleEndAttention(
                q=q,
                k=k,
                v=v,
                visible_end=context.visible_end,
                scale=scale,
                cu_seqlens_q=context.cu_seqlens_q,
                page_table=context.block_table,
                fully_visible=context.fully_visible,
                prefix_k=k_cache,
                prefix_v=v_cache,
                prefix_lens=context.seq_lens,
                ctx=context,
            ),
            selection=selection,
        )


def bind_attention_modules(
    model: nn.Module,
    cache_pool: CachePool,
    selection: AttentionSelection,
) -> None:
    for module in model.modules():
        if isinstance(module, RadixAttention):
            module.bind(cache_pool, selection)


__all__ = ["RadixAttention", "bind_attention_modules"]
