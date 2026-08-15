"""Stateless attention over an explicit forward context."""

from __future__ import annotations

import torch
from torch import nn

import uniserve_worker.ops as ops

from ..execution.forward_batch import (
    EmptyKvView,
    ForwardBatch,
    NoAttention,
    PackedAttentionPlan,
    PagedDecodePlan,
    PagedVarlenPlan,
)


class RadixAttention(nn.Module):
    """Attention equations selected solely by the supplied immutable plan."""

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
        """Execute one closed attention-plan variant.

        Cache access is available only through ``context.kv``; there is no
        alternate cache argument or ambient execution context.
        """

        effective_scale = self.scale if scale is None else float(scale)
        plan = context.attention
        if isinstance(plan, NoAttention):
            if not isinstance(context.kv, EmptyKvView):
                raise ValueError("dense attention must use an empty KV view")
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
                selection=plan.backends,
            )
        if attn_mask is not None:
            raise ValueError("paged attention plans do not accept a dense attention mask")
        if isinstance(context.kv, EmptyKvView):
            raise ValueError("paged attention requires a non-empty KV view")
        if isinstance(plan, PagedDecodePlan):
            return self._decode(q, k, v, context, plan, causal, effective_scale)
        if isinstance(plan, PagedVarlenPlan):
            return self._varlen(q, k, v, context, plan, causal, effective_scale)
        return self._packed(q, k, v, context, plan, effective_scale)

    def _decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        plan: PagedDecodePlan,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError("paged decode expects [rows, heads, dim] queries")
        if bool(causal) is not plan.causal:
            raise ValueError("paged decode causal semantics do not match its plan")
        if int(q.shape[0]) != int(plan.block_table.shape[0]):
            raise ValueError("paged decode query rows do not match its page table")
        k_cache, v_cache = context.kv.layer_kv(self.layer_id)
        out = ops.attention(
            ops.PagedDecodeAttention(
                q=q.unsqueeze(2),
                k=k_cache,
                v=v_cache,
                block_table=plan.block_table,
                cache_seqlens=plan.cache_seqlens,
                current_k=k.unsqueeze(2),
                current_v=v.unsqueeze(2),
                causal=causal,
                scale=scale,
                kv_cache=context.kv,
                ctx=context,
            ),
            selection=plan.backends,
        )
        return out.squeeze(2) if out.ndim == 4 else out

    def _varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: ForwardBatch,
        plan: PagedVarlenPlan,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        raw_tokens = sum(plan.query_lens_cpu)
        if bool(causal) is not plan.causal:
            raise ValueError("paged varlen causal semantics do not match its plan")
        if q.ndim != 3 or raw_tokens < 1 or raw_tokens > int(q.shape[0]):
            raise ValueError("paged varlen query geometry is invalid")
        q_run = q[:raw_tokens]
        k_run = k[:raw_tokens]
        v_run = v[:raw_tokens]
        context.kv.append_varlen(
            self.layer_id,
            k_run,
            v_run,
            plan.query_lens_cpu,
            block_table=plan.block_table,
            cache_seqlens=plan.cache_seqlens,
            query_offsets=plan.cu_seqlens_q,
        )
        k_cache, v_cache = context.kv.layer_kv(self.layer_id)
        out = ops.attention(
            ops.VarlenAttention(
                q=q_run,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=plan.cu_seqlens_q,
                cu_seqlens_k=plan.cu_seqlens_k,
                max_seqlen_q=plan.max_seqlen_q,
                max_seqlen_k=plan.max_seqlen_k,
                causal=causal,
                scale=scale,
                block_table=plan.block_table,
                kv_cache=context.kv,
                ctx=context,
            ),
            selection=plan.backends,
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
        plan: PackedAttentionPlan,
        scale: float,
    ) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError("packed attention expects [tokens, heads, dim] queries")
        context.kv.append_packed(
            self.layer_id,
            k,
            v,
            page_ids=plan.write_page_ids,
            page_offsets=plan.write_page_offsets,
            token_indices=plan.write_token_indices,
        )
        k_cache, v_cache = context.kv.layer_kv(self.layer_id)
        return ops.attention(
            ops.VisibleEndAttention(
                q=q,
                k=k_cache,
                v=v_cache,
                visible_end=plan.visible_end,
                scale=scale,
                cu_seqlens_q=plan.cu_seqlens_q,
                page_table=plan.page_table,
                seqused_k=plan.seqused_k,
                max_seqlen_q=plan.max_seqlen_q,
                max_seqlen_k=plan.max_seqlen_k,
                use_prefix_bounds=plan.use_prefix_bounds,
                fully_visible=plan.fully_visible,
                ctx=context,
            ),
            selection=plan.backends,
        )


__all__ = ["RadixAttention"]
