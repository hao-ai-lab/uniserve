"""Attention execution path planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "AttentionExecutionPlan",
    "AttentionRun",
]


@dataclass(frozen=True)
class AttentionRun:
    """Resolved attention path for one model-facing attention call."""

    path: Any
    preferred_backend: str
    kv_cache: Any | None
    update_cache: bool


class AttentionExecutionPlan:
    """Separates RadixAttention path resolution from path execution."""

    def __init__(self, path_enum: Any) -> None:
        self._path = path_enum

    def plan(
        self,
        owner: Any,
        ctx: Any,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache: Any,
        update_cache: bool,
        attn_mask: torch.Tensor | None,
    ) -> AttentionRun:
        path = self.resolve_path(
            owner,
            ctx,
            preferred,
            q,
            k,
            v,
            kv_cache=kv_cache,
            update_cache=update_cache,
            attn_mask=attn_mask,
        )
        return AttentionRun(
            path=path,
            preferred_backend=str(preferred),
            kv_cache=kv_cache,
            update_cache=bool(update_cache),
        )

    def resolve_path(
        self,
        owner: Any,
        ctx: Any,
        preferred: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache: Any,
        update_cache: bool,
        attn_mask: torch.Tensor | None,
    ) -> Any:
        if kv_cache is None:
            return self._path.DENSE
        paged_eligible = update_cache and callable(getattr(kv_cache, "layer_kv", None))
        if paged_eligible:
            if owner._can_run_paged_varlen_prefill(ctx, preferred, kv_cache, q, k, v):
                return self._path.PAGED_VARLEN
            if owner._can_run_contiguous_varlen_prefill(ctx, preferred, kv_cache, q, k, v):
                return self._path.CONTIGUOUS_VARLEN
            if owner._can_run_transient_paged_varlen(ctx, preferred, kv_cache, q, k, v):
                return self._path.TRANSIENT_PAGED_VARLEN
            if attn_mask is None and owner._can_run_paged_extend(ctx, preferred, kv_cache, q, k, v):
                return self._path.PAGED_EXTEND
        if paged_eligible:
            if owner.can_run_paged_attention(q, attn_mask, kv_cache=kv_cache, preferred=preferred, ctx=ctx):
                return self._path.PAGED_DECODE
            if owner._can_run_empty_paged_prefill(ctx, kv_cache, q, k, v):
                return self._path.EMPTY_PAGED_PREFILL
        return self._path.DENSE
