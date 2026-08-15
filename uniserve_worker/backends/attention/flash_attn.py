"""flash-attn attention backend."""

from __future__ import annotations

import torch

from ...execution.forward_batch import ForwardBatch
from .base import AttentionCapabilities
from .layout import QKVLayout, normalize_kv, normalize_to

__all__ = [
    "FlashAttentionBackend",
]

try:  # pragma: no cover - optional CUDA package.
    from flash_attn import flash_attn_func as _flash_attn_func
except ImportError:  # pragma: no cover
    _flash_attn_func = None

try:  # pragma: no cover - optional CUDA package/version.
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except ImportError:  # pragma: no cover
    _flash_attn_varlen_func = None

try:  # pragma: no cover - optional CUDA package/version.
    from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
except ImportError:  # pragma: no cover
    _flash_attn_with_kvcache = None


class FlashAttentionBackend:
    name = "flash_attn"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            available=any(
                value is not None
                for value in (_flash_attn_func, _flash_attn_varlen_func, _flash_attn_with_kvcache)
            ),
            paged_kv=_flash_attn_with_kvcache is not None,
            varlen_attention=_flash_attn_varlen_func is not None,
            varlen_paged_kv=_flash_attn_varlen_func is not None,
            paged_block_size_multiple=256,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None = None,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor:
        del context
        if _flash_attn_func is None:
            raise RuntimeError("flash-attn backend is not available")
        if attn_mask is not None:
            raise RuntimeError("flash-attn backend does not accept explicit dense masks")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("flash-attn backend expects q/k/v in [B, H, L, D] layout")
        out = _flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
        )
        return out.transpose(1, 2).contiguous()

    def forward_paged(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        causal: bool,
        scale: float,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor:
        del context
        if _flash_attn_with_kvcache is None:
            raise RuntimeError("flash-attn paged KV kernel is not available")
        q_blh, restore = normalize_to(q, QKVLayout.BLHD)
        mult = self.capabilities().paged_block_size_multiple
        if int(k_cache.shape[1]) % mult != 0:
            raise RuntimeError(f"flash-attn paged KV requires a {mult}-multiple page size")
        block_table = block_table.to(device=q_blh.device, dtype=torch.int32).contiguous()
        cache_seqlens = cache_seqlens.to(device=q_blh.device, dtype=torch.int32).contiguous()

        k_blh = normalize_kv(k, QKVLayout.BLHD) if k is not None else None
        v_blh = normalize_kv(v, QKVLayout.BLHD) if v is not None else None
        out = _flash_attn_with_kvcache(
            q_blh.contiguous(),
            k_cache,
            v_cache,
            k=k_blh.contiguous() if k_blh is not None else None,
            v=v_blh.contiguous() if v_blh is not None else None,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=scale,
            causal=causal,
        )
        return restore.apply(out)

    def forward_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        scale: float,
        block_table: torch.Tensor | None = None,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor:
        del context
        if _flash_attn_varlen_func is None:
            raise RuntimeError("flash-attn varlen kernel is not available")
        if q.ndim != 3:
            raise ValueError("flash-attn varlen expects q in [total, heads, dim] layout")
        if block_table is None:
            if k.ndim != 3 or v.ndim != 3:
                raise ValueError("flash-attn varlen expects k/v in [total, heads, dim] layout")
        elif k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "flash-attn paged varlen expects k/v caches in "
                "[num_blocks, page, heads, dim] layout"
            )
        if block_table is not None:
            block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
        return _flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q.contiguous(),
            cu_seqlens_k.contiguous(),
            int(max_seqlen_q),
            int(max_seqlen_k),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
            block_table=block_table,
        )
