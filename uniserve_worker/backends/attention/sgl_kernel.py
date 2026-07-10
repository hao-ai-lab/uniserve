"""SGL kernel FlashAttention backend."""
from __future__ import annotations

import torch

from .base import AttentionCapabilities
from .registry import register_attention_backend

__all__ = [
    'SglKernelAttentionBackend',
]

try:  # pragma: no cover - optional CUDA package.
    from sgl_kernel.flash_attn import (
        flash_attn_varlen_func as _flash_attn_varlen_func,
    )
except Exception:  # pragma: no cover
    _flash_attn_varlen_func = None

try:  # pragma: no cover - optional CUDA package.
    from sgl_kernel.flash_attn import (
        flash_attn_with_kvcache as _flash_attn_with_kvcache,
    )
except Exception:  # pragma: no cover
    _flash_attn_with_kvcache = None


class SglKernelAttentionBackend:
    name = "sgl_kernel"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            segment_batched_cfg=False,
            mixed_mode=False,
            paged_kv=_flash_attn_with_kvcache is not None,
            varlen_attention=_flash_attn_varlen_func is not None,
            varlen_paged_kv=False,
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
    ) -> torch.Tensor:
        if _flash_attn_varlen_func is None:
            raise RuntimeError("sgl_kernel flash attention backend is not available")
        if attn_mask is not None:
            raise RuntimeError("sgl_kernel flash attention does not accept explicit dense masks")
        if q.ndim == 3:
            total = int(q.shape[0])
            cu = torch.tensor([0, total], dtype=torch.int32, device=q.device)
            return self.forward_varlen(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=total,
                max_seqlen_k=total,
                causal=causal,
                scale=scale,
            )
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("sgl_kernel flash attention expects q/k/v in [B,H,L,D] or [L,H,D] layout")
        batch = int(q.shape[0])
        q_len = int(q.shape[2])
        k_len = int(k.shape[2])
        if k_len != int(v.shape[2]):
            raise ValueError("sgl_kernel flash attention requires matching K/V lengths")
        q_flat = q.transpose(1, 2).contiguous().view(batch * q_len, int(q.shape[1]), int(q.shape[3]))
        k_flat = k.transpose(1, 2).contiguous().view(batch * k_len, int(k.shape[1]), int(k.shape[3]))
        v_flat = v.transpose(1, 2).contiguous().view(batch * k_len, int(v.shape[1]), int(v.shape[3]))
        cu_q = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=q.device)
        cu_k = torch.arange(0, (batch + 1) * k_len, k_len, dtype=torch.int32, device=q.device)
        out = self.forward_varlen(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
            causal=causal,
            scale=scale,
        )
        return out.view(batch, q_len, int(q.shape[1]), int(q.shape[3])).transpose(1, 2).contiguous()

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
    ) -> torch.Tensor:
        if _flash_attn_with_kvcache is None:
            raise RuntimeError("sgl_kernel paged KV attention backend is not available")
        if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
            raise ValueError("sgl_kernel paged cache expects k/v [pages, page, heads, dim]")
        mult = self.capabilities().paged_block_size_multiple
        if int(k_cache.shape[1]) % mult != 0:
            raise RuntimeError(f"sgl_kernel paged KV requires a {mult}-multiple page size")

        block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
        cache_seqlens = cache_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        q_blh, restore = _q_to_blh(q, block_table)
        if int(q_blh.shape[-1]) != int(k_cache.shape[-1]):
            raise ValueError("query head dim does not match paged KV cache")
        if int(q_blh.shape[0]) != int(cache_seqlens.shape[0]):
            raise ValueError("cache lengths must have one entry per paged attention row")

        k_blh = _kv_to_blh(k, rows=int(q_blh.shape[0])) if k is not None else None
        v_blh = _kv_to_blh(v, rows=int(q_blh.shape[0])) if v is not None else None
        if (k_blh is None) != (v_blh is None):
            raise ValueError("sgl_kernel paged update requires both k and v")
        if k_blh is not None:
            if v_blh is None:
                raise ValueError("sgl_kernel paged update requires a value tensor")
            if k_blh.shape != v_blh.shape:
                raise ValueError("current paged K/V tensors must have matching shapes")
            if k_blh.shape[0] != q_blh.shape[0]:
                raise ValueError("current K/V batch size must match q batch size")
            if k_blh.shape[2:] != k_cache.shape[2:]:
                raise ValueError("current K/V head geometry does not match paged cache")

        out = _flash_attn_with_kvcache(
            q_blh.contiguous(),
            k_cache,
            v_cache,
            k=k_blh.contiguous() if k_blh is not None else None,
            v=v_blh.contiguous() if v_blh is not None else None,
            cache_seqlens=cache_seqlens,
            page_table=block_table,
            softmax_scale=scale,
            causal=causal,
            ver=3,
        )
        if restore == "lhd":
            return out.squeeze(0).contiguous()
        if restore == "bhd":
            return out.squeeze(1).contiguous()
        return out.transpose(1, 2).contiguous()

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
    ) -> torch.Tensor:
        if _flash_attn_varlen_func is None:
            raise RuntimeError("sgl_kernel flash attention backend is not available")
        if block_table is not None:
            raise RuntimeError("sgl_kernel backend is only used for contiguous varlen prefill")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("sgl_kernel varlen attention expects q/k/v in [total,heads,dim] layout")
        return _flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous(),
            cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous(),
            int(max_seqlen_q),
            int(max_seqlen_k),
            softmax_scale=scale,
            causal=causal,
            ver=3,
        )


def _q_to_blh(q: torch.Tensor, block_table: torch.Tensor) -> tuple[torch.Tensor, str]:
    if q.ndim == 4:
        return q.transpose(1, 2).contiguous(), "bhld"
    if q.ndim != 3:
        raise ValueError("sgl_kernel paged q must be [B,H,D], [L,H,D], or [B,H,L,D]")
    rows = int(block_table.shape[0])
    if rows == 1 and int(q.shape[0]) != 1:
        return q.unsqueeze(0).contiguous(), "lhd"
    if rows == int(q.shape[0]):
        return q.unsqueeze(1).contiguous(), "bhd"
    raise ValueError("ragged batched paged attention requires the varlen backend path")


def _kv_to_blh(x: torch.Tensor | None, *, rows: int) -> torch.Tensor | None:
    if x is None:
        return None
    if x.ndim == 4:
        return x.transpose(1, 2).contiguous()
    if x.ndim != 3:
        raise ValueError("sgl_kernel current K/V must be [B,H,D], [L,H,D], or [B,H,L,D]")
    if rows == 1 and int(x.shape[0]) != 1:
        return x.unsqueeze(0).contiguous()
    if rows == int(x.shape[0]):
        return x.unsqueeze(1).contiguous()
    raise ValueError("current K/V rows do not match paged attention rows")


if _flash_attn_varlen_func is not None or _flash_attn_with_kvcache is not None:  # pragma: no cover
    register_attention_backend("sgl_kernel", SglKernelAttentionBackend())
