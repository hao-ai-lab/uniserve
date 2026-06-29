"""Portable torch SDPA attention backend."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionCapabilities
from .registry import register_attention_backend

__all__ = [
    'TorchSDPAAttentionBackend',
]


class TorchSDPAAttentionBackend:
    name = "torch_sdpa"

    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities(
            segment_batched_cfg=True,
            mixed_mode=True,
            paged_kv=False,
            tree_verify=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        # The portable backend tolerates a missing scale (defers to SDPA's
        # default) so direct callers that omit it still work; the layer always
        # passes the effective scale.
        scale: float | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.ndim == 3:
            return self._forward_lhd(q, k, v, causal=causal, scale=scale, attn_mask=attn_mask)
        if q.ndim == 4:
            return self._forward_bhld(q, k, v, causal=causal, scale=scale, attn_mask=attn_mask)
        raise ValueError(f"unsupported q rank {q.ndim}")

    def _forward_lhd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        lq, n_heads, _ = q.shape
        lk, _, _ = k.shape
        k, v = _expand_gqa(k, v, n_heads, head_axis=1)
        q4 = q.permute(1, 0, 2).unsqueeze(0)
        k4 = k.permute(1, 0, 2).unsqueeze(0)
        v4 = v.permute(1, 0, 2).unsqueeze(0)
        mask = attn_mask
        use_is_causal = False
        if mask is None and causal:
            cache_len = lk - lq
            if cache_len == 0:
                # No KV-cache offset: top-left causal masking is exactly what
                # SDPA's is_causal flag expresses, so let it use the fused
                # causal kernel instead of materializing a dense [Lq, Lk] mask.
                use_is_causal = True
            else:
                iq = torch.arange(lq, device=q.device).unsqueeze(1)
                ik = torch.arange(lk, device=q.device).unsqueeze(0)
                bad = ik > (cache_len + iq)
                mask = torch.zeros(lq, lk, dtype=q.dtype, device=q.device)
                mask.masked_fill_(bad, float("-inf"))
        mask = _normalize_mask(mask, q)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None]
        out = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask, is_causal=use_is_causal, scale=scale
        )
        return out.squeeze(0).permute(1, 0, 2)

    def _forward_bhld(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        k, v = _expand_gqa(k, v, q.shape[1], head_axis=1)
        mask = _normalize_mask(attn_mask, q)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=causal and mask is None,
            scale=scale,
        )
        return out


def _expand_gqa(
    k: torch.Tensor,
    v: torch.Tensor,
    n_heads: int,
    *,
    head_axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat-interleave the KV head groups to match ``n_heads`` query heads.

    The shared grouped-query expansion for both the packed ``[L, H, D]`` and
    batched ``[B, H, L, D]`` paths. ``k``/``v`` are returned unchanged when the
    head counts already match.
    """
    n_kv_heads = int(k.shape[head_axis])
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"num heads {n_heads} is not divisible by kv heads {n_kv_heads}")
    if n_heads == n_kv_heads:
        return k, v
    rep = n_heads // n_kv_heads
    return k.repeat_interleave(rep, dim=head_axis), v.repeat_interleave(rep, dim=head_axis)


def _normalize_mask(mask: torch.Tensor | None, q: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.device != q.device:
        mask = mask.to(device=q.device)
    if mask.dtype.is_floating_point and mask.dtype != q.dtype:
        mask = mask.to(dtype=q.dtype)
    return mask


register_attention_backend("torch_sdpa", TorchSDPAAttentionBackend())
