"""FlashAttention-4 CUTE backend.

The FA4 reference kernel is an attention-only kernel: unlike FA2's
``flash_attn_with_kvcache`` it does not append current K/V into the cache.
This backend keeps that contract explicit by writing the current K/V span into
the worker-owned paged cache before calling the FA4 paged forward.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from importlib import import_module
from typing import Any, Protocol

import torch

from ...execution.forward_batch import ForwardBatch
from ...foundation.sizing import ceil_div
from ..paged_kv_math import paged_kv_write, write_locations
from .base import AttentionCapabilities
from .layout import QKVLayout, normalize_kv, normalize_to

__all__ = [
    "Fa4CuteAttentionBackend",
]

_IMPORT_ERROR: Exception | None = None
# Authoritative table of (q, k, v) head-dim geometries the FA4 unified trunk
# kernel accepts. This is the single source of truth: it is published through
# AttentionCapabilities.trunk_geometries so the registry selector and the
# model-facing RadixAttention layer query it instead of duplicating the literal.
_SUPPORTED_TRUNK_GEOMETRIES: frozenset[tuple[int, int, int]] = frozenset(
    {
        (64, 64, 64),
        (96, 96, 96),
        (128, 128, 128),
        (192, 192, 128),
    }
)
# FA4-cute kernel launch tuning, shared by every forward variant so the tile and
# thread-count tuning lives in exactly one place.
_FA4_TILE_MN = (128, 128)
_FA4_NUM_THREADS = 384
_PREFIX_BOUNDS_CACHE_LIMIT = 16
_PREFIX_BOUNDS_CACHE: OrderedDict[
    tuple[int, int | None, int, int | None, int],
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor],
] = OrderedDict()


class _ComputePrefixBounds(Protocol):
    def __call__(self, visible_end: torch.Tensor, *, q_tile_size: int) -> torch.Tensor: ...


class _ComputePrefixBoundsVarlen(Protocol):
    def __call__(
        self,
        visible_end: torch.Tensor,
        seqlens_q: torch.Tensor,
        *,
        q_tile_size: int,
        num_q_tiles: int | None = None,
    ) -> torch.Tensor: ...


_compute_prefix_bounds: _ComputePrefixBounds | None
_compute_prefix_bounds_varlen: _ComputePrefixBoundsVarlen | None
try:  # pragma: no cover - optional CUDA package.
    mm_attn_varlen = import_module("uniserve_kernel.mm_attn_varlen")

    _fa4_flash_attn_fwd = mm_attn_varlen.flash_attn_fwd
    _compute_prefix_bounds = mm_attn_varlen.compute_prefix_bounds
    _compute_prefix_bounds_varlen = mm_attn_varlen.compute_prefix_bounds_varlen
    _hybrid_multimodal_mask = mm_attn_varlen.hybrid_multimodal_mask
    _IMPORT_ERROR = mm_attn_varlen.import_error()
    _fa4_accepts_prefix_bounds = (
        "prefix_bounds" in inspect.signature(_fa4_flash_attn_fwd).parameters
        if _fa4_flash_attn_fwd is not None
        else False
    )
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    _fa4_flash_attn_fwd = None
    _compute_prefix_bounds = None
    _compute_prefix_bounds_varlen = None
    _hybrid_multimodal_mask = None
    _fa4_accepts_prefix_bounds = False


class Fa4CuteAttentionBackend:
    """FlashAttention-4 CUTE backend with explicit paged KV writes before forward."""

    name = "fa4_cute"

    def capabilities(self) -> AttentionCapabilities:
        available = _fa4_flash_attn_fwd is not None
        return AttentionCapabilities(
            available=available,
            paged_kv=available,
            visible_end=available,
            visible_end_cuda_graph=available,
            paged_block_size_multiple=1,
            trunk_geometries=_SUPPORTED_TRUNK_GEOMETRIES,
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
        if attn_mask is not None:
            raise RuntimeError("fa4_cute backend does not accept explicit dense masks")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("fa4_cute backend expects q/k/v in [B, H, L, D] layout")
        _validate_unified_trunk_geometry(q.shape[-1], k.shape[-1], v.shape[-1], scale=scale)
        _require_fa4()
        out = _fa4_output(
            _fa4_flash_attn_fwd(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                softmax_scale=scale,
                causal=causal,
                tile_mn=_FA4_TILE_MN,
                num_threads=_FA4_NUM_THREADS,
            )
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
        q_blh, restore = normalize_to(q, QKVLayout.BLHD)
        _validate_unified_trunk_geometry(
            q_blh.shape[-1],
            k_cache.shape[-1],
            v_cache.shape[-1],
            scale=scale,
        )
        _require_fa4()
        cache_seqlens = cache_seqlens.to(device=q_blh.device, dtype=torch.int32).contiguous()
        block_table = block_table.to(device=q_blh.device, dtype=torch.int32).contiguous()
        live_seqlens = cache_seqlens.clone()
        if k is not None or v is not None:
            if k is None or v is None:
                raise ValueError("fa4_cute paged update requires both k and v")
            k_blh = normalize_kv(k, QKVLayout.BLHD)
            v_blh = normalize_kv(v, QKVLayout.BLHD)
            if k_blh.shape[:3] != v_blh.shape[:3] or k_blh.shape[0] != q_blh.shape[0]:
                raise ValueError("current paged K/V must match q batch and each other")
            _write_paged_kv_cache(k_cache, v_cache, block_table, cache_seqlens, k_blh, v_blh)
            live_seqlens += int(k_blh.shape[1])
        max_seqlen_k = _metadata_context_len(None if context is None else context.attention)
        if max_seqlen_k <= 0:
            raise ValueError(
                "fa4_cute paged forward requires a positive host-known "
                "plan.max_context_len; a plan without one is a scheduling bug"
            )

        out = _fa4_output(
            _fa4_flash_attn_fwd(
                q_blh,
                k_cache,
                v_cache,
                page_table=block_table,
                seqused_k=live_seqlens,
                max_seqlen_q=int(q_blh.shape[1]),
                max_seqlen_k=max_seqlen_k,
                softmax_scale=scale,
                causal=causal,
                tile_mn=_FA4_TILE_MN,
                num_threads=_FA4_NUM_THREADS,
            )
        )
        return restore.apply(out)

    def forward_visible_end(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        visible_end: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        scale: float | None = None,
        use_prefix_bounds: bool = False,
        fully_visible: bool = False,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor:
        """Run the hybrid ``visible_end`` mask path.

        ``q`` may be fixed [B,L,H,D] or varlen [total,H,D].  ``visible_end`` is
        always padded [B,max_q] and indexed locally per sequence.
        """

        del context
        _validate_unified_trunk_geometry(q.shape[-1], k.shape[-1], v.shape[-1], scale=scale)
        _require_fa4()
        visible_end = visible_end.to(device=q.device, dtype=torch.int32).contiguous()
        setattr(visible_end, "__leading_dim__", 1)
        setattr(visible_end, "__assumed_align__", 4)
        kwargs: dict[str, Any] = {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "page_table": page_table,
            "seqused_k": seqused_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": scale,
            "tile_mn": _FA4_TILE_MN,
            "num_threads": _FA4_NUM_THREADS,
        }
        if fully_visible:
            return _fa4_output(_fa4_flash_attn_fwd(q, k, v, **kwargs))
        kwargs["aux_tensors"] = [visible_end]
        if use_prefix_bounds and _fa4_accepts_prefix_bounds:
            # FA4 query-tile width (kernel-ABI); unrelated to the paged block size.
            q_tile = 256
            qhead_per_kvhead = int(q.shape[-2]) // int(k.shape[-2])
            prefix_bounds = _cached_prefix_bounds(
                visible_end,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                qhead_per_kvhead=qhead_per_kvhead,
                q_tile_size=q_tile,
            )
            kwargs["prefix_bounds"] = prefix_bounds
        else:
            kwargs["mask_mod"] = _hybrid_multimodal_mask
        return _fa4_output(_fa4_flash_attn_fwd(q, k, v, **kwargs))


def _require_fa4() -> None:
    if _fa4_flash_attn_fwd is None:
        detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "fa4_cute backend is not available. Install the uniserve-kernel "
            "provider package with its CUTE runtime dependencies"
            f"{detail}"
        )


def _fa4_output(result: Any) -> torch.Tensor:
    if isinstance(result, tuple):
        return result[0]
    return result


def _cached_prefix_bounds(
    visible_end: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None,
    max_seqlen_q: int | None,
    qhead_per_kvhead: int,
    q_tile_size: int,
) -> torch.Tensor:
    compute_prefix_bounds = _compute_prefix_bounds
    compute_prefix_bounds_varlen = _compute_prefix_bounds_varlen
    if compute_prefix_bounds is None or compute_prefix_bounds_varlen is None:
        detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
        raise RuntimeError(f"FA4 prefix-bound provider is unavailable{detail}")

    key = (
        id(visible_end),
        None if cu_seqlens_q is None else id(cu_seqlens_q),
        int(qhead_per_kvhead),
        None if max_seqlen_q is None else int(max_seqlen_q),
        int(q_tile_size),
    )
    cached = _PREFIX_BOUNDS_CACHE.get(key)
    if cached is not None:
        cached_visible, cached_cu_q, prefix_bounds = cached
        if cached_visible is visible_end and cached_cu_q is cu_seqlens_q:
            _PREFIX_BOUNDS_CACHE.move_to_end(key)
            return prefix_bounds

    bounds_visible_end = visible_end
    bounds_max_seqlen_q = max_seqlen_q
    if qhead_per_kvhead > 1:
        # FA4's packed-GQA scheduler counts query tiles in head-expanded row
        # space. Prefix bounds use that same tile space; prefix_visible_end
        # remains indexed by logical q rows in the kernel mask.
        bounds_visible_end = visible_end.repeat_interleave(
            qhead_per_kvhead,
            dim=1,
        ).contiguous()
        if bounds_max_seqlen_q is not None:
            bounds_max_seqlen_q = int(bounds_max_seqlen_q) * int(qhead_per_kvhead)

    if cu_seqlens_q is None:
        prefix_bounds = compute_prefix_bounds(bounds_visible_end, q_tile_size=q_tile_size)
    else:
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
        if qhead_per_kvhead > 1:
            seqlens_q = seqlens_q * int(qhead_per_kvhead)
        tiles = None if bounds_max_seqlen_q is None else ceil_div(bounds_max_seqlen_q, q_tile_size)
        prefix_bounds = compute_prefix_bounds_varlen(
            bounds_visible_end,
            seqlens_q,
            q_tile_size=q_tile_size,
            num_q_tiles=tiles,
        )

    _PREFIX_BOUNDS_CACHE[key] = (visible_end, cu_seqlens_q, prefix_bounds)
    _PREFIX_BOUNDS_CACHE.move_to_end(key)
    while len(_PREFIX_BOUNDS_CACHE) > _PREFIX_BOUNDS_CACHE_LIMIT:
        _PREFIX_BOUNDS_CACHE.popitem(last=False)
    return prefix_bounds


def _validate_unified_trunk_geometry(
    q_head_dim: int,
    k_head_dim: int,
    v_head_dim: int,
    *,
    scale: float | None = None,
) -> None:
    geometry = (int(q_head_dim), int(k_head_dim), int(v_head_dim))
    if geometry not in _SUPPORTED_TRUNK_GEOMETRIES:
        raise RuntimeError(
            "fa4_cute unified trunk path only supports head geometries "
            f"{sorted(_SUPPORTED_TRUNK_GEOMETRIES)}, got {geometry}. "
            "Vision/VAE attention must use a separate backend."
        )
    if scale is not None:
        expected = float(int(q_head_dim) ** -0.5)
        if abs(float(scale) - expected) > 1e-6:
            raise RuntimeError(
                "fa4_cute unified trunk path received an incompatible softmax scale "
                f"{float(scale)} for q head dim {int(q_head_dim)}"
            )


def _write_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
) -> None:
    if k_cache.shape != v_cache.shape:
        raise ValueError("paged K/V cache tensors must have identical shapes")
    if k_current.shape != v_current.shape:
        raise ValueError("current K/V tensors must have identical shapes")
    if k_cache.ndim != 4 or k_current.ndim != 4:
        raise ValueError("paged cache must be [pages,page,heads,dim] and current K/V [B,L,H,D]")
    batch, tokens, heads, head_dim = k_current.shape
    if k_cache.shape[2:] != (heads, head_dim):
        raise ValueError("current K/V head geometry does not match paged cache")
    if block_table.shape[0] != batch or cache_seqlens.shape[0] != batch:
        raise ValueError("block table and cache lengths must have one row per batch")
    if tokens == 0 or batch == 0:
        return
    page_size = int(k_cache.shape[1])
    device = k_cache.device

    # Vectorized cache append: compute the destination slot for every
    # (batch, token) pair on-device and scatter in a single index_copy_, so the
    # decode hot path incurs no per-row Python loop and no ``.item()`` syncs.
    cache_seqlens = cache_seqlens.to(device=device, dtype=torch.int64)
    block_table = block_table.to(device=device, dtype=torch.int64)
    # Absolute positions per (batch, token): [batch, tokens].
    token_offsets = torch.arange(tokens, device=device, dtype=torch.int64)
    positions = cache_seqlens.unsqueeze(1) + token_offsets.unsqueeze(0)
    # A page slot overflowing ``block_table`` or an out-of-range physical page id
    # is left to surface as a CUDA index error from gather/index_copy_. Validating
    # those on-device here would require ``.item()`` syncs every decode step; the
    # engine sizes block tables so those indices stay in range.
    page_ids, offsets = write_locations(block_table, positions, page_size)
    paged_kv_write(k_cache, v_cache, page_ids, offsets, k_current, v_current)


def _metadata_context_len(plan: object | None) -> int:
    value = getattr(plan, "max_context_len", 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
