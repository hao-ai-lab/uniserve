"""Attention backend interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

__all__ = [
    'AttentionCapabilities',
    'AttentionBackend',
    'PagedAttentionBackend',
    'VarlenAttentionBackend',
    'VisibleEndAttentionBackend',
]


@dataclass(frozen=True)
class AttentionCapabilities:
    # Whether the provider's runtime dependency is currently usable. Optional
    # provider modules may still register an unavailable backend so explicit
    # selection and diagnostics can name it, but dispatch must not route generic
    # dense/paged attention through a backend whose kernel import failed.
    available: bool = True
    segment_batched_cfg: bool = False
    mixed_mode: bool = False
    paged_kv: bool = False
    varlen_attention: bool = False
    varlen_paged_kv: bool = False
    # True when ``forward_varlen`` only supports paged KV cache inputs and must
    # not be selected for contiguous q/k/v varlen prefill.
    requires_paged_varlen: bool = False
    visible_end: bool = False
    tree_verify: bool = False
    paged_block_size_multiple: int = 1
    min_head_dim: int = 1
    # The backend's paged-KV kernel only supports single-token decode (one query
    # token per row), so it must not be dispatched for multi-token paged prefill.
    # Backends whose paged kernel handles arbitrary query lengths leave this False.
    paged_decode_only: bool = False
    # The backend's paged-varlen prefill path can be captured directly in a CUDA
    # graph because it consumes live tensor inputs and does not bake mutable host
    # wrapper plan state that another request can later overwrite.
    paged_varlen_cuda_graph: bool = False
    # (q, k, v) head-dim geometries the backend's kernel can run. Empty means the
    # backend imposes no fixed-geometry restriction (the common case); a non-empty
    # set declares the exact tuples a geometry-restricted kernel (e.g. fa4_cute's
    # unified trunk path) accepts, so callers and the registry can pre-emptively
    # avoid dispatching shapes the kernel would hard-reject. This is the single
    # authoritative source for that table.
    trunk_geometries: frozenset[tuple[int, int, int]] = field(default_factory=frozenset)

    def supports_trunk_geometry(self, q_head_dim: int, k_head_dim: int, v_head_dim: int) -> bool:
        """Whether the backend kernel accepts this exact ``(q, k, v)`` geometry.

        An empty :attr:`trunk_geometries` means the backend is not
        geometry-restricted and accepts any shape it is otherwise capable of.
        """
        if not self.trunk_geometries:
            return True
        return (int(q_head_dim), int(k_head_dim), int(v_head_dim)) in self.trunk_geometries

    def supports_trunk_head_dim(self, q_head_dim: int) -> bool:
        """Whether the backend kernel can run *any* geometry with this q head dim.

        Used on dispatch paths where only the query head dim is known before the
        cache shapes are resolved. An empty :attr:`trunk_geometries` means the
        backend is not geometry-restricted and accepts any q head dim.
        """
        if not self.trunk_geometries:
            return True
        dim = int(q_head_dim)
        return any(q_dim == dim for q_dim, _k, _v in self.trunk_geometries)


class AttentionBackend(Protocol):
    """Universal attention-backend contract.

    Only the members declared here are mandatory for *every* registered
    backend (e.g. ``torch_sdpa`` implements just these). The paged- and
    varlen-specific entry points are intentionally *not* part of this base
    Protocol because they are optional and capability-gated: a backend
    implements ``forward_paged`` only when it advertises
    ``capabilities().paged_kv`` and ``forward_varlen`` only when it advertises
    ``capabilities().varlen_attention``. Callers must narrow via the
    corresponding capability flag (or the ``PagedAttentionBackend`` /
    ``VarlenAttentionBackend`` Protocols below) before invoking those methods.
    """

    name: str

    def capabilities(self) -> AttentionCapabilities:
        ...

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
        ...


class PagedAttentionBackend(AttentionBackend, Protocol):
    """Backends that support paged KV decode.

    Implemented only by backends reporting ``capabilities().paged_kv`` (e.g.
    ``flash_attn``, ``fa4_cute``, ``flashinfer``, ``sgl_kernel``). ``paged_kv``
    being ``True`` is the precondition for calling ``forward_paged``; backends
    such as ``torch_sdpa`` report ``paged_kv=False`` and do not satisfy this
    Protocol.
    """

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
        ...


class VarlenAttentionBackend(AttentionBackend, Protocol):
    """Backends that support variable-length (cu_seqlens) prefill.

    Implemented only by backends reporting ``capabilities().varlen_attention``
    (e.g. ``flash_attn``, ``flashinfer``, ``sgl_kernel``). ``varlen_attention``
    being ``True`` is the precondition for calling ``forward_varlen``; backends
    such as ``torch_sdpa`` and ``fa4_cute`` report ``varlen_attention=False``
    and do not satisfy this Protocol.
    """

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
        ...


class VisibleEndAttentionBackend(AttentionBackend, Protocol):
    """Backends that support the hybrid ``visible_end`` mask path.

    Implemented only by backends reporting ``capabilities().visible_end`` (today
    ``fa4_cute``). Callers must check the capability before invoking
    ``forward_visible_end``.

    ``q`` may be fixed ``[B, L, H, D]`` or varlen ``[total, H, D]``;
    ``visible_end`` is padded ``[B, max_q]`` and indexed locally per sequence.
    """

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
    ) -> torch.Tensor:
        ...
