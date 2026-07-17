"""Modality-neutral attention execution plans published on ForwardContext.

These types describe *how* attention runs for one forward, not which modality
produced the batch. Text, packed multimodal, and denoise paths all publish
through the same seam:

* :class:`PagedDecodePlan` — one query token per row against a paged KV cache
* :class:`PagedVarlenPlan` — variable-length paged attention (extend / mixed /
  generation / transient denoise)

Both variants derive from the abstract :class:`AttentionPlanBase`, which carries
the geometry that every paged forward always has (the page table and the per-row
CPU length summaries). Fields whose presence or optionality differs between
decode and varlen stay on the concrete subclasses, so a consumer that narrows
with ``isinstance`` gets unconditional access to exactly the fields that regime
guarantees.

Plans are frozen: a published plan is an immutable description. CUDA-graph
runners keep their static device buffers — whose addresses a captured graph
depends on — on a separate graph-state object, and before each replay publish a
fresh plan that reuses those buffer handles with the current per-replay CPU
summaries (a host-side snapshot via ``dataclasses.replace``).

A captured CUDA-graph wrapper's identity lives on a separate :class:`GraphBinding`
published on :class:`ForwardContext`, so FlashInfer routes by object identity
while the plan carries only geometry and seqlens.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .forward_context import AttentionCache
    from .forward_mode import ForwardMode

__all__ = [
    "GraphBinding",
    "AttentionPlanBase",
    "PagedDecodePlan",
    "PagedVarlenPlan",
]


class GraphBinding:
    """Stable identity token for a captured CUDA-graph attention wrapper.

    FlashInfer (and other backends with mutable plan state) bind an exclusive
    graph-scoped wrapper to ``id(binding)`` and validate liveness with a weakref.
    The object carries no plan data — geometry and seqlens live on the
    accompanying :class:`AttentionPlanBase`.
    """

    __slots__ = ("__weakref__",)


@dataclass(frozen=True, kw_only=True)
class AttentionPlanBase(ABC):
    """Abstract shared geometry for every paged attention plan.

    Holds the per-row page table and the CPU-side length summaries the host path
    reads without a device sync. Regime-specific device tensors and their
    optionality live on the concrete subclasses; construct one of those, never
    this base.
    """

    block_table: "torch.Tensor"
    cache_seqlens_cpu: tuple[int, ...] = ()
    kv_seqlens_cpu: tuple[int, ...] = ()
    query_lens_cpu: tuple[int, ...] = ()
    max_context_len: int = 0

    def __post_init__(self) -> None:
        if type(self) is AttentionPlanBase:
            raise TypeError(
                "AttentionPlanBase is abstract; construct PagedDecodePlan or PagedVarlenPlan"
            )


@dataclass(frozen=True, kw_only=True)
class PagedDecodePlan(AttentionPlanBase):
    """One-token-per-row paged decode attention plan."""

    residency_cache: "AttentionCache"
    cache_seqlens: "torch.Tensor"
    kv_seqlens: "torch.Tensor"
    query_lens: "torch.Tensor"
    decode_page_ids: "torch.Tensor"
    decode_page_offsets: "torch.Tensor"

    @classmethod
    def for_decode_graph(
        cls,
        *,
        residency_cache: "AttentionCache",
        batch_size: int,
        block_table: "torch.Tensor",
        cache_seqlens: "torch.Tensor",
        kv_seqlens: "torch.Tensor",
        query_lens: "torch.Tensor",
        decode_page_ids: "torch.Tensor",
        decode_page_offsets: "torch.Tensor",
        max_context_len: int = 0,
    ) -> "PagedDecodePlan":
        """Build a fully-initialized decode-graph plan in one step.

        The captured one-token decode graph holds a single graph-state object
        whose device tensors are the runner's static input buffers. This
        constructs the plan that names those buffers with the fixed decode
        initial values: one query token per row and unit kv length per row.
        """

        batch_size = int(batch_size)
        query_lens.fill_(1)
        return cls(
            residency_cache=residency_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cache_seqlens_cpu=tuple(0 for _ in range(batch_size)),
            kv_seqlens=kv_seqlens,
            query_lens=query_lens,
            query_lens_cpu=tuple(1 for _ in range(batch_size)),
            kv_seqlens_cpu=tuple(1 for _ in range(batch_size)),
            decode_page_ids=decode_page_ids,
            decode_page_offsets=decode_page_offsets,
            max_context_len=int(max_context_len),
        )


@dataclass(frozen=True, kw_only=True)
class PagedVarlenPlan(AttentionPlanBase):
    """Variable-length paged attention plan (extend / mixed / transient)."""

    cu_seqlens_q: "torch.Tensor"
    cu_seqlens_k: "torch.Tensor"
    max_seqlen_q: int
    max_seqlen_k: int
    residency_cache: "AttentionCache | None" = None
    cache_seqlens: "torch.Tensor | None" = None
    query_lens: "torch.Tensor | None" = None
    kv_seqlens: "torch.Tensor | None" = None
    mode: "ForwardMode | None" = None
