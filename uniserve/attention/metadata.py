"""Borrowed numerical attention inputs and mathematical expert routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class AttentionMode(StrEnum):
    """Selects dense, paged decode, paged prefill, or packed attention execution."""

    DENSE = "dense"
    PAGED_DECODE = "paged_decode"
    PAGED_VARLEN = "paged_varlen"
    PACKED = "packed"


class ExpertRoute(StrEnum):
    """Selects text or diffusion-flow experts for routed transformer tokens."""

    TEXT = "text"
    FLOW = "flow"


@dataclass(frozen=True, slots=True)
class RouteSpan:
    """One contiguous expert span in packed attention token order."""

    route: ExpertRoute
    token_start: int
    token_count: int

    def __post_init__(self) -> None:
        """Validate a non-empty half-open span for one expert route."""

        if self.token_start < 0 or self.token_count < 1:
            raise ValueError("packed expert span geometry is invalid")

    @property
    def token_end(self) -> int:
        """Give the exclusive packed-token boundary of this expert route."""

        return self.token_start + self.token_count


@dataclass(frozen=True, slots=True)
class AttentionMetadata:
    """Borrowed numerical geometry consumed independently by attention backends.

    Prefix lengths exclude the current query. Paged modes materialize total
    lengths; packed attention reads the prefix and query segments separately.
    The caller retains backing tensors until all numerical readers finish.
    """

    attention_mode: AttentionMode
    prefix_lens: torch.Tensor
    query_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    has_cache_writes: bool = True
    block_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    output_indices: torch.Tensor | None = None
    attention_indexes: torch.Tensor | None = None
    visible_end: torch.Tensor | None = None
    route_spans: tuple[RouteSpan, ...] = ()
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    causal: bool = True
    causal_rows_cpu: tuple[bool, ...] = ()
    prefix_lens_cpu: tuple[int, ...] = ()
    query_lens_cpu: tuple[int, ...] = ()
    seq_lens_cpu: tuple[int, ...] = ()
    fully_visible: bool = False
