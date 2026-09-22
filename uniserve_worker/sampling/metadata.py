"""Numerical sampling inputs and GPU output views shared by eager and graph.

execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from uniserve.sampling import SamplingParams


class TokenSelection(StrEnum):
    """Select final-token logits, all-token logits, or hidden states."""

    LAST_LOGITS = "last_logits"
    ALL_LOGITS = "all_logits"
    HIDDEN = "hidden"


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    """Numerical sampling controls for a call's candidate logits.

    Parameters and terminal policy apply to the complete call. Only allowed
    tokens, penalty histories, and RNG draws vary along a speculative chain;
    those columns align with the first dimension of logits.
    """

    logits: torch.Tensor
    parameters: SamplingParams
    penalty_counts: tuple[torch.Tensor | None, ...]
    allowed: tuple[tuple[int, ...] | None, ...]
    suppress: tuple[int, ...]
    finish_token_ids: tuple[int, ...]
    transition_token_ids: tuple[int, ...]
    force_finish: bool
    draws: torch.Tensor | None
    parameter_values: torch.Tensor | None
    draft_token_ids: tuple[int, ...] = ()
    terminal_draft_prefix: int | None = None
    return_transition: bool = False
    predicate: torch.Tensor | None = None
    tagged_predicate: bool = False
    request_pool_index: torch.Tensor | None = None
    # Committed history receives the accepted selections after sampling.
    penalty_base: torch.Tensor | None = None
