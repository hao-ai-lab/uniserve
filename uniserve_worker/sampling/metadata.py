"""Numerical inputs of worker token sampling.

``SamplingMetadata`` carries one sampled call's candidate logits and controls;
``uniserve_worker.execution.token`` builds it and
``uniserve_worker.sampling.sampler.sample`` consumes it. ``TokenSelection``
names the output a token input row requests from the forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from uniserve.sampling import SamplingParams


class TokenSelection(StrEnum):
    """Select final-token logits, all-token logits, hidden states, or none.

    A ``CACHE`` row consumes only the K/V cache its call writes: its call
    returns an empty value for it, and a call of such rows alone evaluates
    no output at all.
    """

    LAST_LOGITS = "last_logits"
    ALL_LOGITS = "all_logits"
    HIDDEN = "hidden"
    CACHE = "cache"


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    """Numerical sampling controls for a call's candidate logits.

    Parameters and terminal policy apply to the complete call. Only allowed
    tokens, penalty histories, and RNG draws vary along a speculative chain;
    those columns align with the first dimension of logits. The dataclass
    itself validates nothing; ``sample`` checks the logits and row shapes,
    rejects draws or parameter values on device-greedy calls, and checks
    their alignment on the others.
    """

    # [rows, vocab] floating candidate logits. An ordinary call has one row;
    # a speculative verification call has len(draft_token_ids) + 1 rows,
    # where row i verifies draft i and the last row is the bonus position.
    logits: torch.Tensor
    parameters: SamplingParams
    # Per row, a dense [vocab] generated-token count vector for repetition,
    # frequency, and presence penalties, or None when the request uses no
    # penalties. Verification rows also count the draft tokens that precede
    # them.
    penalty_counts: tuple[torch.Tensor | None, ...]
    # Per row, the only selectable token ids, or None for no restriction. A
    # forced-token point narrows its row to that single id.
    allowed: tuple[tuple[int, ...] | None, ...]
    # Call-wide token ids masked to -inf on every row.
    suppress: tuple[int, ...]
    finish_token_ids: tuple[int, ...]
    transition_token_ids: tuple[int, ...]
    # Finish on any valid, active selection regardless of its token.
    force_finish: bool
    # [rows] float32 uniform draws in [0, 1) on the logits device, and the
    # [rows, 3] float32 (temperature, top_p, min_p) matrix. Both are None
    # exactly for device-greedy calls: greedy parameters, no allowed-token
    # restriction, and no drafts. The column order is shared with
    # ``uniserve.sampling.sample_top_k``.
    draws: torch.Tensor | None
    parameter_values: torch.Tensor | None
    draft_token_ids: tuple[int, ...] = ()
    # 1-based length of the draft prefix ending at the first draft token that
    # is a finish token; accepting that whole prefix finishes the call.
    terminal_draft_prefix: int | None = None
    # Whether the call publishes a device transition decision.
    return_transition: bool = False
    # Device tensor whose first element gates the call, or None for an
    # always-active call. With ``tagged_predicate`` that element is an int64
    # token relay whose ``TOKEN_CONTINUATION_BIT`` marks activity; otherwise
    # it is a flag converted to bool.
    predicate: torch.Tensor | None = None
    tagged_predicate: bool = False
    # Device request slot, passed through unchanged to ``SamplerRow``.
    request_pool_index: torch.Tensor | None = None
    # The request's committed row of ``DecodeState.penalty_counts``. The
    # sampler does not read it; commit accumulates the accepted selection
    # into it through ``DecodeState.apply_tokens``.
    penalty_base: torch.Tensor | None = None
