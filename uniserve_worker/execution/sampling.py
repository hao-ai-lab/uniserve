"""Numerical sampling inputs and GPU output views shared by eager and graph execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import torch

from ..protocol.batch import SamplingParams

# Packed integer bit patterns and the row geometry needed to decode logprobs.
# The tensor stays on the producing device; the output owner performs any D2H copy.
LogprobValues: TypeAlias = tuple[
    torch.Tensor, tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...], int, int
]


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    """Numerical sampling controls for an operation's candidate logits.

    Parameters and terminal policy apply to the complete operation. Only allowed
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


@dataclass(frozen=True, slots=True)
class SamplerOutput:
    """Numerical selections with shared packed completion and optional logprob columns.

    A batch has completion_index=-1. Row views retain that batch's complete
    packed tensor so the output owner can issue one host copy for all its rows.
    No request state, storage owner, or host completion is carried here.
    """

    tokens: torch.Tensor
    valid: torch.Tensor
    active: torch.Tensor
    finish: torch.Tensor | None
    continuation: torch.Tensor
    tagged_tokens: torch.Tensor
    completion: torch.Tensor
    request_pool_indices: torch.Tensor | None = None
    accepted_draft_count: torch.Tensor | None = None
    accepted_token_count: torch.Tensor | None = None
    transition: torch.Tensor | None = None
    completion_index: int = -1
    logprobs: LogprobValues | None = None

    def row(self, index: int) -> SamplerOutput:
        """Borrow one batch row while preserving the shared completion allocation."""

        if self.completion_index != -1 or not 0 <= index < self.tokens.numel():
            raise IndexError("sampling row is outside the batch")

        def view(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value[index : index + 1]

        return replace(
            self,
            tokens=self.tokens[index : index + 1],
            valid=self.valid[index : index + 1],
            active=self.active[index : index + 1],
            finish=view(self.finish),
            continuation=self.continuation[index : index + 1],
            tagged_tokens=self.tagged_tokens[index : index + 1],
            request_pool_indices=view(self.request_pool_indices),
            accepted_draft_count=view(self.accepted_draft_count),
            accepted_token_count=view(self.accepted_token_count),
            transition=view(self.transition),
            completion_index=index,
        )

    def clone(self) -> SamplerOutput:
        """Copy numerical outputs before reusable graph storage is overwritten."""

        def copy(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().clone()

        logprobs = self.logprobs
        if logprobs is not None:
            logprobs = (logprobs[0].detach().clone(), *logprobs[1:])
        return replace(
            self,
            tokens=self.tokens.detach().clone(),
            valid=self.valid.detach().clone(),
            active=self.active.detach().clone(),
            finish=copy(self.finish),
            continuation=self.continuation.detach().clone(),
            tagged_tokens=self.tagged_tokens.detach().clone(),
            completion=self.completion.detach().clone(),
            request_pool_indices=copy(self.request_pool_indices),
            accepted_draft_count=copy(self.accepted_draft_count),
            accepted_token_count=copy(self.accepted_token_count),
            transition=copy(self.transition),
            logprobs=logprobs,
        )
