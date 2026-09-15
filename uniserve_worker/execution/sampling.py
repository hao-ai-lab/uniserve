"""Numerical sampling inputs and GPU output views shared by eager and graph execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import torch

from uniserve.model.logits import VocabShard

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

    Row selections retain this complete batch so device and host publication
    can consume its columns without splitting and reconstructing tensor views.
    No request state, storage owner, or host completion is carried here.
    """

    tokens: torch.Tensor
    valid: torch.Tensor
    active: torch.Tensor
    finish: torch.Tensor | None
    continuation: torch.Tensor
    tagged_tokens: torch.Tensor
    completion: torch.Tensor
    accepted_draft_count: torch.Tensor | None = None
    accepted_token_count: torch.Tensor | None = None
    logprobs: LogprobValues | None = None

    def row(
        self,
        index: int,
        *,
        request_pool_index: torch.Tensor | None = None,
        transition: torch.Tensor | None = None,
    ) -> SamplerRow:
        """Associate one selection with its input slot and optional transition payload."""

        if not 0 <= index < self.tokens.numel():
            raise IndexError("sampling row is outside the batch")
        return SamplerRow(self, index, request_pool_index, transition)

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
            accepted_draft_count=copy(self.accepted_draft_count),
            accepted_token_count=copy(self.accepted_token_count),
            logprobs=logprobs,
        )


@dataclass(frozen=True, slots=True)
class SamplerRow:
    """One operation's selection within a retained numerical sampling batch.

    Request slots and transition payloads come from operation metadata. Scalar
    selection views are created only for consumers that actually need one row.
    """

    batch: SamplerOutput
    index: int
    request_pool_index: torch.Tensor | None = None
    transition: torch.Tensor | None = None

    @property
    def tokens(self) -> torch.Tensor:
        return self.batch.tokens[self.index : self.index + 1]

    @property
    def valid(self) -> torch.Tensor:
        return self.batch.valid[self.index : self.index + 1]

    @property
    def active(self) -> torch.Tensor:
        return self.batch.active[self.index : self.index + 1]

    @property
    def continuation(self) -> torch.Tensor:
        return self.batch.continuation[self.index : self.index + 1]

    @property
    def accepted_draft_count(self) -> torch.Tensor | None:
        values = self.batch.accepted_draft_count
        return None if values is None else values[self.index : self.index + 1]

    @property
    def accepted_token_count(self) -> torch.Tensor | None:
        values = self.batch.accepted_token_count
        return None if values is None else values[self.index : self.index + 1]


SampleColumn = Literal["tokens", "valid", "active", "continuation", "tagged_tokens"]


def sample_columns(
    rows: Sequence[SamplerRow], names: tuple[SampleColumn, ...]
) -> tuple[torch.Tensor, ...]:
    """Read aligned columns in operation order, retaining contiguous batch spans.

    A completion group may select reordered rows or combine independent sampling
    batches. Only those discontinuities require concatenation; no storage-address
    inspection or per-row tensor construction is needed for a contiguous span.
    """

    if not rows:
        raise ValueError("sampling columns require at least one row")
    spans: list[tuple[SamplerOutput, int, int]] = []
    batch = rows[0].batch
    start = rows[0].index
    end = start + 1
    for row in rows[1:]:
        if row.batch is batch and row.index == end:
            end += 1
        else:
            spans.append((batch, start, end))
            batch, start, end = row.batch, row.index, row.index + 1
    spans.append((batch, start, end))

    columns = []
    for name in names:
        parts = []
        for batch, start, end in spans:
            values = getattr(batch, name)
            parts.append(values if start == 0 and end == values.numel() else values[start:end])
        columns.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
    return tuple(columns)


def greedy_vocabulary(
    logits: torch.Tensor, vocab: VocabShard | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select global maxima with the dense torch.max tie and NaN rules.

    Only one score/token pair is exchanged per row. Scores retain their exact
    floating representation and token IDs use int64, including padded shards
    with no real tokens. Communicator order is logical vocabulary order.
    """

    if vocab is None:
        return torch.max(logits, dim=-1)
    if logits.ndim != 2 or logits.shape[-1] != vocab.local_slice.stop - vocab.local_slice.start:
        raise ValueError("greedy logits must be rows of the declared vocabulary shard")
    begin = vocab.local_slice.start
    valid = max(0, min(logits.shape[-1], vocab.size - begin))
    if valid:
        values, tokens = torch.max(logits[:, :valid], dim=-1)
        tokens = tokens + begin
    else:
        values = logits.new_full((logits.shape[0],), float("-inf"))
        tokens = torch.full_like(values, vocab.size, dtype=torch.int64)
    if vocab.group.size == 1:
        return values, tokens
    candidates = torch.stack((values.to(torch.float64).view(torch.int64), tokens), dim=-1)
    gathered = vocab.group.all_gather(candidates, dim=0).reshape(vocab.group.size, -1, 2)
    scores = gathered[..., 0].contiguous().view(torch.float64)
    maxima, owners = scores.max(dim=0)
    selected = gathered[..., 1].gather(0, owners.unsqueeze(0)).squeeze(0)
    return maxima.to(logits.dtype), selected
