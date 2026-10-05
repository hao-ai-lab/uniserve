"""Device output views of worker token sampling.

``SamplerOutput`` holds one sampled batch's selections, produced either by
``uniserve_worker.sampling.sampler`` or by graph-replayed greedy decode in
``uniserve_worker.model_executor.graph_inputs``. ``SamplerRow`` addresses one
call within it. Output capture (``uniserve_worker.execution.output``) copies
a batch's shared ``completion`` and ``logprobs`` columns once; the native
executor and ``uniserve_worker.execution.token``
gather per-call columns through ``sample_columns`` without splitting them per
row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import torch

# The completion column packs four field sections, each one row per call:
# [valid | active | token | accepted] (``sampling_columns`` in
# ``uniserve_worker.sampling.sampler``). The native PendingOutput decoder
# uses the same field count.
SAMPLING_COMPLETION_FIELDS = 4

# Tagged token relays set bit 31 to flag continuation; the low 31 bits carry
# the token id, which bounds the vocabulary usable by device-side decisions
# (``sample`` rejects a larger vocabulary). Relay values are non-negative
# int64, so ``value >= TOKEN_CONTINUATION_BIT`` tests the flag.
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1

# Packed integer bit patterns and row dimensions are enough to decode logprobs.
# The tensor stays on the producing device; the output owner performs any D2H
# copy.
LogprobValues: TypeAlias = tuple[
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    int,
    int,
]


@dataclass(frozen=True, slots=True)
class SamplerOutput:
    """Numerical selections with shared completion storage and logprobs.

    Row selections retain this complete batch so device and host export
    can consume its columns without splitting and reconstructing tensor views.
    No request state, storage owner, or host completion is carried here.

    Attributes:
        tokens: Selected token id per call.
        valid: Whether each call's filtered distribution was usable; under
            speculation, whether every consumed candidate row was.
        active: Each call's resolved device predicate.
        finish: Whether each call finishes; false for an invalid or
            inactive call.
        continuation: ``valid & active & ~finish`` per call.
        tagged_tokens: ``tokens`` with ``TOKEN_CONTINUATION_BIT`` set where
            ``continuation`` holds; the int64 value relayed to dependents.
        completion: The packed ``SAMPLING_COMPLETION_FIELDS`` column.
        accepted_draft_count: Accepted draft tokens per call.
        accepted_token_count: Tokens the call emits: accepted drafts plus a
            correction or bonus token unless an accepted draft finished it.
            Both counts are set only by the general sampler path, the one
            that handles speculative verification.
        logprobs: Packed logprob column and layout for the calls that
            request logprobs, or None when none do.
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
        """Associate one selection with its request slot and transition.

        Raises:
            IndexError: ``index`` is outside the batch.
        """
        if not 0 <= index < self.tokens.numel():
            raise IndexError("sampling row is outside the batch")
        return SamplerRow(self, index, request_pool_index, transition)

    def clone(self) -> SamplerOutput:
        """Copy numerical outputs out of reusable graph storage.

        Tensors are detached and cloned so a later replay that overwrites the
        borrowed storage does not change this output.
        """

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
    """One call's selection within a retained numerical sampling batch.

    Request slots and transition payloads come from call metadata. Scalar
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


SampleColumn = Literal[
    "tokens", "valid", "active", "continuation", "tagged_tokens"
]


def sample_columns(
    rows: Sequence[SamplerRow], names: tuple[SampleColumn, ...]
) -> tuple[torch.Tensor, ...]:
    """Read aligned columns in call order, retaining contiguous batch spans.

    A batch may select reordered rows or combine independent sampling
    batches. Only those discontinuities require concatenation; no
    storage-address inspection or per-row tensor construction is needed for a
    contiguous span, and a span covering a whole batch uses that batch's
    column tensor without slicing.

    Raises:
        ValueError: ``rows`` is empty.
    """
    if not rows:
        raise ValueError("sampling columns require at least one row")

    # Coalesce consecutive rows of one batch into [start, end) spans.
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
            parts.append(
                values
                if start == 0 and end == values.numel()
                else values[start:end]
            )
        columns.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
    return tuple(columns)
