"""Numerical inputs of worker token sampling.

``SamplingMetadata`` carries one sampled call's candidate logits and controls;
Rust resolves its host controls and this module prepares their tensor inputs.
The native sampler dispatches numerical selectors in ``sampling.sampler``.
``TokenSelection`` names the output a token input row requests from the forward.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import torch

from uniserve.sampling import SamplingParams
from uniserve_worker._uniserve_ipc import SamplingMetadata as SamplingMetadata
from uniserve_worker.errors import unsupported_setup
from uniserve_worker.sampling.result import TOKEN_VALUE_MASK

if TYPE_CHECKING:
    from uniserve_worker.sampling.result import SamplerRow
    from uniserve_worker.storage.decode_state import DecodeState


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


def prepare_inputs(
    logits: torch.Tensor,
    parameters: SamplingParams,
    slot: int,
    decode_state: DecodeState | None,
    sampled: SamplerRow | None,
    draft_token_ids: list[int],
    uniforms: list[float] | None,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor | None, ...],
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Materialize numerical columns for native sampling controls.

    Penalties borrow the committed slot row. Uncommitted selections and draft
    prefixes accumulate into copies, leaving the committed counts unchanged.
    Draws already use the request's semantic Philox coordinates.
    """
    rows = logits.reshape(1, -1) if logits.ndim == 1 else logits
    penalty_base = None
    if parameters.uses_penalties():
        if decode_state is None:
            raise RuntimeError(
                "token sampling has no request runtime-state owner"
            )
        if (
            decode_state.vocab_size != rows.shape[1]
            or decode_state.device != rows.device
        ):
            raise unsupported_setup(
                "sampling vocabulary or device disagrees with "
                "request runtime state"
            )
        penalty_base = decode_state.penalty_counts[slot]

    penalty_view = penalty_base
    if penalty_base is not None and sampled is not None:
        # A selection counts only when valid and predicate-active. Its tag is
        # continuation state, not part of the vocabulary token.
        penalty_view = penalty_base.clone()
        token = sampled.tokens.reshape(-1)[:1].bitwise_and(TOKEN_VALUE_MASK)
        weight = (
            sampled.valid.reshape(-1)[:1] & sampled.active.reshape(-1)[:1]
        ).to(dtype=penalty_view.dtype)
        penalty_view.scatter_add_(0, token.to(dtype=torch.int64), weight)

    counts: list[torch.Tensor | None] = []
    for index in range(rows.shape[0]):
        if penalty_view is None or index == 0 or not draft_token_ids:
            counts.append(penalty_view)
            continue
        row = penalty_view.clone()
        for token_id in draft_token_ids[:index]:
            row[token_id] += 1
        counts.append(row)

    draws = parameter_values = None
    if uniforms is not None:
        draws = torch.tensor(uniforms, dtype=torch.float32, device=rows.device)
        parameter_values = torch.tensor(
            [(parameters.temperature, parameters.top_p, parameters.min_p)]
            * len(uniforms),
            dtype=torch.float32,
            device=rows.device,
        )
    return rows, tuple(counts), draws, parameter_values, penalty_base
