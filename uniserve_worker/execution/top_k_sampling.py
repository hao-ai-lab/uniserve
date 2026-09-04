"""Capture-eligible exact top-k sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import torch

_SamplingKernel = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True)
class SamplingParameters:
    """Holds device-ready temperature, filtering, and token-penalty vectors for batched sampling."""

    temperature: torch.Tensor
    top_p: torch.Tensor
    min_p: torch.Tensor
    repetition_penalty: torch.Tensor
    frequency_penalty: torch.Tensor
    presence_penalty: torch.Tensor

    @classmethod
    def from_columns(cls, values: torch.Tensor) -> SamplingParameters:
        """View a ``[rows, 6]`` parameter matrix as named row-aligned vectors."""

        if values.ndim != 2 or int(values.shape[-1]) != 6:
            raise ValueError("sampling parameters must have six columns per row")
        return cls(
            temperature=values[:, 0],
            top_p=values[:, 1],
            min_p=values[:, 2],
            repetition_penalty=values[:, 3],
            frequency_penalty=values[:, 4],
            presence_penalty=values[:, 5],
        )


def _sample_top_k_tensor(
    logits: torch.Tensor,
    draws: torch.Tensor,
    penalty_token_ids: torch.Tensor,
    penalty_counts: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample row-wise top-k candidates after penalties and probability filters."""

    # Penalties are applied in flattened row-major space so repeated token IDs
    # in different rows cannot alias each other.
    work = logits.float()
    if penalty_token_ids.numel():
        row_offsets = (
            torch.arange(
                logits.shape[0],
                dtype=torch.long,
                device=logits.device,
            )
            * logits.shape[1]
        )
        penalty_indexes = (penalty_token_ids + row_offsets.unsqueeze(1)).flatten()
        flat = work.flatten()
        values = flat.gather(0, penalty_indexes)
        counts = penalty_counts.flatten()
        repetition = repetition_penalty.unsqueeze(1).expand_as(penalty_counts).flatten()
        frequency = frequency_penalty.unsqueeze(1).expand_as(penalty_counts).flatten()
        presence = presence_penalty.unsqueeze(1).expand_as(penalty_counts).flatten()
        adjusted = (
            torch.where(
                values > 0.0,
                values / repetition,
                values * repetition,
            )
            - frequency * counts
            - presence
        )
        flat = flat.scatter(
            0,
            penalty_indexes,
            torch.where(
                (counts <= 0.0) | torch.isneginf(values),
                values,
                adjusted,
            ),
        )
        work = flat.view_as(work)
    # Zero temperature preserves finite logits for filtering; it later selects
    # the first sorted candidate deterministically.
    divisors = torch.where(
        temperature > 0.0,
        temperature,
        torch.ones_like(temperature),
    )
    work = work / divisors.unsqueeze(1)
    candidates, token_indexes = torch.topk(
        work,
        top_k,
        dim=-1,
        sorted=True,
    )

    # Nucleus filtering retains the candidate that first crosses top-p, while
    # min-p is measured relative to the maximum candidate probability in log space.
    cumulative = torch.softmax(candidates, dim=-1).cumsum(dim=-1)
    over = cumulative > top_p.unsqueeze(1)
    drop = torch.cat((torch.zeros_like(over[:, :1]), over[:, :-1]), dim=1)
    candidates = torch.where(drop, float("-inf"), candidates)
    min_threshold = candidates[:, 0] + torch.log(min_p)
    candidates = torch.where(
        (min_p.unsqueeze(1) <= 0.0) | (candidates >= min_threshold.unsqueeze(1)),
        candidates,
        float("-inf"),
    )
    probabilities = torch.softmax(candidates, dim=-1)
    # Sorting candidates by token ID makes inverse-CDF draws deterministic for
    # a fixed random value independent of top-k kernel ordering.
    token_order = torch.argsort(token_indexes, dim=-1)
    ordered_probabilities = probabilities.gather(1, token_order)
    cumulative = ordered_probabilities.cumsum(dim=-1)
    sampled_order = (
        (cumulative < draws.to(dtype=cumulative.dtype).unsqueeze(1))
        .sum(dim=-1)
        .clamp_max(top_k - 1)
    )
    sampled = token_order.gather(1, sampled_order.unsqueeze(1))[:, 0]
    selected = torch.where(
        temperature > 0.0,
        sampled,
        torch.zeros_like(sampled),
    )
    tokens = token_indexes.gather(1, selected.unsqueeze(1))[:, 0]
    valid = (
        ~torch.isnan(candidates).any(dim=-1)
        & ~torch.isposinf(candidates).any(dim=-1)
        & torch.isfinite(candidates).any(dim=-1)
    )
    return tokens, valid


@lru_cache(maxsize=256)
def _compiled_sampling(top_k: int) -> _SamplingKernel:
    """Compile and cache the fixed-top-k sampling specialization for one k value."""

    def kernel(
        logits: torch.Tensor,
        draws: torch.Tensor,
        penalty_token_ids: torch.Tensor,
        penalty_counts: torch.Tensor,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        min_p: torch.Tensor,
        repetition_penalty: torch.Tensor,
        frequency_penalty: torch.Tensor,
        presence_penalty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the fixed-top-k sampling graph used by the compiled specialization."""

        return _sample_top_k_tensor(
            logits,
            draws,
            penalty_token_ids,
            penalty_counts,
            temperature,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            top_k,
        )

    return cast(
        _SamplingKernel,
        torch.compile(
            kernel,
            fullgraph=True,
            dynamic=True,
        ),
    )


def sample_top_k(
    logits: torch.Tensor,
    draws: torch.Tensor,
    penalty_token_ids: torch.Tensor,
    penalty_counts: torch.Tensor,
    parameters: SamplingParameters,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw exact tokens from rows whose top-k candidate bound is at most 128."""

    if logits.ndim != 2 or draws.shape != logits.shape[:1]:
        raise ValueError("top-k sampling draws must align with logits rows")
    if (
        parameters.temperature.shape != logits.shape[:1]
        or parameters.top_p.shape != logits.shape[:1]
        or parameters.min_p.shape != logits.shape[:1]
        or parameters.repetition_penalty.shape != logits.shape[:1]
        or parameters.frequency_penalty.shape != logits.shape[:1]
        or parameters.presence_penalty.shape != logits.shape[:1]
    ):
        raise ValueError("top-k sampling parameter vectors do not align with logits")
    if (
        penalty_token_ids.ndim != 2
        or penalty_counts.shape != penalty_token_ids.shape
        or penalty_token_ids.shape[0] != logits.shape[0]
    ):
        raise ValueError("top-k sampling penalty vectors do not align")
    if not 0 < int(top_k) <= 128 or int(top_k) >= int(logits.shape[1]):
        raise ValueError("top-k sampling requires an exact top-k candidate bound")
    args = (
        logits,
        draws,
        penalty_token_ids,
        penalty_counts,
        parameters.temperature,
        parameters.top_p,
        parameters.min_p,
        parameters.repetition_penalty,
        parameters.frequency_penalty,
        parameters.presence_penalty,
        int(top_k),
    )
    if logits.device.type != "cuda":
        return _sample_top_k_tensor(*args)
    implementation = _compiled_sampling(int(top_k))
    return implementation(*args[:-1])


__all__ = ["SamplingParameters", "sample_top_k"]
