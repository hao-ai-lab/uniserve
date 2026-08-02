"""Capture-eligible exact top-k sampling provider."""

from __future__ import annotations

from collections.abc import Callable
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
    ],
    tuple[torch.Tensor, torch.Tensor],
]
def _sample_top_k_tensor(
    logits: torch.Tensor,
    draws: torch.Tensor,
    penalty_token_ids: torch.Tensor,
    penalty_counts: torch.Tensor,
    parameters: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        repetition = parameters[:, 3].unsqueeze(1).expand_as(penalty_counts).flatten()
        frequency = parameters[:, 4].unsqueeze(1).expand_as(penalty_counts).flatten()
        presence = parameters[:, 5].unsqueeze(1).expand_as(penalty_counts).flatten()
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
    temperatures = parameters[:, 0]
    divisors = torch.where(
        temperatures > 0.0,
        temperatures,
        torch.ones_like(temperatures),
    )
    work = work / divisors.unsqueeze(1)
    candidates, token_indexes = torch.topk(
        work,
        top_k,
        dim=-1,
        sorted=True,
    )
    min_p = parameters[:, 2]
    min_threshold = candidates[:, 0] + torch.log(min_p)
    candidates = torch.where(
        (min_p.unsqueeze(1) <= 0.0) | (candidates >= min_threshold.unsqueeze(1)),
        candidates,
        float("-inf"),
    )
    cumulative = torch.softmax(candidates, dim=-1).cumsum(dim=-1)
    over = cumulative > parameters[:, 1].unsqueeze(1)
    drop = torch.cat((torch.zeros_like(over[:, :1]), over[:, :-1]), dim=1)
    candidates = torch.where(drop, float("-inf"), candidates)
    probabilities = torch.softmax(candidates, dim=-1)
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
        temperatures > 0.0,
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
def _compiled_sampling(top_k: int, has_penalties: bool) -> _SamplingKernel:
    del has_penalties

    def kernel(
        logits: torch.Tensor,
        draws: torch.Tensor,
        penalty_token_ids: torch.Tensor,
        penalty_counts: torch.Tensor,
        parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _sample_top_k_tensor(
            logits,
            draws,
            penalty_token_ids,
            penalty_counts,
            parameters,
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
    parameters: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw exact tokens from rows whose top-k candidate bound is at most 128."""

    if logits.ndim != 2 or draws.shape != logits.shape[:1]:
        raise ValueError("sampling provider draws must align with logits rows")
    if parameters.shape != (logits.shape[0], 6):
        raise ValueError("sampling provider parameter vectors do not align with logits")
    if (
        penalty_token_ids.ndim != 2
        or penalty_counts.shape != penalty_token_ids.shape
        or penalty_token_ids.shape[0] != logits.shape[0]
    ):
        raise ValueError("sampling provider penalty vectors do not align")
    if not 0 < int(top_k) <= 128 or int(top_k) >= int(logits.shape[1]):
        raise ValueError("sampling provider requires an exact top-k candidate bound")
    if logits.device.type != "cuda":
        return _sample_top_k_tensor(
            logits,
            draws,
            penalty_token_ids,
            penalty_counts,
            parameters,
            int(top_k),
        )
    implementation = _compiled_sampling(
        int(top_k),
        int(penalty_token_ids.shape[1]) > 0,
    )
    return implementation(logits, draws, penalty_token_ids, penalty_counts, parameters)


__all__ = ["sample_top_k"]
