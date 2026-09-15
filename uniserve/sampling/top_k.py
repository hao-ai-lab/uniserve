"""Capture-safe exact top-k sampling."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import cast

import torch

_SamplingKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


def _sample_top_k_tensor(
    logits: torch.Tensor,
    draws: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample row-wise top-k candidates.

    With temperature and probability filters.
    """
    work = logits.float()
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
    # min-p is measured relative to the maximum candidate probability in log
    # space.
    cumulative = torch.softmax(candidates, dim=-1).cumsum(dim=-1)
    over = cumulative > top_p.unsqueeze(1)
    drop = torch.cat((torch.zeros_like(over[:, :1]), over[:, :-1]), dim=1)
    candidates = torch.where(drop, float("-inf"), candidates)
    min_threshold = candidates[:, 0] + torch.log(min_p)
    candidates = torch.where(
        (min_p.unsqueeze(1) <= 0.0)
        | (candidates >= min_threshold.unsqueeze(1)),
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

    # Rows with NaN or +inf candidates, or with no finite candidate at all,
    # have no well-defined distribution; the caller rejects their tokens.
    valid = (
        ~torch.isnan(candidates).any(dim=-1)
        & ~torch.isposinf(candidates).any(dim=-1)
        & torch.isfinite(candidates).any(dim=-1)
    )
    return tokens, valid


@lru_cache(maxsize=256)
def _compiled_sampling(top_k: int) -> _SamplingKernel:
    """Compile and cache the fixed-top-k sampling specialization.

    For one k value.
    """

    def kernel(
        logits: torch.Tensor,
        draws: torch.Tensor,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        min_p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _sample_top_k_tensor(
            logits,
            draws,
            temperature,
            top_p,
            min_p,
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
    parameters: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw from unpenalized rows with an exact top-k candidate bound up to 128.

    ``parameters`` is a floating [rows, 3] matrix of temperature, top-p,
    and min-p. The caller routes penalty-bearing rows to the general sampler.
    """
    if logits.ndim != 2 or draws.shape != logits.shape[:1]:
        raise ValueError("top-k sampling draws must align with logits rows")
    if parameters.shape != (logits.shape[0], 3):
        raise ValueError(
            "top-k sampling requires temperature, top-p, and min-p columns"
        )
    if not 0 < int(top_k) <= 128 or int(top_k) >= int(logits.shape[1]):
        raise ValueError(
            "top-k sampling requires an exact top-k candidate bound"
        )
    args = (
        logits,
        draws,
        parameters[:, 0],
        parameters[:, 1],
        parameters[:, 2],
        int(top_k),
    )
    if logits.device.type != "cuda":
        return _sample_top_k_tensor(*args)
    implementation = _compiled_sampling(int(top_k))
    return implementation(*args[:-1])


__all__ = ["sample_top_k"]
