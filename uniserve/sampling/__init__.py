"""Sampling options and numerical token selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .greedy import greedy
from .top_k import sample_top_k


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Configure token sampling, penalties, constraints, and log probabilities."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    ignore_eos: bool = False
    seed: int | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: tuple[tuple[int, float], ...] = ()
    min_tokens: int = 0
    return_logprobs: bool = False
    n_logprobs: int = 0
    return_prompt_logprobs: bool = False
    n_prompt_logprobs: int = 0
    logprob_token_ids: tuple[int, ...] = ()
    bad_words_ids: tuple[tuple[int, ...], ...] = ()
    allowed_token_ids: tuple[int, ...] | None = None
    typical_p: float = 1.0
    forced_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "temperature",
            "top_p",
            "min_p",
            "typical_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if not 0 < self.typical_p <= 1:
            raise ValueError("typical_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if any(not math.isfinite(float(bias)) for _, bias in self.logit_bias):
            raise ValueError("logit_bias values must be finite")
        if self.allowed_token_ids == ():
            raise ValueError("allowed_token_ids must not be empty when present")
        if any(not values for values in self.bad_words_ids):
            raise ValueError("bad_words_ids entries must not be empty")

        for name in ("top_k", "min_tokens", "n_logprobs", "n_prompt_logprobs"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer")


__all__ = ["SamplingParams", "greedy", "sample_top_k"]
