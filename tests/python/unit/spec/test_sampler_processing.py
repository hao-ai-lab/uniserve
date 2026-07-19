"""Logits-processing behavior of the model-independent token sampler.

Covers the public sampling surface in :mod:`uniserve_worker.nn.sampler`:

* greedy (``temperature == 0``) sampling ignores top-k / top-p / min-p
  truncation and returns the argmax of the processor-adjusted logits;
* min-p / top-k / top-p nucleus truncation force pruned positions to ``-inf``;
* frequency, presence and repetition penalties apply with their distinct
  count-scaled / flat / multiplicative-sign-aware semantics;
* the optional async logits-validity probe ignores ``-inf`` but flags ``NaN``
  and ``+inf`` when enabled.

All tests run on CPU (the default trivial mesh makes TP sync a no-op), so no
GPU marker is needed.
"""
from __future__ import annotations

import importlib
import math
import sys

import pytest
import torch

from uniserve_worker.nn import sampler as sampler_mod
from uniserve_worker.nn.sampler import (
    NEG_INF,
    apply_sampling_batched,
    shape_logits_for_sampling,
)

pytestmark = pytest.mark.unit


def _greedy_token(logits: torch.Tensor, sp: dict) -> int:
    """Single-row greedy draw through the public batched sampler."""
    samples = apply_sampling_batched(
        logits.clone(),
        [sp],
        [[]],
        [None],
        [None],
    )
    return samples[0].token_id


# ---------------------------------------------------------------------------
# Greedy ignores truncation knobs, honors processor adjustments.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "truncation",
    [
        {},
        {"top_k": 1},
        {"top_p": 0.1},
        {"min_p": 0.9},
        {"top_k": 2, "top_p": 0.3, "min_p": 0.5},
    ],
)
def test_greedy_ignores_truncation_knobs_and_returns_argmax(truncation):
    # Arrange: the global maximum is at index 1; truncation would prune most
    # tokens if it applied, but greedy must ignore it.
    logits = torch.tensor([[0.5, 3.0, 1.0, 2.5, 0.1]])
    sp = {"temperature": 0.0, **truncation}

    # Act
    token = _greedy_token(logits, sp)

    # Assert: argmax of the unshaped logits regardless of truncation knobs.
    assert token == 1


def test_greedy_returns_argmax_of_suppressed_logits():
    # Arrange: the natural argmax (index 1) is suppressed; truncation knobs are
    # present to prove greedy still ignores them.
    logits = torch.tensor([[0.5, 3.0, 1.0, 2.5, 0.1]])
    sp = {"temperature": 0.0, "top_k": 1, "top_p": 0.1, "min_p": 0.9}

    # Act: suppress token 1 -> next-highest finite logit is index 3 (2.5).
    samples = apply_sampling_batched(
        logits.clone(),
        [sp],
        [[]],
        [None],
        [[1]],
    )

    # Assert
    assert samples[0].token_id == 3


def test_greedy_returns_argmax_within_allowed_mask():
    # Arrange: restrict to {0, 4}; among those index 0 (0.5) beats index 4 (0.1).
    logits = torch.tensor([[0.5, 3.0, 1.0, 2.5, 0.1]])
    sp = {"temperature": 0.0, "top_k": 1}

    # Act
    samples = apply_sampling_batched(
        logits.clone(),
        [sp],
        [[]],
        [[0, 4]],
        [None],
    )

    # Assert
    assert samples[0].token_id == 0


# ---------------------------------------------------------------------------
# Truncation masking forces pruned positions to -inf.
# ---------------------------------------------------------------------------


def test_top_k_masks_all_but_k_highest_to_neg_inf():
    # Arrange: two highest are index 4 (4.0) and index 1 (3.0).
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 4.0])

    # Act
    shape_logits_for_sampling(
        logits, {"top_k": 2}, recent=[], allowed=None, suppress=None, vocab=5
    )

    # Assert: survivors keep their value; everything else is -inf.
    finite = ~torch.isneginf(logits)
    assert finite.tolist() == [False, True, False, False, True]
    torch.testing.assert_close(logits[1], torch.tensor(3.0))
    torch.testing.assert_close(logits[4], torch.tensor(4.0))


def test_min_p_prunes_tokens_below_relative_probability_threshold():
    # Arrange: softmax([3, 2, -5]) ~= [0.731, 0.269, 0.00025]; with min_p=0.2 the
    # threshold is 0.2 * max_prob, which keeps the top two and prunes the tail.
    logits = torch.tensor([3.0, 2.0, -5.0])

    # Act
    shape_logits_for_sampling(
        logits, {"min_p": 0.2}, recent=[], allowed=None, suppress=None, vocab=3
    )

    # Assert: only the low-probability tail position is forced to -inf.
    assert (~torch.isneginf(logits)).tolist() == [True, True, False]
    torch.testing.assert_close(logits[0], torch.tensor(3.0))
    torch.testing.assert_close(logits[1], torch.tensor(2.0))


def test_top_p_nucleus_drops_tail_outside_cumulative_mass():
    # Arrange: logits whose softmax is exactly [0.7, 0.2, 0.1].
    logits = torch.tensor([math.log(0.7), math.log(0.2), math.log(0.1)])
    expected_survivors = logits[:2].clone()

    # Act: cumulative mass is 0.7, 0.9, 1.0; the shifted nucleus cutoff at 0.75
    # keeps the first two tokens and drops the third.
    shape_logits_for_sampling(
        logits, {"top_p": 0.75}, recent=[], allowed=None, suppress=None, vocab=3
    )

    # Assert: the dropped position is -inf and survivors keep their logits.
    assert (~torch.isneginf(logits)).tolist() == [True, True, False]
    torch.testing.assert_close(logits[:2], expected_survivors)


def test_top_k_one_with_temperature_only_ever_samples_the_single_survivor():
    # Arrange: index 1 is the lone top-1 token; every other position is pruned to
    # zero probability, so multinomial can only draw index 1.
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.0]])

    # Act: draw across many seeds; pruned tokens have zero probability.
    drawn = set()
    for seed in range(16):
        samples = apply_sampling_batched(
            logits.clone(),
            [{"temperature": 1.0, "top_k": 1, "seed": seed}],
            [[]],
            [None],
            [None],
        )
        drawn.add(samples[0].token_id)

    # Assert
    assert drawn == {1}


# ---------------------------------------------------------------------------
# Penalty semantics: frequency vs presence vs repetition are distinct.
# ---------------------------------------------------------------------------


def test_frequency_penalty_scales_with_repeat_count():
    # Arrange: token 0 repeated five times, token 1 once; equal base logits.
    # Frequency subtracts count * penalty, so the more-frequent token loses more.
    recent = [0, 0, 0, 0, 0, 1]
    logits = torch.tensor([5.0, 5.0])

    # Act
    shape_logits_for_sampling(
        logits, {"frequency_penalty": 1.0}, recent=recent, allowed=None, suppress=None, vocab=2
    )

    # Assert: token0 -> 5 - 5 = 0.0, token1 -> 5 - 1 = 4.0.
    torch.testing.assert_close(logits, torch.tensor([0.0, 4.0]))


def test_presence_penalty_is_flat_regardless_of_repeat_count():
    # Arrange: token 0 repeated five times, token 1 once; presence subtracts a
    # single flat penalty no matter the count, so both drop by the same amount.
    recent = [0, 0, 0, 0, 0, 1]
    logits = torch.tensor([5.0, 5.0])

    # Act
    shape_logits_for_sampling(
        logits, {"presence_penalty": 1.0}, recent=recent, allowed=None, suppress=None, vocab=2
    )

    # Assert: both lose exactly 1.0, preserving the tie.
    torch.testing.assert_close(logits, torch.tensor([4.0, 4.0]))


def test_repetition_penalty_is_sign_aware_multiplicative():
    # Arrange: token 0 positive, token 1 negative, each seen once. A positive
    # logit is divided (pulled toward zero); a negative logit is multiplied
    # (pushed further negative).
    recent = [0, 1]
    logits = torch.tensor([4.0, -1.0])

    # Act
    shape_logits_for_sampling(
        logits, {"repetition_penalty": 2.0}, recent=recent, allowed=None, suppress=None, vocab=2
    )

    # Assert: token0 -> 4 / 2 = 2.0, token1 -> -1 * 2 = -2.0.
    torch.testing.assert_close(logits, torch.tensor([2.0, -2.0]))


def test_repetition_penalty_ignores_repeat_count():
    # Arrange: token 0 repeated five times but repetition is multiplicative and
    # applied once, unlike the count-scaled frequency penalty.
    recent = [0, 0, 0, 0, 0]
    logits = torch.tensor([4.0, 2.0])

    # Act
    shape_logits_for_sampling(
        logits, {"repetition_penalty": 2.0}, recent=recent, allowed=None, suppress=None, vocab=2
    )

    # Assert: token0 -> 4 / 2 = 2.0 (count ignored), token1 untouched (not recent).
    torch.testing.assert_close(logits, torch.tensor([2.0, 2.0]))


# ---------------------------------------------------------------------------
# Async logits-validity probe: ignores -inf, flags NaN / +inf when enabled.
# ---------------------------------------------------------------------------


@pytest.fixture()
def sampler_with_async_assert(monkeypatch):
    """Reimport the sampler with the async-validity probe enabled.

    ``_ENABLE_ASYNC_ASSERT`` is read once at import time from the
    ``UNISERVE_ENABLE_ASYNC_ASSERT`` env var, so the enabled path is exercised
    by setting the var and reloading the module. The original module object is
    restored afterward so other tests see the default (disabled) configuration.
    """
    original = sys.modules["uniserve_worker.nn.sampler"]
    monkeypatch.setenv("UNISERVE_ENABLE_ASYNC_ASSERT", "1")
    reloaded = importlib.reload(original)
    try:
        yield reloaded
    finally:
        sys.modules["uniserve_worker.nn.sampler"] = original
        monkeypatch.delenv("UNISERVE_ENABLE_ASYNC_ASSERT", raising=False)
        importlib.reload(original)


def _draw_greedy(module, logits: torch.Tensor) -> int:
    return module.apply_sampling_batched(
        logits, [{"temperature": 0.0}], [[]], [None], [None]
    )[0].token_id


def test_async_probe_disabled_by_default_lets_positive_infinity_through():
    # Arrange: with the probe disabled (default), the batched input check is a
    # no-op, so a +inf logit does not abort the call and is simply the argmax.
    logits = torch.tensor([[1.0, float("inf"), 2.0]])

    # Act: greedy draw over the row runs to completion.
    token = sampler_mod.apply_sampling_batched(
        logits, [{"temperature": 0.0}], [[]], [None], [None]
    )[0].token_id

    # Assert: argmax selects the +inf position; the probe never fired.
    assert token == 1


def test_async_probe_allows_negative_infinity(sampler_with_async_assert):
    # Arrange: -inf is a legitimate masked logit and must not trip the probe.
    logits = torch.tensor([[1.0, NEG_INF, 3.0, 2.0]])

    # Act: greedy draw runs to completion; argmax over finite values is index 2.
    token = _draw_greedy(sampler_with_async_assert, logits)

    # Assert
    assert token == 2


def test_async_probe_flags_positive_infinity(sampler_with_async_assert):
    # Arrange: +inf in the input is an invalid logit the probe must flag.
    logits = torch.tensor([[1.0, float("inf"), 2.0]])

    # Act / Assert
    with pytest.raises(RuntimeError):
        _draw_greedy(sampler_with_async_assert, logits)


def test_async_probe_flags_nan(sampler_with_async_assert):
    # Arrange: NaN in the input is an invalid logit the probe must flag.
    logits = torch.tensor([[1.0, float("nan"), 2.0]])

    # Act / Assert
    with pytest.raises(RuntimeError):
        _draw_greedy(sampler_with_async_assert, logits)
