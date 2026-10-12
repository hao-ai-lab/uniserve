"""The sampler's counter-based generator and its inverse-CDF selection rule.

``philox4x32_10`` reproduces the published Random123 known-answer vectors, so
the draws the token sampler addresses by semantic coordinate come from the
standard Philox4x32-10 generator. ``sample_categorical`` selects the first
positive-mass entry whose cumulative mass exceeds the draw, else the last
positive-mass entry, which the boundary cases below pin.
"""

from __future__ import annotations

import pytest
import torch

from uniserve.nn.rng import philox4x32_10
from uniserve.sampling import sample_categorical

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("counter", "key", "words"),
    [
        # Random123 kat_vectors for philox4x32 with ten rounds.
        (
            (0, 0, 0, 0),
            (0, 0),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        ),
        (
            (0xFFFFFFFF,) * 4,
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_reproduces_the_published_known_answers(
    counter, key, words
) -> None:
    assert tuple(philox4x32_10(counter, key)) == words


@pytest.mark.parametrize(
    ("probabilities", "draw", "token"),
    [
        # A zero draw skips masked leading entries.
        ([0.0, 0.0, 0.5, 0.0, 0.5, 0.0], 0.0, 2),
        ([0.125, 0.875], 0.0, 0),
        # A draw on a cumulative boundary selects the next positive entry.
        ([0.0, 0.0, 0.5, 0.0, 0.5, 0.0], 0.5, 4),
        # A float32 total below the largest canonical draw selects the last
        # positive-mass entry.
        (
            [0.25, 0.25, 0.25, 0.24999988079071045, 0.0],
            0.99999994,
            3,
        ),
        ([1 / 12] * 12 + [0.0], 0.99999994, 11),
    ],
)
def test_inverse_cdf_selection_at_the_distribution_boundaries(
    probabilities, draw, token
) -> None:
    selected = sample_categorical(
        torch.tensor([probabilities], dtype=torch.float32),
        torch.tensor([draw], dtype=torch.float32),
    )

    assert int(selected.item()) == token
