"""Cross-language parity of the canonical sampling RNG and inverse-CDF rule.

``tests/python/generated/sampling_rng_parity.json`` holds two kinds of cases.
Each RNG case records a Philox key, the four Philox output words, the uniform
draw, and the token selected from the shared dyadic ``probs``; its RNG values
were produced by the simulator's ``uniserve_core::philox``. Each boundary case
pairs its own distribution with a draw at an edge of the inverse-CDF rule
(the first positive-mass entry whose cumulative mass exceeds the draw, else
the last positive-mass entry) and an expected token derived from that rule.
The worker recomputes every value here, and the simulator's
``sample_categorical`` unit test selects from the same cases, so both
implementations share one RNG mapping and one selection boundary.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from uniserve.nn.rng import philox4x32_10, sampling_key, sampling_uniform
from uniserve.sampling import sample_categorical

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "generated"
    / "sampling_rng_parity.json"
)


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _draw(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _select_token(probs: list[float], draw: float) -> int:
    # Fixture probabilities and draws are exact float32 values.
    selected = sample_categorical(
        torch.tensor([probs], dtype=torch.float32),
        torch.tensor([draw], dtype=torch.float32),
    )
    return int(selected.item())


def test_sampling_rng_matches_the_rust_reference() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    probs = fixture["probs"]
    assert fixture["cases"], "fixture carries at least one coordinate"
    for case in fixture["cases"]:
        key = sampling_key(
            case["session_seed"],
            case["engine_id"],
            case["request_id"],
            case["request_epoch"],
            case["draw_layout"],
        )
        assert key == case["key"]
        index = case["semantic_token_index"]
        counter = (
            index & 0xFFFFFFFF,
            (index >> 32) & 0xFFFFFFFF,
            case["processor_stage"] & 0xFFFFFFFF,
            case["draw_index"] & 0xFFFFFFFF,
        )
        words = philox4x32_10(
            counter, (key & 0xFFFFFFFF, (key >> 32) & 0xFFFFFFFF)
        )
        assert list(words) == case["words"]
        draw = sampling_uniform(
            key,
            case["semantic_token_index"],
            case["processor_stage"],
            case["draw_index"],
        )
        assert _bits(draw) == case["uniform_bits"]
        assert _select_token(probs, draw) == case["selected_token"]


def test_inverse_cdf_boundaries_match_the_shared_fixture() -> None:
    # Covers a zero draw with masked leading entries, a draw on a cumulative
    # boundary, and float32 totals below the largest canonical draw.
    boundaries = json.loads(_FIXTURE.read_text())["boundaries"]
    assert boundaries, "fixture carries boundary cases"
    selected = {
        case["name"]: _select_token(case["probs"], _draw(case["uniform_bits"]))
        for case in boundaries
    }
    assert selected == {
        case["name"]: case["selected_token"] for case in boundaries
    }
