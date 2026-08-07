"""Cross-language parity of the canonical sampling RNG.

The Rust simulator emits ``tests/python/generated/sampling_rng_parity.json``
from ``crates/foundation/core/tests/rng_parity.rs``. The worker recomputes the
Philox key, the four Philox output words, the uniform draw, and the
inverse-CDF token from the same coordinates and asserts byte-identical
agreement, proving one exact counter-based RNG mapping and one inverse-CDF
boundary are shared across both implementations.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from uniserve_worker.runtime.rng import (
    philox4x32_10,
    sampling_key,
    sampling_uniform,
)

_FIXTURE = (
    Path(__file__).resolve().parents[2] / "generated" / "sampling_rng_parity.json"
)


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _select_token(probs: list[float], draw: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probs):
        cumulative += probability
        if draw <= cumulative:
            return index
    return len(probs) - 1


def test_sampling_rng_matches_the_rust_reference() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    probs = fixture["probs"]
    assert fixture["cases"], "fixture carries at least one coordinate"
    for case in fixture["cases"]:
        key = sampling_key(
            case["session_seed"],
            case["authority_id"],
            case["session_id"],
            case["epoch"],
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
        words = philox4x32_10(counter, (key & 0xFFFFFFFF, (key >> 32) & 0xFFFFFFFF))
        assert list(words) == case["words"]
        draw = sampling_uniform(
            key,
            case["semantic_token_index"],
            case["processor_stage"],
            case["draw_index"],
        )
        assert _bits(draw) == case["uniform_bits"]
        assert _select_token(probs, draw) == case["selected_token"]
