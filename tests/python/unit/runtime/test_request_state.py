"""Stable request-state registration and random-stream invariants."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.runtime.request_state import RequestStateTable

pytestmark = pytest.mark.unit


def test_duplicate_new_request_with_empty_blocks_preserves_the_live_chain():
    states = RequestStateTable()
    state = states.create_or_update(6, {"req_id": 6, "block_ids": [10, 11]})

    same = states.create_or_update(6, {"req_id": 6, "block_ids": []})

    assert same is state
    assert same.block_ids == [10, 11]


def test_duplicate_new_request_merges_a_longer_registered_chain():
    states = RequestStateTable()
    state = states.create_or_update(6, {"req_id": 6, "block_ids": [10, 11]})

    same = states.create_or_update(6, {"req_id": 6, "block_ids": [10, 11, 12]})

    assert same is state
    assert same.block_ids == [10, 11, 12]


def test_request_random_streams_are_persistent_and_domain_separated():
    states = RequestStateTable()
    state = states.create_or_update(6, {"req_id": 6, "sampling": {"seed": 0}})

    text = state.device_rng("cpu", stream="text_sampling")
    image = state.device_rng("cpu", stream="model")

    assert state.device_rng("cpu", stream="text_sampling") is text
    assert state.device_rng("cpu", stream="model") is image
    assert text is not image
    expected = torch.randint(
        0,
        1000,
        (4,),
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).tolist()
    assert torch.randint(0, 1000, (4,), generator=text).tolist() == expected
    assert torch.randint(0, 1000, (4,), generator=image).tolist() == expected
