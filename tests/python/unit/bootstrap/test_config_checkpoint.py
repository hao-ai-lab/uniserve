"""The launch descriptor's checkpoint expectation for a worker rank."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args

pytestmark = pytest.mark.unit


def test_the_descriptor_states_the_checkpoint_the_head_derived(tmp_path):
    identity = "a" * 64
    config = worker_args(tmp_path, checkpoint_identity=identity)

    assert config.model is not None
    assert config.model.checkpoint_identity == identity


def test_a_descriptor_without_an_expectation_leaves_the_rank_unchecked(
    tmp_path,
):
    # The head omits the key when it cannot read the checkpoint locally, such
    # as for a Hub identifier; the ranks are then held to one another.
    config = worker_args(tmp_path)

    assert config.model is not None
    assert config.model.checkpoint_identity is None
