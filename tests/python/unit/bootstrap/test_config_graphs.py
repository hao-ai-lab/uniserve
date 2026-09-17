"""Graph residency declarations carried by a worker launch."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args

pytestmark = pytest.mark.unit


def test_declared_video_shapes_carry_duration_and_prompt_length(tmp_path):
    config = worker_args(
        tmp_path,
        max_batch_tokens=8192,
        video_graph_shapes="5x1000,15.0x10000",
    )
    assert config.execution.video_graph_shapes == (
        (5.0, 1000),
        (15.0, 10000),
    )


def test_a_worker_declares_no_video_shapes_by_default(tmp_path):
    config = worker_args(tmp_path, max_batch_tokens=8192)
    assert config.execution.video_graph_shapes == ()


@pytest.mark.parametrize("declaration", ["5", "5x0", "0x1000", "5x1000,5x1000"])
def test_malformed_video_shape_declarations_are_refused(declaration, tmp_path):
    with pytest.raises(SystemExit):
        worker_args(
            tmp_path,
            max_batch_tokens=8192,
            video_graph_shapes=declaration,
        )
