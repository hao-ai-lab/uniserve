"""Video capacity settings carried by a worker launch."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args

pytestmark = pytest.mark.unit


def test_video_capacities_carry_text_capacities_and_the_admitted_floor(
    tmp_path,
):
    config = worker_args(
        tmp_path,
        max_batch_tokens=8192,
        video_text_capacities="1024,4096,8192",
        min_video_seconds=4.0,
    )
    assert config.execution.video_text_capacities == (1024, 4096, 8192)
    assert config.execution.min_video_seconds == 4.0


def test_a_worker_uses_default_video_capacities_without_settings(tmp_path):
    config = worker_args(tmp_path, max_batch_tokens=8192)
    assert config.execution.video_text_capacities == ()
    assert config.execution.min_video_seconds is None


@pytest.mark.parametrize("capacities", ["0,1024", "4096,1024", "1024,1024"])
def test_malformed_text_capacities_are_refused(capacities, tmp_path):
    with pytest.raises(SystemExit):
        worker_args(
            tmp_path,
            max_batch_tokens=8192,
            video_text_capacities=capacities,
        )
