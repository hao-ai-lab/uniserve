"""Attention backend settings carried by a worker launch."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args

pytestmark = pytest.mark.unit


def test_the_flashinfer_workspace_size_reaches_the_attention_config(
    tmp_path,
):
    config = worker_args(tmp_path, flashinfer_workspace_size=64 << 20)

    assert config.execution.flashinfer.workspace_size == 64 << 20


@pytest.mark.parametrize("size", [0, -5])
def test_a_non_positive_flashinfer_workspace_is_a_usage_error(
    size, tmp_path, capsys
):
    # argparse reports a usage error with exit status 2.
    with pytest.raises(SystemExit) as exit_info:
        worker_args(tmp_path, flashinfer_workspace_size=size)

    assert exit_info.value.code == 2
    assert "--flashinfer-workspace-size" in capsys.readouterr().err
