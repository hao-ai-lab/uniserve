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
    assert "FlashInfer workspace size" in capsys.readouterr().err


# Representations no native CUDA attention kernel computes, with the option
# the usage error names.
UNSERVED_ON_CUDA = [
    ({"model_dtype": "float32"}, "--dtype float32"),
    ({"kv_cache_dtype": "float8_e4m3fn"}, "--kv-cache-dtype float8_e4m3fn"),
    (
        {"quantization_config": {"kv_cache_dtype": "float8_e4m3fn"}},
        "--quantization-config kv_cache_dtype float8_e4m3fn",
    ),
]


@pytest.mark.parametrize(("overrides", "option"), UNSERVED_ON_CUDA)
def test_a_cuda_launch_without_native_attention_is_a_usage_error(
    overrides, option, tmp_path, capsys
):
    with pytest.raises(SystemExit) as exit_info:
        worker_args(tmp_path, device="cuda:0", **overrides)

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert f"{option} has no native CUDA attention kernel" in error


@pytest.mark.parametrize(("overrides", "option"), UNSERVED_ON_CUDA)
def test_a_cpu_launch_keeps_the_portable_representations(
    overrides, option, tmp_path
):
    config = worker_args(tmp_path, device="cpu", **overrides)

    assert config.execution.device == "cpu"
    assert config.execution.model_dtype == overrides.get(
        "model_dtype", "bfloat16"
    )
