"""Device normalization for single-GPU (tp=1) worker launches."""

from __future__ import annotations

import pytest

from uniserve_worker.bootstrap.config import _normalize_device

pytestmark = pytest.mark.unit


def test_unindexed_cuda_is_pinned_to_concrete_index() -> None:
    # The frontend launches tp=1 workers with ``--device cuda`` while tensors
    # materialize on ``cuda:0``; the unindexed form must be pinned so the
    # per-forward device-equality check does not reject every output.
    assert _normalize_device("cuda") == "cuda:0"


def test_indexed_cuda_is_preserved() -> None:
    assert _normalize_device("cuda:0") == "cuda:0"
    assert _normalize_device("cuda:1") == "cuda:1"


def test_cpu_is_preserved() -> None:
    assert _normalize_device("cpu") == "cpu"
