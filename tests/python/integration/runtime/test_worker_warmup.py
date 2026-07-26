"""Startup warmup drives valid batches through the real execution path.

The warmup exists to pay first-use attention-kernel JIT before the worker is
reachable. Its value is only realized on CUDA, but the synthetic prefill /
decode / flow batches it constructs must stay valid: earlier drafts tripped
the ``session_id >= 1`` transaction-identity rule and the flow CFG
``branch_count`` bound. These tests exercise the construction on the CPU stub
so a regression in the batch shapes is caught without a GPU.
"""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.spec import OperationType

pytestmark = pytest.mark.integration


def test_warmup_is_a_safe_noop_off_cuda() -> None:
    # The public entry point guards on CUDA; on the stub's CPU device it must
    # return without raising and without leaving sessions behind.
    worker = execution_worker()
    worker.warmup()
    assert worker.sessions.session_ids() == ()


def test_warmup_image_geometry_fits_the_declared_latent_capacity() -> None:
    worker = execution_worker()
    caps = worker.contract.capabilities
    if caps.max_latent_size <= 0:
        pytest.skip("stub declares no image latent capacity")

    height, width = worker._warmup_image_geometry()

    downsample = caps.latent_downsample
    latent_tokens = (height // downsample) * (width // downsample)
    assert latent_tokens <= caps.max_latent_size
    if caps.max_vae_grid_tokens > 0:
        assert latent_tokens <= caps.max_vae_grid_tokens


def test_warmup_sequence_runs_and_cleans_up() -> None:
    worker = execution_worker()
    types = worker.contract.capabilities.supported_operation_types
    if OperationType.SEQUENCE_EXTEND not in types:
        pytest.skip("stub does not support sequence extend")
    # Directly drive the construction the CUDA gate would otherwise skip.
    worker._warmup_sequence()
    # The warmup owns session id 1 transiently and must drop it before serving.
    assert 1 not in worker.sessions.session_ids()
