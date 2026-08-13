"""Public startup qualification behavior."""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.foundation.runtime_config import ExecutionConfig

pytestmark = pytest.mark.integration


def test_warmup_is_a_safe_noop_off_cuda() -> None:
    worker = execution_worker()
    worker.warmup()
    worker.warmup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prefill_graph_startup_reuses_released_request_slots() -> None:
    worker = execution_worker(
        device="cuda:0",
        max_batch_tokens=2048,
        max_request_pool_size=2,
        execution=ExecutionConfig(
            cuda_graph=True,
            prefill_cuda_graph=True,
            decode_graph_batch_sizes=(),
            prefill_graph_token_sizes=(1, 2, 3, 4),
        ),
    )

    worker.warmup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_decode_graph_startup_advances_across_kv_pages() -> None:
    worker = execution_worker(
        device="cuda:0",
        block_size=16,
        max_batch_tokens=2048,
        max_request_pool_size=32,
        execution=ExecutionConfig(
            cuda_graph=True,
            prefill_cuda_graph=False,
        ),
    )

    worker.warmup()
