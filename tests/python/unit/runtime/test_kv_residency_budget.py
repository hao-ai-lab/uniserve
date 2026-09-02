from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.model_execution import TEST_DEPLOYMENT
from uniserve_worker.bootstrap.worker_info import build_worker_info
from uniserve_worker.models.runtime import CacheGeometry
from uniserve_worker.models.stub import StubModel

pytestmark = pytest.mark.unit

BLOCK_SIZE = 64
TOTAL_BYTES = 184 * 1024**3
STATIC_FRACTION = 0.70


@pytest.fixture(autouse=True)
def fixed_device_total(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True, raising=False)
    monkeypatch.setattr(
        "torch.cuda.mem_get_info",
        lambda device: (TOTAL_BYTES, TOTAL_BYTES),
        raising=False,
    )


def _model() -> StubModel:
    model = StubModel()
    model.cache_geometry = CacheGeometry(
        num_layers=32,
        num_attention_heads=8,
        num_kv_heads=8,
        head_dim=8,
        dtype="bfloat16",
    )
    return model


def _deployment(*, token_capacity: int | None):
    return replace(
        TEST_DEPLOYMENT,
        device="cuda:0",
        block_size=BLOCK_SIZE,
        kv_token_capacity=token_capacity,
        kv_memory_fraction=STATIC_FRACTION,
    )


def test_explicit_kv_capacity_provisions_one_physical_page_pool():
    info = build_worker_info(_model(), _deployment(token_capacity=131072))

    assert info.num_blocks * BLOCK_SIZE == 131072
