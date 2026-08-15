from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.model_execution import TEST_DEPLOYMENT
from uniserve_worker.bootstrap.capabilities import resolve_capabilities
from uniserve_worker.foundation.runtime_config import (
    graph_memory_budget_bytes,
    graph_padding_block_count,
)
from uniserve_worker.models.runtime import CacheGeometry, ScratchGeometry
from uniserve_worker.server.stub import StubModel

pytestmark = pytest.mark.unit

BLOCK_SIZE = 64
BYTES_PER_TOKEN = 8192
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


def _model(scratch: ScratchGeometry | None) -> StubModel:
    model = StubModel()
    model.cache_geometry = CacheGeometry(
        num_layers=32,
        num_attention_heads=8,
        num_kv_heads=8,
        head_dim=8,
        dtype="bfloat16",
    )
    model.resource_geometry = replace(model.resource_geometry, scratch=scratch)
    return model


def _deployment(*, token_capacity: int | None):
    return replace(
        TEST_DEPLOYMENT,
        device="cuda:0",
        block_size=BLOCK_SIZE,
        kv_token_capacity=token_capacity,
        kv_memory_fraction=STATIC_FRACTION,
    )


@pytest.mark.parametrize(
    ("scratch", "expected"),
    [
        (None, 0),
        (ScratchGeometry(fixed_tokens=65536), 65536),
        (ScratchGeometry(minimum_blocks=8, latent_copies=4), 524288),
        (ScratchGeometry(fixed_tokens=8192, mirror_kv=True), 139264),
    ],
)
def test_capabilities_advertise_the_provisioned_scratch_capacity(scratch, expected):
    capabilities = resolve_capabilities(_model(scratch), _deployment(token_capacity=131072))

    assert capabilities.num_blocks * BLOCK_SIZE == 131072
    assert capabilities.scratch_capacity_tokens == expected


@pytest.mark.parametrize("weight_bytes", [0, 35 * 1024**3, 62 * 1024**3])
def test_automatic_capacity_respects_the_static_memory_fraction(monkeypatch, weight_bytes):
    model = _model(ScratchGeometry(fixed_tokens=65536, mirror_kv=True))
    deployment = _deployment(token_capacity=None)
    monkeypatch.setattr(
        "torch.cuda.mem_get_info",
        lambda device: (TOTAL_BYTES - weight_bytes, TOTAL_BYTES),
        raising=False,
    )

    capabilities = resolve_capabilities(model, deployment)

    padding_blocks = graph_padding_block_count(BLOCK_SIZE)
    request_blocks = capabilities.num_blocks + padding_blocks
    scratch_blocks = capabilities.scratch_capacity_tokens // BLOCK_SIZE + padding_blocks
    static_bytes = (
        weight_bytes
        + (request_blocks + scratch_blocks) * BLOCK_SIZE * BYTES_PER_TOKEN
        + graph_memory_budget_bytes(TOTAL_BYTES)
    )
    assert static_bytes <= STATIC_FRACTION * TOTAL_BYTES
