from __future__ import annotations

import dataclasses

import pytest

from tests.python.fixtures.model_execution import TEST_DEPLOYMENT
from uniserve_worker.foundation.runtime_config import (
    decode_graph_padding_block_count,
    graph_memory_budget_bytes,
)
from uniserve_worker.foundation.sizing import ceil_div, derive_runtime_kv_capacity
from uniserve_worker.runtime import capabilities as capabilities_module
from uniserve_worker.runtime.capabilities import _kv_residency_shape, _scratch_capacity_tokens
from uniserve_worker.spec import PerBranch, ResourcePlan

pytestmark = pytest.mark.unit

BLOCK_SIZE = 64
BYTES_PER_TOKEN = 8192
MAX_LATENT_SIZE = 4096
TOTAL_BYTES = 184 * 1024**3
STATIC_FRACTION = 0.70


@pytest.fixture(autouse=True)
def fixed_device_total(monkeypatch):
    monkeypatch.setattr(capabilities_module, "device_total_bytes", lambda _device: TOTAL_BYTES)


def _deployment(scratch: PerBranch | None):
    return dataclasses.replace(
        TEST_DEPLOYMENT,
        device="cuda:0",
        block_size=BLOCK_SIZE,
        kv_token_capacity=None,
        kv_memory_fraction=STATIC_FRACTION,
        resources=ResourcePlan(scratch=scratch),
    )


def _shape(deployment):
    return _kv_residency_shape(
        deployment,
        max_latent_size=MAX_LATENT_SIZE,
        bytes_per_token=BYTES_PER_TOKEN,
    )


def _kv_blocks(deployment, num_blocks: int) -> int:
    """Blocks the residency store provisions for one derived request capacity."""
    padding = decode_graph_padding_block_count(BLOCK_SIZE)
    total = num_blocks + padding
    if deployment.resources.scratch is not None:
        scratch_tokens = _scratch_capacity_tokens(
            deployment,
            num_blocks=num_blocks,
            max_latent_size=MAX_LATENT_SIZE,
        )
        total += ceil_div(scratch_tokens, BLOCK_SIZE) + padding
    return total


@pytest.mark.parametrize(
    "scratch",
    [
        None,
        PerBranch(fixed_tokens=65536, mirror_kv=True),
        PerBranch(minimum_blocks=8, mirror_kv=True, latent_copies=4),
        PerBranch(fixed_tokens=8192),
    ],
)
@pytest.mark.parametrize("num_blocks", [1, 977, 65536])
def test_residency_shape_covers_every_pool_provisioned_for_the_capacity(scratch, num_blocks):
    deployment = _deployment(scratch)

    copies, co_resident = _shape(deployment)

    assert copies * num_blocks + co_resident >= _kv_blocks(deployment, num_blocks)


@pytest.mark.parametrize(
    "scratch",
    [None, PerBranch(fixed_tokens=65536, mirror_kv=True)],
)
def test_residency_shape_reserves_the_graph_executable_budget(scratch):
    deployment = _deployment(scratch)

    _copies, co_resident = _shape(deployment)

    graph_blocks = ceil_div(graph_memory_budget_bytes(TOTAL_BYTES), BLOCK_SIZE * BYTES_PER_TOKEN)
    assert co_resident >= graph_blocks


@pytest.mark.parametrize("weight_bytes", [0, 35 * 1024**3, 62 * 1024**3])
def test_static_fraction_bounds_weights_kv_pools_and_graph_executables(monkeypatch, weight_bytes):
    deployment = _deployment(PerBranch(fixed_tokens=65536, mirror_kv=True))
    copies, co_resident = _shape(deployment)
    monkeypatch.setattr(
        "torch.cuda.mem_get_info",
        lambda device: (TOTAL_BYTES - weight_bytes, TOTAL_BYTES),
        raising=False,
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True, raising=False)

    capacity = derive_runtime_kv_capacity(
        block_size=BLOCK_SIZE,
        kv_token_capacity=None,
        bytes_per_token=BYTES_PER_TOKEN,
        device="cuda:0",
        memory_fraction=STATIC_FRACTION,
        resident_copies=copies,
        co_resident_blocks=co_resident,
    )

    kv_bytes = _kv_blocks(deployment, capacity.num_blocks) * BLOCK_SIZE * BYTES_PER_TOKEN
    static_bytes = weight_bytes + kv_bytes + graph_memory_budget_bytes(TOTAL_BYTES)
    assert static_bytes <= STATIC_FRACTION * TOTAL_BYTES


def test_declared_token_capacity_sizes_the_request_pool_exactly():
    deployment = _deployment(PerBranch(fixed_tokens=65536, mirror_kv=True))
    copies, co_resident = _shape(deployment)

    capacity = derive_runtime_kv_capacity(
        block_size=BLOCK_SIZE,
        kv_token_capacity=131072,
        bytes_per_token=BYTES_PER_TOKEN,
        device="cuda:0",
        memory_fraction=STATIC_FRACTION,
        resident_copies=copies,
        co_resident_blocks=co_resident,
    )

    assert capacity.token_capacity == 131072
