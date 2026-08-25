"""Behavioral contracts for model execution and worker identity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.model_execution import TEST_DEPLOYMENT, TEST_MODEL
from uniserve_worker.batch import WorkVariant
from uniserve_worker.bootstrap.capabilities import resolve_capabilities
from uniserve_worker.bootstrap.capacity import latent_trajectory_bytes, model_arena_capacity
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.foundation.math import ceil_div
from uniserve_worker.models.identity import ModelIdentity, architecture_identity

pytestmark = pytest.mark.unit


def test_architecture_identity_is_stable_and_binds_checkpoint_configuration():
    first = architecture_identity("ConformanceModel", {"layers": 2, "width": 8})

    assert first == architecture_identity("ConformanceModel", {"width": 8, "layers": 2})
    assert first != architecture_identity("ConformanceModel", {"layers": 3, "width": 8})
    assert first != architecture_identity("OtherModel", {"layers": 2, "width": 8})
    assert len(first) == 64
    int(first, 16)


def test_model_identity_requires_exact_checkpoint_digests():
    ModelIdentity("ConformanceModel", "a" * 64, "b" * 64)
    with pytest.raises(WorkerError):
        ModelIdentity("ConformanceModel", "A" * 64, "b" * 64)


def test_capabilities_project_model_behavior_and_resource_geometry():
    capabilities = resolve_capabilities(
        TEST_MODEL,
        TEST_DEPLOYMENT,
        architecture_digest="a" * 64,
        weight_digest="b" * 64,
    )

    assert WorkVariant.TOKEN_EXTEND in capabilities.supported_work
    assert WorkVariant.GEN_FLOW in capabilities.supported_work
    assert capabilities.mixed_buckets == ()
    assert capabilities.num_layers == TEST_MODEL.cache_geometry.num_layers
    assert capabilities.model_identity == "a" * 64
    assert capabilities.weight_digest == "b" * 64


def test_latent_capacity_rounds_to_complete_scheduler_pages() -> None:
    flow = TEST_MODEL.generation
    assert flow is not None
    deployment = replace(
        TEST_DEPLOYMENT,
        kv_token_capacity=int(flow.max_latent_tokens) + 1,
    )

    capabilities = resolve_capabilities(TEST_MODEL, deployment)

    expected_pages = ceil_div(
        int(flow.max_latent_tokens) + 1,
        int(deployment.block_size),
    )
    assert capabilities.num_latent_pages == expected_pages + 1
    assert capabilities.latent_capacity_units == expected_pages * int(deployment.block_size)


def test_transfer_capacity_covers_one_maximum_float32_trajectory_per_ticket() -> None:
    deployment = replace(TEST_DEPLOYMENT, model_dtype="float32")
    capabilities = resolve_capabilities(TEST_MODEL, deployment)
    flow = TEST_MODEL.generation
    assert flow is not None
    assert capabilities.max_latent_feature_bytes == latent_trajectory_bytes(
        int(flow.max_vae_grid_tokens),
        int(flow.latent_channels) * int(flow.latent_patch_size) ** 2,
        4,
    )
    arena = model_arena_capacity(
        TEST_MODEL,
        deployment,
        pipeline_depth=1,
        completion_payload_bytes=1024,
        num_blocks=2,
        request_pool_size=4,
        num_latent_pages=5,
        latent_page_units=4,
        latent_width=1024,
        max_latent_feature_bytes=1,
        max_vision_feature_bytes=1,
        bytes_per_token=1,
    )
    expected = latent_trajectory_bytes(int(flow.max_latent_tokens), 1024, 4)
    assert arena.transfer_bytes == expected * arena.transfer_tickets


@pytest.mark.parametrize(
    "deployment",
    [
        lambda: replace(TEST_DEPLOYMENT, tp_rank=1, tp_size=1),
        lambda: replace(TEST_DEPLOYMENT, tp_size=0),
        lambda: replace(TEST_DEPLOYMENT, block_size=0),
        lambda: replace(TEST_DEPLOYMENT, model_dtype="bf16"),
    ],
)
def test_worker_deployment_rejects_invalid_runtime_geometry(deployment):
    with pytest.raises(WorkerError):
        deployment()
