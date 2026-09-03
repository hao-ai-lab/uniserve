"""Model execution and worker resource behavior."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.model_execution import TEST_DEPLOYMENT, TEST_MODEL
from uniserve_worker.bootstrap.capacity import latent_trajectory_bytes, model_arena_capacity
from uniserve_worker.bootstrap.worker_info_builder import build_worker_layout
from uniserve_worker.execution.batch import RunKind
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.foundation.math import ceil_div

pytestmark = pytest.mark.unit


def test_worker_info_projects_model_behavior_and_resource_geometry():
    layout = build_worker_layout(
        TEST_MODEL,
        TEST_DEPLOYMENT,
        model_name="test-model",
        weight_version=7,
    )
    info = layout.info

    assert RunKind.AR_EXTEND in info.supported_ops
    assert RunKind.DIFFUSION_STEP in info.supported_ops
    assert layout.max_vision_feature_bytes == (
        int(TEST_MODEL.max_vit_grid_tokens) * int(TEST_MODEL.hidden_size) * 2
    )
    assert info.kv_cache is not None
    assert info.kv_cache.num_layers == TEST_MODEL.cache_geometry.num_layers
    assert info.model_name == "test-model"
    assert info.weight_version == 7


def test_latent_capacity_rounds_to_complete_scheduler_pages() -> None:
    flow = TEST_MODEL.generation
    assert flow is not None
    deployment = replace(
        TEST_DEPLOYMENT,
        kv_token_capacity=int(flow.max_latent_tokens) + 1,
    )

    layout = build_worker_layout(TEST_MODEL, deployment)

    expected_pages = ceil_div(
        int(flow.max_latent_tokens) + 1,
        int(deployment.block_size),
    )
    assert layout.info.latent_pages == expected_pages + 1
    assert layout.info.latent_capacity_units == expected_pages * int(deployment.block_size)


def test_persistent_buffer_capacity_includes_active_encoder_output() -> None:
    layout = build_worker_layout(TEST_MODEL, TEST_DEPLOYMENT)
    feature_bytes = max(
        layout.max_latent_feature_bytes,
        layout.max_vision_feature_bytes,
    )
    assert layout.info.buffer_pool_bytes == (
        int(TEST_MODEL.resource_geometry.encoder_cache_entries) + 1
    ) * feature_bytes


def test_transfer_capacity_covers_one_maximum_float32_trajectory_per_ticket() -> None:
    deployment = replace(TEST_DEPLOYMENT, model_dtype="float32")
    layout = build_worker_layout(TEST_MODEL, deployment)
    flow = TEST_MODEL.generation
    assert flow is not None
    assert layout.max_latent_feature_bytes == latent_trajectory_bytes(
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
