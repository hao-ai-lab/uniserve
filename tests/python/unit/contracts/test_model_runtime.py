"""Behavioral contracts for model execution and worker identity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.model_execution import TEST_DEPLOYMENT, TEST_MODEL
from uniserve_worker.batch import WorkVariant
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.models.identity import ModelIdentity, architecture_identity
from uniserve_worker.runtime.capabilities import resolve_capabilities

pytestmark = pytest.mark.unit


def test_architecture_identity_is_stable_and_binds_checkpoint_configuration():
    first = architecture_identity("ConformanceModel", {"layers": 2, "width": 8})

    assert first == architecture_identity("ConformanceModel", {"width": 8, "layers": 2})
    assert first != architecture_identity("ConformanceModel", {"layers": 3, "width": 8})
    assert first != architecture_identity("OtherModel", {"layers": 2, "width": 8})
    assert len(first) == 64
    int(first, 16)


def test_model_identity_requires_exact_checkpoint_digests():
    identity = ModelIdentity("ConformanceModel", "a" * 64, "b" * 64)

    assert identity.architecture_digest == "a" * 64
    assert identity.weight_digest == "b" * 64
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
    assert capabilities.tensorized_mixed
    assert capabilities.num_layers == TEST_MODEL.cache_geometry.num_layers
    assert capabilities.resource_classes == (
        "kv_block",
        "encoder_output",
        "image_latent",
        "scratch",
    )
    assert capabilities.model_identity == "a" * 64
    assert capabilities.weight_digest == "b" * 64


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
