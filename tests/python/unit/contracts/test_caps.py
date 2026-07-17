from __future__ import annotations

import pytest

from uniserve_worker.contracts.caps import Caps, ExecutionConstraints, validate_caps
from uniserve_worker.nn.mesh import DeviceMesh, use_mesh
from uniserve_worker.worker.model import ModelWorker

pytestmark = pytest.mark.unit


def _caps() -> Caps:
    return Caps(
        block_size=256,
        num_blocks=1,
        num_layers=1,
        scratch_capacity_tokens=0,
        supported_ops=("prefill_und",),
        max_latent_size=0,
        latent_downsample=1,
        bytes_per_token=1,
        supported_controls=(),
        adapter_mode="none",
        execution_constraints=ExecutionConstraints(max_batch_ops=1),
        resource_classes=("kv_block",),
    )


def test_caps_rank_survives_validation_and_wire_conversion():
    raw = _caps().to_wire()
    raw["rank"] = {
        "tp_rank": 1,
        "tp_size": 2,
        "pp_rank": 0,
        "pp_size": 1,
        "dp_rank": 0,
        "dp_size": 1,
    }

    caps = validate_caps(raw)

    assert caps.tp_rank == 1
    assert caps.tp_size == 2
    assert caps.to_wire()["rank"]["tp_rank"] == 1
    assert caps.to_wire()["rank"]["tp_size"] == 2


def test_model_worker_caps_use_current_mesh_rank():
    with use_mesh(DeviceMesh.tp(1, 2, device="cpu")):
        caps = ModelWorker._caps_with_current_rank(_caps())

    assert caps.to_wire()["rank"]["tp_rank"] == 1
    assert caps.to_wire()["rank"]["tp_size"] == 2
