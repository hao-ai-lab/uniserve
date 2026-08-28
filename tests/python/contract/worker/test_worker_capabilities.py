"""Worker capability conformance across the Python wire boundary."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import ForwardMode, SamplingOwnership
from uniserve_worker.bootstrap.capabilities import resolve_capabilities
from uniserve_worker.bootstrap.capacity import operation_window
from uniserve_worker.models.runtime import ExecutionModel, ResourceGeometry, WorkerDeployment
from uniserve_worker.server.app import dispatch

pytestmark = pytest.mark.contract


def test_worker_capability_wire_reports_schedulable_work_and_bounds() -> None:
    worker = execution_worker()
    wire = dispatch(worker, {"kind": "get_capabilities"})["capabilities"]

    assert wire["num_layers"] > 0
    assert wire["num_kv_heads"] > 0
    assert wire["head_dim"] > 0
    assert wire["max_batch_operations"] > 0
    assert wire["max_unresolved_window"] == operation_window(
        wire["pipeline_depth"], wire["max_batch_operations"]
    )
    assert wire["incremental_kv_publication"] is True
    assert wire["sampling_ownership"] == SamplingOwnership.DESIGNATED_RANK.value


def test_action_model_reports_zero_kv_geometry() -> None:
    class ActionModel(ExecutionModel):
        architecture = "ActionModel"
        resource_geometry = ResourceGeometry(kv=False)
        supported_work = frozenset({ForwardMode.GEN_DECODE})
        generation = None

    deployment = WorkerDeployment(
        device="cpu",
        model_scope="whole",
        tp_rank=0,
        tp_size=1,
        block_size=64,
        kv_token_capacity=None,
        attention_backend=None,
        model_dtype="bfloat16",
        kv_cache_dtype=None,
        kv_memory_fraction=0.9,
        max_batch_operations=2,
        max_batch_tokens=2,
        max_request_pool_size=2,
        generation_device=None,
    )

    capabilities = resolve_capabilities(ActionModel(), deployment)

    assert capabilities.uses_kv is False
    assert capabilities.block_size == 0
    assert capabilities.num_blocks == 0
    assert capabilities.num_layers == 0
    assert capabilities.num_kv_heads == 0
    assert capabilities.head_dim == 0
    assert capabilities.bytes_per_token == 0
    assert capabilities.groups == ()
    assert capabilities.kv_dtype == ""
    assert capabilities.incremental_kv_publication is False
