"""Worker startup information across the Python IPC boundary."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import ForwardMode, SamplingOwnership
from uniserve_worker.bootstrap.worker_info import build_worker_info
from uniserve_worker.bootstrap.capacity import operation_window
from uniserve_worker.models.runtime import ExecutionModel, ResourceGeometry, WorkerDeployment
from uniserve_worker.server.app import dispatch

pytestmark = pytest.mark.integration


def test_worker_info_reports_schedulable_work_and_bounds() -> None:
    worker = execution_worker()
    info = dispatch(worker, {"kind": "get_info"})["info"]

    assert info["num_layers"] > 0
    assert info["num_kv_heads"] > 0
    assert info["head_dim"] > 0
    assert info["max_batch_operations"] > 0
    assert info["max_unresolved_window"] == operation_window(
        info["pipeline_depth"], info["max_batch_operations"]
    )
    assert info["incremental_kv_publication"] is True
    assert info["sampling_ownership"] == SamplingOwnership.DESIGNATED_RANK.value


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

    info = build_worker_info(ActionModel(), deployment)

    assert info.uses_kv is False
    assert info.block_size == 0
    assert info.num_blocks == 0
    assert info.num_layers == 0
    assert info.num_kv_heads == 0
    assert info.head_dim == 0
    assert info.bytes_per_token == 0
    assert info.groups == ()
    assert info.kv_dtype == ""
    assert info.incremental_kv_publication is False
