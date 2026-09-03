"""Worker startup information across the Python IPC boundary."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.bootstrap.capacity import operation_window
from uniserve_worker.bootstrap.worker_info_builder import build_worker_info
from uniserve_worker.execution.batch import RunKind
from uniserve_worker.models.runtime import ExecutionModel, ResourceGeometry, WorkerDeployment
from uniserve_worker.process import dispatch

pytestmark = pytest.mark.integration


def test_worker_info_reports_schedulable_work_and_bounds() -> None:
    worker = execution_worker()
    info = dispatch(worker, {"kind": "info"})["info"]

    assert info["kv_cache"]["num_layers"] > 0
    assert info["kv_cache"]["num_kv_heads"] > 0
    assert info["kv_cache"]["head_dim"] > 0
    assert info["max_batch_ops"] > 0
    assert info["max_unresolved_ops"] == operation_window(
        info["queue_depth"], info["max_batch_ops"]
    )
    assert info["request_slots"] > 0
    assert info["model_name"]


def test_action_model_reports_zero_kv_geometry() -> None:
    class ActionModel(ExecutionModel):
        architecture = "ActionModel"
        resource_geometry = ResourceGeometry(kv=False)
        supported_work = frozenset({RunKind.DIFFUSION_DECODE})
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
    assert info.kv_cache is None
    assert info.request_slots == 2
    assert info.max_batch_ops == 2
