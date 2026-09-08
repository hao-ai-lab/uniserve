"""Worker startup information across the Python IPC boundary."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.bootstrap.capacity import operation_window
from uniserve_worker.bootstrap.worker_info_builder import build_worker_info
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.batch import OpCode
from uniserve_worker.models.runtime import ExecutionModel, ResourceGeometry
from uniserve_worker.process import dispatch

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backends", (("local",), ("local", "shm")))
def test_worker_info_reports_schedulable_work_and_bounds(backends: tuple[str, ...]) -> None:
    worker = execution_worker(transfer_backends=backends)
    try:
        info = dispatch(worker, {"kind": "info"})["info"]
    finally:
        worker.close()

    assert info["device"] == "cpu"
    assert tuple(info["transfer_backends"]) == backends
    assert info["kv_cache"]["num_layers"] > 0
    assert info["kv_cache"]["num_kv_heads"] > 0
    assert info["kv_cache"]["head_dim"] > 0
    assert info["max_batch_ops"] > 0
    assert info["max_unresolved_ops"] == operation_window(
        info["queue_depth"], info["max_batch_ops"]
    )
    assert info["request_slots"] > 0
    assert info["model_name"]
    assert len(info["configuration_id"]) == 64


def test_action_model_reports_zero_kv_geometry() -> None:
    class ActionModel(ExecutionModel):
        architecture = "ActionModel"
        resource_geometry = ResourceGeometry(kv=False)
        supported_work = frozenset({OpCode.DIFFUSION_DECODE})
        generation = None

    worker_config = WorkerConfig(
        device="cpu",
        rank=0,
        world_size=1,
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

    info = build_worker_info(ActionModel(), worker_config)

    assert info.uses_kv is False
    assert info.kv_cache is None
    assert info.request_slots == 2
    assert info.max_batch_ops == 2


def test_loaded_worker_identity_distinguishes_incarnations_in_one_process() -> None:
    snapshots = []
    for grant in (None, 1 << 40):
        worker = execution_worker(
            worker_id="encoder-0",
            execution=WorkerConfig(
                cuda_graph=False,
                prefill_cuda_graph=False,
                flow_graph_batch_sizes=(1,),
                flow_graph_shapes=((16, 16),),
                pool_memory_bytes=grant,
            ),
        )
        try:
            snapshots.append(dispatch(worker, {"kind": "info"})["info"])
        finally:
            worker.close()
    first, second = snapshots
    assert first["endpoint"]["worker_id"] == second["endpoint"]["worker_id"] == "encoder-0"
    assert first["endpoint"]["rank"] == second["endpoint"]["rank"] == 0
    assert first["world_size"] == second["world_size"] == 1
    assert first["endpoint"]["node"] == second["endpoint"]["node"]
    assert first["endpoint"]["address_space"] == second["endpoint"]["address_space"]
    assert first["endpoint"]["incarnation"] != second["endpoint"]["incarnation"]
    assert first["configuration_id"] == second["configuration_id"]
