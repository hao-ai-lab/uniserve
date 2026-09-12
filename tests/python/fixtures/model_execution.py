"""Canonical model and worker_config for execution-boundary tests."""

from uniserve_worker.execution.model_entry import ModelEntry
from uniserve_worker.models.stub import StubModel, stub_worker_config
from uniserve_worker.nn.mesh import Communicator, DeviceMesh
from uniserve_worker.nn.parallel import ComponentConfig, ParallelConfig

TEST_MODEL = StubModel()
TEST_WORKER_CONFIG = stub_worker_config(64, max_batch_tokens=8192)

__all__ = ["TEST_WORKER_CONFIG", "TEST_MODEL"]


def tensor_parallel_bindings(group: Communicator = Communicator()) -> dict[str, ModelEntry]:
    """Construct the concrete rank geometry for a standalone model checkpoint."""

    parallel = ParallelConfig(tensor_parallel_size=group.world_size)
    mesh = DeviceMesh(group.ranks, group.rank, parallel, group.device, {"tp": group})
    return {
        "model": ModelEntry(
            "model", ComponentConfig(group.ranks, parallel), group, mesh, group.device
        )
    }
