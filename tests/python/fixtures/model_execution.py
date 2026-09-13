"""Canonical model and worker_config for execution-boundary tests."""

from tests.python.fixtures.worker_config import stub_worker_config
from uniserve_worker.execution.model_entry import ModelEntry
from uniserve_worker.models.stub import StubModel
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


def model_context(layers):
    """Supply numerical layer geometry for direct model composition tests."""
    from uniserve_worker.modeling.context import BuildContext
    from uniserve_worker.nn.parallel import SequenceParallel

    return BuildContext(
        parallel={
            "model": ParallelConfig(
                tensor_parallel_size=layers.communicator.world_size,
                pipeline_parallel_size=layers.pipeline.world_size,
                sequence_parallel=(
                    SequenceParallel("ulysses", (layers.sequence.world_size,))
                    if layers.sequence.world_size > 1
                    else SequenceParallel()
                ),
            )
        },
        meshes={},
        layers={"model": layers},
        limits={},
        component_precisions={},
        schedule=None,
    )
