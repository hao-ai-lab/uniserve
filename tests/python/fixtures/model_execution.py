"""Canonical model and worker_config for execution-boundary tests."""

from tests.python.fixtures.worker_config import stub_worker_config
from uniserve.distributed.mesh import Communicator, DeviceMesh
from uniserve.distributed.parallel import ParallelConfig
from uniserve.model.limits import ModelLimits
from uniserve_models.stub import StubModel
from uniserve_worker.config import ComponentConfig
from uniserve_worker.execution.model_entry import ModelEntry

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


def model_arguments(layer):
    """Supply explicit numerical construction arguments for text/image models."""

    from uniserve.distributed.parallel import SequenceParallel

    return dict(
        parallel={
            "": ParallelConfig(
                tensor_parallel_size=layer.communicator.world_size,
                pipeline_parallel_size=layer.pipeline.world_size,
                sequence_parallel=(
                    SequenceParallel("ulysses", (layer.sequence.world_size,))
                    if layer.sequence.world_size > 1
                    else SequenceParallel()
                ),
            )
        },
        meshes={},
        layers={"": layer},
        limits=ModelLimits(text_tokens=1, video_frames=1),
    )


def h3_arguments(*, parallel, meshes, limits, precisions):
    """Bind H3's actual projection formats to borrowed numerical layer inputs."""

    from uniserve.nn.layer import LayerConfig
    from uniserve_models.minimax_h3.weights import configure_layers

    layers = {
        name: LayerConfig(
            mesh.get_group("tp"),
            None,
            pipeline=mesh.get_group("pp"),
            sequence=mesh.get_group("ulysses"),
        )
        for name, mesh in meshes.items()
    }
    return dict(
        parallel=parallel,
        meshes=meshes,
        layers=configure_layers(layers, precisions),
        limits=limits,
    )
