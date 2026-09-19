"""Worker component participation preserves independent mathematical groups."""

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.runtime import initialize_process_groups
from uniserve_worker.bootstrap.config import (
    ComponentConfig,
    ParallelConfig,
    SequenceConfig,
)
from uniserve_worker.bootstrap.distributed import initialize_components

pytestmark = pytest.mark.integration


@torch.inference_mode()
def _run_groups(rank: int, rendezvous: str, backend: str):
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    environment = initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=device,
        backend=backend,
        init_method=rendezvous,
    )
    bindings = initialize_components(
        environment,
        {
            "denoiser": ComponentConfig(
                (0, 1, 2, 3),
                ParallelConfig(
                    2, sequence_parallel=SequenceConfig("ulysses", (2,))
                ),
            ),
            "encoder": ComponentConfig((3, 1), ParallelConfig(2)),
            "output": ComponentConfig((2,)),
            "decoder": ComponentConfig(
                (3, 1), distribution="temporal_units", units_per_rank=2
            ),
        },
    )
    meshes = {
        name: entry.mesh
        for name, entry in bindings.items()
        if entry.mesh is not None
    }
    assert bindings["decoder"].input_ranks == (3, 1)
    assert bindings["decoder"].output_ranks == (3, 1)
    assert ("decoder" in bindings and bindings["decoder"].owns) == (
        rank in (3, 1)
    )
    if "decoder" in bindings and bindings["decoder"].owns:
        value = torch.tensor([rank + 1.0], device=device)
        result = meshes["decoder"].get_group("tp").all_reduce(value.clone())
        torch.testing.assert_close(result, value, rtol=0, atol=0)

    for mesh in meshes.values():
        for dimension in mesh.axes:
            group = mesh.get_group(dimension)
            value = torch.tensor([[rank + 1.0]], device=device)
            expected = torch.tensor(
                [[sum(member + 1.0 for member in group.ranks)]], device=device
            )
            torch.testing.assert_close(
                group.all_reduce(value.clone()), expected, rtol=0, atol=0
            )
            gathered = group.all_gather(value)
            reference = torch.tensor(
                [[member + 1.0] for member in group.ranks], device=device
            )
            torch.testing.assert_close(gathered, reference, rtol=0, atol=0)
            root = group.size - 1
            torch.testing.assert_close(
                group.broadcast(value.clone(), src=root),
                reference[root : root + 1],
                rtol=0,
                atol=0,
            )
    environment.close()


@pytest.mark.parametrize(
    "backend", ("gloo", pytest.param("nccl", marks=pytest.mark.gpu))
)
def test_component_participation_and_collective_values(tmp_path, backend):
    mp.spawn(
        _run_groups,
        args=((tmp_path / "components").as_uri(), backend),
        nprocs=4,
        join=True,
    )
