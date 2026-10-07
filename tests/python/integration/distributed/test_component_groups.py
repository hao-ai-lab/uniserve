"""Worker component participation preserves independent mathematical groups."""

import socket

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.runtime import initialize_process_groups
from uniserve.runtime.process_groups import Rendezvous
from uniserve_worker.bootstrap.distributed import initialize_components
from uniserve_worker.config.deployment import (
    ComponentConfig,
    ParallelConfig,
    SequenceConfig,
)

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

        # A temporally distributed component computes each media unit on one
        # rank, so its mesh states no exchange. Consecutive units still cross
        # the ring between the ranks holding them, and the binding must report
        # that ring for callers that decide behavior from participation.
        (ring,) = tuple(
            group
            for group in bindings["decoder"].communicators
            if group.size > 1
        )
        assert ring.ranks == (3, 1)
        unit_value = torch.tensor([[rank + 1.0]], device=device)
        torch.testing.assert_close(
            ring.all_gather(unit_value),
            torch.tensor(
                [[member + 1.0] for member in ring.ranks], device=device
            ),
            rtol=0,
            atol=0,
        )

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


def _run_replicas(rank, port, backend):
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    replica, worker_rank = divmod(rank, 2)
    with initialize_process_groups(
        rank=worker_rank,
        local_rank=rank,
        world_size=2,
        device=device,
        backend=backend,
        experts=(rank, 4, Rendezvous("127.0.0.1", port)),
    ) as groups:
        value = torch.tensor([[rank + 1.0]], device=device)
        torch.testing.assert_close(
            groups.process_group.broadcast(value.clone(), src=1),
            torch.full_like(value, 2 * replica + 2),
            rtol=0,
            atol=0,
        )
        # Replicas bind different numbers of component groups. Each shares
        # the expert union but requires only its own ranks for TP startup.
        for _ in range(replica + 1):
            bindings = initialize_components(
                groups,
                {
                    "text": ComponentConfig((1, 0), ParallelConfig(2)),
                    "decoder": ComponentConfig(
                        (1, 0),
                        distribution="temporal_units",
                        units_per_rank=2,
                    ),
                    "output": ComponentConfig((1,)),
                },
            )
            assert bindings["output"].owns == (worker_rank == 1)
            offset = (1 - worker_rank) * 2
            assert tuple(bindings["decoder"].media_units(7, 3)) == tuple(
                range(7 + offset, 7 + min(offset + 2, 3))
            )
            torch.testing.assert_close(
                bindings["text"].mesh.get_group("tp").all_gather(value),
                torch.tensor(
                    [[2 * replica + 2.0], [2 * replica + 1.0]], device=device
                ),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            groups.experts.all_reduce(value.clone()),
            torch.full_like(value, 10),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "backend", ("gloo", pytest.param("nccl", marks=pytest.mark.gpu))
)
def test_tensor_replicas_share_an_independent_expert_union(backend):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_run_replicas, args=(port, backend), nprocs=4, join=True)
