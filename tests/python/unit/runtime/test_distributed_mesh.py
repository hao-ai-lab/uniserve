"""Declared component geometry and physical launch validation."""

import pytest
import torch

from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.runtime.distributed import init_distributed_environment

pytestmark = pytest.mark.unit


def test_launch_rejects_unavailable_rank_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(Exception, match=r"cuda device cuda:1 is outside the 1 visible CUDA device"):
        init_distributed_environment(rank=1, local_rank=1, world_size=2, device="cuda")


def test_mesh_maps_ordered_members_and_independent_dimensions():
    mesh = DeviceMesh(
        (7, 3, 5, 1), 5, ParallelConfig(2, sequence_parallel=SequenceParallel("ulysses", (2,)))
    )
    assert mesh.get_coordinate() == (0, 0, 1, 0)
    assert mesh.group_members("tp") == ((7, 5), (3, 1))
    assert mesh.group_members("ulysses") == ((7, 3), (5, 1))
    assert mesh.size("sp") == 2
    with pytest.raises(ValueError, match="unknown mesh dimension"):
        mesh.get_group("typo")


def test_sequence_composite_holds_tensor_coordinates_fixed():
    mesh = DeviceMesh(
        tuple(range(8)), 6, ParallelConfig(2, sequence_parallel=SequenceParallel("hybrid", (2, 2)))
    )
    assert mesh.group_members("sp") == ((0, 1, 4, 5), (2, 3, 6, 7))
    assert mesh.coord("sp") == 2
    assert mesh.size("sp") == 4


@pytest.mark.parametrize(
    "value",
    [
        {"tensor_parallel_size": 0},
        {"tensor_parallel_size": True},
        {"sequence_parallel_size": 4},
        {"sequence_parallel": {"kind": "ulysses", "ring_degree": 2}},
        {"sequence_parallel": {"kind": "unknown"}},
    ],
)
def test_parallel_config_rejects_ambiguous_or_invalid_degrees(value):
    with pytest.raises(ValueError):
        ParallelConfig.from_dict(value)


def test_membership_requires_exact_product():
    with pytest.raises(ValueError, match="requires 4"):
        DeviceMesh(
            (0, 1), 0, ParallelConfig(2, sequence_parallel=SequenceParallel("ulysses", (2,)))
        )
