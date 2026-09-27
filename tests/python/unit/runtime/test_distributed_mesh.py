"""Mathematical topology, tensor placements and physical launch validation."""

import pytest
import torch
from torch.distributed.tensor import Partial, Replicate, Shard

from uniserve.distributed import Communicator, DeviceMesh, Distribution
from uniserve.runtime.process_groups import initialize_process_groups

pytestmark = pytest.mark.unit


def test_launch_rejects_unavailable_rank_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(
        ValueError,
        match=r"cuda device cuda:1 is outside the 1 visible CUDA device",
    ):
        initialize_process_groups(
            rank=1, local_rank=1, world_size=2, device="cuda"
        )


def test_mesh_maps_ordered_members_and_independent_axes():
    mesh = DeviceMesh(
        ranks=(7, 3, 5, 1), shape=(2, 2), axes=("tensor", "heads"), rank=5
    )
    assert mesh.coordinate(5) == (1, 0)
    assert mesh.members(("tensor",)) == ((7, 5), (3, 1))
    assert mesh.members(("heads",)) == ((7, 3), (5, 1))
    assert mesh.members(("heads", "tensor")) == ((7, 5, 3, 1),)
    assert mesh.size(("heads", "tensor")) == 4
    group = mesh.get_group("tensor")
    assert (group.ranks, group.rank, group.global_rank, group.size) == (
        (7, 5),
        1,
        5,
        2,
    )
    with pytest.raises(ValueError, match="unknown or repeated"):
        mesh.get_group("typo")
    with pytest.raises(RuntimeError, match="requires initialized"):
        group.all_reduce(torch.ones(1))


def test_submesh_preserves_selected_order_and_fixes_other_coordinates():
    mesh = DeviceMesh(
        ranks=tuple(range(8)),
        shape=(2, 2, 2),
        axes=("context", "tensor", "heads"),
        rank=6,
    )
    assert mesh.members(("context", "heads")) == ((0, 1, 4, 5), (2, 3, 6, 7))
    selected = mesh.submesh(("heads", "context"))
    assert selected.axes == ("heads", "context")
    assert selected.ranks == (2, 6, 3, 7)
    assert selected.coordinate(6) == (0, 1)
    assert selected.get_group(selected.axes).rank == 1
    assert selected.submesh(()).ranks == (6,)


def test_nonmember_can_inspect_topology_without_executable_groups():
    mesh = DeviceMesh(ranks=(2, 4), shape=(2,), axes=("tensor",), rank=0)
    assert mesh.coordinate(4) == (1,)
    assert mesh.members(("tensor",)) == ((2, 4),)
    for call in (mesh.get_group, mesh.submesh):
        with pytest.raises(ValueError, match="nonparticipating"):
            call(("tensor",))


@pytest.mark.parametrize(
    "ranks,shape,axes",
    [
        ((0, 1), (4,), ("tensor",)),
        ((0, 0), (2,), ("tensor",)),
        ((0, 1), (2, 1), ("x", "x")),
        ((0,), (0,), ("x",)),
        ((0,), (1,), ("",)),
    ],
)
def test_mesh_rejects_inconsistent_topology(ranks, shape, axes):
    with pytest.raises(ValueError):
        DeviceMesh(ranks=ranks, shape=shape, axes=axes, rank=0)


def test_distribution_uses_upstream_tensor_placements():
    mesh = DeviceMesh(
        ranks=(0,),
        shape=(1, 1, 1),
        axes=("tokens", "channels", "replicas"),
        rank=0,
    )
    distribution = Distribution(mesh, (Shard(0), Shard(1), Replicate()))
    assert distribution.shard_axes(0) == ("tokens",)
    assert distribution.shard_axes(1) == ("channels",)
    assert distribution.shard_axes(2) == ()
    assert (
        Distribution(mesh, (Partial(), Replicate(), Replicate())).shard_axes(0)
        == ()
    )
    with pytest.raises(ValueError, match="one entry per mesh axis"):
        Distribution(mesh, (Replicate(),))


def test_one_member_all_gather_returns_its_input_or_fills_out():
    group = Communicator()
    value = torch.arange(6, dtype=torch.float32).view(2, 3)

    for dim in (0, 1):
        # The one-member concatenation is the input itself: same storage.
        assert group.all_gather(value, dim=dim) is value

        out = torch.empty_like(value)
        assert group.all_gather(value, dim=dim, out=out) is out
        torch.testing.assert_close(out, value, rtol=0, atol=0)
