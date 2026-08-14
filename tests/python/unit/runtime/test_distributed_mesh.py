from __future__ import annotations

import pytest
import torch

from uniserve_worker.execution.forward_batch import RouteMeshView
from uniserve_worker.nn.mesh import (
    CollectiveAxisTransport,
    CollectiveTransport,
    DeviceMesh,
    LocalP2PTransport,
    MeshAxis,
    PeerAxisTransport,
)
from uniserve_worker.nn.placement import Replicate, Shard, Sharding, reshard
from uniserve_worker.server.distributed import build_device_mesh

pytestmark = pytest.mark.unit


def test_tp_cuda_mesh_rejects_unavailable_rank_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(Exception, match=r"cuda device cuda:1 is outside the 1 visible CUDA device"):
        build_device_mesh(tp_rank=1, tp_size=2, device="cuda", tp_init_method="tcp://127.0.0.1:1")


def test_tower_cuda_mesh_rejects_unavailable_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(Exception, match=r"cuda device cuda:1 is outside the 1 visible CUDA device"):
        build_device_mesh(tp_rank=0, tp_size=1, device="cpu", tower_devices=["cuda:0", "cuda:1"])


def test_axis_transports_advertise_only_valid_operation_families():
    collective = CollectiveTransport(axis="tp", _size=2, _coord=0)
    peer = LocalP2PTransport(
        axis="tower",
        devices=(torch.device("cpu"), torch.device("cpu")),
        _coord=0,
    )

    assert isinstance(collective, CollectiveAxisTransport)
    assert not isinstance(collective, PeerAxisTransport)
    assert isinstance(peer, PeerAxisTransport)
    assert not isinstance(peer, CollectiveAxisTransport)


def test_mesh_view_rejects_collectives_on_a_routing_axis():
    peer = LocalP2PTransport(
        axis="tower",
        devices=(torch.device("cpu"), torch.device("cpu")),
        _coord=0,
    )
    view = RouteMeshView(
        DeviceMesh.of(MeshAxis("tower", 2, 0, peer), device="cpu"),
        ("tower",),
    )

    with pytest.raises(RuntimeError, match="does not support all-reduce"):
        view.all_reduce(torch.ones(1), "tower")


def test_local_sharding_rejects_uneven_geometry():
    transport = LocalP2PTransport(
        axis="tower",
        devices=(torch.device("cpu"), torch.device("cpu")),
        _coord=0,
    )
    mesh = DeviceMesh.of(MeshAxis("tower", 2, 0, transport), device="cpu")

    with pytest.raises(ValueError, match="not divisible"):
        reshard(
            torch.arange(3),
            Sharding((Replicate("tower"),)),
            Sharding((Shard("tower", 0),)),
            mesh,
        )
