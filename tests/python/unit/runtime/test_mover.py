"""Mover: the per-worker transfer authority.

One Mover per worker owns the single register-once Transport and selects the
und↔gen tower-handoff implementation for the worker's deployment edge: an
in-process edge gets the local peer-copy handoff, a cross-process edge gets the
data-plane handoff riding the worker Transport. A cross-process edge configured
with an in-process transport must fail at construction, not at first crossing.
"""
from __future__ import annotations

import pytest

from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.mover import Mover
from uniserve_worker.runtime.tower_handoff import (
    DataPlaneTowerHandoff,
    LocalP2PTowerHandoff,
    TowerBinding,
)


def _binding() -> TowerBinding:
    return TowerBinding(
        transport=None,
        primary_coord=0,
        gen_coord=0,
        num_layers=1,
        block_size=4,
        target_pool=None,
        target_device="cpu",
        allocate_blocks=lambda n: list(range(n)),
    )


def test_in_process_edge_selects_the_local_peer_copy_handoff():
    mover = Mover(transfer_backend="local")
    handoff = mover.tower_handoff(_binding)
    assert isinstance(handoff, LocalP2PTowerHandoff)
    assert handoff.bind is _binding


def test_cross_process_edge_selects_the_data_plane_over_the_worker_transport():
    mover = Mover(transfer_backend="cuda_ipc", cross_process=True)
    handoff = mover.tower_handoff(_binding)
    assert isinstance(handoff, DataPlaneTowerHandoff)
    assert handoff.data_plane is mover.transport
    assert handoff.bind is _binding


def test_one_transport_per_worker_is_shared_across_consumers():
    mover = Mover(transfer_backend="local")
    assert mover.transport is mover.transport
    second_edge = Mover(transfer_backend="cuda_ipc", cross_process=True)
    assert second_edge.tower_handoff(_binding).data_plane is second_edge.transport


@pytest.mark.parametrize("backend", ["", "local", "shm"])
def test_cross_process_edge_rejects_in_process_backends(backend: str):
    with pytest.raises(WorkerError, match="cross-process"):
        Mover(transfer_backend=backend, cross_process=True)
