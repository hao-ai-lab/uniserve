"""Mapped CUDA storage keeps its published allocation available to readers."""

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.distributed import DeviceMesh
from uniserve.runtime import initialize_process_groups
from uniserve.runtime._peer_storage import (
    allocate_peer_tensor,
    allocate_symmetric_storage,
)
from uniserve_kernels import peer_storage

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("mapping", ["local", "peers"])
def test_mapping_keeps_exported_allocation_importable(mapping):
    """A late reader can import while the producer retains its tensor view."""
    if torch.cuda.device_count() < 2:
        pytest.skip("peer import requires two CUDA devices")
    if not all(peer_storage.exports_fabric_handles(rank) for rank in (0, 1)):
        pytest.skip("this test exercises fabric handle lifetime")

    device = torch.device("cuda:0")
    page = peer_storage.allocation_granularity(device)
    allocation = peer_storage.allocate(
        (page,), dtype=torch.uint8, device=device
    )
    handle = allocation.export_handle()
    source = (
        allocation.map_local()
        if mapping == "local"
        else allocation.map_peers([handle])
    )
    source.fill_(37)
    torch.cuda.synchronize(device)

    # Distributed allocation returns only a tensor; the original allocation
    # object can disappear before another rank imports its published handle.
    del allocation
    imported = peer_storage.import_handle(
        torch.empty(0, dtype=torch.uint8, device="cuda:1"), handle, page
    )
    assert imported.cpu().tolist() == [37] * page

    # Retirement is reader first, then the owner of the published storage.
    del imported
    del source


def _peer_views(rank, rendezvous):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        init_method=rendezvous,
    ) as environment:
        mesh = environment.bind(
            DeviceMesh(ranks=(1, 0), shape=(2,), axes=("tp",), rank=rank),
            device=device,
        )
        group = mesh.get_group("tp")
        page = peer_storage.allocation_granularity(device)
        mapped = allocate_peer_tensor(group, (page,), dtype=torch.uint8)
        mapped.view(2, page)[group.rank].fill_(rank + 1)

        workspace = allocate_symmetric_storage(
            group, (16,), dtype=torch.float32
        )
        workspace.local.fill_(rank + 1)
        arrival = torch.tensor([rank + 1], dtype=torch.int32, device=device)
        arrivals = torch.empty(2, dtype=torch.int32, device=device)
        workspace.fence(arrival, arrivals)

        assert arrivals.cpu().tolist() == [2, 1]
        assert mapped.view(2, page)[:, :4].cpu().tolist() == [[2] * 4, [1] * 4]
        for peer, value in zip(workspace.peers, (2, 1), strict=True):
            torch.testing.assert_close(
                peer, torch.full_like(peer, value), rtol=0, atol=0
            )

        # Every peer finishes reading before either process drops its backing.
        workspace.fence(arrival, arrivals)
        torch.cuda.synchronize(device)


def test_collective_allocations_preserve_logical_rank_order(tmp_path):
    mp.spawn(
        _peer_views,
        args=((tmp_path / "peers").as_uri(),),
        nprocs=2,
        join=True,
    )
