"""Mapped CUDA storage keeps its published allocation available to readers."""

import pytest
import torch

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
