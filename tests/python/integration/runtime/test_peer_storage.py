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


def test_export_names_each_allocation_by_one_identity():
    """Every view of one storage exports the same handle and identity.

    Distinct allocations never share an identity, and storage from another
    allocator is not exported where it lies.
    """
    device = torch.device("cuda:0")
    page = peer_storage.allocation_granularity(device)
    first = peer_storage.empty((page,), dtype=torch.uint8, device=device)
    second = peer_storage.empty((page,), dtype=torch.uint8, device=device)

    exported = peer_storage.export_handle(first)
    assert exported is not None
    handle, capacity, offset, identity = exported
    assert (capacity, offset) == (page, 0)
    assert identity > 0
    assert peer_storage.export_handle(first[64:]) == (
        handle,
        page,
        64,
        identity,
    )

    other = peer_storage.export_handle(second)
    assert other is not None
    assert other[3] != identity

    cached = torch.empty((page,), dtype=torch.uint8, device=device)
    assert peer_storage.export_handle(cached) is None
