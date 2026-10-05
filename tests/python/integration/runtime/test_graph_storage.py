"""Independent graph pools consume one physical device budget."""

import pytest
import torch

from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve_worker.model_executor.graph_storage import GraphStorage

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_independent_graph_owners_share_a_byte_limit_and_retire_capacity():
    device = torch.device("cuda:0")
    storage = GraphStorage(budgets={device: 2 << 20})
    owners = (object(), object())
    try:
        for owner in owners:
            storage.reserve(owner, (device,))
        with storage.allocate(owners[0]):
            first = torch.empty(1 << 19, dtype=torch.uint8, device=device)
        storage.check()
        with storage.allocate(owners[1]):
            second = torch.empty_like(first)
        with pytest.raises(CUDAGraphError, match="exceeds its byte budget"):
            storage.check()
        # A worker can assign unused storage after its fixed pools are sized.
        # The new grant still covers every owner, including existing backing.
        storage.set_budget(device, 4 << 20)
        storage.check()
        with pytest.raises(CUDAGraphError, match="exceeds its byte budget"):
            storage.set_budget(device, 2 << 20)
        del second
        storage.release(owners[1])
        storage.check()
        del first
        storage.release(owners[0])
        storage.check()
    finally:
        storage.close()


def test_shared_pool_is_charged_once_and_survives_the_first_owner():
    device = torch.device("cuda:0")
    storage = GraphStorage(budgets={device: 2 << 20})
    first, second = object(), object()
    storage.reserve(first, (device,))
    storage.reserve(second, (device,), share=first)
    try:
        with storage.allocate(first):
            source = torch.full((1 << 17,), 7, dtype=torch.int32, device=device)
        with storage.allocate(second):
            output = torch.empty_like(source)
        output.copy_(source)
        torch.cuda.synchronize(device)

        storage.check()
        assert storage.pool_bytes() == {device: 2 << 20}
        assert storage.owner_bytes() == {(first, device): 2 << 20}

        del source
        storage.release(first)
        storage.check()
        assert storage.owner_bytes() == {(second, device): 2 << 20}
        assert torch.all(output == 7).item()

        del output
        storage.release(second)
        assert storage.pool_bytes() == {device: 0}
    finally:
        storage.close()


def test_failed_device_allocation_leaves_the_owner_available():
    device = torch.device("cuda:0")
    storage = GraphStorage()
    owner = object()
    try:
        unavailable = torch.device("cuda", torch.cuda.device_count())
        with pytest.raises(RuntimeError):
            storage.reserve(owner, (device, unavailable))

        storage.reserve(owner, ("cpu", device, "cuda:0"))
        with storage.allocate(owner):
            value = torch.ones(16, device=device)
        assert value.cpu().tolist() == [1] * 16
        storage.check()
        del value
        storage.release(owner)
    finally:
        storage.close()
