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
        del second
        storage.release(owners[1])
        storage.check()
        del first
        storage.release(owners[0])
        storage.check()
    finally:
        storage.close()
