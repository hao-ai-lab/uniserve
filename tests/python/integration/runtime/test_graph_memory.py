"""Independent graph pools consume one physical device budget."""

import pytest
import torch

from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve_worker.execution.graph_memory import GraphMemory

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_independent_graph_owners_share_a_byte_limit_and_retire_capacity():
    device = torch.device("cuda:0")
    memory = GraphMemory(budgets={device: 2 << 20})
    owners = (object(), object())
    try:
        for owner in owners:
            memory.reserve(owner, (device,))
        with memory.allocate(owners[0]):
            first = torch.empty(1 << 19, dtype=torch.uint8, device=device)
        memory.check()
        with memory.allocate(owners[1]):
            second = torch.empty_like(first)
        with pytest.raises(CUDAGraphError, match="exceeds its byte budget"):
            memory.check()
        del second
        memory.release(owners[1])
        memory.check()
        del first
        memory.release(owners[0])
        memory.check()
    finally:
        memory.close()
