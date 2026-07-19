"""Correctness contracts for plan/forward stream overlap."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_coordinator_requires_cuda_device():
    from uniserve_worker.execution.runner import PlanStreamOverlap

    with pytest.raises(ValueError):
        PlanStreamOverlap(torch.device("cpu"), max_inflight=3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_prepare_launch_orders_consumption_after_plan_stream():
    from uniserve_worker.execution.runner import PlanStreamOverlap

    device = torch.device("cuda", torch.cuda.current_device())
    overlap = PlanStreamOverlap(device, max_inflight=3)
    outputs = []
    for step in range(8):
        host = torch.full((1024,), float(step), pin_memory=True)

        def prepare(host_tensor=host):
            staged = torch.empty_like(host_tensor, device=device)
            staged.copy_(host_tensor, non_blocking=True)
            return staged

        prepared = overlap.prepare(prepare)
        staged = prepared.value

        def run(value=staged):
            return value * 2.0

        outputs.append(overlap.launch(prepared, run, retain=staged))
    overlap.drain()
    for step, out in enumerate(outputs):
        assert torch.equal(out.cpu(), torch.full((1024,), float(step * 2)))
