"""Plan/forward stream-overlap coordinator (specs/intra-worker-stream-overlap.md)."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts.op_kinds import DECODE_UND, DENOISE_GEN, PREFILL_UND
from uniserve_worker.execution.engine import _overlap_eligible

pytestmark = pytest.mark.unit


def _group(*kinds: str) -> list[tuple[int, dict]]:
    return [(index, {"kind": kind}) for index, kind in enumerate(kinds)]


def test_overlap_covers_text_only_groups():
    assert _overlap_eligible(_group(DECODE_UND, DECODE_UND))
    assert _overlap_eligible(_group(PREFILL_UND))
    assert _overlap_eligible(_group(PREFILL_UND, DECODE_UND))


def test_overlap_excludes_non_text_groups():
    assert not _overlap_eligible(_group(DENOISE_GEN))
    assert not _overlap_eligible(_group(DECODE_UND, DENOISE_GEN))


def test_coordinator_requires_cuda_device():
    from uniserve_worker.execution.engine import PlanStreamOverlap

    with pytest.raises(ValueError):
        PlanStreamOverlap(torch.device("cpu"), max_inflight=3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_prepare_launch_orders_consumption_after_plan_stream():
    from uniserve_worker.execution.engine import PlanStreamOverlap

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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_inflight_forwards_are_bounded_by_ring_depth():
    from uniserve_worker.execution.engine import PlanStreamOverlap

    device = torch.device("cuda", torch.cuda.current_device())
    overlap = PlanStreamOverlap(device, max_inflight=2)
    for _ in range(6):
        prepared = overlap.prepare(lambda: torch.ones(8, device=device))
        overlap.launch(prepared, lambda: None, retain=prepared.value)
        assert len(overlap._inflight) <= 2
    overlap.drain()
    assert not overlap._inflight
