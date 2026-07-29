from __future__ import annotations

import gc
import time
import weakref

import pytest
import torch

from uniserve_worker.runtime.completion_store import CompletionArena

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_completion_copy_progresses_through_event_queries() -> None:
    device = torch.device("cuda:0")
    arena = CompletionArena(depth=2, token_capacity=4, devices=(device,))
    pending = []
    for expected in ((3, 7, 11), (13,)):
        lease = arena.reserve(1, devices=(device,))
        source = torch.tensor(expected, dtype=torch.long, device=device)
        retired: list[bool] = []
        weakref.finalize(source, retired.append, True)
        capture = lease.capture(source)
        lease.seal()
        del source
        gc.collect()
        assert retired == [True]
        pending.append((lease, capture, expected))

    deadline = time.monotonic() + 5.0
    while not all(lease.ready() for lease, _capture, _expected in pending):
        if time.monotonic() >= deadline:
            raise TimeoutError("completion events did not become query-ready")

    for lease, capture, expected in reversed(pending):
        assert capture.values() == expected
        lease.observe(0, lease.generation)
