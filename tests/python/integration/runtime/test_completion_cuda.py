from __future__ import annotations

import time

import pytest
import torch

from uniserve_worker.server.completion import CompletionArena
from uniserve_worker.server.image_codec import quantize_image_hwc

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_completions_become_observable_by_event_query() -> None:
    device = torch.device("cuda:0")
    arena = CompletionArena(depth=2, token_capacity=4, devices=(device,))
    pending = []
    for expected in ((3, 7, 11), (13,)):
        lease = arena.reserve(1, devices=(device,))
        capture = lease.capture(torch.tensor(expected, dtype=torch.long, device=device))
        lease.seal()
        pending.append((lease, capture, expected))

    deadline = time.monotonic() + 5.0
    while not all(lease.ready() for lease, _capture, _expected in pending):
        if time.monotonic() >= deadline:
            raise TimeoutError("completion events did not become query-ready")

    for lease, capture, expected in reversed(pending):
        assert capture.values() == expected
        lease.observe(0, lease.generation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_image_bytes_materialize_after_the_completion_event() -> None:
    device = torch.device("cuda:0")
    arena = CompletionArena(depth=1, token_capacity=16, devices=(device,))
    lease = arena.reserve(1)
    image = torch.tensor(
        [[[-1.0, 1.0]], [[0.0, 0.5]], [[1.0, -1.0]]],
        device=device,
    )
    capture = lease.capture_bytes(quantize_image_hwc(image))
    lease.seal()

    deadline = time.monotonic() + 5.0
    while not capture.ready() and time.monotonic() < deadline:
        time.sleep(0.0001)

    assert capture.ready()
    assert torch.equal(
        capture.tensor(),
        torch.tensor([[[0, 128, 255], [255, 191, 0]]], dtype=torch.uint8),
    )
    lease.observe(0, lease.generation)
