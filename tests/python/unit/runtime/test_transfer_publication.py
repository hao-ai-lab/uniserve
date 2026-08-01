from __future__ import annotations

from multiprocessing import shared_memory

import pytest
import torch

from uniserve_worker.runtime.transfer import ShmTransport


def test_shm_async_publication_round_trips_cpu_storage() -> None:
    transport = ShmTransport()
    source = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    try:
        locator = transport.publish_async(source)
        actual = transport.fetch(locator)
        torch.testing.assert_close(actual, source, rtol=0, atol=0)
        transport.release(locator)
    finally:
        transport.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_shm_async_publication_exposes_an_inflight_ticket_and_exact_snapshot() -> None:
    transport = ShmTransport()
    source = torch.zeros((1024,), dtype=torch.float32, device="cuda")
    torch.cuda._sleep(500_000_000)
    source.add_(1)
    try:
        locator = transport.publish_async(source)
        segment = shared_memory.SharedMemory(name=locator.handle.decode())
        try:
            assert segment.buf[0] == 0
        finally:
            segment.close()
        actual = transport.fetch(locator)
        torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)
        transport.release(locator)
    finally:
        transport.close()
