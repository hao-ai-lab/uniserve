"""GPU verification of CUDA Green Context SM partitioning (plan Phase 3).

Skipped unless a CUDA device is present. Builds a :class:`StreamManager`, checks
it partitions the device's SMs into prefill/decode stream groups, and runs
concurrent work on a partitioned pair. Falls back cleanly (asserts the manager
still yields usable streams) when the driver cannot create green contexts.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")


def test_stream_manager_partitions_and_runs_concurrent_streams():
    _cuda_or_skip()
    torch.cuda.init()
    from uniserve_worker.runtime.stream_manager import StreamManager

    manager = StreamManager(0, sm_groups=8)
    # Endpoints (full-SM prefill-only + decode-only) plus the inner divisions.
    assert len(manager.stream_groups) >= 2
    assert len(manager.stream_groups) == len(manager.sm_counts)

    # select_streams maps decode batch size to a group monotonically.
    assert manager.select_idx(0) == 0
    big = manager.select_idx(10_000)
    assert big >= manager.select_idx(0)

    # Run work on a (possibly partitioned) prefill/decode pair concurrently.
    prefill_s, decode_s = manager.stream_groups[min(1, len(manager.stream_groups) - 1)]
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    with torch.cuda.stream(prefill_s):
        pa = a @ a
    with torch.cuda.stream(decode_s):
        db = b @ b
    torch.cuda.synchronize()
    assert torch.isfinite(pa).all() and torch.isfinite(db).all()

    if manager.using_green_contexts:
        # At least one inner group must reflect a real SM split (prefill > decode > 0).
        assert any(p > d > 0 for p, d in manager.divisions)
