"""SM-partition math for CUDA Green Context stream groups (plan Phase 3).

Covers the pure, GPU-free logic: architecture granularity constraints, the
``divide_sm`` partition selection (ported from sglang), and the per-stream graph
cache. Green-context creation itself is exercised by the GPU verification, not
here.
"""
from __future__ import annotations

import pytest

from uniserve_worker.runtime.stream_manager import (
    PerStreamGraphCache,
    divide_sm,
    get_arch_constraints,
)

pytestmark = pytest.mark.unit


def test_arch_constraints_cover_sm6_through_blackwell():
    assert get_arch_constraints((6, 0)) == (1, 1)
    assert get_arch_constraints((7, 5)) == (2, 2)
    assert get_arch_constraints((8, 0)) == (4, 2)
    assert get_arch_constraints((9, 0)) == (8, 8)
    # Blackwell (GB200) and any newer major share the 8/8 granularity.
    assert get_arch_constraints((10, 0)) == (8, 8)
    assert get_arch_constraints((12, 0)) == (8, 8)


def test_arch_constraints_reject_unknown_pre_pascal():
    with pytest.raises(ValueError):
        get_arch_constraints((5, 0))


def test_divide_sm_empty_for_nonpositive_groups():
    assert divide_sm(132, (9, 0), 0) == []
    assert divide_sm(132, (9, 0), -1) == []


def _assert_valid(divisions, total, multiple):
    for prefill, decode in divisions:
        assert prefill + decode == total
        assert prefill % multiple == 0
        assert prefill >= decode, "prefill partition must be the larger half"
        assert decode >= 16, "decode partition must keep >= 16 SMs"


def test_divide_sm_h100_partitions_are_valid_and_ordered():
    total = 132
    divisions = divide_sm(total, (9, 0), 6)
    assert divisions, "expected partitions for H100"
    _assert_valid(divisions, total, multiple=8)
    # Returned larger-prefill-first.
    prefills = [p for p, _ in divisions]
    assert prefills == sorted(prefills, reverse=True)


def test_divide_sm_blackwell_gb200_partitions_are_valid():
    # GB200 B200 die reports 152 SMs; 6 inner groups (sm_groups=8).
    total = 152
    divisions = divide_sm(total, (10, 0), 6)
    assert divisions, "expected partitions for GB200"
    _assert_valid(divisions, total, multiple=8)
    assert len(divisions) <= 6


def test_divide_sm_respects_group_count_cap():
    divisions = divide_sm(132, (9, 0), 3)
    assert len(divisions) <= 3


def test_per_stream_graph_cache_captures_once():
    cache = PerStreamGraphCache(stream_idx=2)
    calls = {"n": 0}

    def capture():
        calls["n"] += 1
        return f"graph-{calls['n']}"

    first = cache.get_or_capture(("decode", 8), capture)
    second = cache.get_or_capture(("decode", 8), capture)
    third = cache.get_or_capture(("decode", 16), capture)
    assert first == "graph-1"
    assert second == "graph-1", "same key must reuse the captured graph"
    assert third == "graph-2", "a new key captures a fresh graph"
    assert calls["n"] == 2
    assert cache.stream_idx == 2
