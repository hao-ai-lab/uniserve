"""Distributed transaction protocol (Stage 9, dormant).

Two ranks with real transactional residency each, one logical
ExecutionEngine: configuration-hash gating, all-rank prepare agreement with
no leaked reservation, group-atomic launch, designated-rank results,
disagreement poisoning, and epoch poisoning on a post-launch rank failure.
"""

from __future__ import annotations

import pytest

from uniserve_worker.contracts.execution import (
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    NewSession,
    OperationTag,
    SamplingSpec,
    SequenceStep,
    SessionRef,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.contracts.segment_table import GraphCapacity
from uniserve_worker.execution.engine import (
    AdapterRowOutcome,
    DistributedConfigurationError,
    EngineBackpressure,
    EnginePoisoned,
    EngineState,
    ExecutionEngine,
    RankFanOutExecutor,
    RankMember,
    StandardTransactionExecutor,
)
from uniserve_worker.models.cache_registrations import qwen3_cache_registration
from uniserve_worker.runtime.transactional_residency import ArenaConfig, Residency

pytestmark = pytest.mark.unit

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_PAGE_TOKENS = 16
_REGISTRATION = qwen3_cache_registration(
    layer_count=2, query_heads=8, kv_heads=2, qk_head_dim=64, value_head_dim=64,
    page_tokens=_PAGE_TOKENS,
)


class RankAdapter:
    """Deterministic per-rank adapter; optionally faulty or divergent."""

    def __init__(self, rank: int, *, diverge: bool = False) -> None:
        self.rank = rank
        self.diverge = diverge
        self.fail_next = False
        self.calls = 0

    def forward(self, segments, residency, capacity, payload):
        self.calls += 1
        if self.fail_next:
            raise RuntimeError(f"rank {self.rank} device failure")
        rows = 1 + max(
            segments.row_id[i]
            for i in range(capacity.segments)
            if segments.segment_active[i]
        )
        base = 100 + (self.rank if self.diverge else 0)
        return tuple(
            AdapterRowOutcome(sampled_tokens=(base + row,)) for row in range(rows)
        )


def _capacity() -> GraphCapacity:
    return GraphCapacity(
        rows=2, segments=8, tokens=64, branches=3, candidate_tokens=8,
        position_axes=1, visibility_payload_entries=0,
        residency=ResidencyBatchCapacity(bindings=6, page_references=32, tokens=64),
    )


def _member(rank: int, *, fingerprint: str = "cfg-a", pages: int = 17, diverge: bool = False):
    residency = Residency(
        _ENGINE, (ArenaConfig(domain_id=1, page_count=pages, page_tokens=_PAGE_TOKENS),)
    )
    adapter = RankAdapter(rank, diverge=diverge)
    executor = StandardTransactionExecutor(
        residency=residency,
        registration=_REGISTRATION,
        capacities=(_capacity(),),
        adapter=adapter,
        page_tokens=_PAGE_TOKENS,
    )
    return RankMember(rank, fingerprint, executor), residency, adapter


def _group(**kwargs):
    member0, residency0, adapter0 = _member(0, **kwargs)
    member1, residency1, adapter1 = _member(1, **kwargs)
    fan_out = RankFanOutExecutor((member0, member1))
    engine = ExecutionEngine(
        engine=_ENGINE,
        executor=fan_out,
        advertised_operations=frozenset(OperationTag),
        replay_window=8,
    )
    return engine, (residency0, residency1), (adapter0, adapter1)


def _admission_batch(step: int = 0) -> ExecuteBatch:
    return ExecuteBatch(
        engine_epoch=_ENGINE.engine_epoch,
        step_id=step,
        acknowledged_through=step - 1,
        rows=(
            ExecuteRow(
                row_id=0,
                session=SessionRef(_ENGINE, 41, 1, 0),
                operation=SequenceStep((5, 6, 7), 0, 0, 1),
                admission=NewSession(
                    41, 1, SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0), 42, 64
                ),
                cache_leases=(),
                product_leases=(),
                scheduler_op_id=0,
            ),
        ),
    )


def test_configuration_fingerprints_must_agree_before_the_group_exists():
    member0, _, _ = _member(0, fingerprint="cfg-a")
    member1, _, _ = _member(1, fingerprint="cfg-b")
    with pytest.raises(DistributedConfigurationError, match="fingerprint"):
        RankFanOutExecutor((member0, member1))


def test_success_runs_every_rank_and_returns_the_designated_result():
    engine, residencies, adapters = _group()
    result = engine.execute(_admission_batch()).result()
    assert result.row_results[0].sampled_tokens == (100,)
    assert [adapter.calls for adapter in adapters] == [1, 1]
    # Both ranks committed their rank-local physical state.
    for residency in residencies:
        assert residency.pressure()[1]["free_pages"] == 16 - 1


def test_one_rank_prepare_failure_is_noncommitted_with_no_leak_anywhere():
    # Rank 1 has a one-page arena: its reservation for a 3-token prefill fits,
    # so exhaust it instead with a huge operation both ranks see; rank 0 is
    # sized to succeed and must be rolled back when rank 1 rejects.
    member0, residency0, _ = _member(0, pages=17)
    member1, residency1, _ = _member(1, pages=2)
    engine = ExecutionEngine(
        engine=_ENGINE,
        executor=RankFanOutExecutor((member0, member1)),
        advertised_operations=frozenset(OperationTag),
        replay_window=8,
    )
    free0 = residency0.pressure()[1]["free_pages"]
    free1 = residency1.pressure()[1]["free_pages"]
    huge = ExecuteBatch(
        engine_epoch=_ENGINE.engine_epoch,
        step_id=0,
        acknowledged_through=-1,
        rows=(
            ExecuteRow(
                row_id=0,
                session=SessionRef(_ENGINE, 41, 1, 0),
                operation=SequenceStep(tuple(range(30)), 0, 0, 1),
                admission=NewSession(
                    41, 1, SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0), 42, 64
                ),
                cache_leases=(),
                product_leases=(),
                scheduler_op_id=0,
            ),
        ),
    )
    with pytest.raises(EngineBackpressure):
        engine.execute(huge)
    assert engine.state is EngineState.READY
    assert residency0.pressure()[1]["free_pages"] == free0
    assert residency1.pressure()[1]["free_pages"] == free1
    # The group recovers: the same step retries with a smaller batch.
    engine.execute(_admission_batch(0))


def test_post_launch_rank_failure_poisons_the_whole_epoch():
    engine, residencies, adapters = _group()
    adapters[1].fail_next = True
    free0 = residencies[0].pressure()[1]["free_pages"]
    with pytest.raises(EnginePoisoned):
        engine.execute(_admission_batch())
    assert engine.state is EngineState.POISONED
    # Rank 0 launched and committed before rank 1 failed; its pages moved.
    # The group interpretation is unsafe — exactly why the epoch poisons —
    # but no reservation remains provisional anywhere.
    assert residencies[0].pressure()[1]["free_pages"] in (free0, free0 - 1)


def test_rank_disagreement_poisons_the_epoch():
    engine, _, _ = _group(diverge=True)
    with pytest.raises(EnginePoisoned):
        engine.execute(_admission_batch())
    assert engine.state is EngineState.POISONED
