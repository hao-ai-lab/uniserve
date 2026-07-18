"""Capacity-only CUDA graph runtime laws (Stage 7, dormant, GPU).

Proves the graph laws with a captured structural adapter: dominance
selection with no composition keys, exactly one replay per execute, stale
tails killed by refresh (maximal/minimal alternation), pointer stability, no
post-readiness capture or allocation, and capacity rejection — then the full
dormant stack (engine + executor + residency + sessions) runs decode
transactions through a real captured CUDA graph.
"""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts.cache_schema import CacheLifetime
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
from uniserve_worker.execution.cuda_graph import (
    BucketTensors,
    CudaGraphError,
    CudaGraphRuntime,
)
from uniserve_worker.execution.engine import ExecutionEngine
from uniserve_worker.execution.lowering import RoleSequences, lower_rows
from uniserve_worker.execution.transaction import (
    AdapterRowOutcome,
    StandardTransactionExecutor,
)
from uniserve_worker.models.cache_registrations import (
    PRIMARY_ROLE,
    qwen3_cache_registration,
)
from uniserve_worker.runtime.transactional_residency import (
    ArenaConfig,
    Residency,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_SESSION = SessionRef(_ENGINE, request_id=41, incarnation=1, session_version=2)
_PAGE_TOKENS = 16

_QWEN3 = qwen3_cache_registration(
    layer_count=2, query_heads=8, kv_heads=2, qk_head_dim=64, value_head_dim=64
)


def structural_adapter(tensors: BucketTensors) -> None:
    """Deterministic captured reduction over the refreshed structure.

    ``row_results[r] = sum over active segments of row r of
    (context_length + query_count)`` — every input arrives through the
    refreshed columns, so stale-tail leaks and missed refreshes change the
    output and fail the equivalence checks.
    """

    contribution = (
        (tensors.context_length + tensors.query_count)
        * tensors.segment_active
    ).to(torch.int64)
    tensors.row_results.zero_()
    tensors.row_results.scatter_add_(
        0, tensors.row_id.to(torch.int64).clamp_(min=0), contribution
    )


def _capacity(tokens: int, segments: int = 8, rows: int = 4) -> GraphCapacity:
    return GraphCapacity(
        rows=rows,
        segments=segments,
        tokens=tokens,
        branches=3,
        candidate_tokens=8,
        position_axes=1,
        visibility_payload_entries=0,
        residency=ResidencyBatchCapacity(
            bindings=6, page_references=32, tokens=tokens
        ),
    )


def _reference(segments) -> list[int]:
    totals: dict[int, int] = {}
    for index in range(len(segments.segment_active)):
        if segments.segment_active[index]:
            row = segments.row_id[index]
            totals[row] = totals.get(row, 0) + (
                segments.context_length[index] + segments.query_count[index]
            )
    return [totals.get(row, 0) for row in sorted(totals)]


def _lowered(tokens: tuple[int, ...], history: int, residency: Residency, ref):
    operation = SequenceStep(tokens, history, history, 1)
    row = ExecuteRow(
        row_id=0,
        session=_SESSION,
        operation=operation,
        admission=None,
        cache_leases=(),
        product_leases=(),
        scheduler_op_id=0,
    )
    return lower_rows(
        ((row, RoleSequences({PRIMARY_ROLE: ref})),),
        _QWEN3,
    )


def test_capture_replay_and_alternation_respect_graph_laws():
    residency = Residency(
        _ENGINE, (ArenaConfig(domain_id=1, page_count=65, page_tokens=_PAGE_TOKENS),)
    )
    ref = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    small, large = _capacity(16, segments=4, rows=2), _capacity(64)
    runtime = CudaGraphRuntime((large, small), device="cuda")
    runtime.capture_all(structural_adapter)
    assert runtime.capture_count == 2
    with pytest.raises(CudaGraphError, match="after readiness"):
        runtime.capture_all(structural_adapter)

    # Maximal batch on the small bucket's limit, then a minimal one: the
    # second result must show no trace of the first (neutral-fill law).
    for tokens in ((1, 2, 3, 4, 5, 6, 7, 8), (9,), (1, 2), (3,)):
        lowered = _lowered(tokens, 0, residency, ref)
        segments = lowered.fill_segment_table(small)
        reservation = residency.reserve(lowered.plan)
        arrays = reservation.batch_arrays(
            small.residency, lowered.write_token_begins
        )
        view = runtime.execute(
            lowered.demand(page_tokens=_PAGE_TOKENS), segments, arrays
        )
        assert view.bucket_index == 1  # smallest dominating bucket
        expected = _reference(segments)
        assert view.row_results[: len(expected)].tolist() == expected
        reservation.abort()
    assert runtime.replay_counts() == (0, 4)

    # A demand beyond every bucket is a typed capacity miss, never a capture.
    oversize = _lowered(tuple(range(80)), 0, residency, ref)
    with pytest.raises(CudaGraphError, match="dominates"):
        runtime.execute(
            oversize.demand(page_tokens=_PAGE_TOKENS),
            oversize.fill_segment_table(_capacity(128, segments=8)),
            residency.reserve(oversize.plan).batch_arrays(
                _capacity(128).residency, oversize.write_token_begins
            ),
        )


class GraphResidentAdapter:
    """ResidentAdapter over the captured runtime (the Stage 6/7 bridge)."""

    def __init__(self, runtime: CudaGraphRuntime, page_tokens: int) -> None:
        self._runtime = runtime
        self._page_tokens = page_tokens

    def forward(self, segments, residency, capacity):
        demand = capacity  # the executor already selected this capacity
        view = self._runtime.execute(demand, segments, residency)
        rows = 1 + max(
            segments.row_id[index]
            for index in range(len(segments.segment_active))
            if segments.segment_active[index]
        )
        values = view.row_results[:rows].tolist()
        return tuple(
            AdapterRowOutcome(sampled_tokens=(int(value),)) for value in values
        )


def test_full_dormant_stack_runs_transactions_through_a_captured_graph():
    residency = Residency(
        _ENGINE, (ArenaConfig(domain_id=1, page_count=65, page_tokens=_PAGE_TOKENS),)
    )
    capacity = _capacity(64)
    runtime = CudaGraphRuntime((capacity,), device="cuda")
    runtime.capture_all(structural_adapter)
    executor = StandardTransactionExecutor(
        residency=residency,
        registration=_QWEN3,
        capacities=(capacity,),
        adapter=GraphResidentAdapter(runtime, _PAGE_TOKENS),
        page_tokens=_PAGE_TOKENS,
    )
    engine = ExecutionEngine(
        engine=_ENGINE,
        executor=executor,
        advertised_operations=frozenset(OperationTag),
        replay_window=8,
    )
    admission = NewSession(41, 1, SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0), 42, 64)
    prefill = SequenceStep((5, 6, 7), 0, 0, 1)
    result = engine.execute(
        ExecuteBatch(
            engine_epoch=7,
            step_id=0,
            acknowledged_through=-1,
            rows=(
                ExecuteRow(0, SessionRef(_ENGINE, 41, 1, 0), prefill, admission, (), (), 0),
            ),
        )
    ).result()
    # Structural adapter: context 0 + 3 query rows -> sampled "token" 3.
    assert result.row_results[0].sampled_tokens == (3,)
    assert result.session_deltas[0].history_length_after == 3
    # Decode advance replays the same captured graph with fresh structure.
    decode = SequenceStep((101,), 3, 3, 1)
    result = engine.execute(
        ExecuteBatch(
            engine_epoch=7,
            step_id=1,
            acknowledged_through=0,
            rows=(
                ExecuteRow(0, SessionRef(_ENGINE, 41, 1, 1), decode, None, (), (), 0),
            ),
        )
    ).result()
    # context 3 + 1 query row -> 4.
    assert result.row_results[0].sampled_tokens == (4,)
    assert runtime.replay_counts() == (2,)
