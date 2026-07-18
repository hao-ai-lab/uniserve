"""End-to-end dormant transactions: engine + lowering + residency + sessions.

Drives the target stack as one machine — ExecutionEngine over the
StandardTransactionExecutor with real transactional residency and the Qwen3
and SenseNova cache registrations, a deterministic adapter stub standing in
for the captured graph replay. Admission, decode, candidate verification,
CFG denoise overlays, backpressure, and launch-failure abort all flow
through the same seams the production cutover will use.
"""

from __future__ import annotations

import pytest

from uniserve_worker.contracts.execution import (
    CandidateVerification,
    EncodeStep,
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    FlowStep,
    NewSession,
    OperationTag,
    ProductLease,
    ProductLifetime,
    RepresentationKind,
    SamplingSpec,
    SequenceStep,
    SessionRef,
    TransferKind,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.contracts.segment_table import GraphCapacity
from uniserve_worker.execution.engine import (
    EngineBackpressure,
    EnginePoisoned,
    ExecutionEngine,
)
from uniserve_worker.execution.transaction import (
    AdapterRowOutcome,
    StandardTransactionExecutor,
)
from uniserve_worker.models.cache_registrations import (
    qwen3_cache_registration,
    sensenova_cache_registration,
)
from uniserve_worker.runtime.transactional_residency import (
    ArenaConfig,
    ProductDemand,
    ProductStoreConfig,
    ReservationPlan,
    Residency,
    RowDemand,
)

pytestmark = pytest.mark.unit

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_SAMPLING = SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0)
_PAGE_TOKENS = 16


class StubAdapter:
    """Deterministic packed-traversal stand-in for the captured graph."""

    def __init__(self) -> None:
        self.calls = 0
        self.accept = 0
        self.fail_next = False

    def forward(self, segments, residency, capacity, payload):
        self.calls += 1
        if self.fail_next:
            raise RuntimeError("injected replay failure")
        rows = max(
            (
                segments.row_id[index] + 1
                for index in range(capacity.segments)
                if segments.segment_active[index]
            ),
            default=0,
        )
        return tuple(
            AdapterRowOutcome(sampled_tokens=(100 + row,), accepted_candidates=self.accept)
            for row in range(rows)
        )


def _capacity(tokens: int = 64) -> GraphCapacity:
    return GraphCapacity(
        rows=4,
        segments=8,
        tokens=tokens,
        branches=3,
        candidate_tokens=8,
        position_axes=3,
        visibility_payload_entries=0,
        residency=ResidencyBatchCapacity(
            bindings=6, page_references=32, tokens=tokens
        ),
    )


def _stack(registration, *, pages: int = 33):
    residency = Residency(
        _ENGINE,
        (ArenaConfig(domain_id=1, page_count=pages, page_tokens=_PAGE_TOKENS),),
        product_stores=(ProductStoreConfig(schema_id=7, row_capacity=64),),
    )
    adapter = StubAdapter()
    executor = StandardTransactionExecutor(
        residency=residency,
        registration=registration,
        capacities=(_capacity(),),
        adapter=adapter,
        page_tokens=_PAGE_TOKENS,
    )
    engine = ExecutionEngine(
        engine=_ENGINE,
        executor=executor,
        advertised_operations=frozenset(OperationTag),
        replay_window=8,
    )
    return engine, residency, adapter


def _row(operation, *, row_id: int = 0, version: int, admission=None, leases=()):
    return ExecuteRow(
        row_id=row_id,
        session=SessionRef(_ENGINE, 41, 1, version),
        operation=operation,
        admission=admission,
        cache_leases=(),
        product_leases=tuple(leases),
        scheduler_op_id=row_id,
    )


def _batch(step_id: int, *rows) -> ExecuteBatch:
    return ExecuteBatch(
        engine_epoch=_ENGINE.engine_epoch,
        step_id=step_id,
        acknowledged_through=step_id - 1,
        rows=tuple(rows),
    )


def _admission() -> NewSession:
    return NewSession(41, 1, _SAMPLING, 42, 64)


def test_text_lifecycle_prefill_decode_and_verification():
    engine, residency, adapter = _stack(
        qwen3_cache_registration(
            layer_count=2, query_heads=8, kv_heads=2,
            qk_head_dim=64, value_head_dim=64, page_tokens=_PAGE_TOKENS,
        )
    )
    # Prefill via admission: three prompt tokens commit.
    prefill = SequenceStep((5, 6, 7), 0, 0, 1)
    result = engine.execute(
        _batch(0, _row(prefill, version=0, admission=_admission()))
    ).result()
    assert result.session_deltas[0].history_length_after == 3
    # Decode advance at version 1.
    decode = SequenceStep((101,), 3, 3, 1)
    result = engine.execute(_batch(1, _row(decode, version=1))).result()
    assert result.session_deltas[0].history_length_after == 4
    # Target verification: 3 candidates, 2 accepted.
    adapter.accept = 2
    verify = SequenceStep(
        (102,), 4, 4, 1, CandidateVerification((9, 10, 11), (5, 6, 7))
    )
    result = engine.execute(_batch(2, _row(verify, version=2))).result()
    delta = result.session_deltas[0]
    assert result.row_results[0].accepted_candidates == 2
    # 1 input token commits fully; 2 of 3 candidates publish.
    assert delta.history_length_after == 4 + 1 + 2
    assert adapter.calls == 3


def test_denoise_transaction_overlays_and_releases():
    engine, residency, adapter = _stack(
        sensenova_cache_registration(
            layer_count=2, query_heads=8, kv_heads=2,
            qk_head_dim=64, value_head_dim=64, page_tokens=_PAGE_TOKENS,
        )
    )
    # Conditioning prefill first (opens the primary role).
    prefill = SequenceStep((5, 6), 0, 0, 1)
    engine.execute(_batch(0, _row(prefill, version=0, admission=_admission())))
    free_before = residency.pressure()[1]["free_pages"]
    publish = residency.reserve(
        ReservationPlan(
            rows=(
                RowDemand(
                    row_id=0,
                    bindings=(),
                    products=(
                        ProductDemand(
                            schema_id=7,
                            rows=16,
                            producer=SessionRef(_ENGINE, 41, 1, 1),
                        ),
                    ),
                ),
            )
        )
    )
    (latent,) = publish.commit(())
    flow = FlowStep(1, 0, 50, latent.lease_id, (4.0, 1.0, 1.0), (), 7)
    result = engine.execute(
        _batch(1, _row(flow, version=1, leases=(latent,)))
    ).result()
    # Transient overlays release every page after commit; nothing persists.
    assert residency.pressure()[1]["free_pages"] == free_before
    assert result.session_deltas[0].history_length_after == 2


def test_reservation_backpressure_is_retryable_without_a_record():
    engine, residency, adapter = _stack(
        qwen3_cache_registration(
            layer_count=2, query_heads=8, kv_heads=2,
            qk_head_dim=64, value_head_dim=64, page_tokens=_PAGE_TOKENS,
        ),
        pages=2,  # one real page: 16 rows total
    )
    huge = SequenceStep(tuple(range(40)), 0, 0, 1)
    with pytest.raises(EngineBackpressure):
        engine.execute(_batch(0, _row(huge, version=0, admission=_admission())))
    assert adapter.calls == 0
    # The same step retries with a smaller operation and succeeds.
    small = SequenceStep((1, 2), 0, 0, 1)
    receipt = engine.execute(
        _batch(0, _row(small, version=0, admission=_admission()))
    )
    assert receipt.ready()


def test_launch_failure_aborts_the_reservation_and_poisons():
    engine, residency, adapter = _stack(
        qwen3_cache_registration(
            layer_count=2, query_heads=8, kv_heads=2,
            qk_head_dim=64, value_head_dim=64, page_tokens=_PAGE_TOKENS,
        )
    )
    free_before = residency.pressure()[1]["free_pages"]
    adapter.fail_next = True
    prefill = SequenceStep((5, 6, 7), 0, 0, 1)
    with pytest.raises(EnginePoisoned):
        engine.execute(_batch(0, _row(prefill, version=0, admission=_admission())))
    # The reservation aborted: no page leaked despite the poisoned epoch.
    assert residency.pressure()[1]["free_pages"] == free_before


def test_encode_publishes_a_product_only_at_commit():
    engine, residency, adapter = _stack(
        sensenova_cache_registration(
            layer_count=2, query_heads=8, kv_heads=2,
            qk_head_dim=64, value_head_dim=64, page_tokens=_PAGE_TOKENS,
        )
    )
    prefill = SequenceStep((5, 6), 0, 0, 1)
    engine.execute(_batch(0, _row(prefill, version=0, admission=_admission())))
    encode = EncodeStep(RepresentationKind.IMAGE_PATCH, 0, 7, (1, 4, 4))
    result = engine.execute(_batch(1, _row(encode, version=1))).result()
    (lease,) = result.session_deltas[0].product_leases_added
    assert lease.schema_id == 7
    assert lease.extent_rows == 16
    residency.validate_product(lease)
    # The committed product feeds the next flow transaction as a real input.
    flow = FlowStep(1, 0, 50, lease.lease_id, (4.0, 1.0, 1.0), (), 7)
    engine.execute(_batch(2, _row(flow, version=2, leases=(lease,)))).result()
    # Explicit versioned release; a second release is stale.
    from uniserve_worker.runtime.transactional_residency import (
        LeaseReleaseOutcome,
    )
    assert residency.release_product(lease) is LeaseReleaseOutcome.RELEASED
    assert residency.release_product(lease) is LeaseReleaseOutcome.STALE


def test_aborted_products_never_publish_and_stale_inputs_reject():
    engine, residency, adapter = _stack(
        sensenova_cache_registration(
            layer_count=2, query_heads=8, kv_heads=2,
            qk_head_dim=64, value_head_dim=64, page_tokens=_PAGE_TOKENS,
        )
    )
    rows_before = residency.product_rows_used(7)
    publish = residency.reserve(
        ReservationPlan(
            rows=(
                RowDemand(
                    row_id=0,
                    bindings=(),
                    products=(
                        ProductDemand(
                            schema_id=7, rows=16,
                            producer=SessionRef(_ENGINE, 41, 1, 1),
                        ),
                    ),
                ),
            )
        )
    )
    publish.abort()
    assert residency.product_rows_used(7) == rows_before
    # Consuming a never-committed lease is a typed pre-launch rejection.
    from uniserve_worker.execution.engine import PreLaunchRejection

    fabricated = ProductLease(
        lease_id=99, schema_id=7, producer=SessionRef(_ENGINE, 41, 1, 1),
        product_version=1, extent_rows=16, lifetime=ProductLifetime.REQUEST,
        transfer=TransferKind.LOCAL_RESIDENCY,
    )
    prefill = SequenceStep((5, 6), 0, 0, 1)
    engine.execute(_batch(0, _row(prefill, version=0, admission=_admission())))
    flow = FlowStep(1, 0, 50, 99, (4.0, 1.0, 1.0), (), 7)
    with pytest.raises(PreLaunchRejection):
        engine.execute(_batch(1, _row(flow, version=1, leases=(fabricated,))))
