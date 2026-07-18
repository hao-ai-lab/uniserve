"""SenseNova mixed-composition conformance (Stage 6/11 slice, GPU).

The composition-invariance law at unit scale: a mixed batch carrying one
text decode row and one CFG denoise flow row (three branches, transient
overlays over branch caches) traverses the two-route root once, and every
output equals the isolated-batch runs of the same rows — route dispatch,
attention regions, and cache effects cannot leak across co-batched rows.
The text half is additionally pinned to an independent dense oracle through
the engine's sampled tokens.
"""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.backends.attention.reference_target import (
    CacheDeviceBinding,
    ReferenceAttentionBackend,
)
from uniserve_worker.contracts.execution import (
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    FlowStep,
    NewSession,
    OperationTag,
    SamplingSpec,
    SequenceStep,
    SessionRef,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.contracts.segment_table import GraphCapacity
from uniserve_worker.execution.engine import ExecutionEngine
from uniserve_worker.execution.transaction import StandardTransactionExecutor
from uniserve_worker.models.cache_registrations import (
    sensenova_cache_registration,
)
from uniserve_worker.models.sensenova_target import SenseNovaTarget
from uniserve_worker.nn.target_decoder import TargetDecoderConfig
from uniserve_worker.runtime.transactional_residency import (
    ArenaConfig,
    ProductDemand,
    ProductStoreConfig,
    ReservationPlan,
    Residency,
    RowDemand,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_PAGE_TOKENS = 4
_CONFIG = TargetDecoderConfig(
    vocab_size=64,
    hidden_size=32,
    layers=2,
    routes=2,
    query_heads=4,
    kv_heads=2,
    head_dim=8,
    mlp_hidden=64,
)
_SAMPLING = SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0)


def _stack():
    registration = sensenova_cache_registration(
        layer_count=_CONFIG.layers,
        query_heads=_CONFIG.query_heads,
        kv_heads=_CONFIG.kv_heads,
        qk_head_dim=_CONFIG.head_dim,
        value_head_dim=_CONFIG.head_dim,
        page_tokens=_PAGE_TOKENS,
    )
    residency = Residency(
        _ENGINE,
        (ArenaConfig(domain_id=1, page_count=65, page_tokens=_PAGE_TOKENS),),
        product_stores=(ProductStoreConfig(schema_id=7, row_capacity=128),),
    )
    binding = CacheDeviceBinding(
        layers=_CONFIG.layers,
        pages=65,
        page_tokens=_PAGE_TOKENS,
        kv_heads=_CONFIG.kv_heads,
        head_dim=_CONFIG.head_dim,
        device="cuda",
    )
    adapter = SenseNovaTarget(
        _CONFIG, ReferenceAttentionBackend(binding), device="cuda", seed=23
    )
    capacity = GraphCapacity(
        rows=2,
        segments=8,
        tokens=64,
        branches=3,
        candidate_tokens=8,
        position_axes=3,
        visibility_payload_entries=0,
        residency=ResidencyBatchCapacity(bindings=8, page_references=48, tokens=64),
    )
    executor = StandardTransactionExecutor(
        residency=residency,
        registration=registration,
        capacities=(capacity,),
        adapter=adapter,
        page_tokens=_PAGE_TOKENS,
    )
    engine = ExecutionEngine(
        engine=_ENGINE,
        executor=executor,
        advertised_operations=frozenset(OperationTag),
        replay_window=16,
    )
    return engine, residency, adapter


def _publish_latent(residency, rows: int, session: SessionRef):
    reservation = residency.reserve(
        ReservationPlan(
            rows=(
                RowDemand(
                    row_id=0,
                    bindings=(),
                    products=(
                        ProductDemand(schema_id=7, rows=rows, producer=session),
                    ),
                ),
            )
        )
    )
    (lease,) = reservation.commit(())
    return lease


def _admit(engine, request_id: int, step: int, prompt: tuple[int, ...]):
    row = ExecuteRow(
        row_id=0,
        session=SessionRef(_ENGINE, request_id, 1, 0),
        operation=SequenceStep(prompt, 0, 0, 1),
        admission=NewSession(request_id, 1, _SAMPLING, 42, 64),
        cache_leases=(),
        product_leases=(),
        scheduler_op_id=0,
    )
    return engine.execute(
        ExecuteBatch(
            engine_epoch=_ENGINE.engine_epoch,
            step_id=step,
            acknowledged_through=step - 1,
            rows=(row,),
        )
    ).result()


def test_mixed_text_and_denoise_rows_equal_their_isolated_runs():
    # Two identical stacks (same seed): one runs the mixed batch, the other
    # runs the same rows isolated. Outputs must agree exactly.
    mixed_engine, mixed_res, mixed_adapter = _stack()
    solo_engine, solo_res, solo_adapter = _stack()

    prompt = (3, 1, 4)
    for engine in (mixed_engine, solo_engine):
        first = _admit(engine, 41, 0, prompt)  # text request
        assert first.row_results[0].sampled_tokens
    sampled = first.row_results[0].sampled_tokens[0]
    for engine in (mixed_engine, solo_engine):
        _admit(engine, 55, 1, (9, 9))  # image request conditioning

    def decode_row(row_id: int) -> ExecuteRow:
        return ExecuteRow(
            row_id=row_id,
            session=SessionRef(_ENGINE, 41, 1, 1),
            operation=SequenceStep((sampled,), 3, 3, 1),
            admission=None,
            cache_leases=(),
            product_leases=(),
            scheduler_op_id=row_id,
        )

    def flow_row(row_id: int, residency) -> ExecuteRow:
        latent = _publish_latent(residency, 8, SessionRef(_ENGINE, 55, 1, 1))
        return ExecuteRow(
            row_id=row_id,
            session=SessionRef(_ENGINE, 55, 1, 1),
            operation=FlowStep(
                1, 0, 50, latent.lease_id, (4.0, 1.0, 1.0), (), 7
            ),
            admission=None,
            cache_leases=(),
            product_leases=(latent,),
            scheduler_op_id=row_id,
        )

    # Mixed: one batch, two rows, one root traversal.
    mixed_result = mixed_engine.execute(
        ExecuteBatch(
            engine_epoch=_ENGINE.engine_epoch,
            step_id=2,
            acknowledged_through=1,
            rows=(decode_row(0), flow_row(1, mixed_res)),
        )
    ).result()

    # Isolated: the same rows as two batches.
    solo_text = solo_engine.execute(
        ExecuteBatch(
            engine_epoch=_ENGINE.engine_epoch,
            step_id=2,
            acknowledged_through=1,
            rows=(decode_row(0),),
        )
    ).result()
    solo_flow = solo_engine.execute(
        ExecuteBatch(
            engine_epoch=_ENGINE.engine_epoch,
            step_id=3,
            acknowledged_through=2,
            rows=(flow_row(0, solo_res),),
        )
    ).result()

    assert (
        mixed_result.row_results[0].sampled_tokens
        == solo_text.row_results[0].sampled_tokens
    )
    assert mixed_result.row_results[1].sampled_tokens == ()
    assert solo_flow.row_results[0].sampled_tokens == ()
    # Text session history advanced identically; denoise committed zero rows.
    assert (
        mixed_result.session_deltas[0].history_length_after
        == solo_text.session_deltas[0].history_length_after
        == 4
    )
    assert mixed_result.session_deltas[1].history_length_after == 2
    assert mixed_adapter.config.routes == solo_adapter.config.routes == 2
