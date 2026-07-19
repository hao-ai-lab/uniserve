"""Mechanical composition conformance for every registered model family.

Every nonempty advertised operation subset, in every row order, executes
through a fresh transaction stack: the batch must commit, exactly one resident
adapter invocation must serve it, and results stay in scheduler order. The
manifest machinery is validated separately for determinism and stale-hash
rejection.
"""

from __future__ import annotations

import pytest

from uniserve_worker.contracts.execution import (
    EncodeStep,
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    FlowStep,
    MaterializeStep,
    NewSession,
    OperationTag,
    RepresentationKind,
    SamplingSpec,
    SequenceStep,
    SessionRef,
)
from uniserve_worker.contracts.model_family import ModelFamilyDescriptor
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.execution import (
    AdapterRowOutcome,
    ExecutionEngine,
    GraphCapacity,
    ManifestError,
    StandardTransactionExecutor,
    build_manifest,
    generate_cases,
    validate_manifest,
)
from uniserve_worker.models.bagel import BagelForUnifiedGeneration
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration
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
_PAGE_TOKENS = 4
_SAMPLING = SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0)
_LATENT_ROWS = 4


class _CountingResidentAdapter:
    """Minimal resident adapter that exposes transaction orchestration only."""

    def __init__(self) -> None:
        self.calls = 0

    def forward(self, segments, residency, capacity, payload):
        del residency, payload
        self.calls += 1
        row_ids = sorted(
            {
                int(segments.row_id[index])
                for index in range(capacity.segments)
                if segments.segment_active[index]
            }
        )
        return tuple(AdapterRowOutcome(sampled_tokens=(0,)) for _row_id in row_ids)


_FAMILIES = {
    model.family: ModelFamilyDescriptor.from_model_class(model)
    for model in (
        Qwen3ForCausalLM,
        BagelForUnifiedGeneration,
        SenseNovaU1ForUnifiedGeneration,
    )
}


def _stack(family: str):
    descriptor = _FAMILIES[family]
    registration = descriptor.build_cache_registration(
        layer_count=1,
        query_heads=4,
        kv_heads=2,
        qk_head_dim=8,
        value_head_dim=8,
        page_tokens=_PAGE_TOKENS,
    )
    residency = Residency(
        _ENGINE,
        (ArenaConfig(domain_id=1, page_count=129, page_tokens=_PAGE_TOKENS),),
        product_stores=(ProductStoreConfig(schema_id=7, row_capacity=256),),
    )
    adapter = _CountingResidentAdapter()
    capacity = GraphCapacity(
        rows=4,
        segments=16,
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
        advertised_operations=descriptor.operation_tags,
        replay_window=64,
    )
    return engine, residency, adapter


def _publish_latent(residency, session: SessionRef):
    reservation = residency.reserve(
        ReservationPlan(
            rows=(
                RowDemand(
                    row_id=0,
                    bindings=(),
                    products=(ProductDemand(schema_id=7, rows=_LATENT_ROWS, producer=session),),
                ),
            )
        )
    )
    (lease,) = reservation.commit(())
    return lease


def _run_case(case) -> None:
    engine, residency, adapter = _stack(case.family)
    step = 0
    rows = []
    for index, tag in enumerate(case.operations):
        request_id = 100 + index
        # Setup: admit and commit a two-token conditioning prefill.
        engine.execute(
            ExecuteBatch(
                engine_epoch=_ENGINE.engine_epoch,
                step_id=step,
                acknowledged_through=step - 1,
                rows=(
                    ExecuteRow(
                        row_id=0,
                        session=SessionRef(_ENGINE, request_id, 1, 0),
                        operation=SequenceStep((7, 8), 0, 0, 1),
                        admission=NewSession(request_id, 1, _SAMPLING, 42, 64),
                        cache_leases=(),
                        product_leases=(),
                        scheduler_op_id=0,
                    ),
                ),
            )
        ).result()
        step += 1
        session = SessionRef(_ENGINE, request_id, 1, 1)
        leases = ()
        if tag is OperationTag.SEQUENCE_STEP:
            operation = SequenceStep((11,), 2, 2, 1)
        elif tag is OperationTag.FLOW_STEP:
            latent = _publish_latent(residency, session)
            leases = (latent,)
            operation = FlowStep(1, 0, 50, latent.lease_id, (4.0, 1.0, 1.0), (), 7)
        elif tag is OperationTag.ENCODE_STEP:
            operation = EncodeStep(RepresentationKind.IMAGE_PATCH, 0, 7, (1, 2, 2))
        else:
            latent = _publish_latent(residency, session)
            leases = (latent,)
            operation = MaterializeStep(latent.lease_id, 7)
        rows.append(
            ExecuteRow(
                row_id=index,
                session=session,
                operation=operation,
                admission=None,
                cache_leases=(),
                product_leases=leases,
                scheduler_op_id=index,
            )
        )
    calls_before = adapter.calls
    result = engine.execute(
        ExecuteBatch(
            engine_epoch=_ENGINE.engine_epoch,
            step_id=step,
            acknowledged_through=step - 1,
            rows=tuple(rows),
        )
    ).result()
    assert adapter.calls == calls_before + 1, case.case_id
    assert [row.row_id for row in result.row_results] == list(range(len(case.operations))), (
        case.case_id
    )


def test_manifest_generation_is_deterministic_and_stale_hashes_reject():
    for descriptor in _FAMILIES.values():
        manifest = build_manifest(descriptor.family, descriptor.operation_tags)
        again = build_manifest(descriptor.family, descriptor.operation_tags)
        assert manifest == again
        validate_manifest(manifest)
        stale = type(manifest)(
            family=manifest.family,
            advertised_operations=manifest.advertised_operations,
            case_ids=manifest.case_ids,
            case_set_hash="0" * 64,
            benchmark_references=manifest.benchmark_references,
        )
        with pytest.raises(ManifestError, match="hash"):
            validate_manifest(stale)
    qwen3 = build_manifest("qwen3", frozenset({OperationTag.SEQUENCE_STEP}))
    assert qwen3.case_ids == ("qwen3/sequence_step",)
    full = build_manifest("sensenova", frozenset(OperationTag))
    assert len(full.case_ids) == 4 + 12 + 24 + 24
    assert "qwen3_sharegpt_r16" in full.benchmark_references


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_every_generated_case_commits_with_one_root_invocation(family: str):
    descriptor = _FAMILIES[family]
    cases = generate_cases(family, descriptor.operation_tags)
    assert cases
    for case in cases:
        _run_case(case)
