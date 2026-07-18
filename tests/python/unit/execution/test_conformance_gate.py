"""The mechanically generated adapter conformance gate (Stage 11, GPU).

Every nonempty advertised operation subset, in every row order, for all
three registered families, executes through a fresh dormant stack: the batch
must commit, exactly one adapter/root invocation must serve it, and results
stay in scheduler order. The manifest machinery is validated separately for
determinism and stale-hash rejection.
"""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.backends.attention.reference_target import (
    CacheDeviceBinding,
    ReferenceAttentionBackend,
)
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
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.contracts.segment_table import GraphCapacity
from uniserve_worker.execution.conformance import (
    ManifestError,
    build_manifest,
    generate_cases,
    validate_manifest,
)
from uniserve_worker.execution.engine import ExecutionEngine
from uniserve_worker.execution.transaction import StandardTransactionExecutor
from uniserve_worker.models.bagel_target import BagelTarget
from uniserve_worker.models.cache_registrations import (
    bagel_cache_registration,
    qwen3_cache_registration,
    sensenova_cache_registration,
)
from uniserve_worker.models.qwen3_target import Qwen3Target
from uniserve_worker.models.sensenova_target import SenseNovaTarget
from uniserve_worker.models.target_registry import target_registry
from uniserve_worker.nn.target_decoder import TargetDecoderConfig
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


class _CountingAdapter:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def forward(self, segments, residency, capacity, payload):
        self.calls += 1
        return self.inner.forward(segments, residency, capacity, payload)


def _config(routes: int) -> TargetDecoderConfig:
    return TargetDecoderConfig(
        vocab_size=64,
        hidden_size=32,
        layers=1,
        routes=routes,
        query_heads=4,
        kv_heads=2,
        head_dim=8,
        mlp_hidden=64,
    )


_FAMILIES = {
    "qwen3": (qwen3_cache_registration, Qwen3Target, 1),
    "bagel": (bagel_cache_registration, BagelTarget, 2),
    "sensenova": (sensenova_cache_registration, SenseNovaTarget, 2),
}


def _stack(family: str):
    registration_factory, adapter_type, routes = _FAMILIES[family]
    config = _config(routes)
    registration = registration_factory(
        layer_count=config.layers,
        query_heads=config.query_heads,
        kv_heads=config.kv_heads,
        qk_head_dim=config.head_dim,
        value_head_dim=config.head_dim,
        page_tokens=_PAGE_TOKENS,
    )
    residency = Residency(
        _ENGINE,
        (ArenaConfig(domain_id=1, page_count=129, page_tokens=_PAGE_TOKENS),),
        product_stores=(ProductStoreConfig(schema_id=7, row_capacity=256),),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    binding = CacheDeviceBinding(
        layers=config.layers,
        pages=129,
        page_tokens=_PAGE_TOKENS,
        kv_heads=config.kv_heads,
        head_dim=config.head_dim,
        device=device,
    )
    adapter = _CountingAdapter(
        adapter_type(config, ReferenceAttentionBackend(binding), device=device)
    )
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
        advertised_operations=frozenset(OperationTag),
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
                    products=(
                        ProductDemand(
                            schema_id=7, rows=_LATENT_ROWS, producer=session
                        ),
                    ),
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
    assert [row.row_id for row in result.row_results] == list(
        range(len(case.operations))
    ), case.case_id


def test_manifest_generation_is_deterministic_and_stale_hashes_reject():
    registry = target_registry()
    for architecture in ("Qwen3ForCausalLM", "BAGEL", "NEOChatModel"):
        registration = registry.resolve((architecture,))
        manifest = build_manifest(registration.family, registration.operations)
        again = build_manifest(registration.family, registration.operations)
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_every_generated_case_commits_with_one_root_invocation(family: str):
    registry = target_registry()
    architecture = {
        "qwen3": "Qwen3ForCausalLM",
        "bagel": "BAGEL",
        "sensenova": "NEOChatModel",
    }[family]
    registration = registry.resolve((architecture,))
    cases = generate_cases(family, registration.operations)
    assert cases
    for case in cases:
        _run_case(case)
