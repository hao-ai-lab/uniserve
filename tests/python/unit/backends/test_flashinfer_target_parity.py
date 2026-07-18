"""FlashInfer target provider vs the reference provider (GPU parity).

The production-kernel path must reproduce the reference provider — itself
oracle-proven against dense recomputation — across multi-page causal history,
extend+verification stacking, and page-aligned CFG transient overlays, from
identical page storage and packed metadata.
"""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.backends.attention.flashinfer_target import (
    FlashInferTargetBackend,
)
from uniserve_worker.backends.attention.reference_target import (
    CacheDeviceBinding,
    ReferenceAttentionBackend,
)
from uniserve_worker.contracts.cache_schema import CacheLifetime
from uniserve_worker.contracts.execution import (
    CandidateVerification,
    EngineRef,
    ExecuteRow,
    FlowStep,
    SequenceStep,
    SessionRef,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.contracts.segment_table import (
    AttentionLayerSpec,
    GraphCapacity,
)
from uniserve_worker.execution.lowering import RoleSequences, lower_rows
from uniserve_worker.models.cache_registrations import (
    IMAGE_UNCONDITIONAL_ROLE,
    PRIMARY_ROLE,
    TEXT_UNCONDITIONAL_ROLE,
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

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_SESSION = SessionRef(_ENGINE, request_id=41, incarnation=1, session_version=2)
_PAGE_TOKENS = 4
_SITE = AttentionLayerSpec(
    layer_id=0, site_id=1, domain_id=1,
    query_heads=4, kv_heads=2, qk_head_dim=64, value_head_dim=64,
    scale=64**-0.5,
)
_QWEN3 = qwen3_cache_registration(
    layer_count=1, query_heads=4, kv_heads=2, qk_head_dim=64, value_head_dim=64,
    page_tokens=_PAGE_TOKENS,
)
_SENSENOVA = sensenova_cache_registration(
    layer_count=1, query_heads=4, kv_heads=2, qk_head_dim=64, value_head_dim=64,
    page_tokens=_PAGE_TOKENS,
)


def _capacity() -> GraphCapacity:
    return GraphCapacity(
        rows=2, segments=8, tokens=64, branches=3, candidate_tokens=8,
        position_axes=3, visibility_payload_entries=0,
        residency=ResidencyBatchCapacity(bindings=8, page_references=48, tokens=64),
    )


def _pair():
    residency = Residency(
        _ENGINE,
        (ArenaConfig(domain_id=1, page_count=65, page_tokens=_PAGE_TOKENS),),
        product_stores=(ProductStoreConfig(schema_id=7, row_capacity=64),),
    )
    bindings = []
    backends = []
    for maker in (ReferenceAttentionBackend, None):
        binding = CacheDeviceBinding(
            layers=1, pages=65, page_tokens=_PAGE_TOKENS,
            kv_heads=2, head_dim=64, device="cuda", dtype=torch.bfloat16,
        )
        bindings.append(binding)
    backends = [
        ReferenceAttentionBackend(bindings[0]),
        FlashInferTargetBackend(bindings[1], _SITE),
    ]
    return residency, backends


def _row(operation, product_leases=()):
    return ExecuteRow(
        row_id=0, session=_SESSION, operation=operation, admission=None,
        cache_leases=(), product_leases=tuple(product_leases), scheduler_op_id=0,
    )


def _step(residency, backends, registration, roles, operation, leases=()):
    lowered = lower_rows(((_row(operation, leases), roles),), registration)
    capacity = _capacity()
    segments = lowered.fill_segment_table(capacity)
    segments.validate(capacity)
    reservation = residency.reserve(lowered.plan)
    arrays = reservation.batch_arrays(capacity.residency, lowered.write_token_begins)
    tokens = lowered.tokens
    torch.manual_seed(tokens * 977 + arrays.active_page_reference_count)
    q = torch.randn(tokens, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(tokens, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(tokens, 2, 64, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for backend in backends:
        backend.prepare(segments, arrays)
        outputs.append(backend.forward(_SITE, q, k, v))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-2, atol=2e-2)
    return reservation


def test_causal_history_and_verification_parity():
    residency, backends = _pair()
    ref = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    # Multi-page prefill, two decodes, then extend+verification stacking.
    for tokens in (tuple(range(6)), (9,), (10,)):
        roles = RoleSequences({PRIMARY_ROLE: residency.sequence_ref(ref.sequence_id)})
        reservation = _step(
            residency, backends, _QWEN3, roles,
            SequenceStep(tokens, 0, 0, 1),
        )
        reservation.commit((len(tokens),))
    roles = RoleSequences({PRIMARY_ROLE: residency.sequence_ref(ref.sequence_id)})
    verify = SequenceStep(
        (11,), 8, 8, 1, CandidateVerification((1, 2, 3), (9, 10, 11))
    )
    reservation = _step(residency, backends, _QWEN3, roles, verify)
    reservation.commit((1, 2))


def test_cfg_overlay_parity_with_aligned_prefix():
    residency, backends = _pair()
    roles_map = {
        role_id: residency.create_sequence(
            _SESSION, domain_id=1, role_id=role_id, lifetime=CacheLifetime.BRANCH
        )
        for role_id in (PRIMARY_ROLE, TEXT_UNCONDITIONAL_ROLE, IMAGE_UNCONDITIONAL_ROLE)
    }
    # Page-aligned conditioning prefix (8 rows over 4-token pages).
    reservation = _step(
        residency, backends, _SENSENOVA,
        RoleSequences(dict(roles_map)),
        SequenceStep(tuple(range(8)), 0, 0, 1),
    )
    reservation.commit((8,))
    publish = residency.reserve(
        ReservationPlan(
            rows=(
                RowDemand(
                    row_id=0, bindings=(),
                    products=(ProductDemand(schema_id=7, rows=8, producer=_SESSION),),
                ),
            )
        )
    )
    (latent,) = publish.commit(())
    flow = FlowStep(1, 0, 50, latent.lease_id, (4.0, 1.0, 1.0), (), 7)
    roles = RoleSequences(
        {
            role: residency.sequence_ref(state.sequence_id)
            for role, state in roles_map.items()
        }
    )
    reservation = _step(
        residency, backends, _SENSENOVA, roles, flow, leases=(latent,)
    )
    reservation.commit((0, 0, 0))
