"""Reference attention provider vs dense oracle (canonical attention, GPU).

Packed page references, backend-owned K/V writes, causal-prefix and
full-query-prefix regions, verification stacking, and transient overlays are
compared numerically against a dense full-recompute oracle across multiple
committed transactions — the packed ABI must be byte-equivalent truth, not
just structurally valid.
"""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.backends.attention.reference_target import (
    AttentionLayerSpec,
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
from uniserve_worker.contracts.segment_table import GraphCapacity
from uniserve_worker.execution.engine import RoleSequences, lower_rows
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
_PAGE_TOKENS = 4  # small pages exercise multi-page chains quickly
_LAYER = AttentionLayerSpec(
    layer_id=0,
    site_id=1,
    domain_id=1,
    query_heads=4,
    kv_heads=2,
    qk_head_dim=8,
    value_head_dim=8,
    scale=8**-0.5,
)

_QWEN3 = qwen3_cache_registration(
    layer_count=1, query_heads=4, kv_heads=2, qk_head_dim=8, value_head_dim=8,
    page_tokens=_PAGE_TOKENS,
)
_SENSENOVA = sensenova_cache_registration(
    layer_count=1, query_heads=4, kv_heads=2, qk_head_dim=8, value_head_dim=8,
    page_tokens=_PAGE_TOKENS,
)


def _capacity(tokens: int = 32) -> GraphCapacity:
    return GraphCapacity(
        rows=2,
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


def _dense_causal(queries, prefix_k, prefix_v, current_k, current_v):
    """Oracle: causal attention over [prefix + current] at current positions."""

    keys = torch.cat([prefix_k, current_k], dim=0)
    values = torch.cat([prefix_v, current_v], dim=0)
    group = queries.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(group, dim=1)
    values = values.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", queries.float(), keys.float()) * _LAYER.scale
    context = prefix_k.shape[0]
    count = queries.shape[0]
    key_pos = torch.arange(context + count, device=queries.device)
    query_pos = context + torch.arange(count, device=queries.device)
    scores = scores.masked_fill(
        (key_pos[None, :] > query_pos[:, None])[None], float("-inf")
    )
    return torch.einsum(
        "hqk,khd->qhd", torch.softmax(scores, dim=-1), values.float()
    ).to(queries.dtype)


def _dense_full(queries, prefix_k, prefix_v, current_k, current_v):
    """Oracle: every query sees the prefix and the whole current region."""

    keys = torch.cat([prefix_k, current_k], dim=0)
    values = torch.cat([prefix_v, current_v], dim=0)
    group = queries.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(group, dim=1)
    values = values.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", queries.float(), keys.float()) * _LAYER.scale
    return torch.einsum(
        "hqk,khd->qhd", torch.softmax(scores, dim=-1), values.float()
    ).to(queries.dtype)


def _row(operation, product_leases=()):
    return ExecuteRow(
        row_id=0,
        session=_SESSION,
        operation=operation,
        admission=None,
        cache_leases=(),
        product_leases=tuple(product_leases),
        scheduler_op_id=0,
    )


def _run_step(residency, backend, registration, roles, operation, leases=()):
    lowered = lower_rows(((_row(operation, leases), roles),), registration)
    capacity = _capacity()
    segments = lowered.fill_segment_table(capacity)
    segments.validate(capacity)
    reservation = residency.reserve(lowered.plan)
    arrays = reservation.batch_arrays(capacity.residency, lowered.write_token_begins)
    arrays.validate(capacity.residency, page_tokens=_PAGE_TOKENS)
    tokens = lowered.tokens
    torch.manual_seed(tokens * 1000 + arrays.active_page_reference_count)
    q = torch.randn(tokens, _LAYER.query_heads, _LAYER.qk_head_dim, device="cuda")
    k = torch.randn(tokens, _LAYER.kv_heads, _LAYER.qk_head_dim, device="cuda")
    v = torch.randn(tokens, _LAYER.kv_heads, _LAYER.value_head_dim, device="cuda")
    backend.prepare(segments, arrays)
    out = backend.forward(_LAYER, q, k, v)
    return lowered, reservation, segments, (q, k, v), out


def test_causal_history_matches_the_dense_oracle_across_transactions():
    residency = Residency(
        _ENGINE, (ArenaConfig(domain_id=1, page_count=33, page_tokens=_PAGE_TOKENS),)
    )
    binding = CacheDeviceBinding(
        layers=1, pages=33, page_tokens=_PAGE_TOKENS,
        kv_heads=2, head_dim=8, device="cuda",
    )
    backend = ReferenceAttentionBackend(binding)
    ref = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    history_k = torch.zeros(0, 2, 8, device="cuda")
    history_v = torch.zeros(0, 2, 8, device="cuda")

    # Prefill 6 rows (multi-page), then two single-row decodes.
    for step_tokens in (tuple(range(6)), (9,), (10,)):
        roles = RoleSequences({PRIMARY_ROLE: residency.sequence_ref(ref.sequence_id)})
        operation = SequenceStep(
            step_tokens, history_k.shape[0], history_k.shape[0], 1
        )
        lowered, reservation, segments, (q, k, v), out = _run_step(
            residency, backend, _QWEN3, roles, operation
        )
        expected = _dense_causal(q, history_k, history_v, k, v)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        reservation.commit((len(step_tokens),))
        history_k = torch.cat([history_k, k], dim=0)
        history_v = torch.cat([history_v, v], dim=0)


def test_verification_stacks_and_publishes_only_the_accepted_prefix():
    residency = Residency(
        _ENGINE, (ArenaConfig(domain_id=1, page_count=33, page_tokens=_PAGE_TOKENS),)
    )
    binding = CacheDeviceBinding(
        layers=1, pages=33, page_tokens=_PAGE_TOKENS,
        kv_heads=2, head_dim=8, device="cuda",
    )
    backend = ReferenceAttentionBackend(binding)
    ref = residency.create_sequence(
        _SESSION, domain_id=1, role_id=PRIMARY_ROLE, lifetime=CacheLifetime.REQUEST
    )
    roles = RoleSequences({PRIMARY_ROLE: ref})
    prefill = SequenceStep((1, 2, 3), 0, 0, 1)
    _, reservation, _, (q0, k0, v0), _ = _run_step(
        residency, backend, _QWEN3, roles, prefill
    )
    reservation.commit((3,))

    # Extend by one token and verify three candidates in the same row.
    roles = RoleSequences({PRIMARY_ROLE: residency.sequence_ref(ref.sequence_id)})
    verify = SequenceStep(
        (4,), 3, 3, 1, CandidateVerification((7, 8, 9), (4, 5, 6))
    )
    lowered, reservation, segments, (q, k, v), out = _run_step(
        residency, backend, _QWEN3, roles, verify
    )
    # Region 1 (the input token) attends prefix(3); region 2 (candidates)
    # attends prefix(3) + the input token, causally within itself.
    expected_text = _dense_causal(q[:1], k0, v0, k[:1], v[:1])
    torch.testing.assert_close(out[:1], expected_text, rtol=1e-4, atol=1e-4)
    prefix_k = torch.cat([k0, k[:1]], dim=0)
    prefix_v = torch.cat([v0, v[:1]], dim=0)
    expected_candidates = _dense_causal(q[1:], prefix_k, prefix_v, k[1:], v[1:])
    torch.testing.assert_close(out[1:], expected_candidates, rtol=1e-4, atol=1e-4)
    # Accept one candidate; committed history is 3 + 1 + 1.
    reservation.commit((1, 1))
    assert residency.sequence_ref(ref.sequence_id).committed_rows == 5


def test_transient_overlays_read_the_prefix_and_leave_it_untouched():
    residency = Residency(
        _ENGINE,
        (ArenaConfig(domain_id=1, page_count=33, page_tokens=_PAGE_TOKENS),),
        product_stores=(ProductStoreConfig(schema_id=7, row_capacity=64),),
    )
    binding = CacheDeviceBinding(
        layers=1, pages=33, page_tokens=_PAGE_TOKENS,
        kv_heads=2, head_dim=8, device="cuda",
    )
    backend = ReferenceAttentionBackend(binding)
    roles_map = {
        role_id: residency.create_sequence(
            _SESSION, domain_id=1, role_id=role_id, lifetime=CacheLifetime.BRANCH
        )
        for role_id in (PRIMARY_ROLE, TEXT_UNCONDITIONAL_ROLE, IMAGE_UNCONDITIONAL_ROLE)
    }
    # Commit a 5-row conditioning prefix on the primary branch.
    prefill = SequenceStep((1, 2, 3, 4, 5), 0, 0, 1)
    _, reservation, _, (q0, k0, v0), _ = _run_step(
        residency, backend, _SENSENOVA, RoleSequences(dict(roles_map)), prefill
    )
    reservation.commit((5,))

    publish = residency.reserve(
        ReservationPlan(
            rows=(
                RowDemand(
                    row_id=0,
                    bindings=(),
                    products=(
                        ProductDemand(schema_id=7, rows=6, producer=_SESSION),
                    ),
                ),
            )
        )
    )
    (latent,) = publish.commit(())
    flow = FlowStep(1, 0, 50, latent.lease_id, (4.0, 1.0, 1.0), (), 7)
    roles = RoleSequences(
        {
            PRIMARY_ROLE: residency.sequence_ref(
                roles_map[PRIMARY_ROLE].sequence_id
            ),
            TEXT_UNCONDITIONAL_ROLE: residency.sequence_ref(
                roles_map[TEXT_UNCONDITIONAL_ROLE].sequence_id
            ),
            IMAGE_UNCONDITIONAL_ROLE: residency.sequence_ref(
                roles_map[IMAGE_UNCONDITIONAL_ROLE].sequence_id
            ),
        }
    )
    lowered, reservation, segments, (q, k, v), out = _run_step(
        residency, backend, _SENSENOVA, roles, flow, leases=(latent,)
    )
    # Branch 0 (conditional) sees the committed prefix; branches 1 and 2 see
    # empty histories. Every branch region is full-query-prefix visible.
    spans = [(0, 6, k0, v0), (6, 12, None, None), (12, 18, None, None)]
    for begin, end, pk, pv in spans:
        prefix_k = pk if pk is not None else k.new_zeros((0, 2, 8))
        prefix_v = pv if pv is not None else v.new_zeros((0, 2, 8))
        expected = _dense_full(
            q[begin:end], prefix_k, prefix_v, k[begin:end], v[begin:end]
        )
        torch.testing.assert_close(
            out[begin:end], expected, rtol=1e-4, atol=1e-4
        )
    reservation.commit((0, 0, 0))
    # The committed prefix is untouched: a follow-up decode still matches the
    # oracle built from the original prefill tensors.
    decode = SequenceStep((6,), 5, 5, 1)
    roles = RoleSequences(
        {PRIMARY_ROLE: residency.sequence_ref(roles_map[PRIMARY_ROLE].sequence_id)}
    )
    _, reservation, _, (q1, k1, v1), out1 = _run_step(
        residency, backend, _SENSENOVA, roles, decode
    )
    expected = _dense_causal(q1, k0, v0, k1, v1)
    torch.testing.assert_close(out1, expected, rtol=1e-4, atol=1e-4)
    reservation.commit((1,))
