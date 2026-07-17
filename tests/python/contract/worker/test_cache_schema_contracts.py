"""Closed cache/attention registration contracts (unified KV runtime, dormant).

Covers the "Closed contracts and sessions" acceptance obligations from
``specs/unified_kv_attention_runtime.md``: family schemas validate without
model callbacks, registration data fingerprints by field order, the region
expressions stay closed and bounded, and the packed ``ResidencyBatch`` ABI
proves its structural invariants and exact capacity byte formula.
"""

from __future__ import annotations

import dataclasses

import pytest

from uniserve_worker.models.cache_registrations import (
    PRIMARY_ROLE,
    bagel_cache_registration,
    qwen3_cache_registration,
    sensenova_cache_registration,
)
from uniserve_worker.contracts.cache_schema import (
    AttentionPattern,
    AttentionSiteSpec,
    CacheEffect,
    CacheRegionSpec,
    CacheRoleSelector,
    CacheSchemaError,
    CommitExpr,
    ExtentExpr,
    ExtentOperand,
    FamilyCacheSchema,
    OperationTag,
    PositionExpr,
    PositionSchema,
    RegionMultiplicity,
    RouteRunSpec,
    registration_fingerprint,
    validate_registration,
)
from uniserve_worker.contracts.residency_batch import (
    ResidencyBatchArrays,
    ResidencyBatchCapacity,
    ResidencyBatchError,
)

pytestmark = pytest.mark.contract

_QWEN3_GEOMETRY = dict(
    layer_count=64,
    query_heads=64,
    kv_heads=8,
    qk_head_dim=128,
    value_head_dim=128,
)
_BAGEL_GEOMETRY = dict(
    layer_count=28,
    query_heads=28,
    kv_heads=4,
    qk_head_dim=128,
    value_head_dim=128,
)
_SENSENOVA_GEOMETRY = dict(
    layer_count=36,
    query_heads=32,
    kv_heads=8,
    qk_head_dim=128,
    value_head_dim=128,
)


# --------------------------------------------------------------------------- #
# Family registrations validate without model callbacks.
# --------------------------------------------------------------------------- #


def test_qwen3_registration_validates():
    registration = qwen3_cache_registration(**_QWEN3_GEOMETRY)
    assert len(registration.domains) == 1
    assert registration.domains[0].layer_ids == tuple(range(64))
    effects = {region.cache_effect for region in registration.schema.regions}
    assert effects == {CacheEffect.PERSISTENT_APPEND, CacheEffect.TENTATIVE_APPEND}


def test_bagel_registration_validates():
    registration = bagel_cache_registration(**_BAGEL_GEOMETRY)
    flow = [
        region
        for region in registration.schema.regions
        if region.operation_tag is OperationTag.FLOW_STEP
    ]
    assert len(flow) == 1
    assert flow[0].cache_effect is CacheEffect.TRANSIENT_OVERLAY
    assert flow[0].multiplicity is RegionMultiplicity.CFG_BRANCHES
    # Marker, generation body, marker route runs per the spec.
    assert [run.route_id for run in flow[0].route_runs] == [0, 1, 0]


def test_sensenova_registration_validates():
    registration = sensenova_cache_registration(**_SENSENOVA_GEOMETRY)
    site = registration.sites[0]
    assert site.position_schema is PositionSchema.TEMPORAL_GRID
    assert AttentionPattern.VISIBLE_END in site.visibility_domain
    flow = [
        region
        for region in registration.schema.regions
        if region.operation_tag is OperationTag.FLOW_STEP
    ]
    assert flow[0].logical_position_commit.kind.name == "ZERO"


def test_registration_fingerprints_are_stable_and_distinct():
    def fingerprint(registration):
        return registration_fingerprint(
            registration.domains,
            registration.sites,
            registration.schema,
        )

    qwen_a = fingerprint(qwen3_cache_registration(**_QWEN3_GEOMETRY))
    qwen_b = fingerprint(qwen3_cache_registration(**_QWEN3_GEOMETRY))
    assert qwen_a == qwen_b
    assert qwen_a != fingerprint(bagel_cache_registration(**_BAGEL_GEOMETRY))
    other_page = fingerprint(
        qwen3_cache_registration(**_QWEN3_GEOMETRY, page_tokens=32)
    )
    assert qwen_a != other_page


# --------------------------------------------------------------------------- #
# Registration validation rejects contract violations.
# --------------------------------------------------------------------------- #


def _region_with(**overrides) -> CacheRegionSpec:
    tokens = ExtentExpr.operand(ExtentOperand.INPUT_TOKEN_COUNT)
    values = dict(
        operation_tag=OperationTag.SEQUENCE_STEP,
        local_region_id=0,
        multiplicity=RegionMultiplicity.ONE,
        role=CacheRoleSelector.fixed(PRIMARY_ROLE),
        route_runs=(RouteRunSpec(route_id=0, extent=tokens),),
        query_rows=tokens,
        reserve_rows=tokens,
        commit_rows=CommitExpr.all_reserved(),
        cache_effect=CacheEffect.PERSISTENT_APPEND,
        attention_pattern=AttentionPattern.CAUSAL_PREFIX,
        position=PositionExpr.linear_from_cursor(0),
        logical_position_commit=CommitExpr.all_reserved(),
    )
    values.update(overrides)
    return CacheRegionSpec(**values)


def _mutated_qwen3(region: CacheRegionSpec):
    registration = qwen3_cache_registration(**_QWEN3_GEOMETRY)
    schema = FamilyCacheSchema(roles=registration.schema.roles, regions=(region,))
    return registration.domains, registration.sites, schema


def test_transient_regions_must_commit_zero_rows():
    domains, sites, schema = _mutated_qwen3(
        _region_with(
            cache_effect=CacheEffect.TRANSIENT_OVERLAY,
            commit_rows=CommitExpr.all_reserved(),
            logical_position_commit=CommitExpr.zero(),
        )
    )
    with pytest.raises(CacheSchemaError, match="transient"):
        validate_registration(domains, sites, schema)


def test_tentative_regions_use_accepted_prefix_commit():
    domains, sites, schema = _mutated_qwen3(
        _region_with(
            cache_effect=CacheEffect.TENTATIVE_APPEND,
            commit_rows=CommitExpr.all_reserved(),
        )
    )
    with pytest.raises(CacheSchemaError, match="tentative"):
        validate_registration(domains, sites, schema)


def test_read_only_regions_reserve_nothing():
    domains, sites, schema = _mutated_qwen3(
        _region_with(
            cache_effect=CacheEffect.READ_ONLY,
            commit_rows=CommitExpr.zero(),
            logical_position_commit=CommitExpr.zero(),
        )
    )
    with pytest.raises(CacheSchemaError, match="READ_ONLY"):
        validate_registration(domains, sites, schema)


def test_route_runs_must_partition_query_rows():
    tokens = ExtentExpr.operand(ExtentOperand.INPUT_TOKEN_COUNT)
    domains, sites, schema = _mutated_qwen3(
        _region_with(
            route_runs=(
                RouteRunSpec(route_id=0, extent=tokens),
                RouteRunSpec(route_id=1, extent=ExtentExpr.literal(1)),
            ),
        )
    )
    with pytest.raises(CacheSchemaError, match="partition"):
        validate_registration(domains, sites, schema)


def test_site_geometry_must_match_domain_layout():
    registration = qwen3_cache_registration(**_QWEN3_GEOMETRY)
    bad_site = dataclasses.replace(registration.sites[0], kv_heads=16)
    with pytest.raises(CacheSchemaError, match="geometry"):
        validate_registration(
            registration.domains,
            (bad_site,),
            registration.schema,
        )


def test_layers_belong_to_exactly_one_domain():
    registration = qwen3_cache_registration(**_QWEN3_GEOMETRY)
    duplicate = dataclasses.replace(
        registration.domains[0],
        domain_id=2,
    )
    with pytest.raises(CacheSchemaError, match="exactly one domain"):
        validate_registration(
            (*registration.domains, duplicate),
            registration.sites,
            registration.schema,
        )


def test_extent_expressions_are_bounded_affine():
    expr = ExtentExpr(
        constant=2,
        terms=(
            (ExtentOperand.IMAGE_TOKEN_COUNT, 1),
            (ExtentOperand.CFG_BRANCH_COUNT, 3),
        ),
    )
    value = expr.evaluate(
        {
            ExtentOperand.IMAGE_TOKEN_COUNT: 4096,
            ExtentOperand.CFG_BRANCH_COUNT: 3,
        }
    )
    assert value == 2 + 4096 + 9
    with pytest.raises(CacheSchemaError, match="missing operand"):
        expr.evaluate({ExtentOperand.IMAGE_TOKEN_COUNT: 4096})


def test_no_cache_regions_must_be_read_only():
    tokens = ExtentExpr.operand(ExtentOperand.INPUT_TOKEN_COUNT)
    domains, sites, schema = _mutated_qwen3(
        _region_with(
            role=CacheRoleSelector.no_cache(),
            cache_effect=CacheEffect.PERSISTENT_APPEND,
            reserve_rows=tokens,
        )
    )
    with pytest.raises(CacheSchemaError, match="READ_ONLY"):
        validate_registration(domains, sites, schema)


# --------------------------------------------------------------------------- #
# Attention-site sanity used by the fingerprint path.
# --------------------------------------------------------------------------- #


def test_sites_require_an_attention_pattern():
    registration = qwen3_cache_registration(**_QWEN3_GEOMETRY)
    empty_site = dataclasses.replace(
        registration.sites[0],
        visibility_domain=frozenset(),
    )
    with pytest.raises(CacheSchemaError, match="pattern"):
        validate_registration(
            registration.domains,
            (empty_site,),
            registration.schema,
        )
    assert isinstance(registration.sites[0], AttentionSiteSpec)


# --------------------------------------------------------------------------- #
# Packed ResidencyBatch ABI.
# --------------------------------------------------------------------------- #


def _capacity() -> ResidencyBatchCapacity:
    return ResidencyBatchCapacity(bindings=2, page_references=6, tokens=4)


def _valid_arrays() -> ResidencyBatchArrays:
    # One active binding owning three pages; two active write slots.
    return ResidencyBatchArrays(
        active_binding_count=1,
        active_page_reference_count=3,
        binding_active=[0, 1, 0],
        binding_domain_id=[0, 1, 0],
        binding_committed_rows=[0, 40, 0],
        binding_provisional_rows=[0, 42, 0],
        binding_page_indptr=[0, 0, 3, 3],
        page_ids=[7, 9, 11, 0, 0, 0],
        write_page_ids=[11, 11, 0, 0],
        write_page_offsets=[8, 9, 0, 0],
        write_active=[1, 1, 0, 0],
    )


def test_capacity_byte_formula_is_exact():
    capacity = _capacity()
    b, p, t = 2, 6, 4
    assert capacity.canonical_metadata_bytes() == 4 * (4 * b + p + 2 * t + 7) + b + t + 1


def test_valid_packed_mapping_passes():
    _valid_arrays().validate(_capacity(), page_tokens=16)


def test_sink_binding_must_stay_zeroed():
    arrays = dataclasses.replace(_valid_arrays(), binding_active=[1, 1, 0])
    with pytest.raises(ResidencyBatchError, match="sink binding"):
        arrays.validate(_capacity(), page_tokens=16)


def test_active_binding_indptr_must_be_monotonic():
    arrays = dataclasses.replace(_valid_arrays(), binding_page_indptr=[0, 0, 2, 3])
    with pytest.raises(ResidencyBatchError):
        arrays.validate(_capacity(), page_tokens=16)


def test_inactive_bindings_pin_to_active_reference_count():
    arrays = dataclasses.replace(_valid_arrays(), binding_page_indptr=[0, 0, 3, 4])
    with pytest.raises(ResidencyBatchError, match="inactive binding"):
        arrays.validate(_capacity(), page_tokens=16)


def test_inactive_page_references_name_the_sink_page():
    arrays = dataclasses.replace(_valid_arrays(), page_ids=[7, 9, 11, 5, 0, 0])
    with pytest.raises(ResidencyBatchError, match="sink page"):
        arrays.validate(_capacity(), page_tokens=16)


def test_write_offsets_respect_page_geometry():
    arrays = dataclasses.replace(_valid_arrays(), write_page_offsets=[8, 16, 0, 0])
    with pytest.raises(ResidencyBatchError, match="page geometry"):
        arrays.validate(_capacity(), page_tokens=16)


def test_provisional_rows_never_undershoot_committed_rows():
    arrays = dataclasses.replace(
        _valid_arrays(),
        binding_provisional_rows=[0, 39, 0],
    )
    with pytest.raises(ResidencyBatchError, match="inconsistent"):
        arrays.validate(_capacity(), page_tokens=16)
