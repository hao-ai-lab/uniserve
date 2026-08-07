"""Behavioral contracts for immutable model declarations and resolved identity."""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest

from tests.python.fixtures.model_execution import TEST_DEPLOYMENT, TEST_MODEL_SPEC
from uniserve_worker.batch import WorkVariant
from uniserve_worker.capabilities import configured_work_variants
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.runtime.capabilities import prove_depth_one_lowering
from uniserve_worker.spec import (
    ModelSpec,
    OperationSpec,
    OperationStagePurpose,
    OperationStageSpec,
    Rename,
    RoutePlacement,
    RouteRowKind,
    RouteShape,
    RouteSpec,
    Stack,
    WeightSpec,
    resolved_digest,
)

pytestmark = pytest.mark.unit


def _route(**overrides: object) -> RouteSpec:
    values: dict[str, object] = {
        "name": "text",
        "row_kinds": (RouteRowKind.TOKEN,),
        "mixed_combinations": (),
        "dtype": "float32",
        "placement": RoutePlacement.PRIMARY,
        "topology_axes": ("tp",),
        "shape": RouteShape(max_tokens_per_row=128, token_multiple=1),
        "graph_eligible": True,
    }
    values.update(overrides)
    return RouteSpec(**values)  # type: ignore[arg-type]


def _spec(**overrides: object) -> ModelSpec:
    route = _route()
    values: dict[str, object] = {
        "architecture": "ConformanceModel",
        "revision": "revision-a",
        "routes": (route,),
        "operations": (
            OperationSpec(
                WorkVariant.TOKEN_EXTEND,
                (OperationStageSpec(route.name, RouteRowKind.TOKEN),),
            ),
        ),
        "weights": WeightSpec(
            transforms=(
                Rename("model.", "core."),
                Stack("qkv_proj", "q_proj", "q"),
            )
        ),
        "inputs": TEST_MODEL_SPEC.inputs,
        "cache": TEST_MODEL_SPEC.cache,
    }
    values.update(overrides)
    return ModelSpec(**values)  # type: ignore[arg-type]


def test_model_declarations_are_immutable():
    spec = _spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.architecture = "Other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.routes[0].dtype = "float16"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        TEST_DEPLOYMENT.block_size = 32  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.weights.transforms[0].source = "other."  # type: ignore[misc,union-attr]


def test_weight_spec_rejects_duplicate_stack_sources():
    with pytest.raises(WorkerError, match="weight stack sources must be unique"):
        WeightSpec(
            transforms=(
                Stack("first", "q_proj", "q"),
                Stack("second", "q_proj", "q"),
            )
        )


def test_route_requires_a_closed_aligned_row_output_vocabulary():
    with pytest.raises(WorkerError) as repeated:
        _route(row_kinds=(RouteRowKind.TOKEN, RouteRowKind.TOKEN))
    assert repeated.value.code is ErrorCode.INVALID_DESCRIPTOR

    with pytest.raises(WorkerError, match="mixes a row kind it does not accept"):
        _route(mixed_combinations=((RouteRowKind.TOKEN, RouteRowKind.FLOW),))


def test_model_spec_rejects_ambiguous_or_invalid_operation_routes():
    route = _route()
    with pytest.raises(WorkerError, match="route names must be unique"):
        _spec(routes=(route, route))

    with pytest.raises(WorkerError, match="unknown route"):
        _spec(
            operations=(
                OperationSpec(
                    WorkVariant.TOKEN_EXTEND,
                    (OperationStageSpec("missing", RouteRowKind.TOKEN),),
                ),
            )
        )

    with pytest.raises(WorkerError, match="does not accept"):
        _spec(
            operations=(
                OperationSpec(
                    WorkVariant.GEN_TRANSITION,
                    (OperationStageSpec(route.name, RouteRowKind.FLOW),),
                ),
            ),
            flow=TEST_MODEL_SPEC.flow,
        )


def test_depth_one_lowering_accepts_the_canonical_model_spec():
    supported = configured_work_variants(
        operation.kind for operation in TEST_MODEL_SPEC.operations
    )
    lowering = prove_depth_one_lowering(TEST_MODEL_SPEC, supported)
    declared = {route.name for route in TEST_MODEL_SPEC.routes}

    assert set(lowering) == set(supported)
    for stage in lowering.values():
        assert stage is None or stage.route in declared


def test_depth_one_lowering_accepts_a_model_free_materialization_frame():
    route = _route(row_kinds=(RouteRowKind.TOKEN,))
    spec = _spec(
        routes=(route,),
        operations=(
            OperationSpec(
                WorkVariant.TOKEN_EXTEND,
                (OperationStageSpec(route.name, RouteRowKind.TOKEN),),
            ),
            OperationSpec(WorkVariant.MATERIALIZE),
        ),
    )
    supported = configured_work_variants(operation.kind for operation in spec.operations)

    lowering = prove_depth_one_lowering(spec, supported)

    assert lowering[WorkVariant.MATERIALIZE] is None
    assert lowering[WorkVariant.TOKEN_EXTEND].route == route.name


def test_depth_one_lowering_maps_a_state_only_transfer_variant_to_no_route():
    # A KV state-publication op carries only STATE stages and lowers onto no
    # compute route: the proof maps it to None rather than rejecting it, while a
    # neural variant still resolves to its one declared primary route.
    route = _route()
    spec = _spec(
        operations=(
            OperationSpec(
                WorkVariant.TOKEN_EXTEND,
                (OperationStageSpec(route.name, RouteRowKind.TOKEN),),
            ),
            OperationSpec(
                WorkVariant.TRANSFER_KV_PUBLISH,
                (
                    OperationStageSpec(
                        route.name, RouteRowKind.TOKEN, OperationStagePurpose.STATE
                    ),
                ),
            ),
        )
    )
    supported = configured_work_variants(operation.kind for operation in spec.operations)

    lowering = prove_depth_one_lowering(spec, supported)

    assert lowering[WorkVariant.TRANSFER_KV_PUBLISH] is None
    assert lowering[WorkVariant.TOKEN_EXTEND].route == route.name


def test_flow_operation_requires_a_flow_spec():
    flow_route = _route(row_kinds=(RouteRowKind.FLOW,))
    with pytest.raises(WorkerError, match="declares no FlowSpec"):
        _spec(
            routes=(flow_route,),
            operations=(
                OperationSpec(
                    WorkVariant.GEN_TRANSITION,
                    (OperationStageSpec(flow_route.name, RouteRowKind.FLOW),),
                ),
            ),
        )


def test_resolved_digest_is_stable_and_binds_spec_and_deployment():
    first = resolved_digest(_spec(), TEST_DEPLOYMENT)
    assert first == resolved_digest(_spec(), TEST_DEPLOYMENT)
    assert len(first) == 64
    int(first, 16)

    assert resolved_digest(_spec(revision="revision-b"), TEST_DEPLOYMENT) != first
    assert resolved_digest(_spec(), replace(TEST_DEPLOYMENT, block_size=32)) != first
    assert resolved_digest(_spec(), replace(TEST_DEPLOYMENT, device="cuda")) != first


def test_resolved_digest_is_shared_by_tensor_parallel_ranks():
    rank_zero = replace(TEST_DEPLOYMENT, device="cuda:0", tp_rank=0, tp_size=2)
    rank_one = replace(TEST_DEPLOYMENT, device="cuda:1", tp_rank=1, tp_size=2)

    assert resolved_digest(_spec(), rank_zero) == resolved_digest(_spec(), rank_one)
    assert resolved_digest(_spec(), rank_zero) != resolved_digest(
        _spec(), replace(rank_zero, tp_size=4)
    )


def test_deployment_overlay_validates_topology_and_capacity():
    with pytest.raises(WorkerError):
        replace(TEST_DEPLOYMENT, tp_rank=1, tp_size=1)
    with pytest.raises(WorkerError):
        replace(TEST_DEPLOYMENT, tp_size=0)
    with pytest.raises(WorkerError):
        replace(TEST_DEPLOYMENT, block_size=0)


@pytest.mark.parametrize("alias", ["bf16", "fp16", "fp32", "half", "fp8_e4m3"])
def test_dtype_aliases_are_rejected(alias: str):
    with pytest.raises(WorkerError):
        _route(dtype=alias)
    with pytest.raises(WorkerError):
        replace(TEST_MODEL_SPEC.cache, dtype=alias)
    with pytest.raises(WorkerError):
        replace(TEST_DEPLOYMENT, model_dtype=alias)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_canonical_float_dtypes_are_accepted(dtype: str):
    assert _route(dtype=dtype).dtype == dtype
    assert replace(TEST_MODEL_SPEC.cache, dtype=dtype).dtype == dtype
    assert replace(TEST_DEPLOYMENT, model_dtype=dtype).model_dtype == dtype
