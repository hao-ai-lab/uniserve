"""Behavioral contracts for ModelSpec composition and the resolved digest."""

from __future__ import annotations

import dataclasses

import pytest

from uniserve_worker.contracts.model_spec import (
    CacheSpec,
    DeploymentOverlay,
    FlowSpec,
    InputSpec,
    ModelSpec,
    RouteSpec,
    resolved_digest,
)
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.loader.weight_spec import Rename, StackedParamMapping, WeightSpec

pytestmark = pytest.mark.unit


def _text_route(**overrides):
    values = dict(
        name="text",
        op_kinds=("prefill_und", "decode_und"),
        mixed=True,
        dtype="bfloat16",
        graph_eligible=True,
    )
    values.update(overrides)
    return RouteSpec(**values)


def _spec(**overrides):
    values = dict(
        architecture="TinyForCausalLM",
        routes=(_text_route(),),
        weights=WeightSpec(
            renames=(Rename("model.", "core."),),
            stacked=(StackedParamMapping("qkv_proj", "q_proj", "q"),),
        ),
        inputs=InputSpec(requires_worker_tokenizer=True),
        cache=CacheSpec(num_layers=2, num_kv_heads=4, head_dim=64, dtype="bfloat16"),
        revision="rev-a",
    )
    values.update(overrides)
    return ModelSpec(**values)


def _overlay(**overrides):
    values = dict(
        device="cuda:0",
        model_scope="whole",
        tp_rank=0,
        tp_size=1,
        block_size=16,
        kv_token_capacity=4096,
        generation_kv_capacity_tokens=None,
        attention_backend="auto",
        model_dtype="bfloat16",
        kv_cache_dtype=None,
    )
    values.update(overrides)
    return DeploymentOverlay(**values)


def test_model_spec_composition_is_immutable():
    spec = _spec()
    overlay = _overlay()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.architecture = "Other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.routes[0].op_kinds = ("decode_und",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        overlay.block_size = 32


def test_route_rejects_unknown_and_repeated_op_kinds():
    with pytest.raises(WorkerError) as unknown:
        _text_route(op_kinds=("prefill_und", "no_such_op"))
    assert unknown.value.code is ErrorCode.INVALID_DESCRIPTOR
    with pytest.raises(WorkerError):
        _text_route(op_kinds=("prefill_und", "prefill_und"))
    with pytest.raises(WorkerError):
        _text_route(op_kinds=())


def test_model_spec_rejects_ambiguous_route_bindings():
    with pytest.raises(WorkerError) as duplicate_name:
        _spec(routes=(_text_route(), _text_route(op_kinds=("target_verify_und",))))
    assert "unique" in duplicate_name.value.message
    with pytest.raises(WorkerError) as shared_kind:
        _spec(
            routes=(
                _text_route(),
                _text_route(name="text2", op_kinds=("decode_und", "target_verify_und")),
            )
        )
    assert "exactly one physical route" in shared_kind.value.message
    with pytest.raises(WorkerError):
        _spec(routes=())


def test_denoise_route_requires_a_flow_spec():
    denoise_route = _text_route(name="mot", op_kinds=("prefill_und", "denoise_gen"))
    with pytest.raises(WorkerError) as missing_flow:
        _spec(routes=(denoise_route,))
    assert "FlowSpec" in missing_flow.value.message

    spec = _spec(
        routes=(denoise_route,),
        flow=FlowSpec(
            latent_downsample=16,
            prediction="velocity",
            schedule_direction="descending",
            schedule_shift_domain="time",
        ),
    )
    assert spec.op_kinds() == frozenset({"prefill_und", "denoise_gen"})


def test_resolved_digest_is_stable_for_equal_declarations():
    first = resolved_digest(_spec(), _overlay())
    second = resolved_digest(_spec(), _overlay())
    assert first == second
    assert len(first) == 64
    int(first, 16)


def test_resolved_digest_separates_spec_and_overlay_identities():
    base = resolved_digest(_spec(), _overlay())
    assert resolved_digest(_spec(revision="rev-b"), _overlay()) != base
    assert resolved_digest(_spec(), _overlay(block_size=32)) != base
    assert resolved_digest(_spec(), _overlay(device="cuda:1")) != base
    assert resolved_digest(_spec(), _overlay(tp_size=2)) != base
    assert (
        resolved_digest(_spec(routes=(_text_route(graph_eligible=False),)), _overlay()) != base
    )


def test_deployment_overlay_validates_topology():
    with pytest.raises(WorkerError):
        _overlay(tp_rank=1, tp_size=1)
    with pytest.raises(WorkerError):
        _overlay(tp_size=0)
    with pytest.raises(WorkerError):
        _overlay(block_size=0)
