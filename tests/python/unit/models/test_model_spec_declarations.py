"""Every served model family declares one composable ModelSpec."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.execution.flow import schedule_from_flow_spec
from uniserve_worker.models.bagel import BagelForUnifiedGeneration
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration
from uniserve_worker.nn.diffusion import (
    FlowMatchSchedule,
    ScheduleDirection,
    ScheduleShiftDomain,
)
from uniserve_worker.server.stub import StubUniModel

pytestmark = pytest.mark.unit

TINY_QWEN3_CONFIG = {
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "attention_bias": False,
}


def _assert_spec_matches_model(spec, model):
    assert spec.architecture == type(model).architectures[0]
    assert spec.revision == ""
    assert spec.op_kinds() == frozenset(model.supported_ops)
    assert spec.weights is type(model).weight_spec or spec.weights == type(model).weight_spec


def test_qwen3_declares_a_single_graph_eligible_text_route():
    model = Qwen3ForCausalLM(config=TINY_QWEN3_CONFIG)
    spec = model.model_spec()

    _assert_spec_matches_model(spec, model)
    (route,) = spec.routes
    assert route.name == "text"
    assert route.mixed and route.graph_eligible
    assert spec.flow is None
    assert spec.inputs.requires_worker_tokenizer
    assert spec.cache.num_layers == 2
    assert spec.cache.num_kv_heads == 2
    assert spec.cache.head_dim == 4


def test_bagel_declares_mot_vae_and_vit_routes_with_flow_semantics():
    model = BagelForUnifiedGeneration()
    spec = model.model_spec()

    _assert_spec_matches_model(spec, model)
    routes = {route.name: route for route in spec.routes}
    assert set(routes) == {"mot", "vae", "vit"}
    assert routes["mot"].mixed and routes["mot"].graph_eligible
    assert set(routes["mot"].op_kinds) == {"prefill_und", "decode_und", "denoise_gen"}
    assert not routes["vae"].mixed and not routes["vae"].graph_eligible
    assert not spec.inputs.requires_worker_tokenizer
    assert spec.flow is not None
    assert spec.flow.prediction == "velocity"
    assert spec.flow.schedule_direction == "descending"
    assert spec.flow.schedule_shift_domain == "time"
    assert spec.flow.cfg_recipe == "image_over_text"
    assert spec.flow.timestep_shift == model.cfg.timestep_shift
    assert spec.flow.latent_downsample == model.cfg.latent_downsample
    assert spec.cache.num_layers == model.cfg.llm.num_hidden_layers


def test_sensenova_declares_mot_vae_and_vit_routes_with_flow_semantics():
    model = SenseNovaU1ForUnifiedGeneration(
        config={
            "llm_config": {
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
            }
        }
    )
    spec = model.model_spec()

    _assert_spec_matches_model(spec, model)
    routes = {route.name: route for route in spec.routes}
    assert set(routes) == {"mot", "vae", "vit"}
    assert routes["mot"].mixed and routes["mot"].graph_eligible
    assert set(routes["vae"].op_kinds) == {"commit_gen", "commit_writeback"}
    assert spec.inputs.requires_worker_tokenizer
    assert spec.flow is not None
    assert spec.flow.prediction == "velocity"
    assert spec.flow.schedule_direction == "ascending"
    assert spec.flow.schedule_shift_domain == "sigma"
    assert spec.flow.cfg_recipe == "additive_deltas"
    assert spec.flow.timestep_shift is None
    assert spec.flow.latent_downsample == model.latent_downsample
    assert spec.cache.num_layers == 1
    assert spec.cache.num_kv_heads == 1
    assert spec.cache.head_dim == 4


def test_stub_model_declares_one_mixed_route_covering_its_ops():
    model = StubUniModel()
    spec = model.model_spec()

    assert spec.architecture == StubUniModel.architectures[0]
    assert spec.op_kinds() == frozenset(model.supported_ops)
    (route,) = spec.routes
    assert route.mixed and not route.graph_eligible
    assert spec.flow is not None
    assert spec.flow.latent_downsample == model.latent_downsample


def test_declared_flow_spec_produces_each_family_reference_schedule():
    """Spec-driven schedule selection reproduces each family's schedule bit for bit."""
    bagel_flow = BagelForUnifiedGeneration().model_spec().flow
    assert bagel_flow is not None
    bagel_schedule = schedule_from_flow_spec(bagel_flow, num_steps=5, shift=1.7)
    bagel_reference = FlowMatchSchedule(
        num_steps=5,
        shift=1.7,
        direction=ScheduleDirection.DESCENDING,
        shift_domain=ScheduleShiftDomain.TIME,
    )
    assert torch.equal(bagel_schedule.timesteps(), bagel_reference.timesteps())

    sensenova_flow = (
        SenseNovaU1ForUnifiedGeneration(
            config={
                "llm_config": {
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                }
            }
        )
        .model_spec()
        .flow
    )
    assert sensenova_flow is not None
    sensenova_schedule = schedule_from_flow_spec(sensenova_flow, num_steps=5, shift=3.0)
    sensenova_reference = FlowMatchSchedule(
        num_steps=5,
        shift=3.0,
        direction=ScheduleDirection.ASCENDING,
        shift_domain=ScheduleShiftDomain.SIGMA,
    )
    assert torch.equal(sensenova_schedule.timesteps(), sensenova_reference.timesteps())
