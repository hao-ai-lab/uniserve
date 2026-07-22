"""Every served model family declares one composable ModelSpec."""

from __future__ import annotations

import pytest

from uniserve_worker.models.bagel import BagelForUnifiedGeneration
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration
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
