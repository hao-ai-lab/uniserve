"""Canonical declarations exposed by every concrete model root."""

from __future__ import annotations

import dataclasses

import pytest
from torch import nn

from uniserve_worker.models.bagel import BagelConfig, BagelForUnifiedGeneration, LLMConfig
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.sensenova.config import NeoChatConfig
from uniserve_worker.models.sensenova.model import NEOChatModel
from uniserve_worker.nn.layer import LayerSpec
from uniserve_worker.nn.mesh import TensorParallelSpec
from uniserve_worker.server.stub import StubModel
from uniserve_worker.spec import OperationType, RouteRowKind

pytestmark = pytest.mark.unit


def _layer_spec() -> LayerSpec:
    return LayerSpec(TensorParallelSpec(rank=0, size=1), None)


def _qwen_config() -> dict[str, object]:
    return {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "attention_bias": False,
        "max_position_embeddings": 128,
    }


def _bagel_config() -> BagelConfig:
    return BagelConfig(
        llm=LLMConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            max_position_embeddings=128,
        ),
        max_latent_size=2,
        vit_image_size=224,
        vit_patch_size=14,
        vit_max_num_patch_per_side=16,
    )


class _LoadedBagelGraph(nn.Module):
    def __init__(self, config: BagelConfig) -> None:
        super().__init__()
        self.cfg = config


def _sensenova_config() -> NeoChatConfig:
    return NeoChatConfig(
        vision_config={
            "hidden_size": 8,
            "llm_hidden_size": 8,
            "downsample_ratio": 0.5,
            "patch_size": 2,
            "num_channels": 3,
            "rope_theta_vision": 10_000.0,
            "max_position_embeddings_vision": 128,
        },
        llm_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "attention_bias": False,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "max_position_embeddings": 128,
            "rope_theta_hw": 10_000.0,
            "max_position_embeddings_hw": 128,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        downsample_ratio=0.5,
        max_image_seq_len=16,
        fm_head_layers=2,
    )


def test_sensenova_composition_resolves_top_level_token_ids():
    config = NeoChatConfig(
        llm_config={
            "bos_token_id": 11,
            "eos_token_id": 12,
            "pad_token_id": None,
        },
        bos_token_id=None,
        eos_token_id=22,
        pad_token_id=23,
    )

    assert config.llm_config.bos_token_id == 11
    assert config.llm_config.eos_token_id == 12
    assert config.llm_config.pad_token_id == 23


def test_concrete_model_roots_inherit_directly_from_nn_module():
    assert Qwen3ForCausalLM.__bases__ == (nn.Module,)
    assert BagelForUnifiedGeneration.__bases__ == (nn.Module,)
    assert NEOChatModel.__bases__ == (nn.Module,)


def test_qwen_projects_loader_data_into_a_stable_declaration():
    config = _qwen_config()
    model = Qwen3ForCausalLM(config, layer_spec=_layer_spec())
    config["num_hidden_layers"] = 7
    config["max_position_embeddings"] = 4096

    assert model.spec.architecture == "Qwen3ForCausalLM"
    assert model.spec.cache.num_layers == 1
    assert model.spec.routes[0].shape.max_tokens_per_row == 128
    assert model.spec.operation_types() == {
        OperationType.SEQUENCE_EXTEND,
        OperationType.SEQUENCE_DECODE,
        OperationType.SEQUENCE_VERIFY,
    }


def test_qwen_rejects_incomplete_or_untyped_configuration():
    with pytest.raises(ValueError, match="requires integer field"):
        Qwen3ForCausalLM({}, layer_spec=_layer_spec())
    with pytest.raises(TypeError, match="must be a mapping"):
        Qwen3ForCausalLM(object(), layer_spec=_layer_spec())  # type: ignore[arg-type]


def test_bagel_uses_frozen_configuration_and_declares_mixed_mot():
    config = _bagel_config()
    model = BagelForUnifiedGeneration(
        config,
        layer_spec=_layer_spec(),
        graph=_LoadedBagelGraph(config),  # type: ignore[arg-type]
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_latent_size = 4  # type: ignore[misc]
    routes = {route.name: route for route in model.spec.routes}
    assert set(routes) == {"mot", "vae", "vit"}
    assert routes["mot"].mixed_combinations == (
        (RouteRowKind.TOKEN, RouteRowKind.FLOW),
    )
    assert model.spec.flow is not None
    assert model.spec.flow.schedule_direction == "descending"
    assert model.spec.cache.num_layers == 1


def test_sensenova_resolves_mutable_checkpoint_config_at_construction():
    config = _sensenova_config()
    model = NEOChatModel(config, layer_spec=_layer_spec())
    config.downsample_ratio = 0.25
    config.max_image_seq_len = 2048

    routes = {route.name: route for route in model.spec.routes}
    assert set(routes) == {"mot", "vit"}
    assert routes["mot"].mixed_combinations == (
        (RouteRowKind.TOKEN, RouteRowKind.FLOW),
    )
    assert model.spec.flow is not None
    assert model.spec.flow.latent_downsample == 4
    assert model.spec.flow.max_latent_tokens == 16
    assert model.spec.flow.schedule_direction == "ascending"


def test_simulation_model_obeys_the_same_forward_and_spec_boundary():
    model = StubModel()

    assert type(model).__bases__ == (nn.Module,)
    assert model.spec.routes[0].mixed_combinations == (
        (RouteRowKind.TOKEN, RouteRowKind.FLOW),
    )
