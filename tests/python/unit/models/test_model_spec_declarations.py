"""Canonical declarations exposed by every concrete model root."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from uniserve_worker.forward import (
    AttentionSelection,
    EmptyKvView,
    EmptyLatentView,
    EmptyMeshView,
    EmptyOutputView,
    ForwardContext,
    GraphBinding,
    PagedDecodePlan,
    PagedVarlenPlan,
    TokenIds,
    TokenLogits,
    TokenRow,
    TokenSelection,
)
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


class _CountingProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.register_buffer("weight", torch.arange(8 * 32, dtype=torch.float32).view(8, 32))

    def forward(self, value: torch.Tensor, _mesh: object) -> torch.Tensor:
        self.calls += 1
        return value @ self.weight


class _AttentionCausalityRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[bool] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        context: ForwardContext,
        *,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        del key, value, context, scale
        self.values.append(bool(causal))
        return query


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


def _paged_decode_context(rows: int) -> ForwardContext:
    provider = SimpleNamespace(name="test")
    attention = PagedDecodePlan(
        backends=AttentionSelection("test", (provider,)),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        cache_seqlens=torch.zeros(rows, dtype=torch.int32),
        kv_seqlens=torch.ones(rows, dtype=torch.int32),
        query_lens=torch.ones(rows, dtype=torch.int32),
        cache_seqlens_cpu=(0,) * rows,
        kv_seqlens_cpu=(1,) * rows,
        query_lens_cpu=(1,) * rows,
        decode_page_ids=torch.zeros(rows, dtype=torch.long),
        decode_page_offsets=torch.zeros(rows, dtype=torch.long),
        max_context_len=1,
        causal=True,
        binding=GraphBinding(1),
    )
    return ForwardContext(
        kv=EmptyKvView(),
        latent=EmptyLatentView(),
        attention=attention,
        mesh=EmptyMeshView(),
        output=EmptyOutputView(),
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


def test_qwen_projects_one_decode_wave_as_one_logit_matrix():
    model = Qwen3ForCausalLM(_qwen_config(), layer_spec=_layer_spec())
    projection = _CountingProjection()
    model.lm_head = projection
    model.logits = nn.Identity()
    rows = tuple(
        TokenRow(
            row_id=index,
            inputs=TokenIds(torch.tensor([index])),
            positions=torch.tensor([index]),
            output_slot=index,
            selection=TokenSelection.LAST_LOGITS,
        )
        for index in range(4)
    )
    hidden = torch.arange(32, dtype=torch.float32).view(4, 8)

    output = model._outputs(hidden, rows, SimpleNamespace(mesh=object()))

    assert projection.calls == 1
    assert len(output.rows) == 4
    logits = torch.cat(
        tuple(row.value.value for row in output.rows if isinstance(row.value, TokenLogits))
    )
    assert torch.equal(logits, hidden @ projection.weight)


def test_sensenova_projects_one_decode_wave_as_one_logit_matrix():
    model = NEOChatModel(_sensenova_config(), layer_spec=_layer_spec())
    projection = _CountingProjection()
    model.language_model.lm_head = projection
    rows = tuple(
        TokenRow(
            row_id=index,
            inputs=TokenIds(torch.tensor([index])),
            positions=torch.tensor([index]),
            output_slot=index,
            selection=TokenSelection.LAST_LOGITS,
        )
        for index in range(4)
    )
    hidden = torch.arange(32, dtype=torch.float32).view(4, 8)

    output = model._mot_outputs(
        hidden,
        rows,
        tuple((index, index + 1) for index in range(4)),
        SimpleNamespace(mesh=object()),
    )

    assert projection.calls == 1
    assert len(output.rows) == 4
    logits = torch.cat(
        tuple(row.value.value for row in output.rows if isinstance(row.value, TokenLogits))
    )
    assert torch.equal(logits, hidden @ projection.weight)


def test_sensenova_paged_decode_executes_causal_attention():
    model = NEOChatModel(_sensenova_config(), layer_spec=_layer_spec())
    attention = model.language_model.model.layers[0].self_attn
    recorder = _AttentionCausalityRecorder()
    attention.attention = recorder

    model.language_model.model(
        torch.zeros((2, 8)),
        _paged_decode_context(2),
        positions=torch.tensor((7, 11)),
    )

    assert recorder.values == [True]


def test_qwen_prefill_uses_device_output_indices_for_ragged_rows():
    model = Qwen3ForCausalLM(_qwen_config(), layer_spec=_layer_spec())
    projection = _CountingProjection()
    model.lm_head = projection
    model.logits = nn.Identity()
    rows = (
        TokenRow(
            row_id=0,
            inputs=TokenIds(torch.tensor([1, 2])),
            positions=torch.tensor([0, 1]),
            output_slot=0,
            selection=TokenSelection.LAST_LOGITS,
        ),
        TokenRow(
            row_id=1,
            inputs=TokenIds(torch.tensor([3, 4, 5, 6, 7])),
            positions=torch.tensor([0, 1, 2, 3, 4]),
            output_slot=1,
            selection=TokenSelection.LAST_LOGITS,
        ),
    )
    provider = SimpleNamespace(name="test")
    attention = PagedVarlenPlan(
        backends=AttentionSelection("test", (provider,)),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        cache_seqlens=torch.zeros(2, dtype=torch.int32),
        query_lens=torch.tensor([2, 5], dtype=torch.int32),
        kv_seqlens=torch.tensor([2, 5], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2, 7], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 2, 7], dtype=torch.int32),
        output_indices=torch.tensor([1, 6], dtype=torch.int64),
        cache_seqlens_cpu=(0, 0),
        query_lens_cpu=(2, 5),
        kv_seqlens_cpu=(2, 5),
        max_seqlen_q=5,
        max_seqlen_k=5,
        max_context_len=8,
        causal=True,
        binding=GraphBinding(1),
    )
    hidden = torch.arange(56, dtype=torch.float32).view(7, 8)

    output = model._outputs(
        hidden,
        rows,
        SimpleNamespace(attention=attention, mesh=object()),
    )

    assert projection.calls == 1
    logits = torch.cat(
        tuple(row.value.value for row in output.rows if isinstance(row.value, TokenLogits))
    )
    assert torch.equal(logits, hidden[[1, 6]] @ projection.weight)


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
    assert routes["mot"].mixed_combinations == ((RouteRowKind.TOKEN, RouteRowKind.FLOW),)
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
    assert routes["mot"].mixed_combinations == ((RouteRowKind.TOKEN, RouteRowKind.FLOW),)
    assert model.spec.flow is not None
    assert model.spec.flow.latent_downsample == 4
    assert model.spec.flow.max_latent_tokens == 16
    assert model.spec.flow.schedule_direction == "ascending"


def test_simulation_model_obeys_the_same_forward_and_spec_boundary():
    model = StubModel()

    assert type(model).__bases__ == (nn.Module,)
    assert model.spec.routes[0].mixed_combinations == ((RouteRowKind.TOKEN, RouteRowKind.FLOW),)
