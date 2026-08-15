"""Observable execution behavior exposed by concrete model roots."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from uniserve_worker.batch import WorkVariant
from uniserve_worker.execution.forward_batch import (
    AttentionSelection,
    EmptyKvView,
    EmptyMeshView,
    ForwardBatch,
    ModelPhase,
    PagedDecodePlan,
    PagedVarlenPlan,
    TokenSelection,
)
from uniserve_worker.models.bagel import BagelConfig, BagelForConditionalGeneration, LLMConfig
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.sensenova.config import NeoChatConfig
from uniserve_worker.models.sensenova.model import NEOChatModel
from uniserve_worker.nn.diffusion.schedule import ScheduleDirection
from uniserve_worker.nn.layer import LayerSpec
from uniserve_worker.nn.mesh import TensorParallelSpec

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


def _projection_weight(module: nn.Module) -> torch.Tensor:
    weight = torch.arange(module.weight.numel(), dtype=torch.float32).reshape_as(module.weight)
    with torch.no_grad():
        module.weight.copy_(weight)
    return weight[:32]


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


def _decode_attention(rows: int) -> PagedDecodePlan:
    provider = SimpleNamespace(name="test")
    return PagedDecodePlan(
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
        binding=1,
    )


def _text_batch(
    query_lens: tuple[int, ...],
    *,
    attention: PagedDecodePlan | PagedVarlenPlan,
) -> ForwardBatch:
    rows = len(query_lens)
    return ForwardBatch(
        phase=ModelPhase.TEXT,
        row_count=rows,
        request_pool_indices=torch.arange(1, rows + 1),
        token_row_indices=tuple(range(rows)),
        input_ids=torch.zeros(sum(query_lens), dtype=torch.long),
        positions=torch.arange(sum(query_lens), dtype=torch.long),
        query_lens=query_lens,
        token_selections=(TokenSelection.LAST_LOGITS,) * rows,
        kv=EmptyKvView(),
        attention=attention,
        mesh=EmptyMeshView(),
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


def test_qwen_constructs_runtime_behavior_from_checkpoint_configuration():
    config = _qwen_config()
    model = Qwen3ForCausalLM(config, layer_spec=_layer_spec())
    config["num_hidden_layers"] = 7
    config["max_position_embeddings"] = 4096

    assert model.architecture == "Qwen3ForCausalLM"
    assert model.cache_geometry.num_layers == 1
    assert model.text_max_tokens == 128
    assert model.supported_work == {
        WorkVariant.TOKEN_EXTEND,
        WorkVariant.TOKEN_DECODE,
        WorkVariant.TOKEN_VERIFY,
    }


def test_qwen_decode_projection_preserves_row_alignment():
    model = Qwen3ForCausalLM(_qwen_config(), layer_spec=_layer_spec())
    weight = _projection_weight(model.lm_head)
    hidden = torch.arange(32, dtype=torch.float32).view(4, 8)
    batch = _text_batch((1, 1, 1, 1), attention=_decode_attention(4))

    output = model.project(hidden, batch)

    assert len(output.values) == 4
    assert torch.equal(torch.cat(output.values), hidden @ weight.T)


def test_sensenova_decode_projection_preserves_row_alignment():
    model = NEOChatModel(_sensenova_config(), layer_spec=_layer_spec())
    weight = _projection_weight(model.language_model.lm_head)
    hidden = torch.arange(32, dtype=torch.float32).view(4, 8)
    batch = _text_batch((1, 1, 1, 1), attention=_decode_attention(4))

    output = model.project(hidden, batch)

    assert len(output.values) == 4
    assert torch.equal(torch.cat(output.values), hidden @ weight.T)


def test_qwen_prefill_selects_the_last_logit_for_each_ragged_row():
    model = Qwen3ForCausalLM(_qwen_config(), layer_spec=_layer_spec())
    weight = _projection_weight(model.lm_head)
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
        binding=1,
    )
    hidden = torch.arange(56, dtype=torch.float32).view(7, 8)
    batch = _text_batch((2, 5), attention=attention)

    output = model.project(hidden, batch)

    assert torch.equal(torch.cat(output.values), hidden[[1, 6]] @ weight.T)


def test_qwen_rejects_incomplete_or_untyped_configuration():
    with pytest.raises(ValueError, match="requires integer field"):
        Qwen3ForCausalLM({}, layer_spec=_layer_spec())
    with pytest.raises(TypeError, match="must be a mapping"):
        Qwen3ForCausalLM(object(), layer_spec=_layer_spec())  # type: ignore[arg-type]


def test_bagel_exposes_configured_generation_behavior():
    config = _bagel_config()
    model = BagelForConditionalGeneration(
        config,
        layer_spec=_layer_spec(),
        graph=_LoadedBagelGraph(config),  # type: ignore[arg-type]
    )

    assert model.tensorized_mixed
    assert model.generation.schedule_direction is ScheduleDirection.DESCENDING
    assert model.cache_geometry.num_layers == 1
    assert model.text_max_tokens >= config.latent_token_capacity


def test_sensenova_freezes_runtime_behavior_at_construction():
    config = _sensenova_config()
    model = NEOChatModel(config, layer_spec=_layer_spec())
    config.downsample_ratio = 0.25
    config.max_image_seq_len = 2048

    assert model.tensorized_mixed
    assert model.generation.latent_downsample == 4
    assert model.generation.max_latent_tokens == 16
    assert model.generation.schedule_direction is ScheduleDirection.ASCENDING
