"""Observable execution behavior exposed by concrete model roots."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from uniserve_worker.execution.batch import OpCode
from uniserve_worker.execution.forward_batch import (
    AttentionMode,
    ForwardBatch,
    ModelPhase,
    TokenSelection,
)
from uniserve_worker.models.bagel import BagelConfig, BagelForConditionalGeneration, LLMConfig
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.sensenova.config import NeoChatConfig
from uniserve_worker.models.sensenova.model import NEOChatModel
from uniserve_worker.nn.diffusion.schedule import ScheduleDirection
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator

pytestmark = pytest.mark.unit


def _layer_config() -> LayerConfig:
    return LayerConfig(Communicator(), None)


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


def _text_batch(
    query_lens: tuple[int, ...],
    *,
    forward_mode: AttentionMode,
) -> ForwardBatch:
    rows = len(query_lens)
    total = sum(query_lens)
    cumulative = torch.tensor(
        [0, *[sum(query_lens[: index + 1]) for index in range(rows)]],
        dtype=torch.int32,
    )
    return ForwardBatch(
        phase=ModelPhase.TEXT,
        row_count=rows,
        forward_mode=forward_mode,
        req_pool_indices=torch.arange(1, rows + 1),
        seq_lens=torch.zeros(rows, dtype=torch.int32),
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        out_cache_loc=torch.zeros(total, dtype=torch.int64),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        kv_lens=torch.tensor(query_lens, dtype=torch.int32),
        cu_seqlens_q=(cumulative if forward_mode is AttentionMode.PAGED_VARLEN else None),
        cu_seqlens_k=(cumulative if forward_mode is AttentionMode.PAGED_VARLEN else None),
        output_indices=(
            torch.tensor(
                [sum(query_lens[: index + 1]) - 1 for index in range(rows)],
                dtype=torch.int64,
            )
            if forward_mode is AttentionMode.PAGED_VARLEN
            else None
        ),
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(query_lens),
        seq_lens_cpu=(0,) * rows,
        query_lens_cpu=query_lens,
        kv_lens_cpu=query_lens,
        token_row_indices=tuple(range(rows)),
        input_ids=torch.zeros(total, dtype=torch.long),
        positions=torch.arange(total, dtype=torch.long),
        token_selections=(TokenSelection.LAST_LOGITS,) * rows,
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
    model = Qwen3ForCausalLM(config, layer_config=_layer_config())
    config["num_hidden_layers"] = 7
    config["max_position_embeddings"] = 4096

    assert model.architecture == "Qwen3ForCausalLM"
    assert model.cache_geometry.num_layers == 1
    assert model.text_max_tokens == 128
    assert model.supported_work == {
        OpCode.AR_EXTEND,
        OpCode.AR_DECODE,
        OpCode.AR_VERIFY,
    }


def test_qwen_decode_projection_preserves_row_alignment():
    model = Qwen3ForCausalLM(_qwen_config(), layer_config=_layer_config())
    weight = _projection_weight(model.lm_head)
    hidden = torch.arange(32, dtype=torch.float32).view(4, 8)
    batch = _text_batch((1, 1, 1, 1), forward_mode=AttentionMode.PAGED_DECODE)

    output = model.project(hidden, batch).materialize()

    assert len(output.values) == 4
    assert torch.equal(torch.cat(output.values), hidden @ weight.T)


def test_sensenova_decode_projection_preserves_row_alignment():
    model = NEOChatModel(_sensenova_config(), layer_config=_layer_config())
    weight = _projection_weight(model.language_model.lm_head)
    hidden = torch.arange(32, dtype=torch.float32).view(4, 8)
    batch = _text_batch((1, 1, 1, 1), forward_mode=AttentionMode.PAGED_DECODE)

    output = model.project(hidden, batch)

    assert len(output.values) == 4
    assert torch.equal(torch.cat(output.values), hidden @ weight.T)


def test_qwen_prefill_selects_the_last_logit_for_each_ragged_row():
    model = Qwen3ForCausalLM(_qwen_config(), layer_config=_layer_config())
    weight = _projection_weight(model.lm_head)
    hidden = torch.arange(56, dtype=torch.float32).view(7, 8)
    batch = _text_batch((2, 5), forward_mode=AttentionMode.PAGED_VARLEN)

    output = model.project(hidden, batch)

    assert torch.equal(torch.cat(output.values), hidden[[1, 6]] @ weight.T)


def test_qwen_rejects_incomplete_or_untyped_configuration():
    with pytest.raises(ValueError, match="requires integer field"):
        Qwen3ForCausalLM({}, layer_config=_layer_config())
    with pytest.raises(TypeError, match="must be a mapping"):
        Qwen3ForCausalLM(object(), layer_config=_layer_config())  # type: ignore[arg-type]


def test_bagel_exposes_configured_generation_behavior():
    config = _bagel_config()
    model = BagelForConditionalGeneration(
        config,
        layer_config=_layer_config(),
        graph=_LoadedBagelGraph(config),  # type: ignore[arg-type]
    )

    assert model.tensorized_mixed
    assert model.generation.schedule_direction is ScheduleDirection.DESCENDING
    assert model.cache_geometry.num_layers == 1
    assert model.text_max_tokens >= config.latent_token_capacity


def test_sensenova_freezes_runtime_behavior_at_construction():
    config = _sensenova_config()
    model = NEOChatModel(config, layer_config=_layer_config())
    config.downsample_ratio = 0.25
    config.max_image_seq_len = 2048

    assert model.tensorized_mixed
    assert model.generation.latent_downsample == 4
    assert model.generation.max_latent_tokens == 16
    assert model.generation.schedule_direction is ScheduleDirection.ASCENDING


@pytest.mark.parametrize(
    "total_kv_heads,intervals", ((8, ((2, 0), (2, 2), (2, 4), (2, 6))), (2, ((2, 0),) * 4))
)
def test_cache_geometry_preserves_tp_member_order(total_kv_heads, intervals):
    ranks = (7, 3, 11, 5)
    config = {
        **_qwen_config(),
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 8,
        "num_key_value_heads": total_kv_heads,
    }
    for rank, (heads, offset) in zip(ranks, intervals, strict=True):
        model = Qwen3ForCausalLM(
            config, layer_config=LayerConfig(Communicator(ranks=ranks, rank=rank), None)
        )
        geometry = model.cache_geometry
        assert geometry.total_kv_heads == total_kv_heads
        assert (geometry.num_kv_heads, geometry.kv_head_offset) == (heads, offset)
