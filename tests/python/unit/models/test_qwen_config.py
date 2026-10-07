"""Checkpoint normalization fixes architecture values.

The fixes apply before numerical construction.
"""

import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from transformers import Qwen3Config, Qwen3MoeConfig

from uniserve import loading
from uniserve.model import EmbeddingReplacement, TextInput
from uniserve.nn.attention import AttentionBatch, SequenceLengths, VarlenInput
from uniserve_models import qwen3

pytestmark = pytest.mark.unit


def _config():
    return qwen3.Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=128,
        attention_bias=False,
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=(),
    )


_DENSE = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "vocab_size": 32,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "max_position_embeddings": 128,
}

_SPARSE = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "vocab_size": 32,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
}


def _read(tmp_path, metadata):
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    return qwen3.read_config(tmp_path, loading.Config(), sources={})


def test_loading_normalizes_immutable_architecture_fields(tmp_path):
    config = _read(tmp_path, _DENSE)
    (tmp_path / "config.json").write_text(
        json.dumps(_DENSE | {"num_hidden_layers": 7})
    )
    assert config.num_hidden_layers == 1
    assert config.max_position_embeddings == 128
    with pytest.raises(FrozenInstanceError):
        config.num_hidden_layers = 7


def test_checkpoint_configuration_requires_numerical_dimensions(tmp_path):
    metadata = dict(_DENSE)
    del metadata["hidden_size"]
    with pytest.raises(ValueError, match="requires integer field"):
        _read(tmp_path, metadata)


@pytest.mark.parametrize(
    "metadata,reference",
    [(_DENSE, Qwen3Config), (_SPARSE, Qwen3MoeConfig)],
    ids=["dense", "sparse"],
)
def test_absent_fields_take_each_family_s_reference_defaults(
    tmp_path, metadata, reference
):
    """A field a checkpoint omits means what Transformers makes of it."""
    sparse = metadata is _SPARSE
    omitted = {"head_dim", "max_position_embeddings", "rope_theta"}
    trimmed = {
        key: value for key, value in metadata.items() if key not in omitted
    }
    config = _read(tmp_path, trimmed)
    expected = reference(
        **{
            key: value
            for key, value in trimmed.items()
            if key not in {"architectures", "model_type"}
        }
    )

    assert config.head_dim == getattr(
        expected,
        "head_dim",
        expected.hidden_size // expected.num_attention_heads,
    )
    assert config.max_position_embeddings == expected.max_position_embeddings
    assert config.rope_theta == expected.rope_parameters["rope_theta"]
    assert config.rope_scaling is None
    assert config.num_key_value_heads == expected.num_key_value_heads
    if sparse:
        assert config.num_experts == expected.num_experts
        assert config.num_experts_per_tok == expected.num_experts_per_tok
        assert config.moe_intermediate_size == expected.moe_intermediate_size
        assert config.norm_topk_prob == expected.norm_topk_prob
        assert all(config.sparse(layer) for layer in range(2))
    else:
        assert config.num_experts == 0
        assert not config.sparse(0)


def test_null_dense_kv_heads_mean_one_per_query_head(tmp_path):
    config = _read(tmp_path, _DENSE | {"num_key_value_heads": None})
    assert (
        config.num_key_value_heads
        == Qwen3Config(
            num_attention_heads=2, num_key_value_heads=None
        ).num_key_value_heads
        == 2
    )


@pytest.mark.parametrize(
    "metadata,message",
    [
        (_DENSE | {"model_type": "qwen3_moe"}, "model_type"),
        (_DENSE | {"architectures": ["LlamaForCausalLM"]}, "architectures"),
        (_DENSE | {"num_experts": 4}, "expert fields"),
        (_DENSE | {"use_sliding_window": True}, "sliding-window"),
        (_SPARSE | {"use_sliding_window": True}, "sliding-window"),
        (
            _DENSE | {"layer_types": ["sliding_attention"]},
            "layer_types",
        ),
        (
            _DENSE | {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            "default and YaRN",
        ),
        (_DENSE | {"partial_rotary_factor": 0.5}, "full head"),
        (_SPARSE | {"hidden_act": "gelu"}, "experts do not implement"),
        (
            _SPARSE | {"num_experts": 4, "num_local_experts": 8},
            "conflicting expert-count",
        ),
    ],
    ids=[
        "family-mismatch",
        "unknown-architecture",
        "dense-with-experts",
        "dense-window",
        "sparse-window",
        "sliding-layer",
        "linear-rope",
        "partial-rope",
        "erf-experts",
        "expert-aliases",
    ],
)
def test_unserved_semantics_are_rejected(tmp_path, metadata, message):
    with pytest.raises(ValueError, match=message):
        _read(tmp_path, metadata)


@pytest.mark.parametrize(
    "values,message",
    [
        ({"num_key_value_heads": 3}, "divisible by KV heads"),
        ({"num_experts": 2, "num_experts_per_tok": 3}, "must not exceed"),
        ({"head_dim": 3}, "even"),
        ({"rms_norm_eps": float("nan")}, "finite and positive"),
        ({"num_hidden_layers": True}, "positive integer"),
    ],
)
def test_architecture_rejects_invalid_mathematics(values, message):
    with pytest.raises(ValueError, match=message):
        replace(_config(), **values)


@torch.inference_mode()
def test_replacement_embeddings_preserve_tokens_and_independent_head_width():
    generator = torch.Generator().manual_seed(814)
    model = qwen3.Model(
        replace(_config(), hidden_size=10, num_attention_heads=3)
    )
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator) / 8
            )
    lengths = SequenceLengths.from_lengths((3,), device="cpu")
    attention = AttentionBatch.single(VarlenInput(lengths, lengths, (True,)))
    ids = torch.tensor([1, 2, 3])
    positions = torch.arange(3)
    expected = model(TextInput(torch.tensor([1, 7, 3]), positions, attention))
    embeddings = model.embed_input_ids(torch.tensor([19, 7, 29]))
    actual = model(
        TextInput(
            ids,
            positions,
            attention,
            EmbeddingReplacement(
                embeddings, torch.tensor([False, True, False])
            ),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    logits = model.compute_logits(
        actual, token_indices=torch.tensor([2])
    ).gather()
    assert logits.shape == (1, 32)
    assert torch.isfinite(logits).all()
