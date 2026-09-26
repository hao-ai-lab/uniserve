"""Checkpoint normalization fixes architecture values.

The fixes apply before numerical construction.
"""

import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest
import torch

from uniserve import loading
from uniserve.model import EmbeddingReplacement, TextInput
from uniserve.nn.attention import AttentionBatch, SequenceLengths, VarlenInput
from uniserve_models import qwen3

pytestmark = pytest.mark.unit


def _config():
    return qwen3.Config(
        32,
        8,
        16,
        1,
        2,
        1,
        4,
        "silu",
        1e-6,
        10000.0,
        128,
        False,
        False,
        0,
        1,
        16,
    )


def test_loading_normalizes_immutable_architecture_fields(tmp_path):
    metadata = asdict(_config())
    path = tmp_path / "config.json"
    path.write_text(json.dumps(metadata))
    config = qwen3.read_config(tmp_path, loading.Config(), sources={})
    metadata["num_hidden_layers"] = 7
    metadata["max_position_embeddings"] = 4096
    path.write_text(json.dumps(metadata))
    assert config.num_hidden_layers == 1
    assert config.max_position_embeddings == 128
    with pytest.raises(FrozenInstanceError):
        config.num_hidden_layers = 7


def test_checkpoint_configuration_requires_numerical_dimensions(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="requires integer field"):
        qwen3.read_config(tmp_path, loading.Config(), sources={})


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
