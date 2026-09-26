"""Observable full-sequence decoder behavior without resident KV storage."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from uniserve.nn.attention import AttentionBatch, SequenceLengths, VarlenInput
from uniserve_models.qwen3 import Config, Transformer

pytestmark = pytest.mark.unit


def _decoder(*, normalize_output: bool) -> Transformer:
    config = Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=128,
        attention_bias=False,
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
    )
    model = Transformer(config)
    if not normalize_output:
        model.norm = torch.nn.Identity()
    generator = torch.Generator().manual_seed(193)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) / 8
                )
    return model


def _forward(model, embeddings, positions):
    batch, length, width = embeddings.shape
    lengths = SequenceLengths.from_lengths(
        (length,) * batch, device=embeddings.device
    )
    attention = AttentionBatch.single(
        VarlenInput(lengths, lengths, (True,) * batch)
    )
    result = model(
        embeddings.reshape(-1, width),
        positions.expand(batch, -1).flatten(),
        attention,
    )
    return result.view(batch, length, width)


@pytest.mark.parametrize("normalize_output", [False, True])
def test_full_sequence_decoder_preserves_causal_prefix_and_batch_independence(
    normalize_output,
):
    model = _decoder(normalize_output=normalize_output)
    tokens = torch.tensor([[1, 3, 5, 7, 9, 11], [1, 3, 5, 2, 4, 6]])
    positions = torch.arange(tokens.shape[1]).expand_as(tokens)
    with torch.inference_mode():
        together = _forward(model, model.embed_input_ids(tokens), positions)
        first = _forward(
            model, model.embed_input_ids(tokens[:1]), positions[:1]
        )
        prefix = _forward(
            model, model.embed_input_ids(tokens[:1, :3]), positions[:1, :3]
        )

    torch.testing.assert_close(together[0], first[0])
    torch.testing.assert_close(together[0, :3], together[1, :3])
    torch.testing.assert_close(together[0, :3], prefix[0])
    assert not torch.equal(together[0, 3:], together[1, 3:])


@pytest.mark.parametrize("normalize_output", [False, True])
def test_full_sequence_decoder_returns_requested_residual_stream(
    normalize_output,
):
    model = _decoder(normalize_output=normalize_output)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.zero_()
    inputs = torch.arange(1, 49, dtype=torch.float32).reshape(1, 3, 16) / 16
    with torch.inference_mode():
        actual = _forward(model, inputs, torch.arange(3))
    expected = (
        F.rms_norm(inputs, (16,), eps=1e-6) if normalize_output else inputs
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("factor", [None, 2.0])
@pytest.mark.parametrize("dim", [16, 32])
def test_axis_rotary_preserves_full_frequency_range(dim, factor):
    from uniserve.nn.rope import LinearScaling, RotaryEmbedding

    theta = 10000.0
    rotary = RotaryEmbedding(
        dim,
        theta=theta,
        keep_freq_range=True,
        scaling=None if factor is None else LinearScaling(factor=factor),
    )
    positions = torch.tensor([0, 3, 17, 256])
    cosine, sine = rotary(positions, dtype=torch.float32, sequence_length=257)
    # Keeping alternate frequencies of a doubled-width rotary embedding gives
    # each spatial/temporal axis the full frequency range of its source head.
    frequencies = theta ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if factor is not None:
        frequencies = frequencies / factor
    phase = positions.float()[:, None] * frequencies
    torch.testing.assert_close(cosine, phase.cos())
    torch.testing.assert_close(sine, phase.sin())
