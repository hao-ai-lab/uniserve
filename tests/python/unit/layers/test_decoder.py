"""Observable full-sequence decoder behavior without resident KV storage."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from uniserve_worker.backends.attention.torch_sdpa import TorchSDPAAttentionBackend
from uniserve_worker.execution.forward_batch import AttentionSelection
from uniserve_worker.nn.attention import bind_dense_attention_modules
from uniserve_worker.nn.decoder.qwen import Qwen3Config, Qwen3Model
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator

pytestmark = pytest.mark.unit


def _decoder(*, normalize_output: bool) -> Qwen3Model:
    config = Qwen3Config(
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
    model = Qwen3Model(
        config,
        layer_config=LayerConfig(Communicator(), None),
        normalize_output=normalize_output,
    )
    bind_dense_attention_modules(
        model, AttentionSelection("torch_sdpa", (TorchSDPAAttentionBackend(),))
    )
    generator = torch.Generator().manual_seed(193)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) / 8)
    return model


@pytest.mark.parametrize("normalize_output", [False, True])
def test_full_sequence_decoder_preserves_causal_prefix_and_batch_independence(normalize_output):
    model = _decoder(normalize_output=normalize_output)
    tokens = torch.tensor([[1, 3, 5, 7, 9, 11], [1, 3, 5, 2, 4, 6]])
    positions = torch.arange(tokens.shape[1]).expand_as(tokens)
    with torch.inference_mode():
        together = model(tokens, positions)
        first = model(tokens[:1], positions[:1])
        prefix = model(tokens[:1, :3], positions[:1, :3])

    torch.testing.assert_close(together[0], first[0])
    torch.testing.assert_close(together[0, :3], together[1, :3])
    torch.testing.assert_close(together[0, :3], prefix[0])
    assert not torch.equal(together[0, 3:], together[1, 3:])


@pytest.mark.parametrize("normalize_output", [False, True])
def test_full_sequence_decoder_returns_requested_residual_stream(normalize_output):
    model = _decoder(normalize_output=normalize_output)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.zero_()
    inputs = torch.arange(1, 49, dtype=torch.float32).reshape(1, 3, 16) / 16
    with torch.inference_mode():
        actual = model(torch.zeros((1, 3), dtype=torch.long), torch.arange(3), input_embeds=inputs)
    expected = F.rms_norm(inputs, (16,), eps=1e-6) if normalize_output else inputs
    torch.testing.assert_close(actual, expected)
