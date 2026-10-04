"""H3 dense attention excludes padding from the numerical key domain."""

import pytest
import torch

from uniserve.distributed import Communicator
from uniserve.nn.attention import SequenceLengths, VisibleInput
from uniserve_models.minimax_h3.attention import Dense
from uniserve_models.minimax_h3.config import TransformerConfig
from uniserve_models.minimax_h3.inputs import SequenceInput

pytestmark = pytest.mark.unit


@torch.inference_mode()
def test_dense_attention_averages_only_visible_values():
    config = TransformerConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_refiner_layers=1,
        intermediate_size=128,
        text_dim=40,
        frequency_dim=16,
        time_hidden_dim=64,
        time_dim=32,
        rope_frequency_dim=4,
    )
    layer = Dense(config)
    for parameter in layer.parameters():
        parameter.zero_()
    # Zero Q/K make every visible key equiprobable. Identity V and output
    # projections then expose the mean of the visible input rows directly.
    layer.projection.projections["v"].weight[:64].copy_(torch.eye(64))
    layer.output.weight[:, :64].copy_(torch.eye(64))
    hidden = torch.arange(8 * 64, dtype=torch.float32).reshape(8, 64) / 64
    hidden[6:] = 10_000
    lengths = SequenceLengths.from_lengths((8,), device="cpu")
    inputs = SequenceInput(
        slice(0, 8),
        Communicator(),
        VisibleInput(
            lengths,
            lengths,
            torch.tensor([[6]], dtype=torch.int32),
            None,
            True,
            False,
        ),
        torch.arange(6),
        torch.empty(0, dtype=torch.int64),
    )
    cosine = torch.ones(8, 3 * config.rope_frequency_dim)
    actual = layer(
        hidden, cosine, torch.zeros_like(cosine), inputs, workspace={}
    )
    expected = hidden[:6].mean(0).expand_as(hidden)
    torch.testing.assert_close(actual, expected)
