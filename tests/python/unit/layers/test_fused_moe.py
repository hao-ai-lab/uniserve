"""Routed experts evaluate each token's selected experts with its weights."""

import pytest
import torch
from torch.nn import functional as F

from uniserve.model import TextSize
from uniserve.nn.moe import FusedMoE, TopK
from uniserve.runtime import ExecutionContext

pytestmark = pytest.mark.unit


def _module(activation, *, experts=4, hidden=8, intermediate=6, top_k=2):
    generator = torch.Generator().manual_seed(7)
    module = FusedMoE(
        experts, hidden, intermediate, top_k=top_k, activation=activation
    )
    with torch.no_grad():
        for parameter in (module.up_gate.weight, module.down.weight):
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return module


def _expected(module, hidden, ids, weights):
    """The routed equation with explicit per-token expert selection."""
    intermediate = module.intermediate_size
    result = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            up_gate = module.up_gate.weight[expert] @ hidden[token]
            up, gate = up_gate[:intermediate], up_gate[intermediate:]
            activated = (
                F.silu(gate)
                if module.activation == "silu"
                else F.gelu(gate, approximate="tanh")
            ) * up
            result[token] += weights[token, slot] * (
                module.down.weight[expert] @ activated
            )
    return result


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_each_token_combines_its_selected_experts(activation):
    module = _module(activation)
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(5, 8, generator=generator)
    ids = torch.tensor(
        [[0, 3], [1, 1], [2, 0], [3, 2], [1, 0]], dtype=torch.int32
    )
    weights = torch.rand(5, 2, generator=generator)
    torch.testing.assert_close(
        module(hidden, ids, weights), _expected(module, hidden, ids, weights)
    )

    # A prepared execution context evaluates the same equation.
    with ExecutionContext(module) as context:
        context.prepare(TextSize(5, 1))
        with context.activate():
            torch.testing.assert_close(
                module(hidden, ids, weights),
                _expected(module, hidden, ids, weights),
            )


def test_routing_must_match_the_declared_top_k_and_dtypes():
    module = _module("silu")
    hidden = torch.zeros(3, 8)
    weights = torch.ones(3, 2)
    with pytest.raises(ValueError, match="int32 ids and fp32 weights"):
        module(hidden, torch.zeros(3, 2, dtype=torch.int64), weights)
    with pytest.raises(ValueError, match=r"\[tokens, top_k\]"):
        module(hidden, torch.zeros(3, 3, dtype=torch.int32), torch.ones(3, 3))


def test_top_k_selects_and_optionally_renormalizes_softmax_weights():
    scores = torch.tensor([[0.0, 2.0, 1.0, -1.0]])
    probabilities = scores.softmax(-1)
    ids, weights = TopK(2, renormalize=False)(scores)
    assert ids.dtype == torch.int32 and ids.tolist() == [[1, 2]]
    torch.testing.assert_close(weights, probabilities[:, [1, 2]])
    _, normalized = TopK(2)(scores)
    torch.testing.assert_close(
        normalized, weights / weights.sum(-1, keepdim=True)
    )
