"""Programmatic dependent launches keep stream-order results on SM100.

A chain of the kernels that launch as programmatic dependents (rotary
factors, residual-add RMS norm, fused Q/K norm + RoPE, prefix-block
attention, gated activation, RMS norm and sandwich norm), each reading its
predecessor's output, is captured into a CUDA graph. The replay must
reproduce, bit for bit, the same launches run one at a time with every
kernel completed before the next starts. Every intermediate holds NaN
before each replay, so premature reads can reach the compared values.
"""

from __future__ import annotations

import pytest
import torch
from uniserve_kernels.attention import prefix_block
from uniserve_kernels.norm import rms, sandwich

from uniserve_kernels import activation, rope

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not prefix_block.available(),
        reason="requires the CUTLASS DSL and a compute capability 10.x GPU",
    ),
]

# One block over a 16-token-page prefix, with the head layout of the
# model's sliding-window attention layers.
_PREFIX = 700
_PAGE_TOKENS = 16
_QUERY_HEADS, _KV_HEADS, _HEAD_DIM = 16, 8, 256
_ROTATED = 128
_EPS = 1e-6
_REPLAYS = 3


def _operands(tokens):
    """Inputs of the chain and NaN-initialized intermediates."""
    generator = torch.Generator(device="cuda").manual_seed(0)

    def normal(*shape, dtype=torch.bfloat16):
        values = torch.randn(shape, generator=generator, device="cuda")
        return values.to(dtype)

    def empty(*shape, dtype=torch.bfloat16):
        return torch.full(shape, float("nan"), dtype=dtype, device="cuda")

    projected = (_QUERY_HEADS + _KV_HEADS) * _HEAD_DIM
    pages = -(-_PREFIX // _PAGE_TOKENS)
    cache_shape = (pages + 1, _PAGE_TOKENS, _KV_HEADS, _HEAD_DIM)
    width = _QUERY_HEADS * _HEAD_DIM // 2

    inputs = {
        "positions": torch.arange(
            _PREFIX, _PREFIX + tokens, dtype=torch.int32, device="cuda"
        ),
        "frequencies": torch.logspace(
            0, -4, _ROTATED // 2, dtype=torch.float32, device="cuda"
        ),
        "hidden": normal(tokens, projected),
        "residual": normal(tokens, projected),
        "projection_weight": normal(projected),
        "q_weight": normal(_HEAD_DIM),
        "k_weight": normal(_HEAD_DIM),
        "value": normal(tokens, _KV_HEADS, _HEAD_DIM),
        "key_cache": normal(*cache_shape),
        "value_cache": normal(*cache_shape),
        "block_table": torch.arange(
            1, pages + 1, dtype=torch.int32, device="cuda"
        ).unsqueeze(0),
        "query_offsets": torch.tensor(
            (0, tokens), dtype=torch.int32, device="cuda"
        ),
        "prefix_lengths": torch.tensor(
            (_PREFIX,), dtype=torch.int32, device="cuda"
        ),
        "norm_weight": normal(width),
        "branch_weight": normal(width),
        "sandwich_weight": normal(width),
    }
    intermediates = {
        "cosine": empty(tokens, _ROTATED // 2, dtype=torch.float32),
        "sine": empty(tokens, _ROTATED // 2, dtype=torch.float32),
        "normed": empty(tokens, projected),
        "summed": empty(tokens, projected),
        "q": empty(tokens, _QUERY_HEADS, _HEAD_DIM),
        "k": empty(tokens, _KV_HEADS, _HEAD_DIM),
        "attended": empty(tokens, _QUERY_HEADS, _HEAD_DIM),
        "gated": empty(tokens, width),
        "gated_normed": empty(tokens, width),
        "stream": empty(tokens, width),
        "stream_normed": empty(tokens, width),
    }
    return inputs, intermediates


def _chain(inputs, values, *, serialized=False):
    """Launch the chain; ``serialized`` completes each kernel before the next.

    The normalized projections and rotary factors feed attention; later
    steps consume the preceding step's output.
    """

    def step():
        if serialized:
            torch.cuda.synchronize()

    tokens = inputs["positions"].numel()
    rope.rotary_factors(
        inputs["positions"],
        inputs["frequencies"],
        1.0,
        values["cosine"],
        values["sine"],
    )
    step()
    rms.add_rms_norm(
        inputs["hidden"],
        inputs["residual"],
        inputs["projection_weight"],
        _EPS,
        values["normed"],
        values["summed"],
    )
    step()
    # Q and K are strided head views of the normalized projection rows.
    heads = values["normed"].view(tokens, _QUERY_HEADS + _KV_HEADS, _HEAD_DIM)
    rope.qk_norm_rope(
        heads[:, :_QUERY_HEADS],
        heads[:, _QUERY_HEADS:],
        (inputs["q_weight"],),
        (inputs["k_weight"],),
        (values["cosine"],),
        (values["sine"],),
        _EPS,
        values["q"],
        values["k"],
        axis_dims=(_HEAD_DIM,),
    )
    step()
    prefix_block.prefix_block_attention(
        values["q"],
        values["k"],
        inputs["value"],
        inputs["key_cache"],
        inputs["value_cache"],
        inputs["block_table"],
        inputs["query_offsets"],
        inputs["prefix_lengths"],
        max_query_len=tokens,
        scale=_HEAD_DIM**-0.5,
        out=values["attended"],
    )
    step()
    # The attended rows read as packed [gate, value] halves.
    activation.act_and_mul(
        values["attended"].view(tokens, -1),
        values["gated"],
        activation="gelu_tanh",
    )
    step()
    rms.rms_norm(
        values["gated"],
        inputs["norm_weight"],
        _EPS,
        values["gated_normed"],
    )
    step()
    sandwich.sandwich(
        values["gated_normed"],
        ((values["gated"], None),),
        inputs["branch_weight"],
        None,
        ((inputs["sandwich_weight"],),),
        _EPS,
        values["stream"],
        (values["stream_normed"],),
    )
    step()


@pytest.mark.parametrize("tokens", (64, 1536))
@torch.inference_mode()
def test_graph_of_dependent_launches_matches_stream_order(tokens):
    inputs, values = _operands(tokens)
    initial = {name: value.clone() for name, value in values.items()}

    def reset():
        for name, value in values.items():
            value.copy_(initial[name])

    # Launches one at a time: the stream-order result, and the compilation
    # every specialization needs before capture.
    _chain(inputs, values, serialized=True)
    expected = {name: value.clone() for name, value in values.items()}
    for name, value in expected.items():
        assert bool(value.isfinite().all()), name

    reset()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _chain(inputs, values)

    for _ in range(_REPLAYS):
        reset()
        graph.replay()
        torch.cuda.synchronize()
        for name, value in expected.items():
            assert torch.equal(values[name], value), name
