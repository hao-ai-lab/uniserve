"""The public rotary projection preserves head values and caller ownership."""

import pytest
import torch
import torch.nn.functional as F

from contextlib import ExitStack
from torch import nn

from uniserve.nn import QKVParallelLinear, RMSNorm
from uniserve.nn.attention import RotaryQKVProjection
from uniserve.runtime import CUDAGraph, ExecutionContext


class Projection(nn.Module):
    def __init__(self, device, target):
        super().__init__()
        self.input_norm = nn.LayerNorm(8, device=device)
        self.qkv = RotaryQKVProjection(
            QKVParallelLinear(8, 2, 1, 4, device=target),
            RMSNorm(4, 1e-6, device=target),
            RMSNorm(4, 1e-6, device=target),
        )

    def forward(self, hidden, cos, sin):
        return self.qkv(self.input_norm(hidden), cos, sin)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rows", [0, 1, 5])
@pytest.mark.parametrize(
    "flow_device",
    [
        "cpu",
        pytest.param(
            "cuda:1",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
            ],
        ),
    ],
)
def test_rotary_projection_preserves_qkv_values_and_replay(rows, flow_device):
    device = "cpu" if flow_device == "cpu" else "cuda:0"
    module = Projection(device, flow_device)
    projection = module.qkv.projection
    weight = torch.arange(128, dtype=torch.float32).reshape(16, 8) / 128 - 0.5
    bias = torch.linspace(-0.3, 0.3, 16)
    scales = torch.tensor([0.5, 0.75, 1.0, 1.25])
    with torch.no_grad():
        for (name, branch), branch_weight, branch_bias in zip(
            projection.projections.items(),
            weight.split((8, 4, 4)),
            bias.split((8, 4, 4)),
            strict=True,
        ):
            branch.weight.copy_(branch_weight)
            branch.bias.copy_(branch_bias)
        module.qkv.query_norm.weight.copy_(scales)
        module.qkv.key_norm.weight.copy_(scales)
    hidden = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) / 32
    phase = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) / 7
    cosine, sine = phase.cos(), phase.sin()

    def expected(values):
        query, key, value = F.linear(F.layer_norm(values, (8,)), weight, bias).split(
            (8, 4, 4), dim=-1
        )

        def rotate(value, heads):
            value = value.reshape(rows, heads, 4)
            value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * scales
            left, right = value.chunk(2, dim=-1)
            cos, sin = cosine[:, None], sine[:, None]
            return torch.cat((left * cos - right * sin, right * cos + left * sin), -1)

        return rotate(query, 2), rotate(key, 1), value.reshape(rows, 1, 4)

    inputs = hidden.to(device)
    cos, sin = (cosine.to(device),), (sine.to(device),)

    def verify(result, wanted):
        for actual, reference in zip(result, wanted, strict=True):
            torch.testing.assert_close(actual, reference.to(device))

    with ExitStack() as scope, torch.inference_mode():
        stream = None if device == "cpu" else torch.cuda.Stream(device=device)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(device))
        context = scope.enter_context(ExecutionContext(module, stream=stream))
        context.prepare(None)
        verify(module(inputs, cos, sin), expected(hidden))
        if stream is not None:
            with torch.cuda.device(flow_device):
                pool = torch.cuda.MemPool()
            graph = scope.enter_context(
                CUDAGraph(context=context, pools={torch.device(flow_device): pool})
            )
            graph.capture(lambda: module(inputs, cos, sin))
            inputs.mul_(1.25)
            verify(graph.replay(), expected(hidden * 1.25))
            stream.synchronize()
