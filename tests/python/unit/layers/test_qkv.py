"""The public rotary projection preserves head values and caller ownership."""

import pytest
import torch
import torch.nn.functional as F

from uniserve.attention.metadata import ExpertRoute
from uniserve.distributed.mesh import Communicator
from uniserve.nn.branch import branch
from uniserve.nn.layer import LayerConfig
from uniserve.nn.linear import QKVParallelLinear
from uniserve.nn.norm import RMSNorm
from uniserve.nn.qkv import RotaryQKV
from uniserve.runtime.branches import bind_branches
from uniserve.runtime.cuda_graph import CudaGraph, capture_pools

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
    projection = QKVParallelLinear(8, 4, 2, 1, layer_config=LayerConfig(Communicator(), None))
    module = branch(RotaryQKV(projection, RMSNorm(4, 1e-6), RMSNorm(4, 1e-6)), ExpertRoute.FLOW)
    weight = torch.arange(128, dtype=torch.float32).reshape(16, 8) / 128 - 0.5
    bias = torch.linspace(-0.3, 0.3, 16)
    scales = torch.tensor([0.5, 0.75, 1.0, 1.25])
    with torch.no_grad():
        projection.weight.copy_(weight)
        projection.bias.copy_(bias)
        module.query_norm.weight.copy_(scales)
        module.key_norm.weight.copy_(scales)
    module.to(flow_device)
    bind_branches(module, device=device, flow_device=flow_device)
    hidden = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) / 32
    phase = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) / 7
    cosine, sine = phase.cos(), phase.sin()

    def expected(values):
        query, key, value = F.linear(values, weight, bias).split((8, 4, 4), dim=-1)

        def rotate(value, heads):
            value = value.reshape(rows, heads, 4)
            value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * scales
            left, right = value.chunk(2, dim=-1)
            cos, sin = cosine[:, None], sine[:, None]
            return torch.cat((left * cos - right * sin, right * cos + left * sin), -1).bfloat16()

        return rotate(query, 2), rotate(key, 1), value.reshape(rows, 1, 4).bfloat16()

    inputs = hidden.to(device)
    cos, sin = (cosine.to(device),), (sine.to(device),)

    def verify(result, wanted):
        for actual, reference in zip(result, wanted, strict=True):
            torch.testing.assert_close(actual, reference.to(device))

    with torch.inference_mode():
        verify(module(inputs, cos=cos, sin=sin), expected(hidden))
        if device != "cpu":
            # Peer transfers, all rotary inputs, and tuple outputs participate
            # in the same replay; updating the source must update every head.
            calls = []
            for offset in (0.25, 0.5):
                current = inputs.clone()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    module(current, cos=cos, sin=sin)
                stream.synchronize()
                graph = CudaGraph(
                    device=torch.device(device),
                    stream=stream,
                    device_pools=capture_pools((torch.device(flow_device),)),
                )
                graph.capture(lambda current=current: module(current, cos=cos, sin=sin))
                output = graph.output
                current.add_(offset)
                calls.append((stream, graph, current, output, offset))
            for stream, graph, _, _, _ in calls:
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    graph.replay()
            for stream, _, _, output, offset in calls:
                torch.cuda.current_stream(device).wait_stream(stream)
                verify(output, expected(hidden + offset))
            # Reusing one caller's graph must preserve the other caller's
            # still-borrowed output, even though their neural weights are shared.
            stream, graph, current, _, _ = calls[1]
            current.add_(0.25)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                graph.replay()
            torch.cuda.current_stream(device).wait_stream(stream)
            verify(calls[0][3], expected(hidden + 0.25))
            verify(calls[1][3], expected(hidden + 0.75))
            for stream, graph, _, _, _ in calls:
                stream.synchronize()
                graph.close()
