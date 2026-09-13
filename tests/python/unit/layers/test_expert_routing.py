"""Independent expert execution preserves the declared packed token order."""

import pytest
import torch
from torch import nn

from uniserve_worker.modeling.tensors import ExpertRoute, RouteSpan
from uniserve_worker.nn.branch import branch
from uniserve_worker.nn.expert_routing import RoutedTensor
from uniserve_worker.runtime.branches import bind_branches

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "flow_device",
    [
        "cpu",
        pytest.param(
            "cuda:1",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    torch.cuda.device_count() < 2, reason="two CUDA devices are required"
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "routes",
    [
        (ExpertRoute.TEXT,),
        (ExpertRoute.FLOW,),
        (ExpertRoute.FLOW, ExpertRoute.TEXT, ExpertRoute.FLOW, ExpertRoute.TEXT),
    ],
)
def test_expert_modules_preserve_route_values_and_packed_order(routes, flow_device):
    text = nn.Linear(2, 2)
    flow = branch(nn.Linear(2, 2, device=flow_device), ExpertRoute.FLOW)
    bind_branches(flow, device="cpu", flow_device=flow_device)
    with torch.no_grad():
        text.weight.copy_(torch.eye(2) * 2)
        text.bias.fill_(1)
        flow.weight.copy_(torch.eye(2) * 3)
        flow.bias.fill_(-1)
    spans = tuple(RouteSpan(route, 2 * index, 2) for index, route in enumerate(routes))
    values = torch.arange(4 * len(routes), dtype=torch.float32).reshape(-1, 2)
    expected = torch.cat(
        [
            values[span.token_start : span.token_end] * (2 if span.route is ExpertRoute.TEXT else 3)
            + (1 if span.route is ExpertRoute.TEXT else -1)
            for span in spans
        ]
    )
    routed = RoutedTensor.from_packed(values, spans)
    actual = routed.apply(text=text, flow=flow).packed(spans)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 5])
def test_sequence_owned_routes_preserve_empty_experts_and_original_order(rows):
    from uniserve_worker.nn.mesh import Communicator
    from uniserve_worker.nn.parallel_sequence import SequencePartition

    values = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    spans = (RouteSpan(ExpertRoute.TEXT, 0, 1),)
    if rows > 1:
        spans += (RouteSpan(ExpertRoute.FLOW, 1, rows - 1),)
    routes = frozenset(span.route for span in spans)
    outputs = []
    for rank in range(4):
        group = Communicator(ranks=(3, 1, 2, 0), rank=(3, 1, 2, 0)[rank])
        partition = SequencePartition(rows, group)
        local = partition.local(values)
        local_spans = partition.routes(spans)
        text = nn.Linear(2, 2, bias=False)
        flow = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            text.weight.copy_(torch.eye(2) * 2)
            flow.weight.copy_(torch.eye(2) * 3)
        outputs.append(
            RoutedTensor.from_packed(local, local_spans, routes=routes)
            .apply(text=text, flow=flow)
            .packed(local_spans)
        )
    expected = values * torch.tensor([2] + [3] * (rows - 1)).unsqueeze(-1)
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("start,stop", [(0, 8), (1, 7), (2, 5), (4, 4), (8, 8)])
def test_routed_intervals_preserve_experts_and_packed_coordinates(start, stop):
    from uniserve_worker.nn.expert_routing import slice_route_spans

    spans = tuple(
        RouteSpan(route, index * 2, 2)
        for index, route in enumerate(
            (ExpertRoute.FLOW, ExpertRoute.TEXT, ExpertRoute.FLOW, ExpertRoute.TEXT)
        )
    )
    values = torch.arange(24).reshape(8, 3)
    routed = RoutedTensor.from_packed(values, spans)
    interval = slice(start, stop)
    actual = routed.narrow(interval, spans).packed(slice_route_spans(spans, interval))
    torch.testing.assert_close(actual, values[interval], rtol=0, atol=0)


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
def test_nested_branch_preserves_values_after_a_failed_call(flow_device):
    device = "cpu" if flow_device == "cpu" else "cuda:0"
    text = branch(nn.Linear(2, 2, device=device), ExpertRoute.TEXT)
    with torch.no_grad():
        text.weight.copy_(torch.eye(2, device=device) * 2)
        text.bias.fill_(-1)
    model = branch(nn.Sequential(text, nn.ReLU()), ExpertRoute.FLOW)
    bind_branches(model, device=device, flow_device=flow_device)
    with pytest.raises(RuntimeError):
        model(torch.ones((2, 3), device=device))
    values = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    torch.testing.assert_close(model(values), (values * 2 - 1).clamp_min(0), rtol=0, atol=0)
