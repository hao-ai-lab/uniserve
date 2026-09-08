"""Independent expert execution preserves the declared packed token order."""

import pytest
import torch
from torch import nn

from uniserve_worker.execution.forward_batch import ExpertRoute, RouteSpan
from uniserve_worker.nn.expert_routing import RoutedTensor

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
    flow = nn.Linear(2, 2, device=flow_device)
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
    actual = routed.apply(text=text, flow=flow, generation_device=torch.device(flow_device)).packed(
        spans
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
