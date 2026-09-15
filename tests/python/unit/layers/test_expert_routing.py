"""Independent expert execution preserves the declared packed token order."""

import pytest
import torch
from torch import nn

from uniserve.nn.routing import RoutedTensor, RouteSpan
from uniserve.runtime import ExecutionContext

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
                    torch.cuda.device_count() < 2,
                    reason="two CUDA devices are required",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "routes",
    [
        ("text",),
        ("flow",),
        ("flow", "text", "flow", "text"),
    ],
)
def test_expert_modules_preserve_route_values_and_packed_order(
    routes, flow_device
):
    device = "cpu" if flow_device == "cpu" else "cuda:0"
    text = nn.Linear(2, 2, device=device)
    flow = nn.Linear(2, 2, device=flow_device)
    modules = nn.ModuleDict({"text": text, "flow": flow})
    with torch.no_grad():
        text.weight.copy_(torch.eye(2) * 2)
        text.bias.fill_(1)
        flow.weight.copy_(torch.eye(2) * 3)
        flow.bias.fill_(-1)
    spans = tuple(
        RouteSpan(route, 2 * index, 2) for index, route in enumerate(routes)
    )
    values = torch.arange(
        4 * len(routes), dtype=torch.float32, device=device
    ).reshape(-1, 2)
    expected = torch.cat(
        [
            values[span.start : span.stop] * (2 if span.route == "text" else 3)
            + (1 if span.route == "text" else -1)
            for span in spans
        ]
    )
    routed = RoutedTensor.from_packed(
        values, spans, routes=frozenset({"text", "flow"})
    )
    with ExecutionContext(modules) as context, torch.inference_mode():
        context.prepare(None)
        actual = routed.apply(modules).packed(spans)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
