"""Route arithmetic preserves token order and interval boundaries.

It also preserves empty routes.
"""

import pytest
import torch
from torch import nn

from uniserve.nn.routing import RoutedTensor, RouteSpan

pytestmark = pytest.mark.unit


def test_routed_values_compose_and_return_to_packed_order():
    spans = (
        RouteSpan("text", 0, 2),
        RouteSpan("image", 2, 1),
        RouteSpan("text", 3, 2),
    )
    value = torch.arange(15).view(5, 3).float()
    routed = RoutedTensor.from_packed(
        value, spans, routes=frozenset({"text", "image", "audio"})
    )
    torch.testing.assert_close(routed.packed(spans), value, rtol=0, atol=0)
    modules = {
        "text": nn.Identity(),
        "image": nn.ReLU(),
        "audio": nn.Identity(),
    }
    torch.testing.assert_close(
        routed.apply(modules).add(routed).packed(spans),
        value * 2,
        rtol=0,
        atol=0,
    )
    narrow = routed.narrow(slice(1, 4), spans)
    selected = (
        RouteSpan("text", 0, 1),
        RouteSpan("image", 1, 1),
        RouteSpan("text", 2, 1),
    )
    torch.testing.assert_close(
        narrow.packed(selected), value[1:4], rtol=0, atol=0
    )
    assert routed.narrow(slice(5, 5), spans).packed(()).shape == (0, 3)


def test_route_partitions_reject_missing_or_repeated_tokens():
    value = torch.ones(3, 4)
    with pytest.raises(ValueError, match="contiguous"):
        RoutedTensor.from_packed(
            value,
            (RouteSpan("text", 0, 2), RouteSpan("text", 1, 1)),
            routes=frozenset({"text"}),
        )
    with pytest.raises(ValueError, match="same tokens"):
        RoutedTensor.from_packed(
            value, (RouteSpan("text", 0, 2),), routes=frozenset({"text"})
        )
