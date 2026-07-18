"""Grouped routing and overlay laws (Stage 5, dormant, GPU where marked)."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.nn.grouped_routing import (
    GroupedLinear,
    OverlayBankError,
    WeightOverlayBank,
)

pytestmark = pytest.mark.unit

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _grouped(routes: int = 3, out: int = 6, inp: int = 4, slots: int = 4):
    torch.manual_seed(7)
    weights = torch.randn(routes, out, inp, device=_DEVICE)
    bank = WeightOverlayBank(
        slots=slots, rank=2, in_features=inp, out_features=out, device=_DEVICE
    )
    return GroupedLinear(weights, bank), weights, bank


def _oracle(x, weights, routes, bank=None, slots=None):
    """Per-token route-specific projection (the prohibited host-loop shape)."""

    rows = []
    for token in range(x.shape[0]):
        y = x[token] @ weights[routes[token]].T
        if bank is not None and slots is not None:
            slot = slots[token]
            y = y + (x[token] @ bank.down[slot].T) @ bank.up[slot].T
        rows.append(y)
    return torch.stack(rows)


def test_grouped_projection_matches_the_per_route_oracle():
    grouped, weights, bank = _grouped()
    x = torch.randn(10, 4, device=_DEVICE)
    routes = torch.tensor([0, 1, 2, 0, 1, 2, 2, 1, 0, 2], device=_DEVICE)
    out = grouped.forward(x, routes)
    torch.testing.assert_close(out, _oracle(x, weights, routes), rtol=1e-4, atol=1e-5)


def test_route_permutations_preserve_canonical_row_results():
    grouped, weights, _ = _grouped()
    x = torch.randn(6, 4, device=_DEVICE)
    routes = torch.tensor([2, 0, 1, 1, 0, 2], device=_DEVICE)
    baseline = grouped.forward(x, routes)
    order = torch.tensor([3, 1, 5, 0, 4, 2], device=_DEVICE)
    permuted = grouped.forward(x[order], routes[order])
    torch.testing.assert_close(permuted, baseline[order], rtol=1e-4, atol=1e-5)


def test_single_route_and_all_route_batches_use_the_same_operator():
    grouped, weights, _ = _grouped()
    x = torch.randn(5, 4, device=_DEVICE)
    single = grouped.forward(x, torch.full((5,), 1, device=_DEVICE, dtype=torch.long))
    torch.testing.assert_close(single, x @ weights[1].T, rtol=1e-4, atol=1e-5)
    mixed_routes = torch.tensor([0, 1, 2, 1, 0], device=_DEVICE)
    mixed = grouped.forward(x, mixed_routes)
    torch.testing.assert_close(
        mixed, _oracle(x, weights, mixed_routes), rtol=1e-4, atol=1e-5
    )


def test_neutral_slot_contributes_exactly_zero_and_overlays_mix_with_base():
    grouped, weights, bank = _grouped()
    x = torch.randn(6, 4, device=_DEVICE)
    routes = torch.tensor([0, 1, 2, 0, 1, 2], device=_DEVICE)
    base = grouped.forward(x, routes)
    neutral = grouped.forward(
        x, routes, torch.zeros(6, device=_DEVICE, dtype=torch.long)
    )
    torch.testing.assert_close(neutral, base)

    bank.load(2, torch.randn(2, 4, device=_DEVICE), torch.randn(6, 2, device=_DEVICE))
    slots = torch.tensor([0, 2, 0, 2, 0, 2], device=_DEVICE)
    mixed = grouped.forward(x, routes, slots)
    torch.testing.assert_close(
        mixed, _oracle(x, weights, routes, bank, slots), rtol=1e-4, atol=1e-5
    )


def test_overlay_lifecycle_rules():
    _, _, bank = _grouped()
    with pytest.raises(OverlayBankError, match="neutral"):
        bank.load(0, bank.down[1], bank.up[1])
    generation = bank.load(
        1, torch.randn(2, 4, device=_DEVICE), torch.randn(6, 2, device=_DEVICE)
    )
    assert generation == 1
    bank.pin(1)
    with pytest.raises(OverlayBankError, match="pinned"):
        bank.load(1, bank.down[1], bank.up[1])
    with pytest.raises(OverlayBankError, match="pinned"):
        bank.unload(1)
    bank.unpin(1)
    bank.unload(1)
    assert torch.count_nonzero(bank.down[1]) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_operator_is_graph_capturable():
    grouped, weights, bank = _grouped()
    x = torch.randn(8, 4, device="cuda")
    routes = torch.randint(0, 3, (8,), device="cuda")
    slots = torch.randint(0, 4, (8,), device="cuda")
    out = torch.zeros(8, 6, device="cuda")

    def step() -> None:
        out.copy_(grouped.forward(x, routes, slots))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    # Refresh inputs in place and replay: results follow the new values.
    x.copy_(torch.randn(8, 4, device="cuda"))
    routes.copy_(torch.randint(0, 3, (8,), device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out, _oracle(x, weights, routes, bank, slots), rtol=1e-4, atol=1e-5
    )
