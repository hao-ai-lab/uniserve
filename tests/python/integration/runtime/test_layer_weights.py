"""Host-streamed layer weights evaluate exactly as resident ones.

The layers run as a chain of chunk generators, as the H3 transformer does:
each layer projects every upstream chunk, combines them once, and yields its
feed-forward rows chunk by chunk, so a layer's projections interleave with
the previous layer's feed-forward and two layers are live at once.
"""

import pytest
import torch
from torch import nn

from uniserve.nn import Linear, MergedColumnParallelLinear
from uniserve.runtime.layer_weights import LayerWeights, layer_stack

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

WIDTH, HIDDEN, LAYERS, CHUNKS, ROWS = 64, 192, 10, 3, 96


class _Layer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.projection = MergedColumnParallelLinear(
            WIDTH,
            {"a": HIDDEN, "b": HIDDEN},
            bias=True,
            device=device,
            dtype=torch.bfloat16,
        )
        self.up = Linear(WIDTH, HIDDEN, device=device, dtype=torch.bfloat16)
        self.down = Linear(
            HIDDEN, WIDTH, bias=False, device=device, dtype=torch.bfloat16
        )

    def chunks(self, upstream):
        projected = [(value, self.projection(value)) for value in upstream]
        # A cross-chunk combination, as attention needs every row.
        mixed = sum(out["a"].float().mean(0) for _, out in projected)
        for value, out in projected:
            gated = (out["b"] + mixed.to(out["b"].dtype)) * self.up(value)
            yield value + self.down(torch.tanh(gated))


def _stack(device):
    torch.manual_seed(3)
    layers = nn.ModuleList(_Layer(device) for _ in range(LAYERS))
    for parameter in layers.parameters():
        parameter.data.normal_(0, 0.2)
    return layers


def _evaluate(layers, inputs, *, stop=None):
    chunks = iter(inputs)
    for index, layer in enumerate(layers):
        if stop is not None and index == stop:
            break
        chunks = layer.chunks(chunks)
    return torch.cat(list(chunks))


@torch.inference_mode()
def test_streamed_layers_equal_resident_layers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    layers = _stack(device)
    inputs = [
        torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
        for _ in range(CHUNKS)
    ]
    expected = _evaluate(layers, inputs)
    partial = _evaluate(layers, inputs, stop=6)

    resident = torch.cuda.memory_allocated(device)
    weights = LayerWeights(layers)
    try:
        assert weights.slots == 5 and weights.ahead == 3
        # Projection matrices left the device; one buffer per slot remains.
        assert torch.cuda.memory_allocated(device) <= resident - (
            (LAYERS - weights.slots) * weights.layer_bytes
        )
        for _ in range(3):
            assert torch.equal(_evaluate(layers, inputs), expected)
        # A pass interrupted after six layers, then complete passes again.
        assert torch.equal(_evaluate(layers, inputs, stop=6), partial)
        assert torch.equal(_evaluate(layers, inputs), expected)
        assert weights.copied_bytes >= 4 * LAYERS * weights.layer_bytes
    finally:
        weights.close()


def test_streaming_refuses_layers_of_differing_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    layers = _stack(device)
    layers[3].down = Linear(
        HIDDEN, WIDTH, bias=True, device=device, dtype=torch.bfloat16
    )
    layers[3].up = Linear(
        WIDTH, 2 * HIDDEN, device=device, dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="common projection weight layout"):
        LayerWeights(layers)


def test_layer_stack_is_the_largest_container_of_one_class():
    model = nn.Module()
    model.small = nn.ModuleList(nn.Linear(4, 4) for _ in range(6))
    model.body = nn.Module()
    model.body.layers = nn.ModuleDict(
        {str(index): nn.Linear(16, 16) for index in range(4)}
    )
    model.mixed = nn.ModuleList(
        (nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.Linear(64, 64))
    )
    assert layer_stack(model) == tuple(model.body.layers.values())
    with pytest.raises(ValueError, match="no stack of four layers"):
        layer_stack(model.mixed)
