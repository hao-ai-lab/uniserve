from __future__ import annotations

import pytest
import torch

from uniserve.nn.functional import bias_add

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    ),
]


def _operand(generator, layout):
    # Odd extents and 96 channels leave partial position and channel blocks.
    values = torch.randn(2, 96, 5, 19, 23, generator=generator, device="cuda")
    if layout == "channels_last":
        return values.contiguous(memory_format=torch.channels_last_3d)
    return values


@pytest.mark.parametrize(
    ("layout", "residual_layout", "residual_bias"),
    [
        ("contiguous", None, False),
        ("channels_last", None, False),
        ("channels_last", "contiguous", False),
        ("channels_last", "contiguous", True),
        ("contiguous", "channels_last", True),
    ],
)
def test_bias_add_rounds_as_separate_tensor_additions(
    layout, residual_layout, residual_bias
):
    # A convolution's bias, then a residual block's shortcut, each added as
    # its own tensor addition: (residual + residual_bias) + (values + bias).
    generator = torch.Generator(device="cuda").manual_seed(11)
    values = _operand(generator, layout)
    bias = torch.randn(96, generator=generator, device="cuda")
    residual = (
        None
        if residual_layout is None
        else _operand(generator, residual_layout)
    )
    shortcut_bias = (
        torch.randn(96, generator=generator, device="cuda")
        if residual_bias
        else None
    )

    with torch.inference_mode():
        actual = bias_add(
            values, bias, residual=residual, residual_bias=shortcut_bias
        )

    expected = values + bias.view(-1, 1, 1, 1)
    if residual is not None:
        if shortcut_bias is not None:
            residual = residual + shortcut_bias.view(-1, 1, 1, 1)
        expected = residual + expected
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()
