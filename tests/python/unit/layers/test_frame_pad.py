from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from uniserve.nn.functional import frame_norm_pad, frame_pad

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    ),
]


def _padded(values, padding, mode):
    left, right, top, bottom, front = padding
    values = F.pad(values, (left, right, top, bottom, 0, 0), mode=mode)
    return F.pad(values, (0, 0, 0, 0, front, 0))


def _source(layout):
    generator = torch.Generator(device="cuda").manual_seed(31)
    # Odd extents leave partial pixel and channel blocks.
    values = torch.randn(2, 96, 5, 19, 23, generator=generator, device="cuda")
    if layout == "channels_last":
        return values.contiguous(memory_format=torch.channels_last_3d)
    if layout == "frame_major":
        # A [batch, frames, ...] tensor viewed channel-first, as per-frame
        # normalization returns it.
        return values.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4)
    return values


@pytest.mark.parametrize("mode", ["reflect", "replicate"])
@pytest.mark.parametrize("padding", [(1, 1, 1, 1, 2), (0, 1, 0, 1, 2)])
@pytest.mark.parametrize(
    "layout", ["contiguous", "channels_last", "frame_major"]
)
@pytest.mark.parametrize(
    "memory_format", [torch.contiguous_format, torch.channels_last_3d]
)
def test_padding_reads_each_source_pixel_exactly(
    mode, padding, layout, memory_format
):
    values = _source(layout)

    with torch.inference_mode():
        actual = frame_pad(
            values, padding, mode=mode, memory_format=memory_format
        )

    assert torch.equal(actual, _padded(values, padding, mode))
    assert actual.is_contiguous(memory_format=memory_format)


@pytest.mark.parametrize("padding", [(0, 23, 0, 0, 0), (0, 0, 19, 0, 1)])
def test_reflection_beyond_the_extent_is_refused(padding):
    # A reflected side as long as its extent would mirror past the opposite
    # edge; F.pad refuses it, and so must the kernel's single reflection.
    values = _source("contiguous")

    with torch.inference_mode(), pytest.raises(ValueError, match="shorter"):
        frame_pad(values, padding, mode="reflect")


def _normalized(values, padding, mode, memory_format, groups, weight, bias):
    batch, channels, frames, height, width = values.shape
    folded = values.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    activated = F.silu(F.group_norm(folded, groups, weight, bias, 1e-6))
    return frame_pad(
        activated.view(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        ),
        padding,
        mode=mode,
        memory_format=memory_format,
    )


@pytest.mark.parametrize("mode", ["reflect", "replicate"])
@pytest.mark.parametrize(
    "layout", ["contiguous", "channels_last", "frame_major"]
)
@pytest.mark.parametrize(
    "memory_format", [torch.contiguous_format, torch.channels_last_3d]
)
@pytest.mark.parametrize("affine", ["both", "weight", "bias", "none"])
def test_normalized_padding_rounds_as_its_composition(
    mode, layout, memory_format, affine
):
    # PyTorch's group norm takes a different expression for each affine
    # combination; the fused pass must round as each of them does.
    values = _source(layout)
    generator = torch.Generator(device="cuda").manual_seed(7)
    present = (affine in ("both", "weight"), affine in ("both", "bias"))
    weight, bias = (
        torch.randn(96, generator=generator, device="cuda") if given else None
        for given in present
    )
    padding = (1, 1, 1, 1, 2)

    with torch.inference_mode():
        actual = frame_norm_pad(
            values,
            padding,
            groups=32,
            weight=weight,
            bias=bias,
            eps=1e-6,
            mode=mode,
            memory_format=memory_format,
        )

    expected = _normalized(
        values, padding, mode, memory_format, 32, weight, bias
    )
    assert torch.equal(actual, expected)
    assert actual.is_contiguous(memory_format=memory_format)


@pytest.mark.parametrize("affine", [True, False])
def test_normalized_padding_of_single_pixel_frames(affine):
    # One-pixel frames take PyTorch's one-dimensional group norm.
    values = torch.randn(2, 64, 3, 1, 1, device="cuda")
    weight = torch.randn(64, device="cuda") if affine else None
    bias = torch.randn(64, device="cuda") if affine else None

    with torch.inference_mode():
        actual = frame_norm_pad(
            values,
            (0, 0, 0, 0, 2),
            groups=8,
            weight=weight,
            bias=bias,
            eps=1e-6,
            mode="replicate",
        )

    expected = _normalized(
        values,
        (0, 0, 0, 0, 2),
        "replicate",
        torch.contiguous_format,
        8,
        weight,
        bias,
    )
    assert torch.equal(actual, expected)
