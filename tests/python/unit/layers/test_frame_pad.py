from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from uniserve.nn.functional import frame_pad

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
