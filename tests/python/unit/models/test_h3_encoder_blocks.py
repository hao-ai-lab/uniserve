"""H3 video encoder blocks round as their eager composition.

The conditioning latents feed a denoiser held to byte-identical output, so
a block's fused normalization, bias and residual passes must reproduce the
eager group norm, SiLU, padded biased convolutions and residual sum bit for
bit.
"""

import pytest
import torch
from torch.nn import functional as F

from uniserve_models.minimax_h3 import video_vae

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    ),
]


# 128 input channels keep the identity shortcut; 64 take a 1x1 projection.
@pytest.mark.parametrize("in_channels", [128, 64])
def test_causal_block_equals_its_eager_composition(in_channels):
    torch.manual_seed(5)
    block = video_vae.CausalBlock(in_channels, 128, video_vae.Config()).cuda()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(0.0, 0.05)
    values = torch.randn(1, in_channels, 5, 12, 12, device="cuda")

    with torch.inference_mode():
        actual = block(values)

        hidden = values
        for norm, convolution in zip(
            block.norms, block.convolutions, strict=True
        ):
            hidden = convolution(F.silu(norm(hidden)))
        shortcut = values if block.shortcut is None else block.shortcut(values)
        expected = shortcut + hidden

    assert torch.equal(actual, expected)
