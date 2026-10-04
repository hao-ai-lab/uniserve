"""Public patch projection and learned positional lookup values."""

import pytest
import torch
from torch.nn import functional as F
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_position_ids,
)

from uniserve.nn.vision import (
    PatchEmbed,
    PositionEmbedding,
    TubeletEmbed,
    merged_grid_coordinates,
)

pytestmark = pytest.mark.unit


def test_patch_projection_sums_each_spatial_patch():
    patch = PatchEmbed(3, 2, patch_size=2, bias=False)
    with torch.no_grad():
        patch.projection.weight.fill_(1.0)
        pixels = torch.ones(1, 3, 4, 4)
        torch.testing.assert_close(patch(pixels), torch.full((1, 4, 2), 12.0))


def test_position_embedding_returns_requested_learned_rows():
    embedding = PositionEmbedding((2, 3), 4)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(24).reshape(6, 4))
        expected = torch.tensor(
            [
                [20.0, 21.0, 22.0, 23.0],
                [0.0, 1.0, 2.0, 3.0],
                [20.0, 21.0, 22.0, 23.0],
            ]
        )
        torch.testing.assert_close(embedding(torch.tensor([5, 0, 5])), expected)


def test_tubelet_projection_is_the_stride_equal_convolution():
    generator = torch.Generator().manual_seed(41)
    embed = TubeletEmbed(3, 8, patch_size=4, temporal_size=2)
    with torch.no_grad():
        embed.weight.copy_(torch.randn(8, 3, 2, 4, 4, generator=generator))
        embed.bias.copy_(torch.randn(8, generator=generator))

    # Five channel-major tubelets, each a [3, 2, 4, 4] pixel block.
    tubelets = torch.randn(5, 96, generator=generator)
    expected = F.conv3d(
        tubelets.view(5, 3, 2, 4, 4), embed.weight, embed.bias, stride=(2, 4, 4)
    ).view(5, 8)
    torch.testing.assert_close(embed(tubelets), expected)


@pytest.mark.parametrize("grid", [(4, 6), (2, 2), (8, 4), (6, 10)])
def test_merged_coordinates_follow_merge_block_patch_order(grid):
    height, width = grid
    rows, columns = merged_grid_coordinates(height, width, 2)
    expected = get_vision_position_ids(torch.tensor([[1, height, width]]), 2)
    assert torch.equal(torch.stack((rows, columns), dim=-1), expected)


@pytest.mark.parametrize("grid", [(4, 6), (2, 2), (8, 4), (12, 18)])
def test_interpolated_positions_blend_the_corner_aligned_table(grid):
    height, width = grid
    generator = torch.Generator().manual_seed(77)
    embedding = PositionEmbedding((5, 5), 6)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.randn(25, 6, generator=generator).bfloat16().float()
        )

    # Qwen3-VL's resampling: four corner-aligned neighbours per patch,
    # blended with FP32 weights, in merge-block patch order.
    indices, weights = get_vision_bilinear_indices_and_weights(
        torch.tensor([[1, height, width]]),
        num_grid_per_side=5,
        spatial_merge_size=2,
    )
    expected = (embedding.weight[indices] * weights[:, :, None]).sum(0)
    actual = embedding.interpolate(height, width, merge=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_merged_grids_require_whole_merge_blocks():
    with pytest.raises(ValueError, match="whole merge blocks"):
        merged_grid_coordinates(4, 5, 2)
