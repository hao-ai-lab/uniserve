"""Public patch projection and learned positional lookup values."""

import pytest
import torch

from uniserve.nn.vision import PatchEmbed, PositionEmbedding

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
            [[20.0, 21.0, 22.0, 23.0], [0.0, 1.0, 2.0, 3.0], [20.0, 21.0, 22.0, 23.0]]
        )
        torch.testing.assert_close(embedding(torch.tensor([5, 0, 5])), expected)
