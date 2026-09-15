"""VSA numerical tile domains can be inspected without execution resources."""

import pytest
import torch

from uniserve.nn.attention import vsa

pytestmark = pytest.mark.unit


def _input(valid_sizes):
    return vsa.Input(
        256,
        1,
        2,
        3,
        valid_sizes,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor(1, dtype=torch.int32),
    )


def test_sparse_pattern_preserves_prefix_visibility_and_context_offsets():
    inputs = _input(torch.tensor([64, 64, 1, 0], dtype=torch.int32))
    # Prefix queries see all keys; video queries see the dense prefix and one
    # selected video tile. Transport padding retains one masked block.
    output = torch.empty((2, 4), dtype=torch.int32)
    pattern = inputs.pattern(4, selected_tiles=1)
    pattern.counts(num_heads=2, query_tiles=4, index_width=3, out=output)
    torch.testing.assert_close(
        output, torch.tensor([[3, 2, 2, 1], [3, 2, 2, 1]], dtype=torch.int32)
    )
    local = inputs.pattern(2, selected_tiles=1, query_tile_offset=2)
    local.counts(num_heads=2, query_tiles=2, index_width=3, out=output[:, :2])
    torch.testing.assert_close(
        output[:, :2], torch.tensor([[2, 1], [2, 1]], dtype=torch.int32)
    )
    with pytest.raises(ValueError, match="dimensions"):
        pattern.counts(num_heads=2, query_tiles=4, index_width=2, out=output)


@pytest.mark.parametrize(
    "valid_sizes", [torch.ones(3, dtype=torch.int32), torch.ones(4)]
)
def test_sparse_metadata_rejects_incompatible_validity_views(valid_sizes):
    with pytest.raises(ValueError, match="index domains"):
        _input(valid_sizes)
