"""Sparse numerical metadata can be prepared before execution resources bind."""

import pytest
import torch

from uniserve_worker.nn.sparse_attention import (
    SparseAttention,
    VideoSparseAttentionMetadata,
    build_video_sparse_metadata,
)

pytestmark = pytest.mark.unit


def test_sparse_metadata_preserves_prefix_visibility_and_requires_execution_binding():
    metadata = build_video_sparse_metadata(
        padded_rows=256,
        prefix_tiles=1,
        video_tiles=2,
        valid_sizes=torch.tensor([64, 64, 1, 0], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    # Prefix queries see all live tiles; video queries see the prefix and one
    # selected video tile; transport padding retains one masked block.
    assert metadata.pattern(4).row_counts == ((3, 2, 2, 1),)
    with pytest.raises(RuntimeError, match="execution-bound provider"):
        SparseAttention().prepare(metadata)


@pytest.mark.parametrize("valid_sizes", [torch.ones(3, dtype=torch.int32), torch.ones(4)])
def test_sparse_metadata_rejects_incompatible_validity_views(valid_sizes):
    with pytest.raises(ValueError, match="one int32 validity value per tile"):
        VideoSparseAttentionMetadata(256, 1, 2, 3, valid_sizes)
