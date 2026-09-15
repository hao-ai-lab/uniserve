from __future__ import annotations

import pytest
import torch
from uniserve_kernel.flash_attn_jagged import (
    compute_prefix_bounds,
    compute_prefix_bounds_varlen,
)


def test_prefix_bounds_match_full_width_varlen() -> None:
    visible_end = torch.tensor(((5, 6, 1, 8), (2, 3, 4, 0)), dtype=torch.int32)

    actual = compute_prefix_bounds(visible_end, q_tile_size=2)
    expected = compute_prefix_bounds_varlen(
        visible_end,
        torch.tensor((4, 4), dtype=torch.int32),
        q_tile_size=2,
    )
    torch.testing.assert_close(actual, expected)


def test_varlen_prefix_bounds_ignore_padded_query_values() -> None:
    visible_end = torch.tensor(
        ((5, 6, 99, 99), (2, 3, 4, 99)),
        dtype=torch.int32,
    )

    actual = compute_prefix_bounds_varlen(
        visible_end,
        torch.tensor((2, 3), dtype=torch.int32),
        q_tile_size=2,
        num_q_tiles=2,
    )

    expected = torch.tensor(
        (((5, 6), (0, 0)), ((2, 3), (4, 4))),
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_varlen_prefix_bounds_capture_and_replay_live_lengths() -> None:
    device = torch.device("cuda")
    visible_end = torch.tensor(
        ((5, 6, 99, 99),), dtype=torch.int32, device=device
    )
    seqlens_q = torch.tensor((2,), dtype=torch.int32, device=device)
    compute_prefix_bounds_varlen(
        visible_end,
        seqlens_q,
        q_tile_size=2,
        num_q_tiles=2,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        output = compute_prefix_bounds_varlen(
            visible_end,
            seqlens_q,
            q_tile_size=2,
            num_q_tiles=2,
        )

    visible_end.copy_(
        torch.tensor(((8, 7, 6, 99),), dtype=torch.int32, device=device)
    )
    seqlens_q.fill_(3)
    graph.replay()

    torch.testing.assert_close(
        output.cpu(),
        torch.tensor((((7, 8), (6, 6)),), dtype=torch.int32),
    )
