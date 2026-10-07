from __future__ import annotations

import pytest
import torch
from uniserve_kernels.heads import (
    can_run_triton_merge_head_shards,
    can_run_triton_pack_head_shards,
    triton_merge_head_shards,
    triton_pack_head_shards,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    ),
]


@pytest.mark.parametrize("dim", [128, 80])
def test_packing_places_every_members_heads_destination_major(dim):
    """Each member's part holds its query heads, then its K and V heads.

    Q/K/V are strided views of one fused projection; the 2 KV heads are
    replicated across the 4 members, and 70 tokens leave a partial block.
    """
    members, tokens, query_heads, kv_heads = 4, 70, 8, 2
    projection = torch.randn(
        tokens, query_heads + 2 * kv_heads, dim, device="cuda"
    ).bfloat16()
    q, k, v = projection.split((query_heads, kv_heads, kv_heads), dim=1)
    slots = (query_heads // members, 1, 1)
    replicas = (1, members // kv_heads, members // kv_heads)
    out = torch.empty(
        members, tokens, sum(slots), dim, device="cuda", dtype=q.dtype
    )
    assert can_run_triton_pack_head_shards((q, k, v), slots, replicas, out)

    triton_pack_head_shards((q, k, v), slots, replicas, out)

    for member in range(members):
        expected = torch.cat(
            (
                q[:, member * 2 : member * 2 + 2],
                k[:, member // 2 : member // 2 + 1],
                v[:, member // 2 : member // 2 + 1],
            ),
            dim=1,
        )
        assert torch.equal(out[member], expected)


@pytest.mark.parametrize("dim", [128, 80])
def test_merging_restores_token_major_heads(dim):
    """Received head ``j`` of member ``m`` becomes output head ``m * h + j``."""
    members, tokens, heads = 4, 70, 3
    received = torch.randn(members, tokens, heads, dim, device="cuda").half()
    # The destination has strided rows, as a slice of a wider buffer.
    wider = torch.zeros(tokens, members * heads + 2, dim, device="cuda").half()
    out = wider[:, : members * heads]
    assert can_run_triton_merge_head_shards(received, out)

    triton_merge_head_shards(received, out)

    expected = received.transpose(0, 1).reshape(tokens, members * heads, dim)
    assert torch.equal(out, expected)
    assert torch.equal(
        wider[:, members * heads :], torch.zeros_like(wider[:, :2])
    )
