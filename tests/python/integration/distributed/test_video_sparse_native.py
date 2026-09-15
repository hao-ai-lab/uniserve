"""Sparse attention masks, native query partitions and head composition."""

import pytest
import torch
import torch.nn.functional as F

from uniserve.nn.attention import vsa
from uniserve.ops.video_sparse import compose_to_head_shards
from uniserve.runtime import ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("heads", [14, 28, 56, 7])
def test_sparse_attention_heads_preserve_block_mask_and_partial_tiles(heads):
    torch.manual_seed(123)
    rows, width, tile = 256, 128, 64
    # Interleaved projections exercise the actual Q/K/V ingress strides.
    projected = torch.randn(rows, heads, 4, width, device="cuda", dtype=torch.bfloat16)
    query, key, value, _ = projected.unbind(2)
    valid_sizes = torch.tensor([64, 17, 64, 0], device="cuda", dtype=torch.int32)
    indices = torch.zeros((heads, 4, 4), device="cuda", dtype=torch.int32)
    for head in range(heads):
        indices[head, :, 1] = 2 if head % 2 == 0 else 1
    counts = torch.full((heads, 4), 2, device="cuda", dtype=torch.int32)
    output = torch.empty((rows, heads, width), device="cuda", dtype=torch.bfloat16)
    module = vsa.BlockAttention(width**-0.5)
    batch = vsa.BlockInput(vsa.Pattern(((2, 2, 2, 2),), 0, 0), indices, counts, valid_sizes, 0)
    with ExecutionContext(module) as context:
        context.prepare(None)
        actual = module(query, key, value, batch, out=output)
    mask = torch.zeros((heads, rows, rows), device="cuda", dtype=torch.bool)
    for head in range(heads):
        mask[head, :, :64] = True
        if head % 2 == 0:
            mask[head, :, 128:192] = True
        else:
            mask[head, :, 64:81] = True
    reference = F.scaled_dot_product_attention(
        query.transpose(0, 1).double(),
        key.transpose(0, 1).double(),
        value.transpose(0, 1).double(),
        attn_mask=mask,
    ).to(torch.bfloat16)
    valid_queries = torch.arange(rows, device="cuda") % tile < valid_sizes.repeat_interleave(tile)
    # The established BF16 attention tolerance covers softmax/MMA rounding;
    # padded query outputs are not part of the model's valid-token contract.
    torch.testing.assert_close(
        actual.transpose(0, 1)[:, valid_queries], reference[:, valid_queries], rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("heads", [7, 14, 28, 56])
def test_sparse_query_partitions_preserve_complete_key_attention(heads):
    from uniserve_kernel.sparse_attention import block_sparse_attention as native_attention

    torch.manual_seed(972)
    rows, width = 512, 128
    query = torch.randn(1, heads, rows, width, device="cuda", dtype=torch.bfloat16)
    key, value = torch.randn_like(query), torch.randn_like(query)
    valid = torch.tensor([64, 17, 64, 64, 0, 64, 31, 64], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 1, 3, 6], device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 4).expand(heads, rows // 64, 4).contiguous()
    counts = (
        torch.arange(heads, device="cuda", dtype=torch.int32).view(-1, 1) * 3
        + torch.arange(rows // 64, device="cuda", dtype=torch.int32).view(1, -1)
    ).remainder(4) + 1
    full = native_attention(query, key, value, indices, counts, valid)
    strided_key = key.transpose(1, 2).contiguous().transpose(1, 2)
    strided_value = torch.empty(1, rows, heads, 2, width, device="cuda", dtype=torch.bfloat16)
    strided_value = strided_value[:, :, :, 1].transpose(1, 2)
    strided_value.copy_(value)
    strided = native_attention(query, strided_key, strided_value, indices, counts, valid)
    torch.testing.assert_close(strided, full, rtol=2e-2, atol=2e-2)
    partitions = [
        native_attention(
            query[:, :, start:end].contiguous(),
            key,
            value,
            indices[:, start // 64 : end // 64].contiguous(),
            counts[:, start // 64 : end // 64].contiguous(),
            valid,
        )
        for start, end in ((0, 64), (64, 256), (256, 512))
    ]
    torch.testing.assert_close(torch.cat(partitions, dim=2), full, rtol=2e-2, atol=2e-2)
    tail_query = query[:, :, :64].contiguous()
    tail_indices, tail_counts = indices[:, :1].contiguous(), counts[:, :1].contiguous()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = native_attention(
            tail_query, strided_key, strided_value, tail_indices, tail_counts, valid
        )
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(captured, full[:, :, :64], rtol=2e-2, atol=2e-2)
    graph.reset()
    selected = torch.arange(4, device="cuda").view(1, 1, -1) < counts.unsqueeze(-1)
    mask = torch.zeros(heads, rows // 64, rows // 64, device="cuda", dtype=torch.bool)
    mask.scatter_(2, indices.long(), selected)
    mask = mask.repeat_interleave(64, 1).repeat_interleave(64, 2)
    mask &= (torch.arange(rows, device="cuda") % 64 < valid.repeat_interleave(64)).view(1, 1, -1)
    reference = F.scaled_dot_product_attention(
        query.double(), key.double(), value.double(), attn_mask=mask.unsqueeze(0)
    )
    torch.testing.assert_close(full.double(), reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("members", [2, 4, 8])
def test_fused_composition_restores_each_rows_head_interval(members):
    torch.manual_seed(456)
    local_rows, local_heads, width = 64, 7, 128
    rows = local_rows * members
    attended = torch.randn((1, local_heads, rows, width), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((rows, local_heads, width), device="cuda", dtype=torch.bfloat16)
    compressed = torch.randn((local_heads, rows // 64, width), device="cuda")
    outputs = tuple(
        torch.zeros((local_rows, members * local_heads, width), device="cuda", dtype=torch.bfloat16)
        for _ in range(members)
    )
    # The independent formula permits fused or separate floating-point
    # arithmetic within the BF16 output error bound.
    expected = attended[0].transpose(0, 1).double() + gate.double() * compressed.transpose(
        0, 1
    ).double().repeat_interleave(64, dim=0)
    for source in range(members):
        compose_to_head_shards(attended, gate, compressed, outputs, source)
    for member, actual in enumerate(outputs):
        row_interval = expected[member * local_rows : (member + 1) * local_rows]
        torch.testing.assert_close(
            actual.double(), row_interval.repeat(1, members, 1), rtol=2e-2, atol=2e-2
        )
