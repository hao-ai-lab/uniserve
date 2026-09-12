from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from uniserve_kernel import flash_attn_jagged
from uniserve_worker.backends.attention.fa4_cute import Fa4CuteAttentionBackend


@torch.inference_mode()
@pytest.mark.parametrize(("batch", "sequence_length"), [(2, 4), (1, 1)])
def test_visible_end_attention_matches_dense_sdpa(batch: int, sequence_length: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    if not flash_attn_jagged.available():
        pytest.skip(f"jagged FlashAttention kernel unavailable: {flash_attn_jagged.import_error()}")

    heads = 4
    kv_heads = 2
    head_dim = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(778)

    query = torch.randn(batch, sequence_length, heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch, sequence_length, kv_heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch, sequence_length, kv_heads, head_dim, device=device, dtype=dtype)
    visible_end = (
        torch.tensor([[1, 2, 4, 4], [4, 4, 4, 4]], dtype=torch.int32, device=device)
        if (batch, sequence_length) == (2, 4)
        else torch.ones((batch, sequence_length), dtype=torch.int32, device=device)
    )

    output = Fa4CuteAttentionBackend().forward_visible_end(
        query,
        key,
        value,
        visible_end=visible_end,
        scale=head_dim**-0.5,
        use_prefix_bounds=False,
    )
    torch.cuda.synchronize()

    references = []
    key_index = torch.arange(sequence_length, device=device)[None, :]
    for batch_index in range(batch):
        masked = key_index >= visible_end[batch_index, :, None]
        additive_mask = torch.zeros(sequence_length, sequence_length, device=device, dtype=dtype)
        additive_mask.masked_fill_(masked, float("-inf"))
        references.append(
            F.scaled_dot_product_attention(
                query[batch_index].transpose(0, 1).unsqueeze(0),
                key[batch_index]
                .repeat_interleave(heads // kv_heads, dim=1)
                .transpose(0, 1)
                .unsqueeze(0),
                value[batch_index]
                .repeat_interleave(heads // kv_heads, dim=1)
                .transpose(0, 1)
                .unsqueeze(0),
                attn_mask=additive_mask[None, None],
                scale=head_dim**-0.5,
            )
            .squeeze(0)
            .transpose(0, 1)
        )

    reference = torch.stack(references, dim=0)
    torch.testing.assert_close(output.float(), reference.float(), atol=0.01, rtol=0.01)


@pytest.mark.parametrize("variable_length", [False, True])
@torch.inference_mode()
def test_prefix_tiles_preserve_live_visibility_and_sequence_boundaries(variable_length):
    """Block skipping and mask evaluation preserve empty, full and partial tiles."""

    device = torch.device("cuda", 0)
    torch.manual_seed(793)
    query_lengths = [193, 301] if variable_length else [301, 301]
    key_lengths = [129, 385] if variable_length else [385, 385]
    maximum_query, maximum_key = 301, 385
    query = torch.randn(sum(query_lengths), 4, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(sum(key_lengths), 2, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    visible = torch.empty(2, maximum_query, device=device, dtype=torch.int32)
    cu_query = torch.zeros(3, device=device, dtype=torch.int32)
    cu_key = torch.zeros_like(cu_query)
    backend = Fa4CuteAttentionBackend()

    def update(iteration):
        cu_query.copy_(torch.tensor([0, query_lengths[0], sum(query_lengths)], device=device))
        cu_key.copy_(torch.tensor([0, key_lengths[0], sum(key_lengths)], device=device))
        for index, (q_len, k_len) in enumerate(zip(query_lengths, key_lengths, strict=True)):
            positions = torch.arange(maximum_query, device=device)
            # Inactive padding must never enlarge the tile's visible KV range.
            limits = (positions * 7 + iteration * 137).remainder(k_len + 1)
            if iteration == 0 and index == 0:
                limits.zero_()
            else:
                limits[positions < min(256, q_len)] = k_len
            visible[index].copy_(torch.where(positions < q_len, limits, 12345))

    def execute():
        return backend.forward_visible_end(
            query if variable_length else query.view(2, maximum_query, 4, 128),
            key if variable_length else key.view(2, maximum_key, 2, 128),
            value if variable_length else value.view(2, maximum_key, 2, 128),
            visible_end=visible,
            cu_seqlens_q=cu_query if variable_length else None,
            cu_seqlens_k=cu_key if variable_length else None,
            max_seqlen_q=maximum_query,
            max_seqlen_k=maximum_key,
            scale=128**-0.5,
            use_prefix_bounds=True,
        ).reshape_as(query)

    def reference():
        values = []
        q_begin = k_begin = 0
        for index, (q_len, k_len) in enumerate(zip(query_lengths, key_lengths, strict=True)):
            mask = torch.arange(k_len, device=device)[None] < visible[index, :q_len, None]
            values.append(
                F.scaled_dot_product_attention(
                    query[q_begin : q_begin + q_len].double().transpose(0, 1),
                    key[k_begin : k_begin + k_len].double().transpose(0, 1),
                    value[k_begin : k_begin + k_len].double().transpose(0, 1),
                    attn_mask=mask,
                    enable_gqa=True,
                )
                .transpose(0, 1)
                .to(query.dtype)
            )
            q_begin += q_len
            k_begin += k_len
        return torch.cat(values)

    update(0)
    actual = execute()
    torch.testing.assert_close(actual, reference(), atol=0.01, rtol=0.01)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = execute()
    if variable_length:
        query_lengths[:] = [194, 300]
        key_lengths[:] = [257, 257]
    update(1)
    value.mul_(0.75)
    graph.replay()
    torch.testing.assert_close(actual, reference(), atol=0.01, rtol=0.01)
    graph.reset()
