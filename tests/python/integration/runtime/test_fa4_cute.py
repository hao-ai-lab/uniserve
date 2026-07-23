from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from uniserve_kernel import mm_attn_varlen
from uniserve_worker.backends.attention.fa4_cute import Fa4CuteAttentionBackend


@torch.inference_mode()
@pytest.mark.parametrize(("batch", "sequence_length"), [(2, 4), (1, 1)])
def test_visible_end_attention_matches_dense_sdpa(batch: int, sequence_length: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    if not mm_attn_varlen.available():
        pytest.skip(f"FA4 CUTE provider unavailable: {mm_attn_varlen.import_error()}")

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
