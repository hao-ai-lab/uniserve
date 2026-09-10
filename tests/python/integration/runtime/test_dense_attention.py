"""Cache-free attention selects legal numerical providers for each request."""

import pytest
import torch
import torch.nn.functional as F

from uniserve_worker.backends.attention import resolve_attention_selection
from uniserve_worker.backends.attention.tuning import FlashInferTuningConfig
from uniserve_worker.nn.attention import RadixAttention

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize(
    "provider,rank,dtype,masked,layout",
    [
        ("auto", rank, dtype, masked, layout)
        for rank in (3, 4)
        for dtype in (torch.bfloat16, torch.float16, torch.float32)
        for masked in (False, True)
        for layout in ("contiguous", "interleaved")
    ]
    + [
        (provider, rank, torch.bfloat16, False, layout)
        for provider, rank in (("flashinfer", 3), ("fa4_cute", 4))
        for layout in ("contiguous", "interleaved")
    ]
    + [
        (provider, 4, torch.bfloat16, False, "strided_columns") for provider in ("auto", "fa4_cute")
    ],
)
@torch.inference_mode()
def test_dense_attention_preserves_heads_masks_and_graph_inputs(
    provider, rank, dtype, masked, layout
):
    device = torch.device("cuda", 0)
    torch.manual_seed(723)
    batch, rows, query_heads, kv_heads, width = 2 if rank == 4 else 1, 97, 8, 2, 128
    if layout in ("interleaved", "strided_columns"):
        projected = torch.randn(
            (
                batch,
                rows,
                query_heads + 2 * kv_heads,
                width * (2 if layout == "strided_columns" else 1),
            ),
            device=device,
            dtype=dtype,
        )
        if layout == "strided_columns":
            projected = projected[..., ::2]
        q, k, v = (
            value.transpose(1, 2)
            for value in projected.split((query_heads, kv_heads, kv_heads), dim=2)
        )
    else:
        q = torch.randn((batch, query_heads, rows, width), device=device, dtype=dtype)
        k = torch.randn((batch, kv_heads, rows, width), device=device, dtype=dtype)
        v = torch.randn_like(k)
    mask = torch.ones(rows, rows, device=device, dtype=torch.bool).tril() if masked else None
    selection = resolve_attention_selection(
        provider, tuning=FlashInferTuningConfig(), block_size=16
    )
    attention = RadixAttention(query_heads, kv_heads, width)
    attention.bind_dense(selection)

    def execute():
        def layout(value):
            return value.squeeze(0).transpose(0, 1) if rank == 3 else value

        output = attention(layout(q), layout(k), layout(v), None, causal=not masked, attn_mask=mask)
        return output.transpose(0, 1).unsqueeze(0) if rank == 3 else output

    def reference():
        return F.scaled_dot_product_attention(
            q.double(),
            k.double(),
            v.double(),
            attn_mask=mask,
            is_causal=not masked,
            enable_gqa=True,
        ).to(dtype)

    # The existing paged/segmented attention conformance uses 2e-2 for half
    # precision and 2e-5 for FP32 against an independently normalized reference.
    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    actual = execute()
    torch.testing.assert_close(actual, reference(), rtol=tolerance, atol=tolerance)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = execute()
    q.mul_(0.5)
    v.add_(0.25)
    if mask is not None:
        mask[:, ::3] = False
    graph.replay()
    torch.testing.assert_close(actual, reference(), rtol=tolerance, atol=tolerance)
    graph.reset()
