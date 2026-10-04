"""Cache-free attention preserves precision, heads and declared visibility."""

import pytest
import torch
import torch.nn.functional as F

from uniserve.nn.attention import Attention, AttentionBatch, DenseInput
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize(
    "provider,rank,dtype,layout",
    [
        ("auto", rank, dtype, layout)
        for rank in (3, 4)
        for dtype in (torch.bfloat16, torch.float16)
        for layout in ("contiguous", "interleaved")
    ]
    + [
        (provider, rank, torch.bfloat16, layout)
        for provider, rank in (("flashinfer", 3), ("flash_attn_4", 4))
        for layout in ("contiguous", "interleaved")
    ]
    + [
        (provider, 4, torch.bfloat16, "strided_columns")
        for provider in ("auto", "flash_attn_4")
    ],
)
@torch.inference_mode()
def test_dense_attention_preserves_heads_and_graph_inputs(
    provider, rank, dtype, layout
):
    device = torch.device("cuda", 0)
    torch.manual_seed(723)
    batch, rows, query_heads, kv_heads, width = (
        2 if rank == 4 else 1,
        97,
        8,
        2,
        128,
    )
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
            for value in projected.split(
                (query_heads, kv_heads, kv_heads), dim=2
            )
        )
    else:
        q = torch.randn(
            (batch, query_heads, rows, width), device=device, dtype=dtype
        )
        k = torch.randn(
            (batch, kv_heads, rows, width), device=device, dtype=dtype
        )
        v = torch.randn_like(k)
    attention = Attention(query_heads, kv_heads, width)
    inputs = AttentionBatch.single(DenseInput(causal=True, mask=None))

    def execute():
        def layout(value):
            return value.squeeze(0).transpose(0, 1) if rank == 3 else value

        output = attention(layout(q), layout(k), layout(v), inputs)
        return output.transpose(0, 1).unsqueeze(0) if rank == 3 else output

    def reference():
        return F.scaled_dot_product_attention(
            q.double(),
            k.double(),
            v.double(),
            is_causal=True,
            enable_gqa=True,
        ).to(dtype)

    # The existing paged/segmented attention conformance uses 2e-2 for half
    # precision against an independently normalized reference.
    tolerance = 2e-2
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    stream.wait(torch.cuda.current_stream(device))
    with (
        stream,
        ExecutionContext(
            attention, attention=provider, stream=stream
        ) as context,
    ):
        context.prepare(None)
        actual = execute()
        torch.testing.assert_close(
            actual, reference(), rtol=tolerance, atol=tolerance
        )
        with CUDAGraph(context=context) as graph:
            graph.capture(execute)
            q.mul_(0.5)
            v.add_(0.25)
            actual = graph.replay()
            torch.testing.assert_close(
                actual, reference(), rtol=tolerance, atol=tolerance
            )


@pytest.mark.parametrize(
    "heads,kv_heads,width,query_rows,causal",
    [(1, 1, 512, 96, False), (8, 8, 256, 256, True), (8, 2, 128, 256, True)],
)
@torch.inference_mode()
def test_fp32_dense_attention_preserves_precision_and_visibility(
    heads, kv_heads, width, query_rows, causal
):
    """FP32 image and audio attention obey full and causal visibility."""
    device = torch.device("cuda", 0)
    torch.manual_seed(311)
    q = torch.randn((1, heads, query_rows, width), device=device)
    k = torch.randn((1, kv_heads, 256, width), device=device)
    v = torch.randn_like(k)
    attention = Attention(heads, kv_heads, width)
    inputs = AttentionBatch.single(DenseInput(causal=causal, mask=None))

    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    stream.wait(torch.cuda.current_stream(device))
    with (
        stream,
        ExecutionContext(attention, attention="auto", stream=stream) as context,
    ):
        context.prepare(None)
        actual = attention(q, k, v, inputs)

    expected = F.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), is_causal=causal, enable_gqa=True
    ).float()
    # CUDA FP32 products may round their inputs to TF32 (2^-11 unit
    # roundoff); the repository's half-precision tolerance, 2e-2 at BF16's
    # 2^-8, scales to 2.5e-3 at that unit.
    torch.testing.assert_close(actual, expected, rtol=2.5e-3, atol=2.5e-3)


@pytest.mark.parametrize(
    ("dtype", "mask", "message"),
    [
        (
            torch.float64,
            None,
            r"no native attention kernel on sm\d+ serves 8 query and 2 KV "
            r"heads of dimension 128, torch.float64 queries without a prefix "
            r"cache",
        ),
        (
            torch.bfloat16,
            "lower",
            r"no native attention kernel serves dense attention on sm\d+ "
            r"with causal with an explicit mask: 8 query and 2 KV heads of "
            r"dimension 128, torch.bfloat16 queries",
        ),
    ],
)
@torch.inference_mode()
def test_dense_calls_without_a_native_kernel_raise(dtype, mask, message):
    """A CUDA dense call no native kernel serves is rejected, never degraded.

    Automatic CUDA selection rejects FP64 and arbitrary dense masks. The
    error names the path, shape, dtype and mask semantics.
    """
    device = torch.device("cuda", 0)
    q = torch.randn((1, 8, 64, 128), device=device, dtype=dtype)
    k = torch.randn((1, 2, 64, 128), device=device, dtype=dtype)
    v = torch.randn_like(k)
    visible = (
        None
        if mask is None
        else torch.ones(64, 64, device=device, dtype=torch.bool).tril()
    )
    attention = Attention(8, 2, 128)
    inputs = AttentionBatch.single(DenseInput(causal=True, mask=visible))

    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    stream.wait(torch.cuda.current_stream(device))
    with (
        stream,
        ExecutionContext(attention, attention="auto", stream=stream) as context,
    ):
        context.prepare(None)
        with pytest.raises(ValueError, match=message):
            attention(q, k, v, inputs)
