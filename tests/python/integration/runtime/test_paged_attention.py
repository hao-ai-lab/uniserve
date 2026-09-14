"""Paged attention planning and replay with changing logical rows and physical pages."""

from dataclasses import replace
from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from uniserve.attention.flashinfer import FlashInferAttentionBackend
from uniserve.attention.metadata import AttentionMetadata, AttentionMode
from uniserve.attention.tuning import FlashInferTuningConfig

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("causal", [False, True])
def test_paged_prefill_replay_tracks_lengths_and_page_remapping(causal):
    pytest.importorskip("flashinfer")
    torch.manual_seed(618)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    page_size, query_heads, kv_heads, width = 64, 4, 2, 128
    query = torch.randn((259, query_heads, width), dtype=dtype, device=device)
    keys = torch.randn((16, page_size, kv_heads, width), dtype=dtype, device=device)
    values = torch.randn_like(keys)
    context = AttentionMetadata(
        attention_mode=AttentionMode.PAGED_VARLEN,
        prefix_lens=torch.empty(3, dtype=torch.int32, device=device),
        out_cache_loc=torch.zeros(259, dtype=torch.int64, device=device),
        has_cache_writes=False,
        causal=causal,
        binding=0,
        block_table=torch.empty((3, 6), dtype=torch.int32, device=device),
        cu_seqlens_q=torch.empty(4, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.empty(4, dtype=torch.int32, device=device),
        query_lens=torch.empty(3, dtype=torch.int32, device=device),
        seq_lens=torch.empty(3, dtype=torch.int32, device=device),
    )
    backend = FlashInferAttentionBackend(
        tuning=FlashInferTuningConfig(workspace_size=64 * 1024 * 1024, prefill_backend="fa2")
    )
    # The final row represents the one-token padding row used by Graph buckets.
    layouts = (
        ((1, 257, 1), (65, 321, 1), ((2, 4, 0, 0, 0, 0), (1, 3, 5, 7, 9, 11), (0, 0, 0, 0, 0, 0))),
        (
            (129, 129, 1),
            (191, 198, 1),
            ((5, 0, 6, 0, 0, 0), (4, 2, 8, 10, 0, 0), (1, 0, 0, 0, 0, 0)),
        ),
        ((257, 1, 1), (321, 68, 1), ((3, 1, 6, 7, 8, 9), (0, 5, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0))),
    )

    def stage(query_lens, seq_lens, pages):
        nonlocal context
        prefix_lens = tuple(
            total - query for total, query in zip(seq_lens, query_lens, strict=True)
        )
        context = replace(
            context, query_lens_cpu=query_lens, seq_lens_cpu=seq_lens, prefix_lens_cpu=prefix_lens
        )
        for name, data in (
            ("prefix_lens", prefix_lens),
            ("query_lens", query_lens),
            ("seq_lens", seq_lens),
            ("cu_seqlens_q", tuple(accumulate(query_lens, initial=0))),
            ("cu_seqlens_k", tuple(accumulate(seq_lens, initial=0))),
            ("block_table", pages),
        ):
            target = getattr(context, name)
            target.copy_(torch.tensor(data, dtype=torch.int32, pin_memory=True), non_blocking=True)

    def execute():
        return backend.forward_varlen(
            query,
            keys,
            values,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=max(context.query_lens_cpu),
            max_seqlen_k=max(context.seq_lens_cpu),
            causal=causal,
            scale=width**-0.5,
            block_table=context.block_table,
            context=context,
        )

    def reference(query_lens, seq_lens, pages):
        outputs = []
        start = 0
        for query_count, kv_count, row_pages in zip(query_lens, seq_lens, pages, strict=True):
            row_query = query[start : start + query_count].transpose(0, 1).double()
            row_key = keys[list(row_pages)].flatten(0, 1)[:kv_count].transpose(0, 1).double()
            row_value = values[list(row_pages)].flatten(0, 1)[:kv_count].transpose(0, 1).double()
            mask = None
            if causal:
                mask = torch.arange(kv_count, device=device)[None, :] <= (
                    torch.arange(query_count, device=device)[:, None] + kv_count - query_count
                )
            outputs.append(
                F.scaled_dot_product_attention(
                    row_query, row_key, row_value, attn_mask=mask, enable_gqa=True
                ).transpose(0, 1)
            )
            start += query_count
        return torch.cat(outputs).to(dtype)

    def prepare():
        backend.prepare_paged_prefill_cuda_graph(
            context.binding,
            context,
            num_q_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=width,
            page_size=page_size,
            q_dtype=dtype,
            kv_dtype=dtype,
            causal=causal,
        )

    stage(*layouts[0])
    torch.testing.assert_close(execute(), reference(*layouts[0]), rtol=2e-2, atol=2e-2)
    context = replace(context, binding=618)
    backend.bind_paged_prefill_graph_wrapper(context.binding, context, device=device)
    graph = torch.cuda.CUDAGraph()
    try:
        prepare()
        execute()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            output = execute()
        original_query = query.clone()
        expected = []
        for layout in layouts:
            query.add_(0.125)
            expected.append(reference(*layout))
        query.copy_(original_query)
        torch.cuda.synchronize(device)
        # Keep the upload stream occupied while successive CPU plans are built.
        # Each replay must consume the plan submitted for that generation.
        torch.cuda._sleep(1_000_000_000)
        actual = []
        for layout in layouts:
            stage(*layout)
            query.add_(0.125)
            prepare()
            graph.replay()
            actual.append(output.clone())
        for result, reference_output in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference_output, rtol=2e-2, atol=2e-2)
    finally:
        torch.cuda.synchronize(device)
        graph.reset()
        backend.release_paged_prefill_graph_wrapper(context.binding)
