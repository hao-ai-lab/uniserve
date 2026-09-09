"""Paged attention planning and replay with changing logical rows and physical pages."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from uniserve_worker.backends.attention.flashinfer import FlashInferAttentionBackend
from uniserve_worker.backends.attention.tuning import FlashInferTuningConfig

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("causal", [False, True])
def test_paged_prefill_replay_tracks_lengths_and_page_remapping(causal):
    pytest.importorskip("flashinfer")
    torch.manual_seed(618)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    page_size, query_heads, kv_heads, width = 64, 4, 2, 128
    query = torch.randn((5, query_heads, width), dtype=dtype, device=device)
    keys = torch.randn((6, page_size, kv_heads, width), dtype=dtype, device=device)
    values = torch.randn_like(keys)
    context = SimpleNamespace(
        binding=None,
        block_table=torch.empty((3, 2), dtype=torch.int32, device=device),
        cu_seqlens_q=torch.empty(4, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.empty(4, dtype=torch.int32, device=device),
        query_lens=torch.empty(3, dtype=torch.int32, device=device),
        kv_lens=torch.empty(3, dtype=torch.int32, device=device),
    )
    backend = FlashInferAttentionBackend(
        tuning=FlashInferTuningConfig(workspace_size=64 * 1024 * 1024, prefill_backend="fa2")
    )
    # The final row represents the one-token padding row used by Graph buckets.
    layouts = (
        ((1, 3, 1), (65, 7, 1), ((2, 4), (1, 3), (0, 0))),
        ((2, 2, 1), (63, 70, 1), ((5, 0), (4, 2), (1, 0))),
        ((1, 3, 1), (1, 68, 1), ((3, 0), (0, 5), (2, 0))),
    )

    def stage(query_lens, kv_lens, pages):
        context.query_lens_cpu = query_lens
        context.kv_lens_cpu = kv_lens
        for name, data in (
            ("query_lens", query_lens),
            ("kv_lens", kv_lens),
            ("cu_seqlens_q", tuple(accumulate(query_lens, initial=0))),
            ("cu_seqlens_k", tuple(accumulate(kv_lens, initial=0))),
            ("block_table", pages),
        ):
            target = getattr(context, name)
            target.copy_(torch.tensor(data, dtype=torch.int32, device=device))

    def execute():
        return backend.forward_varlen(
            query,
            keys,
            values,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=max(context.query_lens_cpu),
            max_seqlen_k=max(context.kv_lens_cpu),
            causal=causal,
            scale=width**-0.5,
            block_table=context.block_table,
            context=context,
        )

    def reference(query_lens, kv_lens, pages):
        outputs = []
        start = 0
        for query_count, kv_count, row_pages in zip(query_lens, kv_lens, pages, strict=True):
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
    context.binding = 618
    backend.bind_paged_prefill_graph_wrapper(context.binding, context, device=device)
    graph = torch.cuda.CUDAGraph()
    try:
        prepare()
        execute()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            output = execute()
        for layout in layouts:
            stage(*layout)
            query.add_(0.125)
            prepare()
            graph.replay()
            torch.testing.assert_close(output, reference(*layout), rtol=2e-2, atol=2e-2)
    finally:
        torch.cuda.synchronize(device)
        graph.reset()
        backend.release_paged_prefill_graph_wrapper(context.binding)
