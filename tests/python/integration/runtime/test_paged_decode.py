"""Paged decode preserves attention results while replanning captured batches."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from uniserve_worker.backends.attention.flashinfer import FlashInferAttentionBackend
from uniserve_worker.backends.attention.tuning import FlashInferTuningConfig
from uniserve_worker.execution.forward_batch import AttentionMetadata, AttentionMode

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tensor_core_decode_replans_and_replays_changed_pages():
    pytest.importorskip("flashinfer")
    torch.manual_seed(619)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    batch, query_heads, kv_heads, width, page_size = 3, 4, 2, 128, 64
    query = torch.randn((batch, query_heads, width), dtype=dtype, device=device)
    keys = torch.randn((6, page_size, kv_heads, width), dtype=dtype, device=device)
    values = torch.randn_like(keys)
    context = AttentionMetadata(
        attention_mode=AttentionMode.PAGED_DECODE,
        prefix_lens=torch.empty(batch, dtype=torch.int32, device=device),
        query_lens=torch.ones(batch, dtype=torch.int32, device=device),
        out_cache_loc=torch.zeros(batch, dtype=torch.int64, device=device),
        has_cache_writes=False,
        binding=619,
        block_table=torch.empty((batch, 2), dtype=torch.int32, device=device),
        seq_lens=torch.empty(batch, dtype=torch.int32, device=device),
    )
    backend = FlashInferAttentionBackend(
        tuning=FlashInferTuningConfig(
            workspace_size=64 * 1024 * 1024,
            decode_backend="fa2",
            use_tensor_core=True,
        )
    )
    layouts = (
        ((65, 7, 1), ((2, 4), (1, 3), (0, 0))),
        ((63, 70, 1), ((5, 0), (4, 2), (1, 0))),
        ((1, 68, 1), ((3, 0), (0, 5), (2, 0))),
    )

    def stage(lengths, pages):
        nonlocal context
        context = replace(
            context,
            seq_lens_cpu=lengths,
            prefix_lens_cpu=tuple(n - 1 for n in lengths),
            query_lens_cpu=(1,) * batch,
        )
        context.prefix_lens.copy_(
            torch.tensor(context.prefix_lens_cpu, dtype=torch.int32, device=device)
        )
        context.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        context.block_table.copy_(torch.tensor(pages, dtype=torch.int32, device=device))
        backend.prepare_paged_decode_cuda_graph(
            context.binding,
            context,
            batch_size=batch,
            max_indices=6,
            num_q_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=width,
            page_size=page_size,
            q_dtype=dtype,
            kv_dtype=dtype,
            scale=width**-0.5,
        )

    def execute():
        return backend.forward_paged(
            query,
            keys,
            values,
            block_table=context.block_table,
            cache_seqlens=context.seq_lens,
            causal=False,
            scale=width**-0.5,
            context=context,
        )

    def reference(lengths, pages):
        rows = []
        for row, (length, row_pages) in enumerate(zip(lengths, pages, strict=True)):
            row_keys = keys[list(row_pages)].flatten(0, 1)[:length].transpose(0, 1).double()
            row_values = values[list(row_pages)].flatten(0, 1)[:length].transpose(0, 1).double()
            rows.append(
                F.scaled_dot_product_attention(
                    query[row, :, None].double(), row_keys, row_values, enable_gqa=True
                )[:, 0]
            )
        return torch.stack(rows).to(dtype)

    graph = torch.cuda.CUDAGraph()
    try:
        stage(*layouts[0])
        torch.testing.assert_close(execute(), reference(*layouts[0]), rtol=2e-2, atol=2e-2)
        with torch.cuda.graph(graph):
            output = execute()
        for lengths, pages in layouts:
            stage(lengths, pages)
            query.add_(0.125)
            graph.replay()
            torch.testing.assert_close(output, reference(lengths, pages), rtol=2e-2, atol=2e-2)
    finally:
        torch.cuda.synchronize(device)
        graph.reset()
        backend.release_paged_decode_graph_binding(context.binding)
