"""Native paged plans retain each submitted generation through CUDA replay."""

from contextlib import ExitStack
from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from uniserve.model import TextSize
from uniserve.nn.attention import BlockTable, PagedInput, SequenceLengths
from uniserve.runtime import TensorBuffers
from uniserve.runtime.backends.attention.flashinfer import Backend, Config

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
@pytest.mark.parametrize("causal", [(False,) * 3, (True,) * 3, (True, False, True)])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("tensor_cores,fast_plan", [(False, False), (True, False), (True, True)])
def test_paged_replay_retains_each_length_and_page_generation(
    causal, decode, tensor_cores, fast_plan
):
    generator = torch.Generator(device="cuda:0").manual_seed(618)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    page_size, query_heads, kv_heads, width = 64, 4, 2, 128
    num_tokens = 3 if decode else 259
    query = torch.randn(
        (num_tokens, query_heads, width), dtype=dtype, device=device, generator=generator
    )
    keys = torch.randn(
        (16, page_size, kv_heads, width), dtype=dtype, device=device, generator=generator
    )
    values = torch.randn(keys.shape, dtype=dtype, device=device, generator=generator)
    backend = Backend(
        Config(
            workspace_size=64 * 1024 * 1024,
            use_tensor_core=tensor_cores,
            prefill_backend="fa2",
            decode_split_tile_size=1 if tensor_cores else None,
            prefill_split_tile_size=1,
            disable_split_kv=True,
            fast_decode_plan=fast_plan,
        )
    )
    layouts = (
        ((1, 257, 1), (65, 321, 1), ((2, 4, 0, 0, 0, 0), (1, 3, 5, 7, 9, 11), (0, 0, 0, 0, 0, 0))),
        (
            (129, 129, 1),
            (191, 198, 1),
            ((5, 0, 6, 0, 0, 0), (4, 2, 8, 10, 0, 0), (1, 0, 0, 0, 0, 0)),
        ),
        ((257, 1, 1), (321, 68, 1), ((3, 1, 6, 7, 8, 9), (0, 5, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0))),
    )

    if decode:
        layouts = tuple(((1, 1, 1), lengths, pages) for _, lengths, pages in layouts)
    table = BlockTable(torch.empty((3, 4096), device=device, dtype=torch.int32)[:, :6], page_size)
    queries = SequenceLengths.from_lengths(layouts[0][0], device=device)
    prefixes = SequenceLengths.from_lengths(
        tuple(b - a for a, b in zip(layouts[0][0], layouts[0][1], strict=True)), device=device
    )

    def stage(query_lens, seq_lens, pages):
        lengths = tuple(total - count for total, count in zip(seq_lens, query_lens, strict=True))
        for columns, data in ((queries, query_lens), (prefixes, lengths)):
            columns.values.copy_(
                torch.tensor(data, dtype=torch.int32, pin_memory=True), non_blocking=True
            )
            columns.offsets.copy_(
                torch.tensor(
                    tuple(accumulate(data, initial=0)), dtype=torch.int32, pin_memory=True
                ),
                non_blocking=True,
            )
        table.indices.copy_(
            torch.tensor(pages, dtype=torch.int32, pin_memory=True), non_blocking=True
        )
        return PagedInput(
            SequenceLengths(host=query_lens, values=queries.values, offsets=queries.offsets),
            SequenceLengths(host=lengths, values=prefixes.values, offsets=prefixes.offsets),
            table,
            None,
            causal,
        )

    def reference(query_lens, seq_lens, pages):
        outputs = []
        start = 0
        for query_count, kv_count, row_pages, row_causal in zip(
            query_lens, seq_lens, pages, causal, strict=True
        ):
            row_query = query[start : start + query_count].transpose(0, 1).double()
            row_key = keys[list(row_pages)].flatten(0, 1)[:kv_count].transpose(0, 1).double()
            row_value = values[list(row_pages)].flatten(0, 1)[:kv_count].transpose(0, 1).double()
            mask = None
            if row_causal:
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

    arguments = dict(
        num_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=width,
        dtype=dtype,
        size=TextSize(num_tokens, 3),
        cache=None,
    )
    requirements = backend.workspace_buffers(**arguments)
    assert requirements["scratch"].shape == (64 * 1024 * 1024,)
    with ExitStack() as scope:
        buffers = scope.enter_context(TensorBuffers.allocate(requirements, device=device))
        operator = backend.prepare(**arguments, workspace=buffers.view(requirements))
        scope.callback(operator.close)
        graph = torch.cuda.CUDAGraph()
        scope.callback(graph.reset)
        batch = stage(*layouts[0])
        operator.bind(batch)
        output = torch.empty_like(query)
        operator(query, keys, values, batch, scale=width**-0.5, out=output)
        torch.testing.assert_close(output, reference(*layouts[0]), rtol=2e-2, atol=2e-2)
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            operator(query, keys, values, batch, scale=width**-0.5, out=output)
        original_query = query.clone()
        expected = []
        for layout in layouts:
            query.add_(0.125)
            expected.append(reference(*layout))
        query.copy_(original_query)
        torch.cuda.synchronize(device)
        # Successive CPU plans may complete before the preceding upload. Each
        # captured invocation must observe its own plan and retained output.
        torch.cuda._sleep(1_000_000_000)
        actual = []
        for layout in layouts:
            batch = stage(*layout)
            query.add_(0.125)
            operator.bind(batch)
            graph.replay()
            actual.append(output.clone())
        for result, expected_output in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, expected_output, rtol=2e-2, atol=2e-2)
        torch.cuda.synchronize(device)
