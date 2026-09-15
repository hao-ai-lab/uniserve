"""Segmented attention merges live prefix pages and visible current tokens."""

from contextlib import ExitStack
from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from uniserve.cache import Config, mha
from uniserve.model import TextSize
from uniserve.nn.attention import BlockTable, SegmentedInput, SequenceLengths
from uniserve.runtime import PrefixCache, TensorBuffers
from uniserve.runtime.backends.attention import resolve

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
@pytest.mark.parametrize("provider", ["flashinfer", "flash_attn_4"])
def test_segmented_replay_reads_changed_partitions_empty_prefixes_and_page_boundaries(provider):
    device = torch.device("cuda", 0)
    torch.manual_seed(136)
    count, width, page_size = 259, 64, 64
    query = torch.randn(count, 4, width, device=device, dtype=torch.bfloat16)
    key = torch.randn(count, 2, width, device=device, dtype=query.dtype)
    value = torch.randn_like(key)
    visible = torch.empty(3, count, dtype=torch.int32, device=device)
    queries = SequenceLengths.from_lengths((1, 129, 129), device=device)
    prefixes = SequenceLengths.from_lengths((0, 63, 64), device=device)
    table = BlockTable(
        torch.tensor([[0, 1], [2, 3], [4, 5]], device=device, dtype=torch.int32), page_size
    )
    layouts = (
        ((1, 129, 129), (0, 63, 64)),
        ((129, 1, 129), (65, 0, 63)),
        ((0, 1, 258), (0, 1, 65)),
    )

    def stage(query_lengths, prefix_lengths, iteration):
        for columns, lengths in ((queries, query_lengths), (prefixes, prefix_lengths)):
            columns.values.copy_(
                torch.tensor(lengths, dtype=torch.int32, pin_memory=True), non_blocking=True
            )
            columns.offsets.copy_(
                torch.tensor(
                    tuple(accumulate(lengths, initial=0)), dtype=torch.int32, pin_memory=True
                ),
                non_blocking=True,
            )
        for row, length in enumerate(query_lengths):
            limits = (torch.arange(count, device=device) * 7 + iteration).remainder(length + 1)
            limits[: min(128, length)] = length
            limits[0] = 0
            visible[row].copy_(limits)
        return SegmentedInput(
            SequenceLengths(host=query_lengths, values=queries.values, offsets=queries.offsets),
            SequenceLengths(host=prefix_lengths, values=prefixes.values, offsets=prefixes.offsets),
            table,
            None,
            visible,
            False,
        )

    with ExitStack() as scope:
        cache = scope.enter_context(
            PrefixCache(
                Config({"attention": mha.Config(2, width, (0, 1), query.dtype)}),
                num_blocks=6,
                block_size=page_size,
                device=device,
            )
        )
        state = cache.state("attention")
        state.key.normal_()
        state.value.normal_()
        saved = state.key.clone(), state.value.clone()
        backend = resolve(provider, device=device)
        arguments = dict(
            num_heads=4,
            num_kv_heads=2,
            head_dim=width,
            dtype=query.dtype,
            size=TextSize(count, 3),
            cache=state,
        )
        requirements = backend.workspace_buffers(**arguments)
        buffers = scope.enter_context(TensorBuffers.allocate(requirements, device=device))
        operator = backend.prepare(**arguments, workspace=buffers.view(requirements))
        scope.callback(operator.close)
        output = torch.empty_like(query)

        def reference(query_lengths, prefix_lengths):
            outputs = []
            offset = 0
            for row, (length, prefix) in enumerate(zip(query_lengths, prefix_lengths, strict=True)):
                interval = slice(offset, offset + length)
                offset += length
                prefix_key = state.key[row * 2 : row * 2 + 2].flatten(0, 1)[:prefix]
                prefix_value = state.value[row * 2 : row * 2 + 2].flatten(0, 1)[:prefix]
                keys = torch.cat((prefix_key, key[interval]))
                values = torch.cat((prefix_value, value[interval]))
                mask = (
                    torch.arange(prefix + length, device=device)[None]
                    < prefix + visible[row, :length, None]
                )
                outputs.append(
                    F.scaled_dot_product_attention(
                        query[interval].double().transpose(0, 1),
                        keys.double().transpose(0, 1),
                        values.double().transpose(0, 1),
                        attn_mask=mask,
                        enable_gqa=True,
                    )
                    .transpose(0, 1)
                    .to(query.dtype)
                )
            return torch.cat(outputs)

        batch = stage(*layouts[0], 0)
        operator.bind(batch)
        operator(query, key, value, batch, scale=width**-0.5, out=output)
        torch.testing.assert_close(output, reference(*layouts[0]), atol=0.02, rtol=0.02)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        scope.callback(graph.reset)
        with torch.cuda.graph(graph):
            operator(query, key, value, batch, scale=width**-0.5, out=output)
        for iteration, layout in enumerate(layouts[1:], 1):
            batch = stage(*layout, iteration)
            value.mul_(0.75)
            operator.bind(batch)
            graph.replay()
            torch.testing.assert_close(output, reference(*layout), atol=0.02, rtol=0.02)
        torch.testing.assert_close(state.key, saved[0], atol=0, rtol=0)
        torch.testing.assert_close(state.value, saved[1], atol=0, rtol=0)
