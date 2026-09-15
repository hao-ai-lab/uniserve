"""Native attention preserves numerical masks, cache writes and graph inputs."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from uniserve.cache import Config, mha
from uniserve.model import TextSize
from uniserve.nn.attention import (
    DenseInput,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
    VarlenInput,
    VisibleInput,
)
from uniserve.runtime import PrefixCache, TensorBuffers
from uniserve.runtime.backends.attention import resolve
from uniserve.runtime.backends.attention.torch import Backend as TorchBackend

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@contextmanager
def _prepare(backend, state, size=TextSize(5, 2)):
    arguments = dict(
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        size=size,
        cache=state,
    )
    requirements = backend.workspace_buffers(**arguments)
    with TensorBuffers.allocate(requirements, device="cuda:0") as buffers:
        operator = backend.prepare(**arguments, workspace=buffers.view(requirements))
        try:
            yield operator
        finally:
            operator.close()


@torch.inference_mode()
@pytest.mark.parametrize(
    ("provider", "representation", "graphs"),
    [
        (provider, name, False)
        for provider in ("flash_attn_4", "flashinfer", "auto")
        for name in ("dense", "varlen", "paged", "visible", "segmented")
    ]
    + [("trtllm", "paged", False)]
    + [("torch", name, True) for name in ("varlen", "paged", "visible", "segmented")],
)
def test_native_attention_matches_declared_visibility(provider, representation, graphs):
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(37)
    q = torch.randn(5, 4, 64, dtype=torch.bfloat16, device=device, generator=generator)
    k, v = (
        torch.randn(5, 2, 64, dtype=q.dtype, device=device, generator=generator) for _ in range(2)
    )
    cache_layout = Config({"attention": mha.Config(2, 64, (0, 1), q.dtype)})
    with PrefixCache(cache_layout, num_blocks=4, block_size=16, device=device) as cache:
        state = cache.state("attention")
        state.write((0,), start=0, key=k[:2], value=v[:2])
        state.write((1,), start=0, key=k[:1], value=v[:1])
        paged = PagedInput.from_blocks(
            blocks=((0,), (1,)),
            query_lengths=(2, 3),
            prefix_lengths=(2, 1),
            block_size=16,
            causal=(True, False),
            device=device,
        )
        ends = torch.tensor([[1, 2, 0], [0, 2, 3]], dtype=torch.int32, device=device)
        batch = {
            "dense": DenseInput(True, None),
            "varlen": VarlenInput(
                paged.queries, SequenceLengths.from_lengths((3, 2), device=device), (True, False)
            ),
            "paged": paged,
            "visible": VisibleInput(paged.queries, paged.queries, ends, None, True, False),
            "segmented": SegmentedInput(
                paged.queries, paged.prefixes, paged.block_table, None, ends, False
            ),
        }[representation]
        with (
            _prepare(resolve(provider, device=device), state) as native,
            _prepare(TorchBackend(), state) as reference,
        ):
            native.bind(batch)
            actual, expected = torch.empty_like(q), torch.empty_like(q)
            native(q, k, v, batch, scale=0.125, out=actual)
            if graphs:
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    native(q, k, v, batch, scale=0.125, out=actual)
                graph.replay()
            # Compare the already committed cache so the reference invocation
            # does not perform a second write for the same numerical operation.
            reference_batch = (
                replace(batch, write_indices=None) if representation == "paged" else batch
            )
            reference(q, k, v, reference_batch, scale=0.125, out=expected)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_segmented_graph_keeps_nonfinite_values_within_their_sequence():
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(93)
    q = torch.randn(5, 4, 64, dtype=torch.bfloat16, device=device, generator=generator)
    k, v = (
        torch.randn(5, 2, 64, dtype=q.dtype, device=device, generator=generator) for _ in range(2)
    )
    # One independent sequence cannot contaminate another through masked
    # operands, even when the former contains nonfinite current values.
    k[:2].fill_(torch.nan)
    v[:2].fill_(torch.nan)
    batch = PagedInput.from_blocks(
        blocks=((0,), (1,)),
        query_lengths=(2, 3),
        prefix_lengths=(0, 0),
        block_size=16,
        causal=False,
        device=device,
    )
    ends = torch.tensor([[2, 2, 2], [3, 3, 3]], dtype=torch.int32, device=device)
    segmented = SegmentedInput(batch.queries, batch.prefixes, batch.block_table, None, ends, True)
    with (
        PrefixCache(
            Config({"attention": mha.Config(2, 64, (0, 1), q.dtype)}),
            num_blocks=2,
            block_size=16,
            device=device,
        ) as cache,
        _prepare(TorchBackend(), cache.state("attention")) as operator,
    ):
        operator.bind(segmented)
        actual = torch.empty_like(q)
        operator(q, k, v, segmented, scale=0.125, out=actual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                operator(q, k, v, segmented, scale=0.125, out=actual)
            graph.replay()
            expected = (
                torch.nn.functional.scaled_dot_product_attention(
                    q[2:].transpose(0, 1).double(),
                    k[2:].transpose(0, 1).double(),
                    v[2:].transpose(0, 1).double(),
                    scale=0.125,
                    enable_gqa=True,
                )
                .transpose(0, 1)
                .to(q.dtype)
            )
            torch.testing.assert_close(actual[2:], expected, rtol=2e-2, atol=2e-2)
        finally:
            torch.cuda.synchronize()
            graph.reset()


@torch.inference_mode()
@pytest.mark.parametrize("provider", ["torch", "flash_attn_4", "trtllm", "flashinfer", "auto"])
def test_native_paged_replay_reads_changed_block_table_and_writes_slot_zero(provider):
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(73)
    q = torch.randn(1, 4, 64, dtype=torch.bfloat16, device=device, generator=generator)
    k, v = (
        torch.randn(1, 2, 64, dtype=q.dtype, device=device, generator=generator) for _ in range(2)
    )
    with PrefixCache(
        Config({"attention": mha.Config(2, 64, (0, 1), q.dtype)}),
        num_blocks=3,
        block_size=16,
        device=device,
    ) as cache:
        state = cache.state("attention")
        batch = PagedInput.from_blocks(
            blocks=((0,),),
            query_lengths=(1,),
            prefix_lengths=(0,),
            block_size=16,
            causal=True,
            device=device,
        )
        with (
            _prepare(resolve(provider, device=device), state) as native,
            _prepare(TorchBackend(), state) as reference,
        ):
            out = torch.empty_like(q)
            native.bind(batch)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                native(q, k, v, batch, scale=0.125, out=out)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.testing.assert_close(state.key[0, 0], k[0], rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                native(q, k, v, batch, scale=0.125, out=out)
            batch.block_table.indices[0, 0] = 2
            batch.write_indices[0] = 32
            v.mul_(3)
            native.bind(batch)
            graph.replay()
            expected = torch.empty_like(q)
            reference(q, k, v, replace(batch, write_indices=None), scale=0.125, out=expected)
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(state.value[2, 0], v[0], rtol=0, atol=0)
            graph.reset()


@torch.inference_mode()
@pytest.mark.parametrize("destination", ["separate", "strided", "query", "offset"])
@pytest.mark.parametrize("decode", [False, True])
def test_native_paged_output_views_preserve_values_on_replay(destination, decode):
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(618)
    counts = (1, 1) if decode else (2, 3)
    q = torch.randn(sum(counts), 4, 64, dtype=torch.bfloat16, device=device, generator=generator)
    k, v = (
        torch.randn(sum(counts), 2, 64, dtype=q.dtype, device=device, generator=generator)
        for _ in range(2)
    )
    if destination == "query":
        out = q
    elif destination == "strided":
        out = torch.empty(*q.shape[:-1], q.shape[-1] * 2, dtype=q.dtype, device=device)[..., ::2]
    elif destination == "offset":
        out = torch.empty(q.numel() + 1, dtype=q.dtype, device=device)[1:].view_as(q)
    else:
        out = torch.empty_like(q)
    with PrefixCache(
        Config({"attention": mha.Config(2, 64, (0, 1), q.dtype)}),
        num_blocks=2,
        block_size=16,
        device=device,
    ) as cache:
        state = cache.state("attention")
        batch = PagedInput.from_blocks(
            blocks=((0,), (1,)),
            query_lengths=counts,
            prefix_lengths=(1, 2),
            block_size=16,
            causal=True,
            device=device,
        )
        with (
            _prepare(resolve("trtllm", device=device), state) as native,
            _prepare(TorchBackend(), state) as reference,
        ):
            native.bind(batch)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                assert native(q, k, v, batch, scale=0.125, out=out) is out
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                native(q, k, v, batch, scale=0.125, out=out)
            # Restore all live inputs after capture, including a query that
            # also supplies output storage. Expected values use its snapshot.
            q.copy_(torch.randn(q.shape, dtype=q.dtype, device=device, generator=generator))
            query = q.clone()
            v.mul_(2)
            graph.replay()
            expected = torch.empty_like(q)
            reference(query, k, v, replace(batch, write_indices=None), scale=0.125, out=expected)
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
            if destination != "query":
                torch.testing.assert_close(q, query, rtol=0, atol=0)
            graph.reset()


@torch.inference_mode()
@pytest.mark.parametrize("column_stride", [1, 2])
def test_paged_replay_preserves_masked_writes_and_live_sequence_boundaries(column_stride):
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(204)
    q = torch.randn(5, 4, 64, dtype=torch.bfloat16, device=device, generator=generator)
    k, v = (
        torch.randn(5, 2, 64 * column_stride, dtype=q.dtype, device=device, generator=generator)[
            ..., ::column_stride
        ]
        for _ in range(2)
    )
    with PrefixCache(
        Config({"attention": mha.Config(2, 64, (0, 1), q.dtype)}),
        num_blocks=4,
        block_size=16,
        device=device,
    ) as cache:
        state = cache.state("attention")
        batch = PagedInput.from_blocks(
            blocks=((0, 1), (2, 3)),
            query_lengths=(2, 3),
            prefix_lengths=(1, 2),
            block_size=16,
            causal=True,
            device=device,
        )
        # The device columns remain borrowed inputs; no host mirror supplies
        # lengths or offsets to the captured numerical call.
        batch = replace(
            batch,
            queries=replace(batch.queries, host=None),
            prefixes=replace(batch.prefixes, host=None),
        )
        batch.write_indices[0] = -1
        with (
            _prepare(resolve("trtllm", device=device), state) as native,
            _prepare(TorchBackend(), state) as reference,
        ):
            out = torch.empty_like(q)
            native.bind(batch)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                native(q, k, v, batch, scale=0.125, out=out)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                native(q, k, v, batch, scale=0.125, out=out)

            changed = PagedInput.from_blocks(
                blocks=((1, 0), (3, 2)),
                query_lengths=(3, 2),
                prefix_lengths=(2, 1),
                block_size=16,
                causal=True,
                device=device,
            )
            changed.write_indices[1] = -1
            for name in ("queries", "prefixes"):
                getattr(batch, name).values.copy_(getattr(changed, name).values)
                getattr(batch, name).offsets.copy_(getattr(changed, name).offsets)
            batch.block_table.indices.copy_(changed.block_table.indices)
            batch.write_indices.copy_(changed.write_indices)
            k.mul_(2)
            v.mul_(3)
            before = tuple(tensor.clone() for tensor in (q, k, v))
            expected_state = tuple(tensor.clone() for tensor in (state.key, state.value))
            slots = changed.write_indices.cpu().tolist()
            for target, source in zip(expected_state, (k, v), strict=True):
                for row, slot in enumerate(slots):
                    if slot != -1:
                        target.flatten(0, 1)[slot].copy_(source[row])
            graph.replay()
            expected = torch.empty_like(q)
            reference(q, k, v, replace(changed, write_indices=None), scale=0.125, out=expected)
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
            for actual, expected in zip((state.key, state.value), expected_state, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip((q, k, v), before, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert all(flags.tolist() == [True] * 4 for flags in state.initialized.values())
            graph.reset()


@torch.inference_mode()
@pytest.mark.parametrize("provider", ["torch", "flash_attn_4", "flashinfer"])
@pytest.mark.parametrize("paged_keys", [False, True])
def test_visible_paged_replay_uses_changed_endpoints(provider, paged_keys):
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(719)
    count = 257
    q = torch.randn(count, 4, 64, dtype=torch.bfloat16, device=device, generator=generator)
    k, v = (
        torch.randn(count, 2, 64, dtype=q.dtype, device=device, generator=generator)
        for _ in range(2)
    )
    with PrefixCache(
        Config({"attention": mha.Config(2, 64, (0, 1), q.dtype)}),
        num_blocks=17,
        block_size=16,
        device=device,
    ) as cache:
        state = cache.state("attention")
        state.write(tuple(range(17)), start=0, key=k, value=v)
        paged = PagedInput.from_blocks(
            blocks=(tuple(range(17)),),
            query_lengths=(count,),
            prefix_lengths=(0,),
            block_size=16,
            causal=True,
            device=device,
        )
        ends = (torch.arange(count, dtype=torch.int32, device=device) + 1)[None]
        batch = VisibleInput(
            paged.queries,
            paged.queries,
            ends,
            paged.block_table if paged_keys else None,
            True,
            False,
        )
        with (
            _prepare(resolve(provider, device=device), state, TextSize(count, 1)) as native,
            _prepare(TorchBackend(), state, TextSize(count, 1)) as reference,
        ):
            native.bind(batch)
            actual = torch.empty_like(q)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                native(q, k, v, batch, scale=0.125, out=actual)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                native(q, k, v, batch, scale=0.125, out=actual)
            ends.fill_(count)
            native.bind(batch)
            graph.replay()
            expected = torch.empty_like(q)
            reference(q, k, v, batch, scale=0.125, out=expected)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            graph.reset()


@torch.inference_mode()
def test_automatic_long_packed_attention_preserves_causal_prefixes_on_replay():
    device = torch.device("cuda:0")
    rows, heads, kv_heads, width = 10_003, 40, 8, 128
    query = torch.zeros(rows, heads, width, dtype=torch.bfloat16, device=device)
    key = torch.zeros(rows, kv_heads, width, dtype=query.dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(219)
    value = torch.randn(key.shape, dtype=key.dtype, device=device, generator=generator)
    lengths = SequenceLengths.from_lengths((10_000, 3), device=device)
    batch = VarlenInput(lengths, lengths, (True, True))
    backend = resolve("auto", device=device)
    arguments = dict(
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=width,
        dtype=query.dtype,
        size=TextSize(rows, 2),
        cache=None,
    )
    requirements = backend.workspace_buffers(**arguments)
    with TensorBuffers.allocate(requirements, device=device) as buffers:
        native = backend.prepare(**arguments, workspace=buffers.view(requirements))
        graph = torch.cuda.CUDAGraph()
        actual = torch.empty_like(query)
        try:
            native.bind(batch)
            native(query, key, value, batch, scale=width**-0.5, out=actual)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                native(query, key, value, batch, scale=width**-0.5, out=actual)
            for partition in ((10_000, 3), (3, 10_000)):
                lengths.values.copy_(torch.tensor(partition, device=device, dtype=torch.int32))
                lengths.offsets.copy_(
                    torch.tensor((0, partition[0], rows), device=device, dtype=torch.int32)
                )
                current = SequenceLengths(
                    host=partition, values=lengths.values, offsets=lengths.offsets
                )
                native.bind(VarlenInput(current, current, batch.causal))
                value.mul_(0.75)
                graph.replay()
                # Zero Q/K makes each causal softmax uniform over its prefix.
                # This oracle needs no quadratic attention matrix or backend.
                start = 0
                for count in partition:
                    prefix = value[start : start + count].float().cumsum(0)
                    divisor = torch.arange(1, count + 1, device=device).view(-1, 1, 1)
                    expected = (prefix / divisor).repeat_interleave(heads // kv_heads, dim=1)
                    torch.testing.assert_close(
                        actual[start : start + count].float(), expected, rtol=2e-2, atol=2e-2
                    )
                    start += count
        finally:
            graph.reset()
            native.close()


@torch.inference_mode()
def test_automatic_vision_attention_preserves_non_power_of_two_heads_on_replay():
    """BAGEL's SigLIP head width is 72; its packed image boundaries remain live."""

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(214)
    query = torch.randn(56, 4, 72, device=device, dtype=torch.bfloat16, generator=generator)
    key = torch.randn(56, 2, 72, device=device, dtype=query.dtype, generator=generator)
    value = torch.randn_like(key)
    lengths = SequenceLengths.from_lengths((39, 17), device=device)
    batch = VarlenInput(lengths, lengths, (False, False))
    backend = resolve("auto", device=device)
    arguments = dict(
        num_heads=4,
        num_kv_heads=2,
        head_dim=72,
        dtype=query.dtype,
        size=TextSize(56, 2),
        cache=None,
    )
    requirements = backend.workspace_buffers(**arguments)
    with TensorBuffers.allocate(requirements, device=device) as buffers:
        native = backend.prepare(**arguments, workspace=buffers.view(requirements))
        reference = TorchBackend().prepare(**arguments, workspace={})
        graph = torch.cuda.CUDAGraph()
        actual, expected = torch.empty_like(query), torch.empty_like(query)
        try:
            native.bind(batch)
            native(query, key, value, batch, scale=72**-0.5, out=actual)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                native(query, key, value, batch, scale=72**-0.5, out=actual)
            for partition in ((39, 17), (1, 55), (56, 0)):
                lengths.values.copy_(torch.tensor(partition, device=device, dtype=torch.int32))
                lengths.offsets.copy_(
                    torch.tensor((0, partition[0], 56), device=device, dtype=torch.int32)
                )
                current = SequenceLengths(
                    host=partition, values=lengths.values, offsets=lengths.offsets
                )
                changed = VarlenInput(current, current, batch.causal)
                value.mul_(0.75)
                native.bind(changed)
                graph.replay()
                reference(query, key, value, changed, scale=72**-0.5, out=expected)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        finally:
            graph.reset()
            native.close()
            reference.close()
