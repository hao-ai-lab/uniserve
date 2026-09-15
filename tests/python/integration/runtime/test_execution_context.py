"""Independent contexts execute shared modules without shared mutable prefix state."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from uniserve.cache import Config, mha
from uniserve.model import TextSize
from uniserve.nn import Linear
from uniserve.nn.attention import Attention, PagedInput
from uniserve.quantization import Quantizer
from uniserve.runtime import CUDAGraph, ExecutionContext, PrefixCache

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "provider,device,graphs,causal",
    [
        ("torch", "cpu", False, (True, False)),
        pytest.param("torch", "cuda:0", True, (True, False), marks=pytest.mark.gpu),
        pytest.param("flashinfer", "cuda:0", True, (True, True), marks=pytest.mark.gpu),
        pytest.param("trtllm", "cuda:0", True, (True, True), marks=pytest.mark.gpu),
        pytest.param("flash_attn_4", "cuda:0", True, (True, True), marks=pytest.mark.gpu),
        pytest.param("trtllm", "cuda:0", False, (True, False), marks=pytest.mark.gpu),
    ],
)
@torch.inference_mode()
def test_device_lengths_drive_current_attention_values(provider, device, graphs, causal):
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(810)
    query = torch.randn(5, 4, 64, dtype=dtype, device=device, generator=generator)
    key, value = (
        torch.randn(5, 2, 64, dtype=dtype, device=device, generator=generator) for _ in range(2)
    )
    layer = Attention(4, 2, 64, cache_name="attention")
    stream = torch.cuda.Stream(device=device) if graphs else None
    with (
        PrefixCache(
            Config({"attention": mha.Config(2, 64, (0, 1), dtype)}),
            num_blocks=2,
            block_size=16,
            device=device,
        ) as cache,
        ExecutionContext(layer, cache=cache, attention=provider, stream=stream) as context,
    ):
        context.prepare(TextSize(5, 2))

        def indices(counts):
            return PagedInput.from_blocks(
                blocks=((0,), (1,)),
                query_lengths=counts,
                prefix_lengths=(0, 0),
                block_size=16,
                causal=causal,
                device=device,
            )

        initial = indices((2, 3))
        batch = replace(
            initial,
            queries=replace(initial.queries, host=None),
            prefixes=replace(initial.prefixes, host=None),
        )
        output = torch.empty_like(query)
        graph = None
        try:
            context.bind_attention(batch)
            layer(query, key, value, batch, out=output)
            if graphs:
                graph = CUDAGraph(context=context)
                graph.capture(lambda: layer(query, key, value, batch, out=output))
            for counts in ((2, 3), (4, 1), (1, 4)):
                changed = indices(counts)
                batch.queries.values.copy_(changed.queries.values)
                batch.queries.offsets.copy_(changed.queries.offsets)
                batch.write_indices.copy_(changed.write_indices)
                value.mul_(0.75)
                context.bind_attention(batch)
                actual = graph.replay() if graph else layer(query, key, value, batch, out=output)
                expected = []
                for q, k, v, flag in zip(
                    query.split(counts), key.split(counts), value.split(counts), causal, strict=True
                ):
                    expected.append(
                        torch.nn.functional.scaled_dot_product_attention(
                            q.transpose(0, 1).double(),
                            k.transpose(0, 1).double(),
                            v.transpose(0, 1).double(),
                            is_causal=flag,
                            enable_gqa=True,
                        )
                        .transpose(0, 1)
                        .to(dtype)
                    )
                torch.testing.assert_close(
                    actual,
                    torch.cat(expected),
                    rtol=1e-5 if device == "cpu" else 2e-2,
                    atol=1e-6 if device == "cpu" else 2e-2,
                )
        finally:
            if graph is not None:
                stream.synchronize()
                graph.close()


class _SelfAttention(nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.projection = Linear(64, 64, bias=False, device=device, dtype=dtype)
        self.attention = Attention(1, 1, 64, cache_name="attention")

    def forward(self, hidden, batch):
        value = self.projection(hidden).unsqueeze(1)
        return self.attention(value, value, value, batch).squeeze(1)


def test_nested_execution_contexts_keep_prefixes_independent():
    module = _SelfAttention()
    module.projection.weight.copy_(torch.eye(64))
    config = Config({"attention": mha.Config(1, 64, (0,), torch.float32)})
    with (
        PrefixCache(config, num_blocks=1, block_size=16, device="cpu") as first,
        PrefixCache(config, num_blocks=1, block_size=16, device="cpu") as second,
        ExecutionContext(module, cache=first, attention="torch") as a,
    ):
        a.prepare(TextSize(1, 1))
        batch = PagedInput.from_blocks(
            blocks=((0,),),
            query_lengths=(1,),
            prefix_lengths=(0,),
            block_size=16,
            causal=True,
            device="cpu",
        )
        one, two = torch.ones(1, 64), torch.full((1, 64), 2.0)
        a.bind_attention(batch)
        torch.testing.assert_close(module(one, batch), one)
        with ExecutionContext(module, cache=second, attention="torch") as b:
            b.prepare(TextSize(1, 1))
            b.bind_attention(batch)
            torch.testing.assert_close(module(two, batch), two)
        readonly = replace(batch, write_indices=None)
        a.bind_attention(readonly)
        torch.testing.assert_close(module(two, readonly), one)
        torch.testing.assert_close(first.state("attention").value[0, 0], one)
        torch.testing.assert_close(second.state("attention").value[0, 0], two)


def test_shared_weight_call_sites_keep_distinct_activation_representations():
    quantizer = Quantizer("fp8", axis=0)
    first, second = Linear(8, 4, bias=False), Linear(8, 4, bias=False)
    first.weight = nn.Parameter(quantizer.quantize(first.weight), requires_grad=False)
    second.weight = first.weight
    second.input_quantizer = quantizer
    layers = nn.ModuleDict({"dense": first, "encoded": second})
    x = torch.arange(24).view(3, 8).float().sin()
    expected = {name: layer(x) for name, layer in layers.items()}
    with ExecutionContext(layers, matmul="torch") as context:
        context.prepare(TextSize(3, 1))
        for name, layer in layers.items():
            torch.testing.assert_close(layer(x), expected[name], rtol=0, atol=0)


@pytest.mark.gpu
@torch.inference_mode()
@pytest.mark.parametrize("axis", (None, 0))
def test_shared_projection_has_independent_encoded_graph_inputs(axis):
    device = torch.device("cuda", 0)
    quantizer = Quantizer("fp8", axis=axis)
    layer = Linear(64, 32, dtype=torch.bfloat16, device=device)
    layer.weight = nn.Parameter(quantizer.quantize(layer.weight), requires_grad=False)
    layer.input_quantizer = quantizer
    generator = torch.Generator(device=device).manual_seed(17)
    x = torch.randn(5, 64, dtype=torch.bfloat16, device=device, generator=generator)
    y = torch.randn(5, 64, dtype=x.dtype, device=device, generator=generator)
    first, second = (
        torch.empty(5, 32, dtype=x.dtype, device=device),
        torch.empty(5, 32, dtype=x.dtype, device=device),
    )
    with ExecutionContext(layer) as a:
        a.prepare(TextSize(8, 1))
        layer(x, out=first)
        with CUDAGraph(context=a) as ga, ExecutionContext(layer) as b:
            b.prepare(TextSize(8, 1))
            layer(y, out=second)
            with CUDAGraph(context=b) as gb:
                ga.capture(lambda: layer(x, out=first))
                gb.capture(lambda: layer(y, out=second))
                x.mul_(4)
                ga.replay()
                expected_x = layer(x)
                gb.replay()
                expected_y = layer(y)
                torch.testing.assert_close(first, expected_x, rtol=0, atol=0)
                torch.testing.assert_close(second, expected_y, rtol=0, atol=0)


@pytest.mark.gpu
@torch.inference_mode()
def test_eager_attention_replans_changed_sequence_boundaries():
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    layer = Attention(2, 1, 64, cache_name="attention")
    config = Config({"attention": mha.Config(1, 64, (0,), dtype)})
    generator = torch.Generator(device=device).manual_seed(30)
    q = torch.randn(5, 2, 64, dtype=dtype, device=device, generator=generator)
    k, v = (
        torch.randn(5, 1, 64, dtype=dtype, device=device, generator=generator) for _ in range(2)
    )
    with (
        PrefixCache(config, num_blocks=2, block_size=16, device=device) as cache,
        ExecutionContext(layer, cache=cache, attention="flashinfer") as context,
    ):
        context.prepare(TextSize(5, 2))
        for counts in ((4, 1), (2, 3), (1, 4)):
            batch = PagedInput.from_blocks(
                blocks=((0,), (1,)),
                query_lengths=counts,
                prefix_lengths=(0, 0),
                block_size=16,
                causal=True,
                device=device,
            )
            actual = layer(q, k, v, batch)
            expected = []
            for query, key, value in zip(
                q.split(counts), k.split(counts), v.split(counts), strict=True
            ):
                expected.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        query.transpose(0, 1).unsqueeze(0),
                        key.transpose(0, 1).unsqueeze(0),
                        value.transpose(0, 1).unsqueeze(0),
                        is_causal=True,
                        enable_gqa=True,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
            torch.testing.assert_close(actual, torch.cat(expected), rtol=2e-2, atol=2e-2)
