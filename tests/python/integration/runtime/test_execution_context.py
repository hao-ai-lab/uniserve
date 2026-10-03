"""Independent contexts execute shared modules.

They do so without shared mutable prefix state.
"""

from contextlib import ExitStack, nullcontext
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from uniserve.cache import Config, mha
from uniserve.model import TextSize
from uniserve.nn import Linear, MergedColumnParallelLinear
from uniserve.nn.attention import (
    Attention,
    AttentionBatch,
    DenseInput,
    PagedInput,
)
from uniserve.quantization import Quantizer
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    PrefixCache,
    Scratch,
)
from uniserve.tensors import BufferConfig

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "provider,device,graphs,causal",
    [
        ("torch", "cpu", False, (True, False)),
        pytest.param(
            "torch", "cuda:0", True, (True, False), marks=pytest.mark.gpu
        ),
        pytest.param(
            "flashinfer", "cuda:0", True, (True, True), marks=pytest.mark.gpu
        ),
        pytest.param(
            "trtllm", "cuda:0", True, (True, True), marks=pytest.mark.gpu
        ),
        pytest.param(
            "flash_attn_4", "cuda:0", True, (True, True), marks=pytest.mark.gpu
        ),
        pytest.param(
            "trtllm", "cuda:0", False, (True, False), marks=pytest.mark.gpu
        ),
    ],
)
@torch.inference_mode()
def test_device_lengths_drive_current_attention_values(
    provider, device, graphs, causal
):
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(810)
    query = torch.randn(
        5, 4, 64, dtype=dtype, device=device, generator=generator
    )
    key, value = (
        torch.randn(5, 2, 64, dtype=dtype, device=device, generator=generator)
        for _ in range(2)
    )
    layer = Attention(4, 2, 64, cache_name="attention")
    stream = (
        CUDAStream.external(torch.cuda.Stream(device=device))
        if graphs
        else None
    )
    with (
        stream if stream is not None else nullcontext(),
        PrefixCache(
            Config({"attention": mha.Config(2, 64, (0, 1), dtype)}),
            num_units=2,
            block_size=16,
            device=device,
        ) as cache,
        ExecutionContext(
            layer, cache=cache, attention=provider, stream=stream
        ) as context,
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
            context.bind_attention(AttentionBatch.single(batch))
            layer(query, key, value, AttentionBatch.single(batch), out=output)
            if graphs:
                graph = CUDAGraph(context=context)
                graph.capture(
                    lambda: layer(
                        query,
                        key,
                        value,
                        AttentionBatch.single(batch),
                        out=output,
                    )
                )
            for counts in ((2, 3), (4, 1), (1, 4)):
                changed = indices(counts)
                batch.queries.values.copy_(changed.queries.values)
                batch.queries.offsets.copy_(changed.queries.offsets)
                batch.write_indices.copy_(changed.write_indices)
                value.mul_(0.75)
                context.bind_attention(AttentionBatch.single(batch))
                actual = (
                    graph.replay()
                    if graph
                    else layer(
                        query,
                        key,
                        value,
                        AttentionBatch.single(batch),
                        out=output,
                    )
                )
                expected = []
                for q, k, v, flag in zip(
                    query.split(counts),
                    key.split(counts),
                    value.split(counts),
                    causal,
                    strict=True,
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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10,
    reason="TensorRT-LLM attention requires an SM100 GPU",
)
@torch.inference_mode()
def test_replay_binding_still_rejects_inputs_outside_the_captured_contract():
    # Two layers read each of two cache tables through a provider whose
    # kernels read every length on the device, so a replay's binding only
    # checks the batch; those checks still guard every table.
    device, dtype = "cuda:0", torch.bfloat16
    names = ("a", "b", "c", "d")
    module = nn.ModuleDict(
        {
            name: Attention(4, 2, 64 if name in "ab" else 128, cache_name=name)
            for name in names
        }
    )
    config = Config(
        {
            name: mha.Config(2, 64 if name in "ab" else 128, (0, 1), dtype)
            for name in names
        }
    )
    with (
        PrefixCache(config, num_units=8, block_size=16, device=device) as cache,
        ExecutionContext(module, cache=cache, attention="trtllm") as context,
    ):
        context.prepare(TextSize(5, 2))
        tables = sorted({cache.table(name) for name in names})
        assert len(tables) == 2

        def batch(counts, scale=1):
            # Each table pages at its group's page size; ``scale`` breaks
            # that contract for the second table alone.
            entries = {}
            for number, table in enumerate(tables):
                pages = cache.groups[cache.tables[table].group].page_tokens
                paged = PagedInput.from_blocks(
                    blocks=((1 + 2 * number,), (2 + 2 * number,)),
                    query_lengths=counts,
                    prefix_lengths=(0, 0),
                    block_size=pages * (scale if number else 1),
                    causal=True,
                    device=device,
                )
                entries[table] = paged
            first = entries[tables[0]]
            entries = {
                table: replace(
                    entry, queries=first.queries, prefixes=first.prefixes
                )
                for table, entry in entries.items()
            }
            return AttentionBatch(entries, first.queries)

        context.bind_attention(batch((2, 3)), replay=True)
        with pytest.raises(ValueError, match="prepared token"):
            context.bind_attention(batch((3, 3)), replay=True)
        with pytest.raises(ValueError, match="block sizes differ"):
            context.bind_attention(batch((2, 3), scale=2), replay=True)


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
        PrefixCache(config, num_units=1, block_size=16, device="cpu") as first,
        PrefixCache(config, num_units=1, block_size=16, device="cpu") as second,
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
        a.bind_attention(AttentionBatch.single(batch))
        torch.testing.assert_close(
            module(one, AttentionBatch.single(batch)), one
        )
        with ExecutionContext(module, cache=second, attention="torch") as b:
            b.prepare(TextSize(1, 1))
            b.bind_attention(AttentionBatch.single(batch))
            torch.testing.assert_close(
                module(two, AttentionBatch.single(batch)), two
            )
        readonly = replace(batch, write_indices=None)
        a.bind_attention(AttentionBatch.single(readonly))
        torch.testing.assert_close(
            module(two, AttentionBatch.single(readonly)), one
        )
        torch.testing.assert_close(first.state("attention").value[0, 0], one)
        torch.testing.assert_close(second.state("attention").value[0, 0], two)


def test_shared_weight_call_sites_keep_distinct_activation_representations():
    quantizer = Quantizer("fp8", axis=0)
    first, second = Linear(8, 4, bias=False), Linear(8, 4, bias=False)
    first.weight = nn.Parameter(
        quantizer.quantize(first.weight), requires_grad=False
    )
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
    layer.weight = nn.Parameter(
        quantizer.quantize(layer.weight), requires_grad=False
    )
    layer.input_quantizer = quantizer
    generator = torch.Generator(device=device).manual_seed(17)
    x = torch.randn(
        5, 64, dtype=torch.bfloat16, device=device, generator=generator
    )
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
        torch.randn(5, 1, 64, dtype=dtype, device=device, generator=generator)
        for _ in range(2)
    )
    with (
        PrefixCache(config, num_units=2, block_size=16, device=device) as cache,
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
            actual = layer(q, k, v, AttentionBatch.single(batch))
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
            torch.testing.assert_close(
                actual, torch.cat(expected), rtol=2e-2, atol=2e-2
            )


@pytest.mark.gpu
@torch.inference_mode()
@pytest.mark.parametrize("provider", ("torch", "flashinfer"))
def test_bound_metadata_serves_one_call_and_later_changes_are_planned(provider):
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    layer = Attention(2, 1, 64, cache_name="attention")
    config = Config({"attention": mha.Config(1, 64, (0,), dtype)})
    generator = torch.Generator(device=device).manual_seed(31)
    q = torch.randn(5, 2, 64, dtype=dtype, device=device, generator=generator)
    k, v = (
        torch.randn(5, 1, 64, dtype=dtype, device=device, generator=generator)
        for _ in range(2)
    )

    def lengths(counts):
        return PagedInput.from_blocks(
            blocks=((0,), (1,)),
            query_lengths=counts,
            prefix_lengths=(0, 0),
            block_size=16,
            causal=True,
            device=device,
        )

    def expected(counts):
        return torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(
                    query.transpose(0, 1).unsqueeze(0),
                    key.transpose(0, 1).unsqueeze(0),
                    value.transpose(0, 1).unsqueeze(0),
                    is_causal=True,
                    enable_gqa=True,
                )
                .squeeze(0)
                .transpose(0, 1)
                for query, key, value in zip(
                    q.split(counts),
                    k.split(counts),
                    v.split(counts),
                    strict=True,
                )
            ]
        )

    # Device-only lengths: every plan reads the live columns.
    initial = lengths((4, 1))
    batch = replace(
        initial,
        queries=replace(initial.queries, host=None),
        prefixes=replace(initial.prefixes, host=None),
    )
    with (
        PrefixCache(config, num_units=2, block_size=16, device=device) as cache,
        ExecutionContext(layer, cache=cache, attention=provider) as context,
    ):
        context.prepare(TextSize(5, 2))
        context.bind_attention(AttentionBatch.single(batch))
        torch.testing.assert_close(
            layer(q, k, v, AttentionBatch.single(batch)),
            expected((4, 1)),
            rtol=2e-2,
            atol=2e-2,
        )
        for counts in ((2, 3), (1, 4)):
            # Columns change in place without another bind; the next call
            # plans from the live lengths instead of the bound ones.
            changed = lengths(counts)
            batch.queries.values.copy_(changed.queries.values)
            batch.queries.offsets.copy_(changed.queries.offsets)
            batch.write_indices.copy_(changed.write_indices)
            torch.testing.assert_close(
                layer(q, k, v, AttentionBatch.single(batch)),
                expected(counts),
                rtol=2e-2,
                atol=2e-2,
            )


@pytest.mark.gpu
@torch.inference_mode()
def test_context_without_derived_lengths_plans_only_with_host_mirrors():
    """A context that reads no lengths from the device uses the batch's own.

    Planning with the host lengths a batch carries serves its calls, while a
    batch whose lengths exist only on the device fails to bind and to run
    rather than copying them to the host.
    """
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    layer = Attention(2, 1, 64, cache_name="attention")
    config = Config({"attention": mha.Config(1, 64, (0,), dtype)})
    generator = torch.Generator(device=device).manual_seed(37)
    q = torch.randn(5, 2, 64, dtype=dtype, device=device, generator=generator)
    k, v = (
        torch.randn(5, 1, 64, dtype=dtype, device=device, generator=generator)
        for _ in range(2)
    )
    batch = PagedInput.from_blocks(
        blocks=((0,), (1,)),
        query_lengths=(4, 1),
        prefix_lengths=(0, 0),
        block_size=16,
        causal=True,
        device=device,
    )
    stripped = replace(
        batch,
        queries=replace(batch.queries, host=None),
        prefixes=replace(batch.prefixes, host=None),
    )
    expected = torch.cat(
        [
            F.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                is_causal=True,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            for query, key, value in zip(
                q.split((4, 1)), k.split((4, 1)), v.split((4, 1)), strict=True
            )
        ]
    )
    with (
        PrefixCache(config, num_units=2, block_size=16, device=device) as cache,
        ExecutionContext(
            layer, cache=cache, attention="torch", derive_host_lengths=False
        ) as context,
    ):
        context.prepare(TextSize(5, 2))
        context.bind_attention(AttentionBatch.single(batch))
        torch.testing.assert_close(
            layer(q, k, v, AttentionBatch.single(batch)),
            expected,
            rtol=2e-2,
            atol=2e-2,
        )
        with pytest.raises(ValueError, match="host query lengths"):
            context.bind_attention(AttentionBatch.single(stripped))
        with pytest.raises(ValueError, match="host query lengths"):
            layer(q, k, v, AttentionBatch.single(stripped))


class _Workspace(nn.Module):
    """Declare one large workspace buffer and no numerical layers."""

    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(1, device="cuda:0"))

    def workspace_buffers(self, size):
        del size
        return {"scratch": BufferConfig((16 << 20,), torch.uint8)}


@pytest.mark.gpu
@torch.inference_mode()
@pytest.mark.parametrize("pending", (False, True))
def test_caller_exception_releases_backing_only_after_its_work_finished(
    pending,
):
    device = torch.device("cuda:0")
    module = _Workspace()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)

    with CUDAStream.external(torch.cuda.Stream(device=device)) as stream:
        with pytest.raises(ValueError, match="caller"):
            with ExecutionContext(module, stream=stream) as context:
                context.prepare(None)
                context.workspace["scratch"].fill_(1)
                if pending:
                    # Keep the stream busy past the exit decision.
                    torch.cuda._sleep(1 << 30)
                else:
                    stream.synchronize()
                raise ValueError("caller rejected its own input")
        stream.synchronize()

    retained = torch.cuda.memory_allocated(device) - baseline
    # Released backing returns to the allocator; storage an unfinished access
    # may still read stays owned until process exit.
    assert retained >= (16 << 20) if pending else retained == 0


@pytest.mark.gpu
@torch.inference_mode()
def test_contexts_sharing_scratch_replay_their_own_values():
    """Contexts on one stream borrow one ``Scratch`` across sizes and graphs.

    The merged projection stages its branches in a scratch output area. The
    smaller context captures first, so the larger one grows the shared area
    after that capture; both graphs then replay alternately with new inputs
    and each matches eager evaluation of its own input.
    """
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(29)
    layer = MergedColumnParallelLinear(
        128,
        {"gate": 64, "up": 64},
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    )
    for branch in layer.projections.values():
        branch.weight.copy_(
            torch.randn(
                branch.weight.shape,
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            / 16
        )

    def reference(x):
        return {
            name: (x.float() @ branch.weight.float().T).bfloat16()
            for name, branch in layer.projections.items()
        }

    sizes = (16, 96)
    inputs = {
        rows: torch.randn(
            rows, 128, device=device, dtype=torch.bfloat16, generator=generator
        )
        for rows in sizes
    }
    outputs = {
        rows: {
            name: torch.empty(rows, 64, device=device, dtype=torch.bfloat16)
            for name in layer.projections
        }
        for rows in sizes
    }
    scratch = Scratch()
    torch.cuda.synchronize(device)
    with (
        CUDAStream.external(torch.cuda.Stream(device=device)) as stream,
        ExitStack() as scope,
    ):
        graphs = {}
        for rows in sizes:
            context = scope.enter_context(
                ExecutionContext(layer, stream=stream, scratch=scratch)
            )
            context.prepare(TextSize(rows, 1))
            graph = scope.enter_context(CUDAGraph(context=context))
            with context.activate():
                layer(inputs[rows], out=outputs[rows])
            graph.capture(
                lambda rows=rows: layer(inputs[rows], out=outputs[rows])
            )
            graphs[rows] = graph

        # FP32 accumulation over K=128, rounded once to BF16.
        for _ in range(2):
            for rows in (*sizes, sizes[0]):
                inputs[rows].copy_(
                    torch.randn(
                        rows,
                        128,
                        device=device,
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                )
                # Inputs are written on the default stream; the graph
                # replays on the context's.
                torch.cuda.synchronize(device)
                graphs[rows].replay()
                torch.cuda.synchronize(device)
                for name, value in reference(inputs[rows]).items():
                    torch.testing.assert_close(
                        outputs[rows][name], value, rtol=2**-7, atol=2**-10
                    )
    scratch.close()


def _reference(query, key, value, allowed):
    """Grouped-query attention over ``[tokens, heads, dim]`` with a mask."""
    return (
        F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            attn_mask=allowed,
            enable_gqa=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )


@torch.inference_mode()
def test_table_batches_plan_cached_layers_beside_cacheless_layers():
    # A windowed and a full-attention layer read two cache tables of one
    # unit pool; a layer without a cache table reads its own single-entry
    # batch. Planning the table batch must leave that layer alone.
    generator = torch.Generator().manual_seed(47)
    windowed = Attention(2, 1, 4, cache_name="windowed", window=1)
    full = Attention(2, 1, 8, cache_name="full")
    dense = Attention(2, 2, 4)
    module = nn.ModuleDict({"windowed": windowed, "full": full, "dense": dense})
    config = Config(
        {
            "windowed": mha.Config(1, 4, (0,), torch.float32, window=1),
            "full": mha.Config(1, 8, (0,), torch.float32),
        }
    )
    with (
        PrefixCache(config, num_units=4, block_size=2, device="cpu") as cache,
        ExecutionContext(module, cache=cache, attention="torch") as context,
    ):
        context.prepare(TextSize(3, 1))
        # The full rows are the widest: two-token full pages, four-token
        # windowed pages, and disjoint units for the two groups' tables.
        assert (cache.table("windowed"), cache.table("full")) == (0, 1)
        first = PagedInput.from_blocks(
            blocks=((1,),),
            query_lengths=(3,),
            prefix_lengths=(0,),
            block_size=4,
            causal=True,
            device="cpu",
        )
        second = PagedInput.from_blocks(
            blocks=((2, 3),),
            query_lengths=(3,),
            prefix_lengths=(0,),
            block_size=2,
            causal=True,
            device="cpu",
        )
        batch = AttentionBatch(
            {
                0: first,
                1: replace(
                    second, queries=first.queries, prefixes=first.prefixes
                ),
            },
            first.queries,
        )
        context.bind_attention(batch)

        positions = torch.arange(3)
        causal = positions[None] <= positions[:, None]
        for layer, width, allowed in (
            (windowed, 4, causal & (positions[None] >= positions[:, None] - 1)),
            (full, 8, causal),
        ):
            query = torch.randn(3, 2, width, generator=generator)
            key, value = (
                torch.randn(3, 1, width, generator=generator) for _ in range(2)
            )
            out = torch.empty_like(query)
            layer(query, key, value, batch, out=out)
            torch.testing.assert_close(
                out, _reference(query, key, value, allowed)
            )

        single = AttentionBatch.single(DenseInput(causal=False, mask=None))
        context.bind_attention(single)
        query = torch.randn(3, 2, 4, generator=generator)
        key, value = (
            torch.randn(3, 2, 4, generator=generator) for _ in range(2)
        )
        out = torch.empty_like(query)
        dense(query, key, value, single, out=out)
        torch.testing.assert_close(out, _reference(query, key, value, None))
