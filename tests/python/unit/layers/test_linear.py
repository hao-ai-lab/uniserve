"""Projection values follow logical branches, statistics, and output storage."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.nn.linear import (
    ColumnParallelLinear,
    Linear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.quantization import Quantizer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "dtype,large", ((torch.float16, 2048), (torch.bfloat16, 256))
)
@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda", marks=pytest.mark.gpu))
)
@torch.inference_mode()
def test_unsharded_row_projection_preserves_dense_affine_rounding(
    dtype, large, device
):
    from uniserve.model import TextSize
    from uniserve.runtime import ExecutionContext

    layer = RowParallelLinear(2, 1, device=device, dtype=dtype)
    layer.weight.copy_(torch.tensor([[large, 1]], device=device, dtype=dtype))
    layer.bias.fill_(-large)
    source = torch.tensor([[1, 1], [1, -1]], device=device, dtype=dtype)
    # Rounding the product before adding bias loses the unit contribution.
    # A single shard is the ordinary dense affine map, with one final store.
    expected = torch.tensor([[1], [-1]], device=device, dtype=dtype)
    with ExecutionContext(layer) as context:
        context.prepare(TextSize(2, 1))
        torch.testing.assert_close(layer(source), expected, rtol=0, atol=0)
        out = torch.empty(2, 2, device=device, dtype=dtype)[:, :1]
        assert layer(source, out=out) is out
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "quantizer",
    (Quantizer("fp8"), pytest.param(Quantizer("nvfp4"), marks=pytest.mark.gpu)),
)
@torch.inference_mode()
def test_encoded_projection_chunks_preserve_the_source_scale_domain(quantizer):
    device = "cpu" if quantizer.format == "fp8" else "cuda"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    source = (
        torch.arange(160, dtype=torch.float32, device=device)
        .reshape(5, 32)
        .sin()
        .to(dtype)
    )
    # Deliberately retain a source domain wider than this local value range.
    # Concatenating through dequantization and fresh statistics changes it.
    encoded = quantizer.quantize(
        source, amax=torch.tensor(3.718, device=device)
    )
    layer = ColumnParallelLinear(32, 32, bias=False, device=device, dtype=dtype)
    if device == "cuda":
        layer.weight = nn.Parameter(
            quantizer.quantize(layer.weight), requires_grad=False
        )
    layer.input_quantizer = quantizer
    expected = layer(encoded)

    def chunks():
        for start, stop in ((0, 2), (2, 5)):
            fields = {
                name: value[start:stop]
                if name in {"values", "block_scale"}
                else value.clone()
                for name, value in encoded.buffers().items()
            }
            yield (
                slice(start, stop),
                quantizer.from_tensors(
                    fields, shape=(stop - start, 32), dtype=dtype
                ),
            )

    result = torch.empty_like(expected)
    for interval, value in layer.forward_chunks(
        chunks(), token_slice=slice(0, 5), num_tokens=5
    ):
        result[interval].copy_(value)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("axis", [None, 0])
def test_encoded_input_preserves_scales_shape_and_bias(axis):
    quantizer = Quantizer("fp8", axis=axis)
    layer = Linear(2, 2)
    weights = torch.tensor([[448.0, 2.0], [-4.0, 448.0]])
    layer.weight = nn.Parameter(
        quantizer.quantize(weights), requires_grad=False
    )
    layer.bias.copy_(torch.tensor([2.0, -4.0]))
    layer.input_quantizer = quantizer
    source = torch.tensor([[[1.0, 2.0], [-2.0, 4.0]]])
    encoded = quantizer.quantize(source.reshape(2, 2))
    expected = F.linear(
        encoded.dequantize(), layer.weight.dequantize(), layer.bias
    ).reshape(1, 2, 2)
    out = torch.empty(1, 2, 2).transpose(-1, -2)
    assert layer(source, output_dtype=torch.float32, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        layer(encoded), expected.reshape(2, 2), rtol=0, atol=0
    )


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 3])
def test_named_branches_preserve_independent_weights_and_bias(quantized, rows):
    layer = MergedColumnParallelLinear(2, {"q": 2, "k": 1, "v": 1})
    weights = (
        torch.tensor([[448.0, 0.0], [0.0, 448.0]]),
        torch.tensor([[224.0, 224.0]]),
        torch.tensor([[112.0, -112.0]]),
    )
    source = torch.tensor([[224.0, 448.0]]).expand(rows, 2)
    for index, (branch, weight) in enumerate(
        zip(layer.projections.values(), weights, strict=True)
    ):
        branch.weight = nn.Parameter(
            Quantizer("fp8").quantize(weight) if quantized else weight,
            requires_grad=False,
        )
        branch.bias.fill_(index + 1)
        branch.input_quantizer = Quantizer("fp8", axis=0) if quantized else None
    result = layer(source)
    for index, (name, weight) in enumerate(
        zip(layer.projections, weights, strict=True)
    ):
        torch.testing.assert_close(
            result[name], F.linear(source, weight) + index + 1, rtol=0, atol=0
        )


@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (4, 1), (8, 8)])
def test_query_shards_select_their_grouped_kv_heads(heads, kv_heads):
    head_dim, width = 4, 8
    weights = tuple(
        torch.arange(count * head_dim * width, dtype=torch.float32).view(
            -1, width
        )
        for count in (heads, kv_heads, kv_heads)
    )
    values = torch.eye(width)[:3]
    ranks = (3, 1, 0, 2)
    for owner, physical in enumerate(ranks):
        layer = QKVParallelLinear(width, heads, kv_heads, head_dim, bias=False)
        for branch, weight in zip(
            layer.projections.values(), weights, strict=True
        ):
            branch.weight.copy_(weight)
        parallelize_(
            layer,
            DeviceMesh(ranks=ranks, shape=(4,), axes=("tp",), rank=physical),
        )
        queries = range(owner * heads // 4, (owner + 1) * heads // 4)
        keys = sorted({head // (heads // kv_heads) for head in queries})
        actual = layer(values)
        for name, selected, weight in zip(
            ("q", "k", "v"), (queries, keys, keys), weights, strict=True
        ):
            columns = [
                head * head_dim + dim
                for head in selected
                for dim in range(head_dim)
            ]
            torch.testing.assert_close(
                actual[name],
                F.linear(values, weight)[:, columns],
                rtol=0,
                atol=0,
            )


def test_shared_parameters_survive_parallel_binding():
    modules = nn.ModuleList([Linear(4, 4), Linear(4, 4)])
    modules[1].weight = modules[0].weight
    parallelize_(modules, DeviceMesh(ranks=(0,), shape=(), axes=(), rank=0))
    assert modules[0].weight is modules[1].weight
    modules[0].weight.fill_(3)
    torch.testing.assert_close(
        modules[0](torch.ones(1, 4)) - modules[0].bias,
        modules[1](torch.ones(1, 4)) - modules[1].bias,
    )


@pytest.mark.parametrize("shape", [(0, 7), (0, 0), ()])
def test_invalid_contraction_width_is_rejected_for_empty_inputs(shape):
    layer = Linear(8, 4)
    with pytest.raises(ValueError, match="dimension"):
        layer(torch.empty(shape))


def test_merged_projection_rejects_bias_outside_its_branch_channels():
    from uniserve.nn.functional import merged_linear

    with pytest.raises(ValueError, match="bias.*channels"):
        merged_linear(
            torch.ones(3, 4),
            {"first": torch.ones(2, 4), "second": torch.ones(3, 4)},
            {"first": torch.ones(1), "second": None},
        )


def test_gated_mlp_uses_named_gate_and_up_branches():
    model = GatedMLP(4, 8, bias=True)
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 16
    gate = model.gate_up.projections["gate"]
    up = model.gate_up.projections["up"]
    expected = F.linear(
        F.silu(F.linear(source, gate.weight, gate.bias))
        * F.linear(source, up.weight, up.bias),
        model.down.weight,
        model.down.bias,
    )
    torch.testing.assert_close(model(source), expected)


def test_row_chunks_keep_one_tensor_scale_across_intervals():
    layer = RowParallelLinear(2, 2)
    layer.input_quantizer = Quantizer("fp8")
    layer.weight = nn.Parameter(
        Quantizer("fp8").quantize(torch.eye(2)), requires_grad=False
    )
    source = torch.tensor([[1.1, 2.3], [100.0, 200.0], [3.2, 4.7]])
    actual = list(
        layer.forward_chunks(
            iter(((slice(0, 1), source[:1]), (slice(1, 3), source[1:])))
        )
    )
    torch.testing.assert_close(
        torch.cat([value for _, value in actual]), layer(source), rtol=0, atol=0
    )
