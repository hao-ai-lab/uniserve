"""Public projections preserve TP reductions.

They also preserve complete token scale domains.
"""

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.model import TextSize
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.nn.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from uniserve.quantization import Quantizer
from uniserve.runtime import ExecutionContext, initialize_process_groups

pytestmark = pytest.mark.integration


def _run(rank, rendezvous):
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device="cpu",
        init_method=rendezvous,
    ) as owner:
        tensor_mesh = owner.bind(
            DeviceMesh(ranks=(3, 1, 0, 2), shape=(4,), axes=("tp",), rank=rank),
            device="cpu",
        )
        group = tensor_mesh.get_group("tp")
        weight = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 16
        x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 8
        x *= torch.tensor([1, 1, 10, 10, 100, 100, 1000, 1000])
        for quantizer in (None, Quantizer("fp8"), Quantizer("fp8", axis=0)):
            layer = RowParallelLinear(8, 4)
            layer.weight.copy_(weight)
            layer.bias.copy_(torch.arange(4))
            parallelize_(layer, tensor_mesh)
            layer.input_quantizer = quantizer
            if quantizer is not None:
                layer.weight = nn.Parameter(
                    quantizer.quantize(
                        layer.weight, distribution=layer.weight_distribution
                    ),
                    requires_grad=False,
                )
                expected = F.linear(
                    quantizer.quantize(x).dequantize(),
                    quantizer.quantize(weight).dequantize(),
                    layer.bias,
                )
            else:
                expected = F.linear(x, weight, layer.bias)
            with ExecutionContext(layer, matmul="torch") as context:
                context.prepare(TextSize(x.shape[0] + 1, 1))
                torch.testing.assert_close(
                    layer(x.chunk(4, dim=1)[group.rank]), expected
                )

        qkv = QKVParallelLinear(8, 8, 2, 4, bias=False)
        for index, branch in enumerate(qkv.projections.values()):
            branch.weight.copy_(
                torch.arange(branch.weight.numel()).reshape_as(branch.weight)
                + index
            )
        full = qkv(x)
        parallelize_(qkv, tensor_mesh)
        for branch in qkv.projections.values():
            maximum = Quantizer("fp8").amax(
                branch.weight, distribution=branch.weight_distribution
            )
            expected_max = (
                255
                if branch is qkv.projections["q"]
                else 64
                if branch is qkv.projections["k"]
                else 65
            )
            torch.testing.assert_close(
                maximum, torch.tensor(float(expected_max)), rtol=0, atol=0
            )
        actual = qkv(x)
        torch.testing.assert_close(
            actual["q"], full["q"].chunk(4, dim=-1)[group.rank]
        )
        for name in ("k", "v"):
            torch.testing.assert_close(
                actual[name], full[name].chunk(2, dim=-1)[group.rank // 2]
            )

        token_mesh = owner.bind(
            DeviceMesh(
                ranks=(3, 1, 0, 2), shape=(4,), axes=("tokens",), rank=rank
            ),
            device="cpu",
        )
        tokens = token_mesh.get_group("tokens")
        for count in (0, 2, 7, 521):
            source = (
                torch.arange(count * 8, dtype=torch.float32)
                .reshape(count, 8)
                .square()
                / 17
            )
            capacity = (count + 3) // 4
            start = min(count, tokens.rank * capacity)
            stop = min(count, start + capacity)
            for quantizer in (None, Quantizer("fp8"), Quantizer("fp8", axis=0)):
                layer = ColumnParallelLinear(8, 4, bias=False)
                layer.weight.copy_(weight)
                if quantizer is not None:
                    layer.weight = nn.Parameter(
                        quantizer.quantize(weight), requires_grad=False
                    )
                # Binding precedes encoding; the weight in this token-only
                # mesh remains replicated, so load it after numerical binding.
                encoded_weight = layer.weight
                layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
                parallelize_(
                    layer,
                    token_mesh,
                    attention=AttentionParallelConfig(heads=Ulysses("tokens")),
                )
                layer.weight = encoded_weight
                layer.input_quantizer = quantizer
                expected_input = (
                    source
                    if quantizer is None
                    else quantizer.quantize(source).dequantize()
                )
                expected_weight = (
                    weight if quantizer is None else encoded_weight.dequantize()
                )
                expected = F.linear(expected_input, expected_weight)
                result = torch.empty_like(expected)
                with ExecutionContext(layer, matmul="torch") as context:
                    context.prepare(TextSize(count + 1, 1))
                    for streamed in (False, True):
                        local = source[start:stop]
                        # Different producer boundaries on each rank must still
                        # result in the same collective payload boundaries.
                        chunk_size = tokens.rank + 1
                        inputs = (
                            (
                                (
                                    slice(begin, min(begin + chunk_size, stop)),
                                    source[
                                        begin : min(begin + chunk_size, stop)
                                    ],
                                )
                                for begin in range(start, stop, chunk_size)
                            )
                            if streamed
                            else local
                        )
                        for interval, values in layer.forward_chunks(
                            inputs,
                            token_slice=slice(start, stop),
                            num_tokens=count,
                        ):
                            result[interval].copy_(values)
                        torch.testing.assert_close(result, expected)

                torch.testing.assert_close(result, expected)

        count = 7
        source = (
            torch.arange(count * 8, dtype=torch.float32)
            .reshape(count, 8)
            .square()
            / 17
        )
        capacity = (count + tokens.size - 1) // tokens.size
        start, stop = (
            tokens.rank * capacity,
            min(count, (tokens.rank + 1) * capacity),
        )
        merged = MergedColumnParallelLinear(
            8, {"dense": 4, "tensor": 4, "rows": 4}, bias=False
        )
        parallelize_(
            merged,
            token_mesh,
            attention=AttentionParallelConfig(heads=Ulysses("tokens")),
        )
        expected = {}
        for (name, branch), quantizer in zip(
            merged.projections.items(),
            (None, Quantizer("fp8"), Quantizer("fp8", axis=0)),
            strict=True,
        ):
            branch.weight.copy_(weight)
            branch.input_quantizer = quantizer
            values = (
                source
                if quantizer is None
                else quantizer.quantize(source).dequantize()
            )
            expected[name] = F.linear(values, weight)
        with ExecutionContext(merged, matmul="torch") as context:
            context.prepare(TextSize(count, 1))
            for inputs in (
                source[start:stop],
                iter(((slice(start, stop), source[start:stop]),)),
            ):
                for interval, values in merged.forward_chunks(
                    inputs, token_slice=slice(start, stop), num_tokens=count
                ):
                    for name in merged.projections:
                        torch.testing.assert_close(
                            values[name], expected[name][interval]
                        )


def test_parallel_projection_values(tmp_path):
    mp.spawn(_run, args=((tmp_path / "linear").as_uri(),), nprocs=4, join=True)
