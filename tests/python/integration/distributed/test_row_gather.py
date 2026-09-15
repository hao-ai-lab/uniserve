"""Streamed public projections retain global values across graph replay."""

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.model import TextSize
from uniserve.nn import ColumnParallelLinear
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.quantization import Quantizer
from uniserve.runtime import CUDAGraph, ExecutionContext, initialize_process_groups

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def _run(rank, rendezvous):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank, local_rank=rank, world_size=2, device=device, init_method=rendezvous
    ) as groups:
        mesh = groups.bind(
            DeviceMesh(ranks=(1, 0), shape=(2,), axes=("tokens",), rank=rank), device=device
        )
        group = mesh.get_group("tokens")
        source = torch.arange(257 * 128, device=device).reshape(257, 128).float().sin().bfloat16()
        weight = torch.arange(64 * 128, device=device).reshape(64, 128).float().cos().bfloat16()
        interval = slice(group.rank * 129, min((group.rank + 1) * 129, 257))
        for quantizer in (None, Quantizer("fp8"), Quantizer("fp8", axis=0)):
            layer = ColumnParallelLinear(128, 64, bias=False, device=device, dtype=torch.bfloat16)
            parallelize_(layer, mesh, attention=AttentionParallelConfig(heads=Ulysses("tokens")))
            layer.weight = nn.Parameter(
                weight.clone() if quantizer is None else quantizer.quantize(weight),
                requires_grad=False,
            )
            layer.input_quantizer = quantizer
            result = torch.empty(257, 64, device=device, dtype=torch.bfloat16)
            nested_result = torch.empty_like(result)
            local = source[interval].clone()
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with ExecutionContext(layer, stream=stream) as context:
                context.prepare(TextSize(0, 0))
                empty = torch.cat(
                    [
                        value
                        for _, value in layer.forward_chunks(
                            source[:0], token_slice=slice(0, 0), num_tokens=0
                        )
                    ]
                )
                torch.testing.assert_close(empty, result[:0], rtol=0, atol=0)
                context.prepare(TextSize(257, 1))

                if quantizer is None:
                    # Closing after a local result must retire peer writers
                    # before a later projection reuses its transport storage.
                    cancelled = layer.forward_chunks(local, token_slice=interval, num_tokens=257)
                    selected, value = next(cancelled)
                    torch.testing.assert_close(
                        value,
                        F.linear(source[selected].float(), weight.float()).bfloat16(),
                        rtol=2e-2,
                        atol=2e-2,
                    )
                    cancelled.close()
                    local.neg_()
                    for selected, value in layer.forward_chunks(
                        local, token_slice=interval, num_tokens=257
                    ):
                        torch.testing.assert_close(
                            value,
                            -F.linear(source[selected].float(), weight.float()).bfloat16(),
                            rtol=2e-2,
                            atol=2e-2,
                        )
                    local.copy_(source[interval])

                def invoke():
                    chunk_size = 37 + group.rank
                    chunks = (
                        (
                            slice(
                                interval.start + start,
                                interval.start + min(start + chunk_size, local.shape[0]),
                            ),
                            local[start : start + chunk_size],
                        )
                        for start in range(0, local.shape[0], chunk_size)
                    )
                    for selected, value in layer.forward_chunks(
                        chunks, token_slice=interval, num_tokens=257
                    ):
                        result[selected].copy_(value)
                        if quantizer is None:
                            # The outer iterator still holds remote input rows.
                            # A nested projection must not overwrite those rows.
                            for nested_slice, nested_value in layer.forward_chunks(
                                -local, token_slice=interval, num_tokens=257
                            ):
                                nested_result[nested_slice].copy_(nested_value)
                    return result

                invoke()
                with CUDAGraph(context=context) as graph:
                    graph.capture(invoke)
                    for multiplier in (1.0, 3.0):
                        local.copy_(source[interval] * multiplier)
                        actual = graph.replay()
                        inputs = source * multiplier
                        if quantizer is not None:
                            # Native scaled GEMM accumulates encoded values
                            # before its output conversion. BF16 dequantization
                            # would add an operand rounding absent from that math.
                            inputs = quantizer.quantize(inputs).dequantize(dtype=torch.float32)
                            matrix = layer.weight.dequantize(dtype=torch.float32)
                        else:
                            matrix = weight
                        expected = F.linear(inputs.float(), matrix.float()).bfloat16()
                        torch.testing.assert_close(
                            actual,
                            expected,
                            rtol=2e-2,
                            atol=2e-2,
                            msg=lambda message: f"{quantizer=}, {multiplier=}: {message}",
                        )
                        if quantizer is None:
                            torch.testing.assert_close(
                                nested_result, -expected, rtol=2e-2, atol=2e-2
                            )
                torch.cuda.synchronize(device)
                # Re-preparation retires the captured exchange resources. New
                # storage and communicator registrations must accept live values.
                context.prepare(TextSize(385, 1))
                torch.testing.assert_close(invoke(), expected, rtol=2e-2, atol=2e-2)
                torch.cuda.synchronize(device)


def test_streamed_projection_replay_preserves_complete_scale_domains(tmp_path):
    mp.spawn(_run, args=((tmp_path / "projection").as_uri(),), nprocs=2, join=True)
