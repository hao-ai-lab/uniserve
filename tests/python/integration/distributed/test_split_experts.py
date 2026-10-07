"""Quantized split experts preserve encodings and weight completed routes."""

import argparse
import os
import socket
from contextlib import ExitStack
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.distributed import Communicator, partition_experts
from uniserve.model import TextSize
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import Quantizer
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    Microbatches,
)
from uniserve.runtime.expert_exchange import ExpertExchange
from uniserve.runtime.process_groups import initialize_process_groups

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def _exercise(
    rank,
    local_rank,
    world_size,
    attention_ranks,
    init_method,
    format,
    width,
    intermediate,
):
    device = torch.device("cuda", local_rank)
    capacity, top_k = 32, 2
    expert_ranks = tuple(range(attention_ranks, world_size))
    experts = 2 * len(expert_ranks)
    source = rank < attention_ranks
    with initialize_process_groups(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
    ) as groups:
        activation = (
            Quantizer(format, calibrated_scale=1.0)
            if format == "nvfp4"
            else Quantizer(format)
        )
        model = torch.nn.ModuleList(
            FusedMoE(
                experts,
                width,
                intermediate,
                top_k=top_k,
                activation="silu",
                device="meta",
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        )
        if not source:
            partition_experts(
                model,
                Communicator(
                    expert_ranks, rank - attention_ranks, "experts", device
                ),
            )
        for index, layer in enumerate(model):
            layer.up_gate.input_quantizer = activation
            layer.down.input_quantizer = activation
            if source:
                continue
            resident = layer.up_gate.num_experts
            up_gate = torch.zeros(
                (resident, 2 * intermediate, width),
                device=device,
                dtype=torch.bfloat16,
            )
            down = torch.zeros(
                (resident, width, intermediate),
                device=device,
                dtype=torch.bfloat16,
            )
            rows = torch.arange(width, device=device)
            up_gate[:, rows, rows] = 1
            # x[0]=6 and x[1]=2 make every gate 32 (64 at layer 1),
            # where SiLU equals the identity in FP32. All intermediate
            # blocks and both weight encodings are exact powers of two.
            up_gate[:, intermediate:, :2] = 4 * 2**index
            for expert in range(resident):
                global_expert = layer.expert_slice.start + expert
                down[expert, rows, rows] = (global_expert % 2 + 1) / 8
            for linear, value in ((layer.up_gate, up_gate), (layer.down, down)):
                linear.weight = torch.nn.Parameter(
                    Quantizer(format).quantize(value), requires_grad=False
                )

        exchanges = [
            ExpertExchange(
                groups.process_group,
                max_tokens=capacity,
                top_k=top_k,
                num_experts=experts,
                hidden_size=width,
                device=device,
                transport="megamoe",
                attention_ranks=attention_ranks,
                intermediate_size=intermediate,
                activation="silu",
            )
            for _ in range(2)
        ]
        with ExitStack() as scope:
            contexts = []
            for exchange in exchanges:
                stream = scope.enter_context(
                    CUDAStream.external(torch.cuda.Stream(device=device))
                )
                stream.wait(torch.cuda.current_stream(device))
                context = scope.enter_context(
                    ExecutionContext(model, stream=stream, experts=exchange)
                )
                context.prepare(TextSize(capacity, 1))
                contexts.append(context)
            run = Microbatches(contexts)
            scope.callback(run.close)
            for empty in (False, True):
                count = 0 if not source or empty and rank == 0 else 5 + rank % 3
                magnitudes = torch.tensor(
                    (0.5, 1, 1.5, 2, 3, 4, 6), device=device
                )
                hidden = (
                    magnitudes[torch.arange(count * width, device=device) % 7]
                    .reshape(count, width)
                    .bfloat16()
                )
                hidden[:, ::16] = 6
                hidden[:, 1] = 2
                ids = torch.arange(
                    count * top_k, device=device, dtype=torch.int32
                ).reshape(count, top_k)
                ids.add_(2 * rank).remainder_(experts)
                weights = (
                    torch.tensor((0.3, 0.7), device=device)
                    .expand(count, top_k)
                    .contiguous()
                )

                def call(lane):
                    exchange, context = exchanges[lane], contexts[lane]
                    exchange.begin(capacity)
                    try:
                        output = (
                            torch.stack(
                                [
                                    layer(hidden, ids, weights * 2**lane)
                                    for layer in model
                                ]
                            )
                            if source
                            else hidden.expand(len(model), -1, -1)
                        )
                        context.join_expert_layers()
                        return output
                    finally:
                        exchange.end()

                def compare(outputs):
                    # The exact routed equation; six BF16 roundoffs cover
                    # projections, remote weighted rows and combine, matching
                    # the distributed grouped-expert numerical contract.
                    expected = torch.stack(
                        [
                            hidden.double()
                            * (32 * 2**layer)
                            * (
                                weights[:, 0, None].double() / 8
                                + weights[:, 1, None].double() / 4
                            )
                            for layer in range(len(model))
                        ]
                    )
                    for lane, output in enumerate(outputs):
                        torch.testing.assert_close(
                            output.double(),
                            expected * 2**lane,
                            rtol=6 / 250,
                            atol=0,
                        )

                calls = [partial(call, lane) for lane in range(2)]
                outputs = run(calls)
                torch.cuda.synchronize(device)
                compare(outputs)
                with CUDAGraph(context=contexts[0]) as graph:
                    graph.capture(lambda: run(calls))
                    for _ in range(4):
                        outputs = graph.replay()
                        contexts[0].stream.synchronize()
                        compare(outputs)
        for exchange in exchanges:
            exchange.close()


def _local(rank, port, format, width, intermediate):
    _exercise(
        rank, rank, 2, 1, f"tcp://127.0.0.1:{port}", format, width, intermediate
    )


@pytest.mark.parametrize("format", ("mxfp8", "nvfp4"))
@pytest.mark.parametrize("width,intermediate", ((512, 512), (384, 768)))
def test_split_experts_preserve_quantization_and_routes(
    format, width, intermediate
):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    if any(
        torch.cuda.get_device_capability(i) not in {(10, 0), (10, 3)}
        for i in range(2)
    ):
        pytest.skip("requires two Blackwell CUDA devices")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _local, args=(port, format, width, intermediate), nprocs=2, join=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-ranks", type=int, required=True)
    parser.add_argument("--format", choices=("mxfp8", "nvfp4"), required=True)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--intermediate", type=int, default=768)
    arguments = parser.parse_args()
    _exercise(
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
        arguments.attention_ranks,
        "env://",
        arguments.format,
        arguments.width,
        arguments.intermediate,
    )
