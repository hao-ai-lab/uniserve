"""Expert modules preserve their equation across disaggregated ranks."""

import argparse
import os
import socket
from contextlib import ExitStack
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from uniserve.distributed import Communicator, partition_experts
from uniserve.model import TextSize
from uniserve.nn.moe import FusedMoE
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
    rank, local_rank, world_size, attention_ranks, init_method, microbatches
):
    device = torch.device("cuda", local_rank)
    hidden_size, intermediate, top_k, capacity = 256, 128, 2, 32
    expert_ranks = tuple(range(attention_ranks, world_size))
    num_experts = 2 * len(expert_ranks)
    source = rank < attention_ranks
    with initialize_process_groups(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
    ) as groups:
        generator = torch.Generator().manual_seed(418)
        # Positive operands give a relative forward-error bound without
        # cancellation; weights are rounded before forming the FP64 reference.
        up_gate = (
            (
                torch.rand(
                    num_experts,
                    2 * intermediate,
                    hidden_size,
                    generator=generator,
                )
                + 0.125
            )
            / hidden_size
        ).to(torch.bfloat16)
        down = (
            (
                torch.rand(
                    num_experts, hidden_size, intermediate, generator=generator
                )
                + 0.125
            )
            / intermediate
        ).to(torch.bfloat16)
        module = torch.nn.ModuleList(
            FusedMoE(
                num_experts,
                hidden_size,
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
                module,
                Communicator(
                    expert_ranks, rank - attention_ranks, "experts", device
                ),
            )
            module.to_empty(device=device)
            for layer in module:
                layer.up_gate.weight.copy_(up_gate[layer.expert_slice])
                layer.down.weight.copy_(down[layer.expert_slice])

        exchanges = [
            ExpertExchange(
                groups.process_group,
                max_tokens=capacity,
                top_k=top_k,
                num_experts=num_experts,
                hidden_size=hidden_size,
                device=device,
                transport="deepep",
                attention_ranks=attention_ranks,
            )
            for _ in range(microbatches)
        ]
        with ExitStack() as scope:
            contexts = []
            for exchange in exchanges:
                stream = scope.enter_context(
                    CUDAStream.external(torch.cuda.Stream(device=device))
                )
                stream.wait(torch.cuda.current_stream(device))
                context = scope.enter_context(
                    ExecutionContext(module, stream=stream, experts=exchange)
                )
                context.prepare(TextSize(capacity, 1))
                contexts.append(context)
            run = Microbatches(contexts)
            scope.callback(run.close)
            for empty in (False, True):
                count = (
                    0 if not source or empty and rank == 0 else 3 * (rank + 1)
                )
                hidden = (
                    torch.rand(count, hidden_size, generator=generator) + 0.125
                ).to(device, torch.bfloat16)
                ids = torch.arange(
                    count * top_k, device=device, dtype=torch.int32
                ).reshape(count, top_k)
                ids.remainder_(num_experts)
                weights = (
                    torch.tensor([0.25, 0.75], device=device)
                    .expand(count, top_k)
                    .contiguous()
                )
                expected = torch.zeros_like(hidden, dtype=torch.float64)
                for slot in range(top_k):
                    expert = ids[:, slot].long().cpu()
                    projected = torch.einsum(
                        "th,toh->to",
                        hidden.cpu().double(),
                        up_gate[expert].double(),
                    )
                    up, gate = projected.split(intermediate, -1)
                    expected += weights[:, slot, None].double() * torch.einsum(
                        "ti,thi->th",
                        F.silu(gate) * up,
                        down[expert].double(),
                    ).to(device)

                def step(context, route_scale):
                    exchange = context.experts
                    exchange.begin(capacity)
                    try:
                        result = (
                            torch.stack(
                                [
                                    layer(hidden, ids, weights * route_scale)
                                    for layer in module
                                ]
                            )
                            if source
                            else hidden.expand(len(module), -1, -1)
                        )
                        context.join_expert_layers()
                        return result
                    finally:
                        exchange.end()

                # Distinct results expose communication buffer sharing across
                # microbatches. Scaling routes by powers of two introduces
                # no additional BF16 rounding error.
                calls = [
                    partial(step, context, 2**index)
                    for index, context in enumerate(contexts)
                ]
                eager = run(calls)
                torch.cuda.synchronize(device)
                expected = expected.expand(len(module), -1, -1)
                # Match the existing grouped-expert contract: six BF16
                # roundoffs cover products, gating, partial and final sums.
                unit = 2**-8
                tolerance = 6 * unit / (1 - 6 * unit)
                for index, actual in enumerate(eager):
                    torch.testing.assert_close(
                        actual.double(),
                        expected * 2**index,
                        rtol=tolerance,
                        atol=0,
                    )
                with CUDAGraph(context=contexts[0]) as graph:
                    graph.capture(lambda: run(calls))
                    for _ in range(4):
                        outputs = graph.replay()
                        contexts[0].stream.synchronize()
                        for index, actual in enumerate(outputs):
                            torch.testing.assert_close(
                                actual.double(),
                                expected * 2**index,
                                rtol=tolerance,
                                atol=0,
                            )
        for exchange in exchanges:
            exchange.close()


def _local(rank, port, microbatches):
    _exercise(rank, rank, 2, 1, f"tcp://127.0.0.1:{port}", microbatches)


@pytest.mark.parametrize("microbatches", (1, 2))
def test_disaggregated_expert_modules_match_the_routed_equation(microbatches):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_local, args=(port, microbatches), nprocs=2, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-ranks", type=int, required=True)
    parser.add_argument(
        "--microbatches", type=int, choices=(1, 2), required=True
    )
    arguments = parser.parse_args()
    _exercise(
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
        arguments.attention_ranks,
        "env://",
        arguments.microbatches,
    )
