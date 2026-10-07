"""Asymmetric expert transport preserves weighted rows and collective order.

The same check runs through torchrun across hosts: source ranks route rows,
expert ranks evaluate a per-expert diagonal map, and the complete weighted
sum must return to each source. Eager calls and graph replays include an
empty source and repeated buffer reuse.
"""

import argparse
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.runtime.backends.deepep import DeepEP
from uniserve.runtime.process_groups import initialize_process_groups

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def _exercise(rank, local_rank, world_size, attention_ranks, init_method):
    device = torch.device("cuda", local_rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
    ) as groups:
        experts = 2 * (world_size - attention_ranks)
        # A width below the transport alignment exercises zero extension.
        hidden_size, capacity, top_k = 192, 32, 2
        exchange = DeepEP(
            groups.process_group,
            attention_ranks=attention_ranks,
            num_experts=experts,
            hidden_size=hidden_size,
            max_tokens=capacity,
            top_k=top_k,
        )
        stream = torch.cuda.Stream(device=device)
        for empty in (False, True):
            count = (
                0
                if rank >= attention_ranks or empty and rank == 0
                else 3 * (rank + 1)
            )
            hidden = (
                torch.arange(count * hidden_size, device=device)
                .reshape(count, hidden_size)
                .remainder(16)
                .float()
                .mul_(0.125)
                .add_(rank + 1)
                .to(torch.bfloat16)
            )
            ids = (
                torch.arange(count * top_k, device=device, dtype=torch.int32)
                .reshape(count, top_k)
                .remainder(experts)
            )
            weights = (
                torch.tensor([0.25, 0.75], device=device)
                .expand(count, top_k)
                .contiguous()
            )
            expected = hidden.float() * ((ids + 1).float() * weights).sum(
                -1, keepdim=True
            )

            def forward():
                received, routes, scales = exchange.dispatch(
                    hidden, ids, weights
                )
                # Each expert multiplies by its global id + 1. Invalid
                # slots have zero weight and contribute no output.
                partial = (
                    received.float()
                    * ((routes + 1).float() * scales).sum(-1, keepdim=True)
                ).to(torch.bfloat16)
                return exchange.combine(partial)

            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                eager = forward()
            stream.synchronize()
            # One BF16 partial per owning rank and a BF16 final combine:
            # three unit roundoffs bound this positive two-route equation.
            unit = 2**-8
            tolerance = 3 * unit / (1 - 3 * unit)
            torch.testing.assert_close(
                eager.float(), expected, rtol=tolerance, atol=0
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = forward()
            for _ in range(4):
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(
                    captured.float(), expected, rtol=tolerance, atol=0
                )
            del graph, captured
        torch.cuda.synchronize(device)
        exchange.close()


def _local(rank, port):
    _exercise(rank, rank, 2, 1, f"tcp://127.0.0.1:{port}")


def test_disaggregated_experts_return_each_sources_weighted_rows():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_local, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-ranks", type=int, required=True)
    arguments = parser.parse_args()
    _exercise(
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
        arguments.attention_ranks,
        "env://",
    )
