"""Expert-parallel experts evaluate the routed equation across ranks.

Two ranks each keep half of the experts of one ``FusedMoE`` and exchange
tokens through the NVLink all-to-all at every call. Each rank's tokens return
with the complete routed sum over all experts, whichever rank holds them, and
a rank without tokens of its own still serves the other rank's tokens routed
to its experts by joining the step. Replaying a captured step reproduces it.
"""

import socket

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from uniserve.distributed import partition_experts
from uniserve.model import TextSize
from uniserve.nn.moe import FusedMoE
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve.runtime.expert_exchange import ExpertExchange
from uniserve.runtime.process_groups import (
    Rendezvous,
    initialize_process_groups,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 8, 256, 128, 2
# Tokens each rank routes: the second rank has none and joins the step.
TOKENS = (40, 0)
CAPACITY = 64


def _gamma(roundings):
    """Relative bound of ``roundings`` BF16 unit roundoffs (Higham gamma)."""
    unit = 2**-8
    return roundings * unit / (1 - roundings * unit)


def _reference(hidden, up_gate, down, ids, weights):
    """The routed tanh-GELU equation in FP64 over every expert."""
    hidden, up_gate, down = hidden.double(), up_gate.double(), down.double()
    result = torch.zeros_like(hidden)
    for slot in range(ids.shape[1]):
        expert = ids[:, slot].long()
        projected = torch.einsum("th,toh->to", hidden, up_gate[expert])
        up, gate = projected.split(INTERMEDIATE, dim=-1)
        activated = F.gelu(gate, approximate="tanh") * up
        result += weights[:, slot, None].double() * torch.einsum(
            "ti,thi->th", activated, down[expert]
        )
    return result


def _inputs(rank, device):
    """Every rank's identical experts and this rank's own tokens."""
    generator = torch.Generator().manual_seed(29)
    # Positive operands scaled so each projection stays near unit size.
    up_gate = (
        torch.rand(EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator)
        + 0.125
    ) / HIDDEN
    down = (
        torch.rand(EXPERTS, HIDDEN, INTERMEDIATE, generator=generator) + 0.125
    ) / INTERMEDIATE
    tokens = []
    for count in TOKENS:
        hidden = torch.rand(count, HIDDEN, generator=generator) + 0.125
        # Distinct experts per token.
        ids = torch.tensor(
            [
                torch.randperm(EXPERTS, generator=generator)[:TOP_K].tolist()
                for _ in range(count)
            ],
            dtype=torch.int64,
        ).reshape(count, TOP_K)
        weights = torch.rand(count, TOP_K, generator=generator) + 0.25
        tokens.append((hidden, ids, weights))
    hidden, ids, weights = tokens[rank]
    return (
        up_gate.to(device, torch.bfloat16),
        down.to(device, torch.bfloat16),
        hidden.to(device, torch.bfloat16),
        ids.to(device, torch.int32),
        weights.to(device),
    )


@torch.inference_mode()
def _run(rank, port):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=0,
        local_rank=rank,
        world_size=1,
        device=device,
        experts=(rank, len(TOKENS), Rendezvous("127.0.0.1", port)),
    ) as groups:
        up_gate, down, hidden, ids, weights = _inputs(rank, device)
        module = FusedMoE(
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            top_k=TOP_K,
            activation="gelu_tanh",
            device=device,
            dtype=torch.bfloat16,
        )
        module.up_gate.weight.copy_(up_gate)
        module.down.weight.copy_(down)
        partition_experts(module, groups.experts)

        exchange = ExpertExchange(
            groups.experts,
            max_tokens=CAPACITY,
            top_k=TOP_K,
            num_experts=EXPERTS,
            hidden_size=HIDDEN,
            device=device,
        )
        kind = exchange.register(frozenset({CAPACITY}))
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        expected = _reference(hidden, up_gate, down, ids, weights)
        with (
            stream,
            ExecutionContext(
                module, stream=stream, experts=exchange
            ) as context,
        ):
            context.prepare(TextSize(CAPACITY, 1))

            def step():
                """One expert step: this rank's tokens, or a join."""
                exchange.begin(CAPACITY)
                try:
                    output = (
                        module(hidden, ids, weights)
                        if hidden.shape[0]
                        else hidden.new_empty((0, HIDDEN))
                    )
                    context.join_expert_layers()
                finally:
                    exchange.end()
                return output

            capacity = exchange.agree(
                kind if hidden.shape[0] else None, hidden.shape[0]
            )
            assert capacity == CAPACITY
            with context.activate():
                eager = step()

            # Every rank captures the same step, then replays it.
            with CUDAGraph(context=context) as graph:
                graph.capture(step)
                captured = graph.replay()
                stream.synchronize()

        if hidden.shape[0]:
            # The local kernel's roundings, the BF16 partial sum each rank
            # returns, and the combined output's rounding.
            for actual in (eager, captured):
                torch.testing.assert_close(
                    actual.double(), expected, rtol=_gamma(6), atol=0
                )
        else:
            assert eager.shape == captured.shape == (0, HIDDEN)


def test_expert_parallel_ranks_return_the_complete_routed_sum():
    if torch.cuda.device_count() < len(TOKENS):
        pytest.fail(f"expert exchange needs {len(TOKENS)} GPUs")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_run, (port,), nprocs=len(TOKENS), join=True)
