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
from uniserve.quantization import Quantizer, ScaleLayout
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


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_expert_parallel_ranks_return_the_complete_routed_sum():
    if torch.cuda.device_count() < len(TOKENS):
        pytest.fail(f"expert exchange needs {len(TOKENS)} GPUs")
    mp.spawn(_run, (_free_port(),), nprocs=len(TOKENS), join=True)


# E4M3 byte of 2**-6: the block scale of every expert weight.
E4M3_2_NEG_6 = 0x08
# Static NVFP4 input scales of the two projections.
INPUT_SCALES = (2.0**-4, 1.0)


def _nvfp4(codes, device):
    """Encode positive E2M1 ``codes [E, rows, K]`` at block scale 2**-6."""
    experts, rows, width = codes.shape
    values = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return (
        Quantizer("nvfp4")
        .from_tensors(
            {
                "values": values.to(device).contiguous(),
                "block_scale": torch.full(
                    (experts * rows, width // 16),
                    E4M3_2_NEG_6,
                    dtype=torch.uint8,
                    device=device,
                ),
                "tensor_scale": torch.ones(experts, device=device),
            },
            shape=(experts, rows, width),
            dtype=torch.bfloat16,
        )
        .repack(scale_layout=ScaleLayout.SWIZZLED_128X4)
    )


def _nvfp4_moe(codes, device, group=None):
    """NVFP4 experts of ``codes``, sharded over ``group`` when given.

    Positive E2M1 codes 1..7 (0.5 to 6) keep every product and sum
    positive, so each output element is a positive sum without
    cancellation.
    """
    module = FusedMoE(
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        top_k=TOP_K,
        activation="gelu_tanh",
        device=device,
        dtype=torch.bfloat16,
    )
    experts = slice(None)
    if group is not None:
        partition_experts(module, group)
        experts = module.expert_slice
    up_gate, down = codes
    module.up_gate.weight = torch.nn.Parameter(
        _nvfp4(up_gate[experts], device), requires_grad=False
    )
    module.down.weight = torch.nn.Parameter(
        _nvfp4(down[experts], device), requires_grad=False
    )
    module.up_gate.input_quantizer, module.down.input_quantizer = (
        Quantizer("nvfp4", calibrated_scale=scale) for scale in INPUT_SCALES
    )
    return module


def _encoded_inputs(counts, device):
    """Expert codes and every rank's positive tokens with distinct routes."""
    generator = torch.Generator().manual_seed(43)
    codes = (
        torch.randint(
            1, 8, (EXPERTS, 2 * INTERMEDIATE, HIDDEN), generator=generator
        ),
        torch.randint(
            1, 8, (EXPERTS, HIDDEN, INTERMEDIATE), generator=generator
        ),
    )
    tokens = []
    for count in counts:
        hidden = torch.rand(count, HIDDEN, generator=generator) + 0.125
        ids = torch.tensor(
            [
                torch.randperm(EXPERTS, generator=generator)[:TOP_K].tolist()
                for _ in range(count)
            ],
            dtype=torch.int32,
        ).reshape(count, TOP_K)
        # Distinct random weights identify each token after the exchange.
        weights = torch.rand(count, TOP_K, generator=generator) + 0.25
        tokens.append(
            (
                hidden.to(device, torch.bfloat16),
                ids.to(device),
                weights.to(device),
            )
        )
    return codes, tokens


def _check_received(group, exchange, tokens, device):
    """Encoded rows arrive with the values and block scales each rank sent."""
    rank, size = group.rank, len(tokens)
    encoding = Quantizer("nvfp4", calibrated_scale=INPUT_SCALES[0])
    local = range(rank * EXPERTS // size, (rank + 1) * EXPERTS // size)
    hidden, ids, weights = tokens[rank]
    exchange.begin(CAPACITY)
    try:
        received, received_ids, received_weights = exchange.dispatch(
            0, encoding.quantize(hidden), ids, weights, invalid_expert=-1
        )
        assert received.quantizer == encoding
        assert received.shape == (size * CAPACITY, HIDDEN)
        fields = received.buffers()
        for source, (sent, sent_ids, sent_weights) in enumerate(tokens):
            sent_fields = encoding.quantize(sent).buffers()
            rows = range(source * CAPACITY, (source + 1) * CAPACITY)
            arrived = [row for row in rows if (received_ids[row] >= 0).any()]
            routed = [
                token
                for token in range(sent.shape[0])
                if any(expert in local for expert in sent_ids[token].tolist())
            ]
            assert len(arrived) == len(routed)
            for row in arrived:
                # The route weights name the sent token.
                (token,) = [
                    token
                    for token in routed
                    if torch.equal(received_weights[row], sent_weights[token])
                ]
                for name in ("values", "block_scale"):
                    assert torch.equal(
                        fields[name][row], sent_fields[name][token]
                    )
        # The exchange's dispatch pairs with a combine.
        exchange.combine(
            torch.zeros(
                size * CAPACITY, HIDDEN, device=device, dtype=torch.bfloat16
            ),
            hidden.shape[0],
        )
    finally:
        exchange.end()


@torch.inference_mode()
def _run_encoded(rank, port, provider):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=0,
        local_rank=rank,
        world_size=1,
        device=device,
        experts=(rank, len(TOKENS), Rendezvous("127.0.0.1", port)),
    ) as groups:
        exchange = ExpertExchange(
            groups.experts,
            max_tokens=CAPACITY,
            top_k=TOP_K,
            num_experts=EXPERTS,
            hidden_size=HIDDEN,
            device=device,
        )
        # Both ranks send tokens, so each receives rows from both.
        _, tokens = _encoded_inputs((24, 16), device)
        _check_received(groups.experts, exchange, tokens, device)

        # The first rank's tokens through the experts; the second rank has
        # none and joins every step with its empty rows.
        codes, tokens = _encoded_inputs(TOKENS, device)
        hidden, ids, weights = tokens[rank]
        module = _nvfp4_moe(codes, device, groups.experts)
        # A rank without tokens only joins, with no rows to encode.
        encoded = (
            module.up_gate.input_quantizer.quantize(hidden)
            if hidden.shape[0]
            else hidden
        )
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        with (
            stream,
            ExecutionContext(
                module, stream=stream, moe=provider, experts=exchange
            ) as context,
        ):
            context.prepare(TextSize(CAPACITY, 1))

            def step(states):
                """One expert step: this rank's tokens, or a join."""
                exchange.begin(CAPACITY)
                try:
                    output = (
                        module(states, ids, weights)
                        if hidden.shape[0]
                        else hidden.new_empty((0, HIDDEN))
                    )
                    context.join_expert_layers()
                finally:
                    exchange.end()
                return output

            with context.activate():
                dense = step(hidden).clone()
                eager = step(encoded).clone()
            with CUDAGraph(context=context) as graph:
                graph.capture(lambda: step(encoded))
                captured = graph.replay()
                stream.synchronize()

        if not hidden.shape[0]:
            assert eager.shape == captured.shape == (0, HIDDEN)
            return
        # Encoded rows give the output of the BF16 rows bit for bit.
        assert torch.equal(eager, dense)
        assert torch.equal(captured, dense)

        # The same experts unsharded on one GPU. Both evaluate the same
        # route products; one GPU rounds each route and the sum to BF16
        # (within gamma(2) of the exact positive sum), expert parallelism
        # also rounds each rank's partial sum (within gamma(3)).
        single = _nvfp4_moe(codes, device)
        with ExecutionContext(single, moe=provider) as context:
            context.prepare(TextSize(CAPACITY, 1))
            with context.activate():
                expected = single(hidden, ids, weights)
        torch.testing.assert_close(
            dense.double(), expected.double(), rtol=_gamma(5), atol=0
        )


@pytest.mark.parametrize("provider", ["cutedsl", "trtllm"])
def test_nvfp4_hidden_states_cross_the_exchange_encoded(provider):
    """NVFP4 experts exchange their input encoding in place of BF16 rows.

    Encoded rows arrive with the values and block scales their rank sent,
    and the experts' output of encoded rows equals that of the BF16 rows
    bit for bit, eager and replayed, while a rank without tokens joins.
    """
    if torch.cuda.device_count() < len(TOKENS):
        pytest.fail(f"expert exchange needs {len(TOKENS)} GPUs")
    mp.spawn(
        _run_encoded,
        (_free_port(), provider),
        nprocs=len(TOKENS),
        join=True,
    )
