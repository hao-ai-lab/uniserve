"""Expert-parallel experts evaluate the routed equation across ranks.

Two ranks each keep half of the experts of one ``FusedMoE`` and exchange
tokens through the NVLink all-to-all at every call. Each rank's tokens return
with the complete routed sum over all experts, whichever rank holds them, and
a rank without tokens of its own still serves the other rank's tokens routed
to its experts by joining the step, in one launch of the join captured at
the step's capacity. Replaying a captured step reproduces it.
The fused MegaMoE exchange serves NVFP4 experts with the same step protocol
and gates with each layer's declared nonlinearity.
"""

import socket

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from uniserve.distributed import partition_experts
from uniserve.model import TextSize
from uniserve.nn import functional
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import Quantizer, ScaleLayout
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve.runtime.expert_exchange import ExpertExchange, JoinGraphs
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
def _run(rank, port, grouped=False):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=0,
        local_rank=rank,
        world_size=1,
        device=device,
        experts=(rank, len(TOKENS), Rendezvous("127.0.0.1", port)),
    ) as groups:
        up_gate, down, hidden, ids, weights = _inputs(
            0 if grouped else rank, device
        )
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
            source_group=(0, 1) if grouped else None,
        )
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

            if grouped:
                # The first member received the shared request, while the
                # second is still polling IPC. Starting a forward here would
                # leave its tensor collectives without a peer.
                assert exchange.agree(hidden.shape[0] if rank == 0 else 0) == 0
                assert not exchange.active
            capacity = exchange.agree(hidden.shape[0])
            assert capacity == CAPACITY
            assert exchange.active == bool(hidden.shape[0])
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


def test_tensor_sources_wait_for_their_shared_request():
    if torch.cuda.device_count() < 2:
        pytest.fail("tensor source agreement needs two GPUs")
    mp.spawn(_run, (_free_port(), True), nprocs=2, join=True)


@torch.inference_mode()
def _run_joined(rank, port):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=0,
        local_rank=rank,
        world_size=1,
        device=device,
        experts=(rank, len(TOKENS), Rendezvous("127.0.0.1", port)),
    ) as groups:
        up_gate, down, hidden, ids, weights = _inputs(rank, device)
        # Two expert layers of the same experts: a join covers every layer.
        layers = torch.nn.ModuleList(
            FusedMoE(
                EXPERTS,
                HIDDEN,
                INTERMEDIATE,
                top_k=TOP_K,
                activation="gelu_tanh",
                device=device,
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        )
        for layer in layers:
            layer.up_gate.weight.copy_(up_gate)
            layer.down.weight.copy_(down)
        partition_experts(layers, groups.experts)

        exchange = ExpertExchange(
            groups.experts,
            max_tokens=CAPACITY,
            top_k=TOP_K,
            num_experts=EXPERTS,
            hidden_size=HIDDEN,
            device=device,
        )
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        expected = _reference(hidden, up_gate, down, ids, weights)
        with (
            stream,
            ExecutionContext(
                layers, stream=stream, experts=exchange
            ) as context,
        ):
            context.prepare(TextSize(CAPACITY, 1))
            # Collective: every rank captures its joins at the same point.
            joins = JoinGraphs(context, exchange, {CAPACITY})
            try:
                if hidden.shape[0]:
                    # This rank's eager step meets the other rank's replay.
                    with context.activate():
                        exchange.begin(CAPACITY)
                        try:
                            outputs = [
                                layer(hidden, ids, weights) for layer in layers
                            ]
                        finally:
                            exchange.end()
                    stream.synchronize()
                else:
                    joins.replay(CAPACITY)
                    stream.synchronize()
            finally:
                joins.close()

        if hidden.shape[0]:
            # The joining rank's experts served both layers' routes.
            for output in outputs:
                torch.testing.assert_close(
                    output.double(), expected, rtol=_gamma(6), atol=0
                )


def test_a_rank_without_tokens_joins_every_layer_in_a_captured_step():
    """A captured join serves the other rank's complete expert step.

    The rank without tokens takes part in a two-layer step by replaying the
    join captured at the step's capacity. Its experts serve the other rank's
    tokens in both layers, whose outputs match the routed equation.
    """
    if torch.cuda.device_count() < len(TOKENS):
        pytest.fail(f"expert exchange needs {len(TOKENS)} GPUs")
    mp.spawn(_run_joined, (_free_port(),), nprocs=len(TOKENS), join=True)


# E4M3 byte of 2**-6: the block scale of every expert weight.
E4M3_2_NEG_6 = 0x08
# Static NVFP4 input scales of the two projections.
INPUT_SCALES = (2.0**-4, 1.0)


def _nvfp4(codes, device, block_scale=None):
    """Encode E2M1 ``codes [E, rows, K]`` with unit tensor scales.

    ``block_scale`` holds one E4M3 byte per ``[E, rows, K / 16]`` block;
    every block takes 2**-6 when it is omitted.
    """
    experts, rows, width = codes.shape
    values = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    if block_scale is None:
        block_scale = torch.full(
            (experts, rows, width // 16), E4M3_2_NEG_6, dtype=torch.long
        )
    return (
        Quantizer("nvfp4")
        .from_tensors(
            {
                "values": values.to(device).contiguous(),
                "block_scale": block_scale.to(torch.uint8)
                .reshape(experts * rows, width // 16)
                .to(device)
                .contiguous(),
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
        expected = []
        for sent, sent_ids, sent_weights in tokens:
            sent_fields = encoding.quantize(sent).buffers()
            expected.extend(
                (sent_fields, token, sent_weights[token])
                for token in range(sent.shape[0])
                if any(expert in local for expert in sent_ids[token].tolist())
            )
        arrived = [
            row
            for row in range(received.shape[0])
            if (received_ids[row] >= 0).any()
        ]
        assert len(arrived) == len(expected)
        for sent_fields, token, sent_weights in expected:
            # The route weights identify each token independently of the
            # transport's receive order, which has no numerical significance.
            (row,) = [
                row
                for row in arrived
                if torch.equal(
                    received_weights[row][received_ids[row] >= 0],
                    sent_weights[received_ids[row] >= 0],
                )
            ]
            for name in ("values", "block_scale"):
                assert torch.equal(fields[name][row], sent_fields[name][token])
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
def _run_encoded(rank, port, provider, transport):
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
            transport=transport,
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

        exchange.close()
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
@pytest.mark.parametrize("transport", ["alltoall", "deepep"])
def test_nvfp4_hidden_states_cross_the_exchange_encoded(provider, transport):
    """NVFP4 experts exchange their input encoding in place of BF16 rows.

    Encoded rows arrive with the values and block scales their rank sent,
    and the experts' output of encoded rows equals that of the BF16 rows
    bit for bit, eager and replayed, while a rank without tokens joins.
    """
    if torch.cuda.device_count() < len(TOKENS):
        pytest.fail(f"expert exchange needs {len(TOKENS)} GPUs")
    mp.spawn(
        _run_encoded,
        (_free_port(), provider, transport),
        nprocs=len(TOKENS),
        join=True,
    )


E4M3_ONE = 0x38  # the E4M3 byte encoding 1.0
# Probe gates. At -3.75 and -3.375 erf GELU departs from the tanh
# approximation by more than a quarter; the others span SiLU's and GELU's
# curvature elsewhere.
GATES = (-3.75, -3.375, -1.5, -0.75, 0.375, 0.75, 1.5, 3.0)


def _probe_experts():
    """E2M1 codes and block scales whose channels isolate each gate.

    Expert e's channel 16m has up value 6 and gate ``GATES[(m + e) % 8]``
    for a token whose hidden column 0 holds 6, and every other channel is
    zero: up rows read column 0 with weight one, gate rows with weight +-0.5
    at block scale ``|g| / 3``, which E4M3 represents exactly for these
    gates. The FC2 input encoding then stores ``act(g) * 6`` as six times
    its block scale, and the diagonal down projection copies it to output
    column 16m. Each expert rotates the gates, so a token served by another
    expert than its route names returns other values.
    """
    probes = torch.arange(0, INTERMEDIATE, 16)
    rotation = (probes[None, :] // 16 + torch.arange(EXPERTS)[:, None]) % len(
        GATES
    )
    gates = torch.tensor(GATES)[rotation]
    up_gate = torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.long)
    up_gate_scale = torch.full(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16), E4M3_ONE, dtype=torch.long
    )
    up_gate[:, probes, 0] = 2
    up_gate[:, INTERMEDIATE + probes, 0] = torch.where(gates > 0, 1, 9)
    up_gate_scale[:, INTERMEDIATE + probes, 0] = (
        (gates.abs() / 3).to(torch.float8_e4m3fn).view(torch.uint8).long()
    )

    down = torch.zeros(EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.long)
    down[:, torch.arange(INTERMEDIATE), torch.arange(INTERMEDIATE)] = 2
    down_scale = torch.full(
        (EXPERTS, HIDDEN, INTERMEDIATE // 16), E4M3_ONE, dtype=torch.long
    )
    return (up_gate, up_gate_scale), (down, down_scale)


@torch.inference_mode()
def _run_megamoe(rank, port, activation):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=0,
        local_rank=rank,
        world_size=1,
        device=device,
        experts=(rank, len(TOKENS), Rendezvous("127.0.0.1", port)),
    ) as groups:
        up_gate, down = _probe_experts()
        quantizers = (
            Quantizer("nvfp4", calibrated_scale=1.0),
            # The FC2 input scale keeps every probe's block scale a normal
            # E4M3.
            Quantizer("nvfp4", calibrated_scale=2.0**-7),
        )
        module = FusedMoE(
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            top_k=1,
            activation=activation,
            device=device,
            dtype=torch.bfloat16,
        )
        partition_experts(module, groups.experts)
        local = module.expert_slice
        module.up_gate.weight = torch.nn.Parameter(
            _nvfp4(up_gate[0][local], device, up_gate[1][local]),
            requires_grad=False,
        )
        module.down.weight = torch.nn.Parameter(
            _nvfp4(down[0][local], device, down[1][local]),
            requires_grad=False,
        )
        module.up_gate.input_quantizer, module.down.input_quantizer = quantizers

        exchange = ExpertExchange(
            groups.experts,
            max_tokens=CAPACITY,
            top_k=1,
            num_experts=EXPERTS,
            hidden_size=HIDDEN,
            device=device,
            transport="megamoe",
            intermediate_size=INTERMEDIATE,
            activation=activation,
        )
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        with (
            stream,
            ExecutionContext(
                module, stream=stream, experts=exchange
            ) as context,
        ):
            context.prepare(TextSize(CAPACITY, 1))

            # Unequal and empty senders share one resident expert binding.
            # Moving from full to short calls must not reuse prior routes;
            # each rank also serves its peer while it has no local tokens.
            for counts in (
                (CAPACITY, 3),
                (5, CAPACITY),
                (EXPERTS, 0),
                (0, EXPERTS),
            ):
                tokens = counts[rank]
                hidden = torch.zeros(
                    tokens, HIDDEN, device=device, dtype=torch.bfloat16
                )
                hidden[:, 0] = 6.0
                ids = (
                    (
                        torch.arange(tokens, dtype=torch.int32, device=device)
                        + 3 * rank
                    )
                    % EXPERTS
                )[:, None]
                weights = torch.ones(tokens, 1, device=device)

                def step(states):
                    """One expert step: this rank's tokens, or a join."""
                    exchange.begin(CAPACITY)
                    try:
                        output = (
                            module(states, ids, weights)
                            if tokens
                            else hidden.new_empty((0, HIDDEN))
                        )
                        context.join_expert_layers()
                    finally:
                        exchange.end()
                    return output

                # These probe rows encode exactly. Stored input must retain
                # those bytes, including across different sender lengths.
                encoded = (
                    module.up_gate.input_quantizer.quantize(hidden)
                    if tokens
                    else hidden
                )
                with context.activate():
                    eager = step(hidden).clone()
                    stored = step(encoded).clone()
                with CUDAGraph(context=context) as graph:
                    graph.capture(lambda: step(hidden))
                    captured = graph.replay().clone()
                    stream.synchronize()

                for actual in (eager, stored, captured):
                    assert actual.shape == (tokens, HIDDEN)
                if not tokens:
                    continue
                assert torch.equal(stored, eager)

                # The portable reference encodes FC2 input with exact FP32
                # division and evaluates the declared gated nonlinearity.
                expected = functional.fused_moe(
                    hidden.float(),
                    _nvfp4(up_gate[0], device, up_gate[1]),
                    _nvfp4(down[0], device, down[1]),
                    ids,
                    weights,
                    activation=activation,
                    input_quantizers=quantizers,
                )
                # Approximate activation/reciprocal may round a scale near
                # an E4M3 midpoint to its neighbor: one E4M3 step plus BF16
                # output rounding, as in the existing numerical contract.
                for actual in (eager, captured):
                    torch.testing.assert_close(
                        actual.float(), expected, rtol=2**-3 + _gamma(1), atol=0
                    )


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_megamoe_ranks_preserve_routed_results_across_local_sizes(activation):
    """Unequal, full and empty senders retain the declared routed equation.

    Tokens reach experts on both ranks through eager calls, stored-input
    calls and graph replays. Every nonempty result matches the portable
    reference with the declared gate nonlinearity.
    """
    if torch.cuda.device_count() < len(TOKENS):
        pytest.fail(f"expert exchange needs {len(TOKENS)} GPUs")
    mp.spawn(
        _run_megamoe,
        (_free_port(), activation),
        nprocs=len(TOKENS),
        join=True,
    )
