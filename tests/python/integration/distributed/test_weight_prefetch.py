"""DWDP evaluates routed experts while peers progress independently."""

import socket
from contextlib import ExitStack

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from uniserve.distributed import partition_experts
from uniserve.model import TextSize
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import Quantizer, ScaleLayout
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext
from uniserve.runtime.process_groups import (
    Rendezvous,
    initialize_process_groups,
)
from uniserve.runtime.weight_prefetch import WeightPrefetch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# Non-page-aligned shard boundaries exercise immutable edge bytes as well as
# both reusable remote slots. Four layers reuse each slot within one call.
EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 16, 768, 704, 2
RANKS = 4


def _model(device, quantized, group=None):
    layers = nn.ModuleList(
        FusedMoE(
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            top_k=TOP_K,
            activation="gelu_tanh",
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(4)
    )
    if group is not None:
        partition_experts(layers, group)
    generator = torch.Generator().manual_seed(73)
    for layer in layers:
        interval = layer.expert_slice
        for name, rows, width in (
            ("up_gate", 2 * INTERMEDIATE, HIDDEN),
            ("down", HIDDEN, INTERMEDIATE),
        ):
            linear = getattr(layer, name)
            if quantized:
                values = torch.randint(
                    0,
                    256,
                    (EXPERTS, rows, width // 2),
                    dtype=torch.uint8,
                    generator=generator,
                )
                scales = torch.randint(
                    8,
                    13,
                    (EXPERTS, rows, width // 16),
                    dtype=torch.uint8,
                    generator=generator,
                )
                tensor_scales = torch.linspace(0.5, 1.0, EXPERTS)
                count = interval.stop - interval.start
                weight = (
                    Quantizer("nvfp4")
                    .from_tensors(
                        {
                            "values": values[interval].to(device).contiguous(),
                            "block_scale": scales[interval]
                            .to(device)
                            .reshape(count * rows, width // 16),
                            "tensor_scale": tensor_scales[interval].to(device),
                        },
                        shape=(count, rows, width),
                        dtype=torch.bfloat16,
                    )
                    .repack(scale_layout=ScaleLayout.SWIZZLED_128X4)
                )
                linear.weight = nn.Parameter(weight, requires_grad=False)
                linear.input_quantizer = Quantizer(
                    "nvfp4",
                    calibrated_scale=0.125 if name == "up_gate" else 1.0,
                )
            else:
                weight = (
                    torch.randn(EXPERTS, rows, width, generator=generator)
                    * 0.02
                )
                linear.weight.copy_(weight[interval].to(device, torch.bfloat16))
    return layers


@torch.inference_mode()
def _run(rank, port, quantized, prepared, finished):
    device = torch.device("cuda", rank)
    with (
        initialize_process_groups(
            rank=0,
            local_rank=rank,
            world_size=1,
            device=device,
            experts=(rank, RANKS, Rendezvous("127.0.0.1", port)),
        ) as groups,
        ExitStack() as scope,
    ):
        reference = _model(device, quantized)
        distributed = _model(device, quantized, groups.experts)
        owner = WeightPrefetch(distributed)
        scope.callback(owner.close)
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        scope.enter_context(stream)
        expected_context = scope.enter_context(
            ExecutionContext(reference, stream=stream)
        )
        context = scope.enter_context(
            ExecutionContext(distributed, stream=stream, weights=owner)
        )
        expected_context.prepare(TextSize(32, 1))
        context.prepare(TextSize(32, 1))
        generator = torch.Generator(device=device).manual_seed(89 + rank)
        pool = torch.cuda.MemPool()
        cases = []
        for count, depth in ((7 + rank * 6, 4), (3 + rank * 2, 3)):
            hidden = torch.rand(
                count,
                HIDDEN,
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
            )
            # Routes cover every expert, including those on the idle peer.
            ids = torch.arange(count * TOP_K, device=device).reshape(
                count, TOP_K
            )
            ids = ((ids + rank) % EXPERTS).to(torch.int32)
            weights = torch.rand(
                count, TOP_K, device=device, generator=generator
            )

            def forward(
                layers=distributed,
                hidden=hidden,
                ids=ids,
                weights=weights,
                depth=depth,
            ):
                # Later layers consume intermediate results from earlier
                # layers; captured outputs and temporaries must remain valid
                # across every prefetch and pool reuse boundary.
                outputs = []
                for layer in layers[:depth]:
                    hidden = layer(hidden, ids, weights)
                    outputs.append(hidden)
                return tuple(outputs)

            with expected_context.activate():
                expected = forward(layers=reference)
            with context.activate():
                eager = forward()
            stream.synchronize()
            for actual, wanted in zip(eager, expected, strict=True):
                torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
            graph = scope.enter_context(
                CUDAGraph(context=context, pools={device: pool})
            )
            graph.capture(forward)
            cases.append((graph, expected))

        def replay(graph, expected):
            actual = graph.replay()
            stream.synchronize()
            for value, wanted in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, wanted, atol=0, rtol=0)

        for _ in range(2 + rank * 3):
            for graph, expected in cases:
                replay(graph, expected)

        # Queue different invocations without a host synchronization between
        # them. Their stream dependencies must protect shared weight slots.
        pending = []
        with torch.cuda.stream(stream.stream):
            for graph, expected in (*cases, *reversed(cases)):
                pending.append(
                    (tuple(value.clone() for value in graph.replay()), expected)
                )
        stream.synchronize()
        for actual, expected in pending:
            for value, wanted in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, wanted, atol=0, rtol=0)

        # This setup fence starts the idle-peer phase after all ranks have
        # prepared their graphs. Only the last rank submits CUDA work until
        # it finishes: numerical progress cannot depend on rank rendezvous.
        stream.synchronize()
        prepared.wait(timeout=240)
        if rank != RANKS - 1:
            assert finished.wait(240), (
                "DWDP needs an idle peer to make progress"
            )
        else:
            for _ in range(5):
                for graph, expected in reversed(cases):
                    replay(graph, expected)
            finished.set()

        # Closing one captured invocation must preserve other invocations
        # borrowing the same execution context and allocator pool.
        cases[0][0].close()
        replay(*cases[1])
        # Every source process must retain its published pages until all
        # peers have finished this final read.
        prepared.wait(timeout=240)


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "nvfp4"])
def test_independent_expert_weights(quantized):
    """Eager/graph calls equal resident experts despite asymmetric progress."""
    if torch.cuda.device_count() < RANKS:
        pytest.fail(f"DWDP shard boundary coverage requires {RANKS} GPUs")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    spawn = mp.get_context("spawn")
    prepared, finished = spawn.Barrier(RANKS), spawn.Event()
    mp.spawn(
        _run, (port, quantized, prepared, finished), nprocs=RANKS, join=True
    )
