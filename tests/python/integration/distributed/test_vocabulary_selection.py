"""Sharded logits preserve dense selection and complete-logit consumers."""

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from uniserve_worker.execution.forward_batch import ForwardOutput
from uniserve_worker.execution.graph.full import FullCudaGraphBackend
from uniserve_worker.loader.handles import TensorWeightHandle
from uniserve_worker.loader.weight_loaders import load_parameter_weight
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.logits import greedy_vocabulary
from uniserve_worker.nn.parallel import ParallelConfig
from uniserve_worker.nn.vocab_parallel_embedding import ParallelLMHead
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_vocabulary_selection(rank: int, rendezvous: str) -> None:
    device = torch.device("cuda", rank)
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    meshes = initialize_model_parallel(
        environment,
        {"ordered": ((0, 1), ParallelConfig(2)), "reversed": ((1, 0), ParallelConfig(2))},
    )
    current = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    entry = None
    try:
        with torch.inference_mode():
            for mesh in meshes.values():
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    head = ParallelLMHead(
                        4, 65, layer_config=LayerConfig(mesh.get_group("tp"), None)
                    ).to(device=device, dtype=dtype)
                    weight = (torch.arange(260, device=device).reshape(65, 4) % 13 / 16).to(dtype)
                    load_parameter_weight(head.weight, TensorWeightHandle("weight", weight))
                    inputs = (torch.arange(24, device=device).reshape(6, 4) / 16).to(dtype)
                    local = head.forward_local(inputs)
                    partition = head.vocabulary_partition()
                    projected = ForwardOutput((local[:2], local[2:]), (partition, partition))
                    expected = F.linear(inputs.float(), weight.float()).to(dtype)
                    actual = torch.cat(projected.materialize().values)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    continuous = torch.arange(15, device=device, dtype=torch.float32).view(3, 5)
                    mixed = ForwardOutput(
                        (local[:2], continuous, local[2:]), (partition, None, partition)
                    )
                    retained_mixed = mixed.clone()
                    mixed_values = mixed.materialize().values
                    torch.testing.assert_close(mixed_values[0], expected[:2], rtol=0, atol=0)
                    torch.testing.assert_close(mixed_values[1], continuous, rtol=0, atol=0)
                    torch.testing.assert_close(mixed_values[2], expected[2:], rtol=0, atol=0)
                    values, tokens = greedy_vocabulary(local, partition)
                    reference_values, reference_tokens = expected.max(dim=-1)
                    torch.testing.assert_close(values, reference_values, rtol=0, atol=0)
                    torch.testing.assert_close(tokens, reference_tokens, rtol=0, atol=0)

                    # Padding, cross-rank ties, infinities and NaNs are input
                    # cases at the public logits boundary, independent of GEMM.
                    full = torch.full((6, 128), -2.0, device=device, dtype=dtype)
                    full[:, 65:] = 100
                    full[0, :65] = -1
                    full[1, 64] = 3
                    full[2, (7, 64)] = 5
                    full[3, :65] = float("-inf")
                    full[4, (3, 64)] = float("nan")
                    full[5, (8, 64)] = float("inf")
                    begin = partition.rank * partition.width
                    local.copy_(full[:, begin : begin + partition.width])
                    continuous.add_(1)
                    saved_values = retained_mixed.materialize().values
                    torch.testing.assert_close(saved_values[0], expected[:2], rtol=0, atol=0)
                    torch.testing.assert_close(saved_values[1], continuous - 1, rtol=0, atol=0)
                    torch.testing.assert_close(saved_values[2], expected[2:], rtol=0, atol=0)
                    stream.wait_stream(current)
                    with torch.cuda.stream(stream):
                        greedy_vocabulary(local, partition)
                    current.wait_stream(stream)
                    current.synchronize()
                    entry = FullCudaGraphBackend(device=device, stream=stream)
                    entry.capture_one(
                        "greedy", lambda: greedy_vocabulary(local, partition), keepalive=(local,)
                    )
                    retained = projected.clone()
                    retained_expected = full[:, :65].clone()
                    for iteration in range(2):
                        if iteration:
                            full[1, 9] = 4
                            full[2, (2, 63)] = 6
                            full[4, :65] = -2
                            full[4, 10] = float("nan")
                        stream.wait_stream(current)
                        with torch.cuda.stream(stream):
                            local.copy_(full[:, begin : begin + partition.width])
                            values, tokens = entry.replay("greedy")
                        current.wait_stream(stream)
                        reference_values, reference_tokens = full[:, :65].max(dim=-1)
                        torch.testing.assert_close(
                            values, reference_values, rtol=0, atol=0, equal_nan=True
                        )
                        torch.testing.assert_close(tokens, reference_tokens, rtol=0, atol=0)
                        torch.testing.assert_close(
                            torch.cat(projected.materialize().values),
                            full[:, :65],
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
                    torch.testing.assert_close(
                        torch.cat(retained.materialize().values),
                        retained_expected,
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                    entry.close()
                    entry = None
    finally:
        torch.cuda.synchronize(device)
        if entry is not None:
            entry.close()
        environment.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
def test_vocabulary_selection_and_materialization_preserve_dense_results(tmp_path: Path):
    mp.spawn(
        _run_vocabulary_selection,
        ((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
