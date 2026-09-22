"""Sharded worker selection preserves dense logit results.

So does output retention.
"""

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from uniserve.distributed import DeviceMesh
from uniserve.model import TextSize, VocabShard
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    initialize_process_groups,
)
from uniserve.sampling import greedy
from uniserve_worker.model_executor.output import ExecutionOutput

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_vocabulary_selection(rank: int, rendezvous: str) -> None:
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    ) as environment:
        meshes = tuple(
            environment.bind(
                DeviceMesh(ranks=ranks, shape=(2,), axes=("tp",), rank=rank),
                device=device,
            )
            for ranks in ((0, 1), (1, 0))
        )
        current = torch.cuda.current_stream(device)
        # Every context below borrows this stream's communicators; the owner
        # retires them after the last context has closed.
        owner = CUDAStream.external(torch.cuda.Stream(device=device))
        stream = owner.stream
        with owner, torch.inference_mode():
            for mesh in meshes:
                group = mesh.get_group("tp")
                for size in (65, 37):
                    vocab = VocabShard(
                        size,
                        slice(group.rank * 64, (group.rank + 1) * 64),
                        128,
                        group,
                    )
                    for dtype in (torch.bfloat16, torch.float16, torch.float32):
                        full = torch.full(
                            (6, 128), -2.0, device=device, dtype=dtype
                        )
                        full[:, size:] = 100
                        full[0, :size] = -1
                        full[1, size - 1] = 3
                        full[2, (7, size - 1)] = 5
                        full[3, :size] = float("-inf")
                        full[4, (3, size - 1)] = float("nan")
                        full[5, (8, size - 1)] = float("inf")
                        local = full[:, vocab.local_slice].contiguous()
                        projected = ExecutionOutput(
                            (local[:2], local[2:]), (vocab, vocab)
                        )
                        continuous = torch.arange(
                            15, device=device, dtype=dtype
                        ).reshape(3, 5)
                        mixed = ExecutionOutput(
                            (local[:2], continuous, local[2:]),
                            (vocab, None, vocab),
                        )
                        retained = mixed.clone()
                        expected_retained = full[:, :size].clone()
                        continuous.add_(1)

                        # This callable has no layers or scratch requirements
                        # beyond its logits and its declared communicator.
                        module = nn.Module()
                        module.register_buffer("logits", local)
                        module.communication_groups = (group,)
                        with ExecutionContext(module, stream=owner) as context:
                            context.prepare(
                                TextSize(num_tokens=6, batch_size=6)
                            )
                            stream.wait_stream(current)
                            with context.activate():
                                greedy(local, vocab)
                            current.wait_stream(stream)
                            current.synchronize()
                            with CUDAGraph(context=context) as graph:
                                graph.capture(lambda: greedy(local, vocab))
                                for iteration in range(2):
                                    if iteration:
                                        full[1, 9] = 4
                                        full[2, (2, size - 2)] = 6
                                        full[4, :size] = -2
                                        full[4, 10] = float("nan")
                                    local.copy_(full[:, vocab.local_slice])
                                    stream.wait_stream(current)
                                    values, tokens = graph.replay()
                                    current.wait_stream(stream)
                                    expected_values, expected_tokens = full[
                                        :, :size
                                    ].max(dim=-1)
                                    torch.testing.assert_close(
                                        values,
                                        expected_values,
                                        rtol=0,
                                        atol=0,
                                        equal_nan=True,
                                    )
                                    torch.testing.assert_close(
                                        tokens, expected_tokens, rtol=0, atol=0
                                    )
                                    torch.testing.assert_close(
                                        torch.cat(
                                            projected.materialize().values
                                        ),
                                        full[:, :size],
                                        rtol=0,
                                        atol=0,
                                        equal_nan=True,
                                    )
                                saved = retained.materialize().values
                                torch.testing.assert_close(
                                    saved[0],
                                    expected_retained[:2],
                                    rtol=0,
                                    atol=0,
                                    equal_nan=True,
                                )
                                torch.testing.assert_close(
                                    saved[1], continuous - 1, rtol=0, atol=0
                                )
                                torch.testing.assert_close(
                                    saved[2],
                                    expected_retained[2:],
                                    rtol=0,
                                    atol=0,
                                    equal_nan=True,
                                )
                                current.synchronize()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="two CUDA devices are required"
)
def test_vocabulary_selection_and_materialization_preserve_dense_results(
    tmp_path: Path,
):
    mp.spawn(
        _run_vocabulary_selection,
        ((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
