"""VSA preserves full key domains through Ulysses partitions.

Gather and peer context partitions preserve them too.
"""

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.nn import MergedColumnParallelLinear
from uniserve.nn.attention import (
    AttentionParallelConfig,
    ContextParallelConfig,
    Ulysses,
    vsa,
)
from uniserve.runtime import (
    CUDAGraph,
    ExecutionContext,
    initialize_process_groups,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _reference(projections, valid):
    q, k, v, gate = projections.unbind(2)
    live = torch.arange(512, device=q.device) % 64 < valid.repeat_interleave(64)
    means = []
    for value in (q, k, v):
        tiles = value.double().view(8, 64, 8, 128)
        means.append(
            (
                tiles.masked_fill(~live.view(8, 64, 1, 1), 0).sum(1)
                / valid.clamp_min(1).view(8, 1, 1)
            ).transpose(0, 1)
        )
    scores = (means[0] @ means[1].transpose(-1, -2) / 128**0.5).masked_fill(
        valid.view(1, 1, -1) == 0, -torch.inf
    )
    compressed = (
        (scores.softmax(-1) @ means[2])
        .transpose(0, 1)
        .repeat_interleave(64, dim=0)
    )
    fine = F.scaled_dot_product_attention(
        q.transpose(0, 1).double(),
        k.transpose(0, 1).double(),
        v.transpose(0, 1).double(),
        attn_mask=live.view(1, 1, -1),
    )
    return (fine.transpose(0, 1) + gate.double() * compressed).to(q.dtype), live


def _run(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=device,
        init_method=rendezvous,
    ) as groups:
        generator = torch.Generator(device="cpu").manual_seed(47)
        hidden = torch.randn(
            512, 64, dtype=torch.bfloat16, generator=generator
        ).to(device)
        matrices = {
            name: (torch.randn(8 * 128, 64, generator=generator) / 8).to(
                device, torch.bfloat16
            )
            for name in ("q", "k", "v", "gate")
        }

        def projections():
            return torch.stack(
                tuple(
                    F.linear(hidden, value).view(512, 8, 128)
                    for value in matrices.values()
                ),
                dim=2,
            )

        valid = torch.tensor(
            [64, 17, 64, 5, 64, 64, 7, 0], device=device, dtype=torch.int32
        )
        cases = (
            ((4,), ("heads",), AttentionParallelConfig(heads=Ulysses("heads"))),
            (
                (2, 2),
                ("context", "heads"),
                AttentionParallelConfig(
                    heads=Ulysses("heads"),
                    context=ContextParallelConfig(gather_axis="context"),
                ),
            ),
            (
                (4,),
                ("context",),
                AttentionParallelConfig(
                    context=ContextParallelConfig(peer_axis="context")
                ),
            ),
            (
                (2, 2),
                ("rows", "columns"),
                AttentionParallelConfig(
                    context=ContextParallelConfig(
                        gather_axis="columns", peer_axis="rows"
                    )
                ),
            ),
        )
        for shape, axes, config in cases:
            mesh = groups.bind(
                DeviceMesh(
                    ranks=(3, 1, 0, 2), shape=shape, axes=axes, rank=rank
                ),
                device=device,
            )
            heads = mesh.get_group(
                () if config.heads is None else config.heads.axis
            )
            context_axes = (
                ()
                if config.context is None
                else tuple(
                    axis
                    for axis in axes
                    if axis
                    in (config.context.gather_axis, config.context.peer_axis)
                )
            )
            context = mesh.get_group(context_axes)
            query_tokens, local_heads = 512 // context.size, 8 // heads.size
            begin = context.rank * query_tokens

            def tensor(shape, dtype=torch.float32):
                return torch.empty(shape, device=device, dtype=dtype)

            workspace = vsa.Workspace(
                tensor((query_tokens, local_heads, 128), torch.bfloat16),
                tensor((local_heads, query_tokens // 64, 8)),
                tensor((local_heads, query_tokens // 64), torch.int32),
                tensor((local_heads, query_tokens // 64, 7), torch.int32),
                tensor((query_tokens // 64, local_heads, 128)),
                tensor((8, local_heads, 128)),
                tensor((8, local_heads, 128)),
                tensor((local_heads, query_tokens // 64, 128)),
            )
            inputs = vsa.Input(
                512,
                1,
                6,
                7,
                valid,
                torch.tensor([0], device=device, dtype=torch.int32),
                torch.arange(7, device=device, dtype=torch.int32),
                torch.tensor(1, device=device, dtype=torch.int32),
            )
            layer = torch.nn.Module()
            layer.projection = MergedColumnParallelLinear(
                64,
                dict.fromkeys(matrices, 8 * 128),
                branch_width=128,
                bias=False,
                dtype=torch.bfloat16,
                device=device,
            )
            for name, value in matrices.items():
                layer.projection.projections[name].weight.data.copy_(value)
            layer.attention = vsa.Attention(vsa.BlockAttention(128**-0.5))
            parallelize_(layer, mesh, attention=config)
            local_begin = begin + heads.rank * (query_tokens // heads.size)
            local_end = local_begin + query_tokens // heads.size
            result = tensor((local_end - local_begin, 8, 128), torch.bfloat16)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            # The contract under test is the value each partition produces,
            # not which kernel produces it, so the provider is the one the
            # device selects rather than one architecture's kernel.
            with ExecutionContext(layer, stream=stream) as execution:
                execution.prepare(None)

                def chunks():
                    for interval, values in layer.projection.forward_chunks(
                        hidden[local_begin:local_end],
                        token_slice=slice(local_begin, local_end),
                        num_tokens=512,
                    ):
                        yield (
                            interval,
                            tuple(
                                values[name].view(-1, local_heads, 128)
                                for name in matrices
                            ),
                        )

                def invoke():
                    for interval, output in layer.attention.forward_chunks(
                        chunks(), inputs, selected_tiles=6, workspace=workspace
                    ):
                        result[
                            interval.start - local_begin : interval.stop
                            - local_begin
                        ].copy_(output)
                    return result

                invoke()
                expected, live = _reference(projections(), valid)
                torch.testing.assert_close(
                    result[live[local_begin:local_end]],
                    expected[local_begin:local_end][
                        live[local_begin:local_end]
                    ],
                    rtol=2e-2,
                    atol=2e-2,
                )
                with CUDAGraph(context=execution) as graph:
                    graph.capture(invoke)
                    hidden.mul_(0.75)
                    graph.replay()
                    expected, live = _reference(projections(), valid)
                    torch.testing.assert_close(
                        result[live[local_begin:local_end]],
                        expected[local_begin:local_end][
                            live[local_begin:local_end]
                        ],
                        rtol=2e-2,
                        atol=2e-2,
                    )
                torch.cuda.synchronize(device)


def test_video_sparse_partitions_and_replay(tmp_path):
    mp.spawn(
        _run, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True
    )
