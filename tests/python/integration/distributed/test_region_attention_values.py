"""Region-wise VSA keeps its values through a Ulysses head partition.

Each rank projects the gathered sequence for its head shard, attends every
row of it and receives its own token shard back with every head; the rows
must equal the dense equations of the unpartitioned layer.
"""

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from tests.python.fixtures.vsa import (
    REGION_TILE,
    REGION_VALID,
    region_reference,
    region_tables,
)
from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.nn import MergedColumnParallelLinear
from uniserve.nn.attention import AttentionParallelConfig, Ulysses, vsa
from uniserve.runtime import ExecutionContext, initialize_process_groups

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

MEMBERS, HEADS, WIDTH, HIDDEN = 2, 4, 128, 64
ROWS = len(REGION_VALID) * REGION_TILE


def _run(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=MEMBERS,
        device=device,
        init_method=rendezvous,
    ) as groups:
        generator = torch.Generator(device="cpu").manual_seed(91)
        hidden = torch.randn(
            ROWS, HIDDEN, dtype=torch.bfloat16, generator=generator
        ).to(device)
        matrices = {
            name: (
                torch.randn(HEADS * WIDTH, HIDDEN, generator=generator) / 8
            ).to(device, torch.bfloat16)
            for name in ("q", "k", "v", "gate")
        }
        mesh = groups.bind(
            DeviceMesh(
                ranks=tuple(range(MEMBERS)),
                shape=(MEMBERS,),
                axes=("heads",),
                rank=rank,
            ),
            device=device,
        )
        layer = torch.nn.Module()
        layer.projection = MergedColumnParallelLinear(
            HIDDEN,
            dict.fromkeys(matrices, HEADS * WIDTH),
            branch_width=WIDTH,
            bias=False,
            dtype=torch.bfloat16,
            device=device,
        )
        for name, value in matrices.items():
            layer.projection.projections[name].weight.data.copy_(value)
        layer.attention = vsa.RegionAttention(
            vsa.BlockAttention(WIDTH**-0.5, tile_size=REGION_TILE)
        )
        parallelize_(
            layer,
            mesh,
            attention=AttentionParallelConfig(heads=Ulysses("heads")),
        )
        regions = vsa.Regions(REGION_TILE, ROWS, *region_tables(device))
        shard = ROWS // MEMBERS
        rows = slice(rank * shard, (rank + 1) * shard)
        local = HEADS // MEMBERS

        with torch.inference_mode(), ExecutionContext(layer) as execution:
            execution.prepare(None)
            # The gathered sequence of this rank's head shard, in row order.
            chunks = sorted(
                layer.projection.forward_chunks(
                    hidden[rows], token_slice=rows, num_tokens=ROWS
                ),
                key=lambda chunk: chunk[0].start,
            )
            q, k, v, gate = (
                torch.cat([values[name] for _, values in chunks]).view(
                    ROWS, local, WIDTH
                )
                for name in matrices
            )
            with execution.activate():
                actual = layer.attention(q, k, v, gate, regions)

            projections = [
                F.linear(hidden, value).view(ROWS, HEADS, WIDTH)
                for value in matrices.values()
            ]
            expected, attending = region_reference(*projections)
            assert actual.shape == (shard, HEADS, WIDTH)
            # The established BF16 attention tolerance of the VSA tests.
            torch.testing.assert_close(
                actual[attending[rows]],
                expected[rows][attending[rows]],
                rtol=2e-2,
                atol=2e-2,
            )
            torch.cuda.synchronize(device)


def test_region_attention_keeps_values_through_ulysses(tmp_path):
    if torch.cuda.device_count() < MEMBERS:
        pytest.skip(f"Ulysses region attention needs {MEMBERS} devices")
    mp.spawn(
        _run,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=MEMBERS,
        join=True,
    )
