"""A streamed projection gathers the OmniRef reference layout's sequence.

The FastH3 OmniRef reference layout gathers a 95,744-row sequence of
5376-wide BF16 rows over a four-member Ulysses group: a 1,029,439,488-byte
gather transport. Every gathered row must equal the unpartitioned
projection.
"""

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.nn import ColumnParallelLinear
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.runtime import (
    CUDAStream,
    ExecutionContext,
    initialize_process_groups,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

MEMBERS, ROWS, WIDTH, OUTPUT = 4, 95_744, 5376, 64


def _rows(rows: slice, device) -> torch.Tensor:
    """Deterministic BF16 input rows of the complete sequence."""
    index = torch.arange(rows.start, rows.stop, device=device)[:, None]
    column = torch.arange(WIDTH, device=device)[None, :]
    return ((index * 7 + column) % 1009 / 1009 - 0.5).bfloat16()


@torch.inference_mode()
def _run(rank, rendezvous):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=MEMBERS,
        device=device,
        init_method=rendezvous,
    ) as groups:
        mesh = groups.bind(
            DeviceMesh(
                ranks=tuple(range(MEMBERS)),
                shape=(MEMBERS,),
                axes=("tokens",),
                rank=rank,
            ),
            device=device,
        )
        layer = ColumnParallelLinear(
            WIDTH, OUTPUT, bias=False, device=device, dtype=torch.bfloat16
        )
        parallelize_(
            layer,
            mesh,
            attention=AttentionParallelConfig(heads=Ulysses("tokens")),
        )
        weight = (
            torch.arange(OUTPUT * WIDTH, device=device)
            .reshape(OUTPUT, WIDTH)
            .float()
            .cos()
            .bfloat16()
        )
        layer.weight = nn.Parameter(weight.clone(), requires_grad=False)

        shard = ROWS // MEMBERS
        interval = slice(rank * shard, (rank + 1) * shard)
        local = _rows(interval, device)
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        covered = 0
        # A context prepared without a token bound sizes the gather transport
        # by each call's own payload, as a denoiser's context does.
        with stream, ExecutionContext(layer, stream=stream) as context:
            context.prepare(None)
            for selected, value in layer.forward_chunks(
                local, token_slice=interval, num_tokens=ROWS
            ):
                expected = F.linear(
                    _rows(selected, device).float(), weight.float()
                ).bfloat16()
                # The established BF16 projection tolerance of the streamed
                # projection tests.
                torch.testing.assert_close(
                    value, expected, rtol=2e-2, atol=2e-2
                )
                covered += value.shape[0]
            torch.cuda.synchronize(device)
        assert covered == ROWS


def test_projection_gathers_a_reference_layout_sequence(tmp_path):
    if torch.cuda.device_count() < MEMBERS:
        pytest.skip(f"the reference layout gather needs {MEMBERS} devices")
    mp.spawn(
        _run,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=MEMBERS,
        join=True,
    )
