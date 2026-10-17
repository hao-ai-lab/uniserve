"""A chunked head exchange returns the whole exchange's rows by interval.

Each member receives, interval by interval and in token order, exactly the
rows ``HeadExchange.tokens`` returns for the same head-shard rows: with the
logical members opposite to backend order, with a shard spanning several
exchange intervals, and when a captured exchange replays over new values. A
consumer that stops after the first interval leaves no transfer writing the
exchange's storage.
"""

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from uniserve.distributed import DeviceMesh
from uniserve.distributed.chunks import exchange_rows
from uniserve.distributed.tokens import HeadExchange
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    initialize_process_groups,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

HEADS, FEATURES = 2, 128


class _Exchange(nn.Module):
    """Declare the exchange group so an execution context binds it."""

    def __init__(self, group):
        super().__init__()
        self.exchange = HeadExchange(group)
        self.communication_groups = (group,)


@torch.inference_mode()
def _run(rank, rendezvous):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        init_method=rendezvous,
    ) as groups:
        # Logical members run opposite to backend order.
        mesh = groups.bind(
            DeviceMesh(ranks=(1, 0), shape=(2,), axes=("heads",), rank=rank),
            device=device,
        )
        group = mesh.get_group("heads")
        module = _Exchange(group)
        # Two whole exchange intervals and a partial third one per member.
        step = exchange_rows(HEADS * FEATURES * 2, 1)
        rows = 2 * step + 384
        generator = torch.Generator(device=device).manual_seed(rank)
        values = torch.randn(
            2 * rows, HEADS, FEATURES, device=device, generator=generator
        ).bfloat16()

        def compose(interval, payload, slots):
            # Member m's tokens of this member's heads are its rows onward.
            for member, slot in enumerate(slots):
                start = member * rows + interval.start
                payload[slot].copy_(
                    values[start : start + interval.stop - interval.start]
                )

        def chunks():
            return module.exchange.token_chunks(
                rows,
                HEADS,
                FEATURES,
                dtype=values.dtype,
                device=device,
                compose=compose,
            )

        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        with stream, ExecutionContext(module, stream=stream) as context:
            context.prepare(None)
            expected = module.exchange.tokens(values)
            received = list(chunks())
            assert [interval for interval, _ in received] == [
                slice(start, min(start + step, rows))
                for start in range(0, rows, step)
            ]
            assert torch.equal(
                torch.cat([value for _, value in received]), expected
            )

            partial = chunks()
            first, value = next(partial)
            assert torch.equal(value, expected[first])
            partial.close()

            captured = torch.empty_like(expected)
            with CUDAGraph(context=context) as graph:
                graph.capture(
                    lambda: captured.copy_(
                        torch.cat([value for _, value in chunks()])
                    )
                )
                values.neg_()
                graph.replay()
                assert torch.equal(captured, module.exchange.tokens(values))
            torch.cuda.synchronize(device)


def test_chunked_head_exchange_returns_the_whole_exchange(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("the head exchange needs two devices")
    mp.spawn(
        _run,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
