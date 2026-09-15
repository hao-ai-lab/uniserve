"""Collective values and output storage follow logical rank order."""

import pytest
import torch
import torch.multiprocessing as mp

from torch.distributed.tensor import Replicate, Shard

from uniserve.distributed import DeviceMesh, Distribution
from uniserve.quantization import Quantizer
from uniserve.runtime.process_groups import initialize_process_groups

pytestmark = pytest.mark.integration


def _communicate(rank, rendezvous):
    with initialize_process_groups(
        rank=rank, local_rank=rank, world_size=2, device="cpu", init_method=rendezvous
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(ranks=(1, 0), shape=(2, 1), axes=("tensor", "replica"), rank=rank),
            device="cpu",
        )
        group = mesh.submesh(("replica", "tensor")).get_group(("replica", "tensor"))
        assert group.global_rank == rank
        assert group.rank == 1 - rank
        value = torch.arange(6, dtype=torch.float32).view(2, 3) + group.rank * 10
        peers = [
            torch.arange(6, dtype=torch.float32).view(2, 3) + member * 10 for member in range(2)
        ]
        for op, expected in (("sum", peers[0] + peers[1]), ("min", peers[0]), ("max", peers[1])):
            out = torch.empty_like(value)
            assert group.all_reduce(value, op=op, out=out) is out
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(value, peers[group.rank], rtol=0, atol=0)
        for dim in (0, 1):
            expected = torch.cat(peers, dim=dim)
            # A transposed destination exercises the public arbitrary-dimension
            # contract independently of the backend's contiguous transport.
            out = torch.empty(tuple(reversed(expected.shape))).T
            assert group.all_gather(value, dim=dim, out=out) is out
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            destination = torch.empty_like(out) if group.rank == 1 else None
            result = group.gather(value, dst=1, dim=dim, out=destination)
            if group.rank == 1:
                assert result is destination
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
            else:
                assert result is None
        broadcast = torch.empty_like(value)
        assert group.broadcast(value, src=1, out=broadcast) is broadcast
        torch.testing.assert_close(broadcast, peers[1], rtol=0, atol=0)
        reduced = torch.empty(1, 3)
        assert group.reduce_scatter(value, out=reduced) is reduced
        torch.testing.assert_close(
            reduced, (peers[0] + peers[1]).chunk(2)[group.rank], rtol=0, atol=0
        )
        splits = (1, 2) if group.rank == 0 else (2, 1)
        source = torch.arange(3, dtype=torch.float32) + group.rank * 10
        exchanged = torch.empty_like(source)
        assert (
            group.all_to_all(source, input_splits=splits, output_splits=splits, out=exchanged)
            is exchanged
        )
        expected = torch.tensor([0.0, 10.0, 11.0] if group.rank == 0 else [1.0, 2.0, 12.0])
        torch.testing.assert_close(exchanged, expected, rtol=0, atol=0)
        received = torch.empty_like(value)
        assert (
            group.send_recv(value, dst=1 - group.rank, src=1 - group.rank, out=received) is received
        )
        torch.testing.assert_close(received, peers[1 - group.rank], rtol=0, atol=0)
        if group.rank == 0:
            group.send(value, dst=1)
        else:
            assert group.recv(src=0, out=received) is received
            torch.testing.assert_close(received, peers[0], rtol=0, atol=0)

        # K shards must share per-token statistics. An empty token shard still
        # participates in tensor-wide maxima for the same logical source.
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) * (100 if group.rank else 1)
        per_token = Quantizer("fp8", axis=0)
        distribution = Distribution(mesh, (Shard(1), Replicate()))
        torch.testing.assert_close(
            per_token.amax(source, distribution=distribution),
            torch.tensor([[200.0], [400.0]]),
            rtol=0,
            atol=0,
        )
        source = torch.empty((0, 2)) if group.rank == 0 else torch.tensor([[1.0, 200.0]])
        distribution = Distribution(mesh, (Shard(0), Replicate()))
        torch.testing.assert_close(
            Quantizer("fp8").amax(source, distribution=distribution),
            torch.tensor(200.0),
            rtol=0,
            atol=0,
        )


def test_collectives_preserve_values_and_outputs(tmp_path):
    mp.spawn(_communicate, args=((tmp_path / "world").as_uri(),), nprocs=2, join=True)
