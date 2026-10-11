"""Collective rendezvous configuration for worker launches."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args

pytestmark = pytest.mark.unit


def test_an_expert_parallel_replica_names_its_world_and_store(tmp_path):
    args = worker_args(
        tmp_path,
        expert_parallel={
            "rank": 0,
            "size": 4,
            "attention_ranks": 0,
            "address": "10.0.0.2:29600",
            "listen_fd": 9,
            "exchange": "megamoe",
        },
    )
    world = args.expert_parallel
    assert (world.rank, world.size) == (0, 4)
    assert (world.rendezvous.host, world.rendezvous.port) == (
        "10.0.0.2",
        29600,
    )
    assert world.rendezvous.listen_fd == 9
    assert args.execution.expert_exchange == "megamoe"


@pytest.mark.parametrize(
    "role,rank,world_size,union_rank",
    [("model", 1, 2, 3), ("experts", 1, 4, 5)],
)
def test_disaggregated_workers_keep_local_ranks_and_request_ownership(
    tmp_path, role, rank, world_size, union_rank
):
    args = worker_args(
        tmp_path,
        role=role,
        expert_microbatches=2,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        components={}
        if role == "experts"
        else {
            "model": {
                "ranks": list(range(world_size)),
                "parallel_config": {"tensor_parallel_size": world_size},
            }
        },
        expert_parallel={
            "rank": union_rank,
            "size": 8,
            "attention_ranks": 4,
            "address": "10.0.0.2:29600",
            "listen_fd": None,
            "exchange": "deepep",
        },
    )
    assert (args.execution.rank, args.execution.world_size) == (
        rank,
        world_size,
    )
    assert args.expert_parallel.rank == union_rank
    assert args.expert_parallel.attention_ranks == 4
    assert args.execution.expert_microbatches == 2
