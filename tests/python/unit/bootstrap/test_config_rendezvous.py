"""Collective rendezvous configuration for worker launches."""

from __future__ import annotations

import pytest

from tests.python.fixtures.launch import worker_args

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("rank", "address", "listen_fd"),
    [
        # A first rank without its inherited socket would bind the store's
        # port itself, which another process can take while the rank starts.
        (0, "127.0.0.1:29500", None),
        # Only the first rank serves the store; the others connect to it.
        (1, "127.0.0.1:29500", 7),
        # A socket names no store without the address the ranks connect to.
        (0, None, 7),
    ],
)
def test_the_store_socket_is_inherited_by_exactly_the_first_rank(
    rank, address, listen_fd, tmp_path, capsys
):
    with pytest.raises(SystemExit):
        worker_args(
            tmp_path,
            rank=rank,
            local_rank=rank,
            world_size=2,
            components={
                "model": {
                    "ranks": [0, 1],
                    "parallel_config": {"tensor_parallel_size": 2},
                }
            },
            rendezvous_address=address,
            rendezvous_listen_fd=listen_fd,
        )
    assert "rendezvous socket" in capsys.readouterr().err


def test_an_expert_parallel_replica_names_its_world_and_store(tmp_path):
    args = worker_args(
        tmp_path,
        expert_parallel={
            "rank": 0,
            "size": 4,
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
    ("world", "message"),
    [
        # Only the world's rank 0 serves its store, from an inherited socket.
        (
            {
                "rank": 1,
                "size": 4,
                "address": "h:1",
                "listen_fd": 9,
                "exchange": "alltoall",
            },
            "rank 0",
        ),
        (
            {
                "rank": 0,
                "size": 4,
                "address": "h:1",
                "listen_fd": None,
                "exchange": "alltoall",
            },
            "rank 0",
        ),
        # One replica shares no experts.
        (
            {
                "rank": 0,
                "size": 1,
                "address": "h:1",
                "listen_fd": 9,
                "exchange": "alltoall",
            },
            "two",
        ),
        (
            {
                "rank": 4,
                "size": 4,
                "address": "h:1",
                "listen_fd": None,
                "exchange": "alltoall",
            },
            "rank",
        ),
    ],
)
def test_a_malformed_expert_parallel_world_is_refused(
    world, message, tmp_path, capsys
):
    with pytest.raises(SystemExit):
        worker_args(tmp_path, expert_parallel=world)
    assert message in capsys.readouterr().err


def test_an_expert_parallel_replica_is_one_rank(tmp_path, capsys):
    with pytest.raises(SystemExit):
        worker_args(
            tmp_path,
            rank=0,
            local_rank=0,
            world_size=2,
            components={
                "model": {
                    "ranks": [0, 1],
                    "parallel_config": {"tensor_parallel_size": 2},
                }
            },
            rendezvous_address="127.0.0.1:29500",
            rendezvous_listen_fd=7,
            expert_parallel={
                "rank": 0,
                "size": 2,
                "address": "h:1",
                "listen_fd": 9,
                "exchange": "alltoall",
            },
        )
    assert "only rank" in capsys.readouterr().err
