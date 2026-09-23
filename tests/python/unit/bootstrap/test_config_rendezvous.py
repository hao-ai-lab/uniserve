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
