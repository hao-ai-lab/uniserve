"""Request slots fit every rank, including stateless component ranks."""

import pytest
import torch.multiprocessing as mp

from uniserve.runtime.process_groups import initialize_process_groups
from uniserve_worker.bootstrap.capacity import tensor_slot_capacity

pytestmark = pytest.mark.integration


def _agree_capacity(rank, rendezvous):
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device="cpu",
        backend="gloo",
        init_method=rendezvous,
    ) as environment:
        # The stateless owner fits three 100-byte product/arena reservations;
        # the stateful owner fits four complete 112-byte reservations.
        capacity = tensor_slot_capacity(
            [slots * (100 if rank == 0 else 112) for slots in range(2, 5)],
            environment.process_group,
            minimum=2,
            available_bytes=300 if rank == 0 else 448,
        )
        assert capacity == 3

        # Rank zero fits two or four requests; rank one fits two or three.
        # The common maximum is two, not the minimum of the local maxima.
        capacity = tensor_slot_capacity(
            [100, 400, 300] if rank == 0 else [100, 200, 400],
            environment.process_group,
            minimum=2,
            available_bytes=300,
        )
        assert capacity == 2

        # Both owners must reject admission when either cannot fit the minimum.
        with pytest.raises(
            RuntimeError, match="common request tensor slot count"
        ):
            tensor_slot_capacity(
                [slots * (100 if rank == 0 else 112) for slots in range(2, 5)],
                environment.process_group,
                minimum=2,
                available_bytes=199 if rank == 0 else 448,
            )


def test_request_capacity_includes_stateless_owners(tmp_path):
    mp.spawn(
        _agree_capacity,
        ((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
