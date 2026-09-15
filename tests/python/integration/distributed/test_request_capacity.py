"""Request capacity agreement includes stateless component ranks."""

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.runtime.process_groups import initialize_process_groups
from uniserve.tensors import BufferConfig
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
        schema = {} if rank == 0 else {"state": BufferConfig((4,), torch.float32)}
        # The stateless owner fits three 100-byte product/arena reservations;
        # the stateful owner fits four complete 112-byte reservations.
        capacity = tensor_slot_capacity(
            schema,
            environment.process_group,
            maximum=4,
            minimum=2,
            available_bytes=300 if rank == 0 else 448,
            auxiliary_bytes=lambda slots: slots * (100 if rank == 0 else 96),
        )
        assert capacity == 3

        # Both owners must reject admission when either cannot fit the minimum.
        with pytest.raises(RuntimeError, match="common request tensor slot count"):
            tensor_slot_capacity(
                schema,
                environment.process_group,
                maximum=4,
                minimum=2,
                available_bytes=199 if rank == 0 else 448,
                auxiliary_bytes=lambda slots: slots * (100 if rank == 0 else 96),
            )


def test_request_capacity_includes_stateless_owners(tmp_path):
    mp.spawn(_agree_capacity, ((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
