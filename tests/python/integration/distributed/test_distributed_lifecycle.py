"""Distributed construction scopes preserve ownership.

They also preserve the original failure.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from uniserve.runtime.process_groups import initialize_process_groups

pytestmark = pytest.mark.integration


def _construction_scope(rank, directory, outcome):
    with pytest.MonkeyPatch.context() as patch:
        failure = RuntimeError("execution construction failed")
        destroy_process_group = dist.destroy_process_group

        def failing_destroy(group):
            destroy_process_group(group)
            raise RuntimeError("distributed teardown failed")

        try:
            if outcome == "cleanup_error":
                # PyTorch is the external resource boundary. Release the real
                # group before reporting a teardown error to the caller.
                patch.setattr(dist, "destroy_process_group", failing_destroy)

            try:
                with initialize_process_groups(
                    rank=rank,
                    local_rank=rank,
                    world_size=2,
                    device="cpu",
                    init_method=f"file://{directory}/{outcome}",
                ):
                    if outcome == "body_error":
                        # Both callers must finish construction before either
                        # exits; otherwise this tests a rendezvous failure.
                        dist.barrier()
                        raise failure
                    value = torch.tensor([rank + 1])
                    dist.all_reduce(value)
                    assert value.item() == 3

                    # A nested owner borrows this world. Closing its own scope
                    # must leave the enclosing caller's collective usable.
                    with initialize_process_groups(
                        rank=rank,
                        local_rank=rank,
                        world_size=2,
                        device="cpu",
                    ):
                        borrowed = torch.tensor([rank + 1])
                        dist.all_reduce(borrowed)
                        assert borrowed.item() == 3
                    assert dist.is_initialized()
            except RuntimeError as error:
                if outcome == "body_error":
                    assert error is failure
                else:
                    assert outcome == "cleanup_error"
                    assert str(error) == "distributed teardown failed"
            else:
                assert outcome == "success"

        finally:
            patch.undo()
    if outcome == "body_error":
        # Aborted scopes retain physical groups until process exit. Peers may
        # still be serving, so teardown cannot wait on their collectives.
        assert dist.is_initialized()
        os._exit(0)
    assert not dist.is_initialized()


@pytest.mark.parametrize("outcome", ("success", "body_error", "cleanup_error"))
def test_distributed_construction_scope_ownership_and_errors(tmp_path, outcome):
    mp.spawn(
        _construction_scope,
        args=(str(tmp_path), outcome),
        nprocs=2,
        join=True,
    )
