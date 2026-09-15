"""Distributed construction scopes preserve ownership and the original failure."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from uniserve.runtime.process_groups import initialize_process_groups

pytestmark = pytest.mark.integration


def _construction_scope(rank, directory):
    for outcome in ("success", "body_error", "cleanup_error"):
        failure = RuntimeError("execution construction failed")
        destroy_process_group = dist.destroy_process_group

        def failing_destroy(group):
            destroy_process_group(group)
            raise RuntimeError("distributed teardown failed")

        with pytest.MonkeyPatch.context() as patch:
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
                    if outcome != "success":
                        raise failure
                    value = torch.tensor([rank + 1])
                    dist.all_reduce(value)
                    assert value.item() == 3
            except RuntimeError as error:
                assert error is failure
                if outcome == "cleanup_error":
                    assert any("distributed teardown failed" in note for note in error.__notes__)
            else:
                assert outcome == "success"

        assert not dist.is_initialized()


def test_distributed_construction_scope_ownership_and_errors(tmp_path):
    mp.spawn(_construction_scope, args=(str(tmp_path),), nprocs=2, join=True)
