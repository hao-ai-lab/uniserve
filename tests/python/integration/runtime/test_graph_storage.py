"""Independent graph pools consume one physical device budget."""

import pytest
import torch

from uniserve.runtime import CUDAStream, ExecutionContext
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve_worker.model_executor.cuda_graph import (
    CUDAGraphRunner,
    Execution,
    GraphBucket,
)
from uniserve_worker.model_executor.graph_storage import GraphStorage

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_independent_graph_owners_share_a_byte_limit_and_retire_capacity():
    device = torch.device("cuda:0")
    storage = GraphStorage(budgets={device: 2 << 20})
    owners = (object(), object())
    try:
        for owner in owners:
            storage.reserve(owner, (device,))
        with storage.allocate(owners[0]):
            first = torch.empty(1 << 19, dtype=torch.uint8, device=device)
        storage.check()
        with storage.allocate(owners[1]):
            second = torch.empty_like(first)
        with pytest.raises(CUDAGraphError, match="exceeds its byte budget"):
            storage.check()
        # A worker can assign unused storage after its fixed pools are sized.
        # The new grant still covers every owner, including existing backing.
        storage.set_budget(device, 4 << 20)
        storage.check()
        with pytest.raises(CUDAGraphError, match="exceeds its byte budget"):
            storage.set_budget(device, 2 << 20)
        del second
        storage.release(owners[1])
        storage.check()
        del first
        storage.release(owners[0])
        storage.check()
    finally:
        storage.close()


def test_shared_pool_is_charged_once_and_survives_the_first_owner():
    device = torch.device("cuda:0")
    storage = GraphStorage(budgets={device: 2 << 20})
    first, second = object(), object()
    storage.reserve(first, (device,))
    storage.reserve(second, (device,), share=first)
    try:
        with storage.allocate(first):
            source = torch.full((1 << 17,), 7, dtype=torch.int32, device=device)
        with storage.allocate(second):
            output = torch.empty_like(source)
        output.copy_(source)
        torch.cuda.synchronize(device)

        storage.check()
        assert storage.pool_bytes() == {device: 2 << 20}
        assert storage.owner_bytes() == {(first, device): 2 << 20}

        del source
        storage.release(first)
        storage.check()
        assert storage.owner_bytes() == {(second, device): 2 << 20}
        assert torch.all(output == 7).item()

        del output
        storage.release(second)
        assert storage.pool_bytes() == {device: 0}
    finally:
        storage.close()


def test_failed_device_allocation_leaves_the_owner_available():
    device = torch.device("cuda:0")
    storage = GraphStorage()
    owner = object()
    try:
        unavailable = torch.device("cuda", torch.cuda.device_count())
        with pytest.raises(RuntimeError):
            storage.reserve(owner, (device, unavailable))

        storage.reserve(owner, ("cpu", device, "cuda:0"))
        with storage.allocate(owner):
            value = torch.ones(16, device=device)
        assert value.cpu().tolist() == [1] * 16
        storage.check()
        del value
        storage.release(owner)
    finally:
        storage.close()


@torch.inference_mode()
def test_execution_retirement_preserves_shared_graph_results():
    device = torch.device("cuda:0")
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    model = torch.nn.Linear(8, 8, bias=False).to(device)
    storage = GraphStorage()
    first = Execution(
        "first",
        ExecutionContext(model, stream=stream),
        devices=(device,),
        storage=storage,
    )
    second = Execution(
        "second",
        ExecutionContext(model, stream=stream),
        devices=(device,),
        storage=storage,
        share=first,
    )
    try:
        inputs = []
        for owner in (first, second):
            owner.context.prepare(None)
            with storage.allocate(owner):
                inputs.append(torch.ones((4, 8), device=device))
        # All persistent inputs precede capture in the shared pool; subsequent
        # graphs may reuse only temporary storage from preceding captures.
        torch.cuda.synchronize(device)
        for owner, value in zip((first, second), inputs, strict=True):
            owner.buckets[4] = GraphBucket(
                {
                    None: CUDAGraphRunner.capture(
                        owner.context,
                        value,
                        model,
                        pools=owner.pools,
                    )
                }
            )

        with first.context.activate():
            retained = first.buckets[4][None].replay().clone()
        torch.cuda.synchronize(device)
        first.close()

        live = torch.full((4, 8), 3.0, device=device)
        stream.wait(torch.cuda.current_stream(device))
        actual = second.buckets[4][None].replay(live)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, model(live))
        torch.testing.assert_close(retained, model(inputs[0]))

        second.close()
        assert storage.pool_bytes() == {device: 0}
    finally:
        torch.cuda.synchronize(device)
        first.close()
        second.close()
        storage.close()
        stream.close()
