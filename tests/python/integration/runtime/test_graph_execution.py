"""CUDA capture storage and delayed output consumption through public entries."""

from contextlib import nullcontext

import pytest
import torch

from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.batch import DType, ShapeBound, StaticDim, TensorSpec
from uniserve_worker.execution.bounded_storage import TensorSchema
from uniserve_worker.execution.cuda_graph import GraphEntry, GraphExecutionError
from uniserve_worker.execution.forward_batch import ForwardOutput
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.execution.trace import ExecutionTrace
from uniserve_worker.models.runtime import ExecutionModel, ResourceGeometry

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("compute_fails", [False, True])
def test_initial_inputs_are_ready_for_consumption_after_preparation(compute_fails):
    device = torch.device("cuda", 0)
    model = ExecutionModel()
    model.resource_geometry = ResourceGeometry(
        kv=False, request_tensors={"input": TensorSchema((4096,), torch.float32)}
    )
    runner = ModelRunner(model, WorkerConfig(device=str(device)), ExecutionTrace("initial_inputs"))
    source = torch.empty(4 * 1024 * 1024, dtype=torch.float32, pin_memory=True).fill_(7)
    destination = torch.empty_like(source, device=device)
    try:
        for value in (7, 13):
            source.fill_(value)
            error = (
                pytest.raises(ValueError, match="compute rejected")
                if compute_fails
                else nullcontext()
            )
            with error:
                with runner.preparing_inputs(((destination, source),)):
                    if compute_fails:
                        raise ValueError("compute rejected")
            # This GPU consumer uses the calling stream. Copy completion must
            # precede it even when independent preparation computation failed.
            actual = (destination + 1).cpu()
            torch.testing.assert_close(actual, torch.full_like(source, value + 1), rtol=0, atol=0)
    finally:
        runner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shared_pool", [False, True])
def test_graph_outputs_survive_independent_and_ordered_replays(shared_pool):
    device = torch.device("cuda", 0)
    first_input = torch.zeros(4096, device=device)
    second_input = torch.zeros_like(first_input)
    first_stream = torch.cuda.Stream(device=device)
    second_stream = first_stream if shared_pool else torch.cuda.Stream(device=device)
    pool = torch.cuda.graph_pool_handle() if shared_pool else None
    current = torch.cuda.current_stream(device)
    first_stream.wait_stream(current)
    second_stream.wait_stream(current)
    first = GraphEntry.capture(
        lambda: first_input * 2,
        inputs=(first_input,),
        stream=first_stream,
        pool=pool,
    )
    second = GraphEntry.capture(
        lambda: second_input + 7,
        inputs=(second_input,),
        stream=second_stream,
        pool=pool,
    )
    # Shared storage requires publication before a different executable runs.
    # Independent entries retain their outputs across the other stream's work.
    retained = []
    source = torch.arange(first_input.numel(), device=device, dtype=first_input.dtype)
    first_stream.wait_stream(current)
    second_stream.wait_stream(current)
    for offset in (1, 3):
        with torch.cuda.stream(first_stream):
            first_value = first.replay(source)
            if shared_pool:
                published = ForwardOutput((first_value,)).clone().values[0]
                retained.append((published, source * 2))
        with torch.cuda.stream(second_stream):
            second_value = second.replay(source)
        current.wait_stream(first_stream)
        current.wait_stream(second_stream)
        torch.testing.assert_close(published if shared_pool else first_value, source * 2)
        torch.testing.assert_close(second_value, source + 7)
        source.add_(offset)
        first_stream.wait_stream(current)
        second_stream.wait_stream(current)
    current.synchronize()
    for published, expected in retained:
        torch.testing.assert_close(published, expected, rtol=0, atol=0)
    first.close()
    second.close()
    with pytest.raises(GraphExecutionError, match="retired"):
        first.replay(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_model_modules_replay_with_independent_outputs_and_retire():
    device = torch.device("cuda", 0)
    model = ExecutionModel()
    model.resource_geometry = ResourceGeometry(kv=False)
    model.add_module("squared_error", torch.nn.MSELoss(reduction="none"))
    projection = torch.nn.Linear(32, 32, device=device)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(32, device=device) * 2)
        projection.bias.fill_(7)
    model.add_module("projection", projection)
    schema = TensorSchema((4, 32), torch.float32)
    model.capture_inputs = {
        "squared_error": (schema, schema),
        "projection": (schema,),
    }
    result = TensorSpec("values", DType.F32, ShapeBound((StaticDim(4), StaticDim(32))))
    model.entry_outputs = {"squared_error": (result,), "projection": (result,)}
    model.scratch_schema = {"conditioning": TensorSchema((4, 32), torch.float32, fill=3)}
    runner = ModelRunner(model, WorkerConfig(device=str(device)), ExecutionTrace("tensor_modules"))
    try:
        runner.capture_modules()
        runner.complete_startup()
        assert runner.scratch is not None
        conditioning = runner.scratch.capacity["conditioning"]
        torch.testing.assert_close(conditioning, torch.full((4, 32), 3.0, device=device))
        source = torch.arange(128, device=device, dtype=torch.float32).reshape(4, 32)
        current = torch.cuda.current_stream(device)
        first_stream = torch.cuda.Stream(device=device)
        second_stream = torch.cuda.Stream(device=device)
        for offset in (1, 3):
            first_stream.wait_stream(current)
            second_stream.wait_stream(current)
            with torch.cuda.stream(first_stream):
                (squared,) = runner.run_entry("squared_error", source, conditioning).values
            with torch.cuda.stream(second_stream):
                (projected,) = runner.run_entry("projection", source).values
            current.wait_stream(first_stream)
            current.wait_stream(second_stream)
            torch.testing.assert_close(squared, (source - 3).square())
            torch.testing.assert_close(projected, source * 2 + 7)
            source.add_(offset)

        with pytest.raises(GraphExecutionError, match="geometry"):
            runner.run_entry("projection", source[:1])
        (projected,) = runner.run_entry("projection", source).values
        torch.testing.assert_close(projected, source * 2 + 7)
        current.synchronize()
        runner.synchronize()
        runner.invalidate_graphs(1)
        with pytest.raises(GraphExecutionError, match="no resident capture"):
            runner.run_entry("projection", source)
    finally:
        torch.cuda.synchronize(device)
        runner.close()
