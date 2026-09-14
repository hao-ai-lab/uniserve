"""CUDA capture storage and delayed output consumption through public entries."""

from contextlib import nullcontext

import pytest
import torch

from uniserve.attention.metadata import ExpertRoute
from uniserve.model.model import Model
from uniserve.nn.branch import branch
from uniserve.runtime.branches import bind_branches
from uniserve.runtime.cuda_graph import CudaGraph, GraphExecutionError, capture_pools
from uniserve.tensors import BufferConfig
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.batch import ExecutionOutput
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.protocol.batch import DType, OutputInfo, ShapeBound, StaticDim

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("compute_fails", [False, True])
def test_initial_inputs_are_ready_for_consumption_after_preparation(compute_fails):
    device = torch.device("cuda", 0)
    model = Model()
    runner = ModelRunner(model, WorkerConfig(device=str(device)))
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
@pytest.mark.parametrize(
    "branch_device",
    [
        "cuda:0",
        pytest.param(
            "cuda:1",
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
        ),
    ],
)
@torch.inference_mode()
def test_graph_outputs_survive_independent_and_ordered_replays(shared_pool, branch_device):
    device = torch.device("cuda", 0)
    modules = torch.nn.ModuleList(
        branch(torch.nn.Linear(1, 1, device=branch_device), ExpertRoute.FLOW) for _ in range(2)
    )
    modules[0].weight.fill_(2)
    modules[0].bias.zero_()
    modules[1].weight.fill_(1)
    modules[1].bias.fill_(7)
    bind_branches(modules, device=device, flow_device=branch_device)
    first_input = torch.zeros((4096, 1), device=device)
    second_input = torch.zeros_like(first_input)
    first_stream = torch.cuda.Stream(device=device)
    second_stream = first_stream if shared_pool else torch.cuda.Stream(device=device)
    pool = torch.cuda.graph_pool_handle() if shared_pool else None
    current = torch.cuda.current_stream(device)
    first_stream.wait_stream(current)
    second_stream.wait_stream(current)
    additional = (torch.device(branch_device),) if branch_device != str(device) else ()
    first_pools = capture_pools(additional)
    second_pools = first_pools if shared_pool else capture_pools(additional)
    first = CudaGraph(device=device, stream=first_stream, pool=pool, device_pools=first_pools)
    second = CudaGraph(device=device, stream=second_stream, pool=pool, device_pools=second_pools)
    first.capture(lambda: modules[0](first_input), keepalive=(first_input,))
    second.capture(lambda: modules[1](second_input), keepalive=(second_input,))
    # Shared storage requires publication before a different executable runs.
    # Independent entries retain their outputs across the other stream's work.
    retained = []
    source = torch.arange(first_input.numel(), device=device, dtype=first_input.dtype).reshape_as(
        first_input
    )
    first_stream.wait_stream(current)
    second_stream.wait_stream(current)
    for offset in (1, 3):
        with torch.cuda.stream(first_stream):
            first_input.copy_(source)
            first_value = first.replay()
            if shared_pool:
                published = ExecutionOutput((first_value,)).clone().values[0]
                retained.append((published, source * 2))
        with torch.cuda.stream(second_stream):
            second_input.copy_(source)
            second_value = second.replay()
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
    with pytest.raises(GraphExecutionError, match="closed"):
        first.replay()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "branch_device",
    [
        "cuda:0",
        pytest.param(
            "cuda:1",
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
        ),
    ],
)
def test_model_modules_replay_with_independent_outputs_and_retire(branch_device):
    device = torch.device("cuda", 0)
    model = Model()
    model.add_module("squared_error", branch(torch.nn.MSELoss(reduction="none"), ExpertRoute.FLOW))
    projection = torch.nn.Linear(32, 32, device=branch_device)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(32, device=branch_device) * 2)
        projection.bias.fill_(7)
    model.add_module("projection", branch(projection, ExpertRoute.FLOW))
    bind_branches(model, device=device, flow_device=branch_device)
    schema = BufferConfig((4, 32), torch.float32)
    result = OutputInfo("values", DType.F32, ShapeBound((StaticDim(4), StaticDim(32))))
    runner = ModelRunner(model, WorkerConfig(device=str(device), generation_device=branch_device))
    try:
        inputs = tuple(
            torch.zeros(schema.shape, dtype=schema.dtype, device=device) for _ in range(2)
        )
        runner.bind_module("squared_error", model.squared_error, inputs=inputs, outputs=(result,))
        runner.bind_module(
            "projection", model.projection, inputs=(torch.zeros_like(inputs[0]),), outputs=(result,)
        )
        runner.prepare_fixed_modules()
        runner.complete_startup()
        conditioning = torch.full((4, 32), 3.0, device=device)
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

        source = source.transpose(0, 1).contiguous().transpose(0, 1)
        with pytest.raises(GraphExecutionError, match="geometry"):
            runner.run_entry("projection", source[:1])
        (projected,) = runner.run_entry("projection", source).values
        torch.testing.assert_close(projected, source * 2 + 7)
    finally:
        torch.cuda.synchronize(device)
        runner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
@pytest.mark.parametrize("during_capture", [False, True])
def test_capture_restores_state_and_failed_capture_preserves_other_computations(during_capture):
    device = torch.device("cuda", 0)
    state = torch.full((32,), 7.0, device=device)
    baseline = state.clone()
    stream = torch.cuda.Stream(device=device)
    graph = CudaGraph(device=device, stream=stream)
    failed = CudaGraph(device=device, stream=stream)

    def advance():
        state.add_(3)
        return state * 2

    def rejected():
        state.add_(100)
        if not during_capture or torch.cuda.is_current_stream_capturing():
            raise ValueError("invalid numerical input")
        return state * 2

    try:
        graph.capture(advance, keepalive=(state,), restore=lambda: state.copy_(baseline))
        torch.testing.assert_close(state, baseline, rtol=0, atol=0)
        with pytest.raises(GraphExecutionError, match="already captured"):
            graph.capture(advance)
        with pytest.raises(ValueError, match="invalid numerical input"):
            failed.capture(rejected, restore=lambda: state.copy_(baseline))
        torch.testing.assert_close(state, baseline, rtol=0, atol=0)
        with pytest.raises(GraphExecutionError, match="not captured"):
            failed.replay()
        torch.testing.assert_close(graph.replay(), (baseline + 3) * 2, rtol=0, atol=0)
        torch.testing.assert_close(state, baseline + 3, rtol=0, atol=0)
    finally:
        torch.cuda.current_stream(device).synchronize()
        graph.close()
        graph.close()
        failed.close()
    with pytest.raises(GraphExecutionError, match="closed"):
        graph.capture(advance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize(
    "branch_device",
    [
        "cuda:0",
        pytest.param(
            "cuda:1",
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
        ),
    ],
)
@torch.inference_mode()
def test_denoising_first_use_and_slot_geometry_changes_advance_one_step(graphs, branch_device):
    from tests.python.fixtures.diffusion import LinearDenoiser
    from uniserve.model.media import ImageSize
    from uniserve.nn.diffusion.schedule import DiffusionSchedule
    from uniserve_worker.execution.denoising import denoising_batch
    from uniserve_worker.execution.diffusion_runner import DiffusionRunner

    device = torch.device("cuda", 0)
    model = LinearDenoiser().to(branch_device)
    branch(model.projection, ExpertRoute.FLOW)
    bind_branches(model, device=device, flow_device=branch_device)
    schedule = DiffusionSchedule.build((1000, 500), (1.0,), scale=1000.0, device=device)
    capture_stream = torch.cuda.Stream(device=device) if graphs else None
    runner = DiffusionRunner(
        model,
        device=device,
        capture_stream=capture_stream,
        groups=(),
        capacity=2,
        additional_devices=(torch.device(branch_device),) if branch_device != str(device) else (),
    )
    try:
        for index, (slot, width, geometry) in enumerate(
            ((1, 32, 2), (1, 64, 3), (2, 32, 2), (1, 32, 2))
        ):
            sample = torch.full((width,), 7.0, device=device)
            reference = sample.clone()
            # Reusing a released input key must consume the replacement constants.
            offset = float(geometry + index)
            constants = runner.prepare_inputs(
                geometry, lambda: {"offset": torch.tensor(offset, device=device)}
            )
            shape = ImageSize(1, width)
            runner.warmup(
                denoising_batch(model, shape, {"image": sample}, schedule, 0),
                schedule,
                state={},
                constants=constants,
                scratch={},
                input_key=geometry,
            )
            torch.testing.assert_close(sample, reference, rtol=0, atol=0)
            for step in (0, 1):
                # The two declared sigma intervals are both exactly one half.
                reference.add_(0.5 * (reference * 0.25 + offset))
                (actual,), _ = runner.step(
                    denoising_batch(model, shape, {"image": sample}, schedule, step),
                    schedule,
                    state={},
                    constants=constants,
                    scratch={},
                    slot=slot,
                    input_key=geometry,
                )
                torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
            torch.cuda.current_stream(device).synchronize()
            runner.release_inputs(geometry)
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
