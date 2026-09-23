"""Prepared input lifetime and numerical denoising.

Both are exercised across graph residency changes.
"""

from contextlib import nullcontext

import pytest
import torch

from tests.python.fixtures.diffusion import LinearDenoiser, Size
from tests.python.fixtures.encoding import Model
from uniserve.model import DenoiserInput, EntryPoint, LatentInput
from uniserve.runtime import CUDAStream, partition_streams
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.component_binding import Call
from uniserve_worker.model_executor.diffusion_runner import DiffusionRunner
from uniserve_worker.model_executor.graph_storage import GraphStorage

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("compute_fails", [False, True])
def test_initial_inputs_are_ready_for_consumption_after_preparation(
    compute_fails,
):
    device = torch.device("cuda", 0)
    model = Model().to(device)
    runner = ModelExecutor(model, WorkerConfig(device=str(device)))
    source = torch.empty(
        4 * 1024 * 1024, dtype=torch.float32, pin_memory=True
    ).fill_(7)
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
            torch.testing.assert_close(
                actual, torch.full_like(source, value + 1), rtol=0, atol=0
            )
    finally:
        runner.close()


def _runner(model, size, *, device, stream, devices=(), bank=None, slots=2):
    """Prepare the linear denoiser's diffusion runner for one size."""
    return DiffusionRunner.for_layout(
        "denoiser",
        Call("denoiser", model, EntryPoint("forward")),
        size,
        device=device,
        stream=stream,
        storage=GraphStorage(),
        devices=devices,
        bank=bank,
        slots=slots,
    )


def _inputs(schedules, sample, size, steps):
    return tuple(
        DenoiserInput(
            {
                "image": (
                    LatentInput(sample, schedules["image"].timesteps[step]),
                )
            },
            (size,),
            step,
        )
        for step in range(steps)
    )


def _advanced(size, steps, *, device):
    """The linear denoiser's samples after ``steps`` unit-shift steps."""
    value = torch.full((size.width,), 7.0, device=device)
    for _ in range(steps):
        value.add_(0.5 * (value * 0.25 + size.offset))
    return value


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("execution", ["cuda:0", "cuda:1", "green"])
@torch.inference_mode()
def test_denoising_reprepared_constants_and_slot_sizes_advance_one_step(
    graphs, execution
):
    device = torch.device("cuda:0")
    branch_device = "cuda:0" if execution == "green" else execution
    model = LinearDenoiser().to(device)
    model.projection.to(branch_device)
    schedules = model.make_schedules(2, shift=1.0, device=device)
    partition = (
        partition_streams(device, (64,))[0] if execution == "green" else None
    )
    stream = (
        partition
        if partition is not None
        else CUDAStream.external(torch.cuda.Stream(device=device))
    )
    devices = (
        (device,)
        + ((torch.device(branch_device),) if branch_device != "cuda:0" else ())
        if graphs
        else ()
    )
    # Slot storage is one bank with a row per request slot; a captured graph
    # reaches a slot's row through the device slot index.
    bank = torch.zeros((2, 64), device=device)
    try:
        for index, (slot, width, key) in enumerate(
            ((1, 32, 2), (1, 64, 3), (2, 32, 2), (1, 32, 2))
        ):
            sample = bank[slot - 1, :width]
            sample.fill_(7.0)
            reference = sample.clone()
            size = Size(width, float(key + index))
            runner = _runner(
                model,
                size,
                device=device,
                stream=stream,
                devices=devices,
                bank={"image": bank},
            )
            try:
                inputs = _inputs(schedules, sample, size, 2)
                runner.warmup(inputs[0], schedules, state={})
                torch.testing.assert_close(sample, reference, rtol=0, atol=0)
                ladder = runner.bind(
                    inputs, schedules, state={"image": sample}, slot=slot
                )
                if graphs:
                    for step in (0, 1):
                        runner.capture(ladder, step)
                for step in (0, 1):
                    reference.add_(0.5 * (reference * 0.25 + size.offset))
                    actual, path = runner.step(ladder, step)
                    assert path == ("graph_replay" if graphs else "eager")
                    torch.testing.assert_close(
                        actual["image"][0], reference, rtol=1e-6, atol=1e-6
                    )
            finally:
                torch.cuda.current_stream(device).synchronize()
                stream.synchronize()
                runner.close()
    finally:
        stream.close()


@torch.inference_mode()
def test_captured_ladders_replay_on_every_slot_with_eager_values():
    """Startup capture leaves one ladder resident that every slot replays.

    Warmup captures one graph per ladder step; the graph gathers whichever
    slot the device slot index names, so the steps a request runs on either
    slot replay rather than capture, and their values match an eager
    evaluation of the same ladder from the same samples.
    """
    device = torch.device("cuda:0")
    model = LinearDenoiser().to(device)
    steps = 2
    schedules = model.make_schedules(steps, shift=1.0, device=device)
    size = Size(32, 2.0)
    slots = (1, 2)

    def ladder(runner, samples):
        values, paths = {}, []
        for slot in slots:
            bound = runner.bind(
                _inputs(schedules, samples[slot], size, steps),
                schedules,
                state={"image": samples[slot]},
                slot=slot,
            )
            for step in range(steps):
                result, path = runner.step(bound, step)
                paths.append(path)
            values[slot] = result["image"][0].clone()
        return values, paths

    eager = _runner(model, size, device=device, stream=None)
    try:
        expected, _ = ladder(
            eager,
            {slot: torch.full((32,), 7.0, device=device) for slot in slots},
        )
    finally:
        torch.cuda.current_stream(device).synchronize()
        eager.close()

    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    bank = torch.zeros((2, 64), device=device)
    runner = _runner(
        model,
        size,
        device=device,
        stream=stream,
        devices=(device,),
        bank={"image": bank},
    )
    try:
        samples = {slot: bank[slot - 1, :32] for slot in slots}
        for sample in samples.values():
            sample.fill_(7.0)
        for slot in slots:
            resting = samples[slot].clone()
            bound = runner.bind(
                _inputs(schedules, samples[slot], size, steps),
                schedules,
                state={"image": samples[slot]},
                slot=slot,
            )
            for step in range(steps):
                runner.capture(bound, step)
            # Capture must leave the slot's samples where it found them.
            torch.testing.assert_close(samples[slot], resting, rtol=0, atol=0)

        actual, paths = ladder(runner, samples)
        assert paths == ["graph_replay"] * (len(slots) * steps)
        for slot in slots:
            torch.testing.assert_close(
                actual[slot], expected[slot], rtol=1e-6, atol=1e-6
            )
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
        stream.close()


@torch.inference_mode()
def test_steps_without_a_captured_ladder_run_eagerly():
    """Stepping never captures: a step replays only after ``capture``.

    On a capturing runner a request's steps run eagerly, with the eager
    values, until startup capture makes their graphs resident; the same
    ladder then replays with the same values.
    """
    device = torch.device("cuda:0")
    model = LinearDenoiser().to(device)
    steps = 2
    schedules = model.make_schedules(steps, shift=1.0, device=device)
    size = Size(32, 2.0)
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    bank = torch.zeros((1, 64), device=device)
    runner = _runner(
        model,
        size,
        device=device,
        stream=stream,
        devices=(device,),
        bank={"image": bank},
        slots=1,
    )
    try:
        sample = bank[0, : size.width]
        bound = runner.bind(
            _inputs(schedules, sample, size, steps),
            schedules,
            state={"image": sample},
            slot=1,
        )
        for expected_path in ("eager", "graph_replay"):
            sample.fill_(7.0)
            paths = [runner.step(bound, step)[1] for step in range(steps)]
            assert paths == [expected_path] * steps
            torch.testing.assert_close(
                sample,
                _advanced(size, steps, device=device),
                rtol=1e-6,
                atol=1e-6,
            )
            for step in range(steps):
                runner.capture(bound, step)
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
        stream.close()
