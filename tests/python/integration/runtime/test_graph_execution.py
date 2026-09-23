"""Prepared input lifetime and numerical denoising.

Both are exercised across graph residency changes.
"""

from contextlib import nullcontext

import pytest
import torch

from tests.python.fixtures.diffusion import LinearDenoiser, Size
from tests.python.fixtures.encoding import Model
from uniserve.model import DenoiserInput, LatentInput
from uniserve.runtime import CUDAStream, partition_streams
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.diffusion_runner import TrajectoryRunner

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
    runner = TrajectoryRunner(
        model,
        device=device,
        stream=stream,
        capture=graphs,
        groups=(),
        capacity=2,
        additional_devices=(torch.device(branch_device),)
        if branch_device != str(device)
        else (),
    )
    # Slot storage is one bank with a row per request slot; a captured graph
    # reaches a slot's row through the device slot index.
    bank = torch.zeros((2, 64), device=device)
    runner.bind_bank({"image": bank})
    try:
        for index, (slot, width, key) in enumerate(
            ((1, 32, 2), (1, 64, 3), (2, 32, 2), (1, 32, 2))
        ):
            sample = bank[slot - 1, :width]
            sample.fill_(7.0)
            reference = sample.clone()
            size = Size(width, float(key + index))
            runner.prepare_inputs(key, size, pin=graphs)

            def inputs(step):
                return DenoiserInput(
                    {
                        "image": (
                            LatentInput(
                                sample, schedules["image"].timesteps[step]
                            ),
                        )
                    },
                    (size,),
                    step,
                )

            runner.warmup(inputs(0), schedules, state={}, input_key=key)
            torch.testing.assert_close(sample, reference, rtol=0, atol=0)
            trajectory = runner.bind_inputs(
                key,
                tuple(inputs(step) for step in (0, 1)),
                schedules,
                state={"image": sample},
                slot=slot,
            )
            if graphs:
                for step in (0, 1):
                    runner.capture(trajectory, step)
            for step in (0, 1):
                reference.add_(0.5 * (reference * 0.25 + size.offset))
                actual, path = runner.step(trajectory, step)
                assert path == ("graph_replay" if graphs else "eager")
                torch.testing.assert_close(
                    actual["image"][0], reference, rtol=1e-6, atol=1e-6
                )
            torch.cuda.current_stream(device).synchronize()
            runner.release_inputs(key)
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
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

    draws = {
        slot: torch.zeros(32, dtype=torch.float32).pin_memory()
        for slot in slots
    }

    def bind(runner, samples, slot):
        return runner.bind_inputs(
            size,
            tuple(
                DenoiserInput(
                    {
                        "image": (
                            LatentInput(
                                samples[slot],
                                schedules["image"].timesteps[step],
                            ),
                        )
                    },
                    (size,),
                    step,
                )
                for step in range(steps)
            ),
            schedules,
            state={"image": samples[slot], "image_noise": draws[slot]},
            slot=slot,
        )

    def ladder(runner, samples):
        values, paths = {}, []
        for slot in slots:
            trajectory = bind(runner, samples, slot)
            for step in range(steps):
                result, path = runner.step(trajectory, step)
                paths.append(path)
            values[slot] = result["image"][0].clone()
        return values, paths

    eager = TrajectoryRunner(
        model, device=device, stream=None, groups=(), capacity=2
    )
    try:
        eager.prepare_inputs(size, size)
        expected, _ = ladder(
            eager,
            {slot: torch.full((32,), 7.0, device=device) for slot in slots},
        )
    finally:
        torch.cuda.current_stream(device).synchronize()
        eager.close()

    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    runner = TrajectoryRunner(
        model,
        device=device,
        stream=stream,
        groups=(),
        capacity=2,
        shapes=2,
    )
    bank = torch.zeros((2, 64), device=device)
    runner.bind_bank({"image": bank})
    try:
        runner.prepare_inputs(size, size, pin=True)
        samples = {slot: bank[slot - 1, :32] for slot in slots}
        for sample in samples.values():
            sample.fill_(7.0)
        for slot in slots:
            resting = samples[slot].clone()
            trajectory = bind(runner, samples, slot)
            for step in range(steps):
                runner.capture(trajectory, step)
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
def test_steps_without_a_resident_ladder_run_eagerly():
    """Serving never captures: an uncaptured size steps eagerly.

    A size whose ladder startup did not capture runs every step eagerly on a
    capturing runner, with the eager values, and preparing it retires no
    pinned ladder, which still replays afterwards.
    """
    device = torch.device("cuda:0")
    model = LinearDenoiser().to(device)
    steps = 2
    schedules = model.make_schedules(steps, shift=1.0, device=device)
    declared, undeclared = Size(32, 2.0), Size(16, 3.0)

    def bind(runner, size, sample):
        return runner.bind_inputs(
            size,
            tuple(
                DenoiserInput(
                    {
                        "image": (
                            LatentInput(
                                sample, schedules["image"].timesteps[step]
                            ),
                        )
                    },
                    (size,),
                    step,
                )
                for step in range(steps)
            ),
            schedules,
            state={"image": sample},
            slot=1,
        )

    def expected(size):
        value = torch.full((size.width,), 7.0, device=device)
        for step in range(steps):
            value.add_(0.5 * (value * 0.25 + size.offset))
        return value

    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    # One unpinned context at a time, so the undeclared sizes below would
    # retire the declared one if pinning did not hold it.
    runner = TrajectoryRunner(
        model, device=device, stream=stream, groups=(), capacity=1, shapes=1
    )
    bank = torch.zeros((1, 64), device=device)
    runner.bind_bank({"image": bank})
    try:
        runner.prepare_inputs(declared, declared, pin=True)
        sample = bank[0, : declared.width]
        pinned = bind(runner, declared, sample)
        for step in range(steps):
            runner.capture(pinned, step)

        for size in (undeclared, Size(8, 4.0)):
            runner.prepare_inputs(size, size)
            sample = bank[0, : size.width]
            sample.fill_(7.0)
            trajectory = bind(runner, size, sample)
            paths = [runner.step(trajectory, step)[1] for step in range(steps)]
            assert paths == ["eager"] * steps
            torch.testing.assert_close(
                sample, expected(size), rtol=1e-6, atol=1e-6
            )

        sample = bank[0, : declared.width]
        sample.fill_(7.0)
        paths = [runner.step(pinned, step)[1] for step in range(steps)]
        assert paths == ["graph_replay"] * steps
        torch.testing.assert_close(
            sample, expected(declared), rtol=1e-6, atol=1e-6
        )
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
        stream.close()
