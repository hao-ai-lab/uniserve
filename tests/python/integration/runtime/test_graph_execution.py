"""Prepared input lifetime and numerical denoising.

Both are exercised across graph residency changes.
"""

from contextlib import nullcontext

import pytest
import torch

from tests.python.fixtures.diffusion import LinearDenoiser, Size
from tests.python.fixtures.encoding import Model
from uniserve.model import DenoiserInput, LatentInput
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.denoising_runner import DenoisingRunner
from uniserve_worker.execution.model_runner import ModelRunner

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("compute_fails", [False, True])
def test_initial_inputs_are_ready_for_consumption_after_preparation(
    compute_fails,
):
    device = torch.device("cuda", 0)
    model = Model().to(device)
    runner = ModelRunner(model, WorkerConfig(device=str(device)))
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
    from uniserve.runtime import partition_streams

    device = torch.device("cuda:0")
    branch_device = "cuda:0" if execution == "green" else execution
    model = LinearDenoiser().to(device)
    model.projection.to(branch_device)
    schedules = model.make_schedules(2, shift=1.0, device=device)
    partition = (
        partition_streams(device, (64,))[0] if execution == "green" else None
    )
    stream = (
        partition.stream
        if partition is not None
        else torch.cuda.Stream(device=device)
    )
    runner = DenoisingRunner(
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
            runner.prepare_inputs(key, size)

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
            for step in (0, 1):
                reference.add_(0.5 * (reference * 0.25 + size.offset))
                actual, _ = runner.step(
                    inputs(step), schedules, state={}, slot=slot, input_key=key
                )
                torch.testing.assert_close(
                    actual["image"][0], reference, rtol=1e-6, atol=1e-6
                )
            torch.cuda.current_stream(device).synchronize()
            runner.release_inputs(key)
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
        if partition is not None:
            partition.close()


@torch.inference_mode()
def test_one_captured_ladder_serves_a_slot_that_owns_host_state():
    """A slot's own host storage does not give that slot its own ladder.

    Preparation draws each request slot's noise into storage that slot owns
    and no bank holds, and hands the same mapping to the step. A ladder
    captured while one slot is resident must still replay for the next slot:
    if the slot's own addresses reached the capture, every slot would pay a
    capture of its own and a deployment's graphs would scale with residency.
    """
    device = torch.device("cuda:0")
    model = LinearDenoiser().to(device)
    steps = 2
    schedules = model.make_schedules(steps, shift=1.0, device=device)
    size = Size(32, 2.0)
    runner = DenoisingRunner(
        model,
        device=device,
        stream=torch.cuda.Stream(device=device),
        groups=(),
        capacity=2,
        shapes=2,
    )
    bank = torch.zeros((2, 64), device=device)
    runner.bind_bank({"image": bank})
    # One draw per slot, in the slot's own host storage, as preparation makes
    # it: outside every bank and at a different address for every slot.
    draws = {
        slot: torch.zeros(32, dtype=torch.float32).pin_memory()
        for slot in (1, 2)
    }
    try:
        runner.prepare_inputs(size, size)
        samples = {slot: bank[slot - 1, :32] for slot in (1, 2)}
        for sample in samples.values():
            sample.fill_(7.0)

        def call(runner_method, slot, step):
            return runner_method(
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
                ),
                schedules,
                state={"image_noise": draws[slot]},
                slot=slot,
                input_key=size,
            )

        for step in range(steps):
            call(runner.capture, 1, step)
        paths = [call(runner.step, 2, step)[1] for step in range(steps)]
        assert paths == ["graph_replay"] * steps
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()


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
        """Advance every slot one full ladder and report its step paths."""
        values, paths = {}, []
        for slot in slots:
            for step in range(steps):
                result, path = runner.step(
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
                    ),
                    schedules,
                    state={},
                    slot=slot,
                    input_key=size,
                )
                paths.append(path)
            values[slot] = result["image"][0].clone()
        return values, paths

    eager = DenoisingRunner(
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

    runner = DenoisingRunner(
        model,
        device=device,
        stream=torch.cuda.Stream(device=device),
        groups=(),
        capacity=2,
        shapes=2,
    )
    bank = torch.zeros((2, 64), device=device)
    runner.bind_bank({"image": bank})
    try:
        runner.prepare_inputs(size, size)
        samples = {slot: bank[slot - 1, :32] for slot in slots}
        for sample in samples.values():
            sample.fill_(7.0)
        for slot in slots:
            resting = samples[slot].clone()
            for step in range(steps):
                runner.capture(
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
                    ),
                    schedules,
                    state={},
                    slot=slot,
                    input_key=size,
                )
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
