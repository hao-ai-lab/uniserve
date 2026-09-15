"""Prepared input lifetime and numerical denoising across graph residency changes."""

from contextlib import nullcontext

import pytest
import torch

from tests.python.fixtures.encoding import Model
from tests.python.fixtures.diffusion import LinearDenoiser, Size
from uniserve.model import DenoiserInput, LatentInput
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.diffusion_runner import DiffusionRunner
from uniserve_worker.execution.model_runner import ModelRunner

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("compute_fails", [False, True])
def test_initial_inputs_are_ready_for_consumption_after_preparation(compute_fails):
    device = torch.device("cuda", 0)
    model = Model().to(device)
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


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("branch_device", ["cuda:0", "cuda:1"])
@torch.inference_mode()
def test_denoising_reprepared_constants_and_slot_sizes_advance_one_step(graphs, branch_device):
    device = torch.device("cuda:0")
    model = LinearDenoiser().to(device)
    model.projection.to(branch_device)
    schedules = model.make_schedules(2, shift=1.0, device=device)
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
        for index, (slot, width, key) in enumerate(
            ((1, 32, 2), (1, 64, 3), (2, 32, 2), (1, 32, 2))
        ):
            sample = torch.full((width,), 7.0, device=device)
            reference = sample.clone()
            size = Size(width, float(key + index))
            runner.prepare_inputs(key, size)

            def inputs(step):
                return DenoiserInput(
                    {"image": (LatentInput(sample, schedules["image"].timesteps[step]),)},
                    (size,),
                    step,
                )

            runner.warmup(inputs(0), schedules, state={}, input_key=key)
            torch.testing.assert_close(sample, reference, rtol=0, atol=0)
            for step in (0, 1):
                reference.add_(0.5 * (reference * 0.25 + size.offset))
                actual, _ = runner.step(inputs(step), schedules, state={}, slot=slot, input_key=key)
                torch.testing.assert_close(actual["image"][0], reference, rtol=1e-6, atol=1e-6)
            torch.cuda.current_stream(device).synchronize()
            runner.release_inputs(key)
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
