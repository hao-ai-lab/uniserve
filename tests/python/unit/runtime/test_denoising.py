"""Public diffusion calls preserve ordered modalities and solver arithmetic."""

import pytest
import torch

from tests.python.fixtures.diffusion import LinearDenoiser, Size
from uniserve.diffusion import DenoisingStep, EulerSolver
from uniserve.model import DenoiserInput, LatentInput

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("solver", ["clean_sample_euler", "euler"])
@torch.inference_mode()
def test_named_denoising_updates_each_sequence_with_its_modality_schedule(
    solver,
):
    model = LinearDenoiser(("video", "audio"), solver=solver)
    schedules = model.make_schedules(2, shift=None, device="cpu")
    samples = {
        "video": (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
        "audio": (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])),
    }
    reference = {
        name: tuple(value.double() for value in rows)
        for name, rows in samples.items()
    }
    constants = {"offset": torch.tensor(2.0)}
    for step in range(2):
        batch = DenoiserInput(
            {
                name: tuple(
                    LatentInput(value, schedules[name].timesteps[step])
                    for value in rows
                )
                for name, rows in samples.items()
            },
            (Size(2),) * 2,
            schedules["video"].step(step),
        )
        output = DenoisingStep(model, batch, schedules, {}, constants, {})()
        for name, rows in reference.items():
            interval = (
                schedules[name].sigmas[step].double()
                - schedules[name].sigmas[step + 1].double()
            )
            for expected, observed in zip(rows, samples[name], strict=True):
                expected.add_(interval * (expected * 0.25 + 2.0))
                torch.testing.assert_close(
                    observed.double(), expected, rtol=1e-6, atol=1e-6
                )
        for observed, expected in zip(
            (row for rows in output.values() for row in rows),
            (row for rows in reference.values() for row in rows),
            strict=True,
        ):
            torch.testing.assert_close(
                observed.double(), expected, rtol=1e-6, atol=1e-6
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_clean_prediction_reaches_its_terminal_sample(dtype):
    sample = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    clean = torch.tensor([4.0, 5.0, 6.0], dtype=dtype)
    EulerSolver("sample").step_(
        clean,
        sample,
        torch.tensor(0.25),
        torch.tensor(1.0),
        sigma=torch.tensor(0.75),
        next_sigma=torch.tensor(0.0),
    )
    torch.testing.assert_close(sample, clean, rtol=0, atol=0)
