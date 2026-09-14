"""Public diffusion calls preserve ordered modalities and solver arithmetic."""

import pytest
import torch

from tests.python.fixtures.diffusion import LinearDenoiser
from uniserve.model.batch import DiffusionBatch
from uniserve.model.denoising import DenoisingStep
from uniserve.model.media import ImageSize
from uniserve.nn.diffusion.integrator import EulerSolver
from uniserve.nn.diffusion.schedule import DiffusionSchedule

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("solver", ["clean_sample_euler", "euler"])
@torch.inference_mode()
def test_named_denoising_updates_each_sequence_with_its_modality_schedule(solver):
    model = LinearDenoiser(("video", "audio"), solver=solver)
    schedule = DiffusionSchedule.build((1000, 500), (1.0, 3.0), scale=1000.0, device="cpu")
    samples = {
        "video": (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
        "audio": (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])),
    }
    reference = {name: tuple(value.double() for value in rows) for name, rows in samples.items()}
    constants = {"offset": torch.tensor(2.0)}
    for step in range(2):
        batch = DiffusionBatch(
            samples,
            (ImageSize(1, 2),) * 2,
            timesteps={
                name: (schedule.timesteps[index][step],) * 2 for index, name in enumerate(samples)
            },
            ladder_index=step,
        )
        output = DenoisingStep(model, batch, {}, constants, {}, schedule)()
        for modality, (name, rows) in enumerate(reference.items()):
            interval = (
                schedule.sigmas[modality][step].double()
                - schedule.sigmas[modality][step + 1].double()
            )
            for expected, observed in zip(rows, samples[name], strict=True):
                expected.add_(interval * (expected * 0.25 + 2.0))
                torch.testing.assert_close(observed.double(), expected, rtol=1e-6, atol=1e-6)
        for observed, expected in zip(
            output, (row for rows in reference.values() for row in rows), strict=True
        ):
            torch.testing.assert_close(observed.double(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_clean_prediction_reaches_its_terminal_sample(dtype):
    sample = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    clean = torch.tensor([4.0, 5.0, 6.0], dtype=dtype)
    EulerSolver("sample").step(clean, sample, torch.tensor(0.25), torch.tensor(1.0))
    torch.testing.assert_close(sample, clean, rtol=0, atol=0)
