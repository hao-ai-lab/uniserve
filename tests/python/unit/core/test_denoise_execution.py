"""Observable denoise behavior at the model-runner boundary."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts import UniModel
from uniserve_worker.execution import ModelRunner, RunnerConfig

pytestmark = pytest.mark.unit


class RecordingVelocityModel(UniModel):
    supported_ops = ("denoise_gen",)
    device = "cpu"

    def __init__(self) -> None:
        self.branches: list[str] = []
        self.latents: list[torch.Tensor] = []

    def predict_velocity(self, ctx, t, latent, branch):
        del ctx, t
        self.branches.append(branch)
        self.latents.append(latent.detach().clone())
        return torch.full_like(latent, float(branch.rsplit("_", 1)[-1]))


def _runner(model: UniModel) -> ModelRunner:
    return ModelRunner(model, config=RunnerConfig(simulation=True))


def _new_request(req_id: int, *, seed: int, steps: int = 1) -> dict:
    return {
        "req_id": req_id,
        "seed": seed,
        "image": {
            "steps": steps,
            "latent_shape": [1, 2, 2],
            "schedule_direction": "ascending",
        },
        "cfg": {"branch_count": 2, "scales": [2.0], "renorm": "none"},
    }


def test_denoise_request_advances_its_schedule_across_runner_steps():
    model = RecordingVelocityModel()
    runner = _runner(model)

    first = runner.execute(
        {
            "step_id": 1,
            "new_reqs": [_new_request(11, seed=123, steps=2)],
            "ops": [{"req_id": 11, "kind": "denoise_gen"}],
        }
    )
    second = runner.execute(
        {"step_id": 2, "ops": [{"req_id": 11, "kind": "denoise_gen"}]}
    )

    assert first["per_seq"] == [
        {"req_id": 11, "denoise_done": False, "num_steps_done": 1}
    ]
    assert second["per_seq"] == [
        {"req_id": 11, "denoise_done": True, "num_steps_done": 2}
    ]
    assert model.branches == ["branch_0", "branch_1", "branch_0", "branch_1"]
    assert not torch.equal(model.latents[0], model.latents[2])


def test_seeded_denoise_requests_expose_reproducible_initial_latents_to_the_model():
    models = [RecordingVelocityModel() for _ in range(3)]
    seeds = [7, 7, 8]

    for index, (model, seed) in enumerate(zip(models, seeds, strict=True), start=1):
        _runner(model).execute(
            {
                "step_id": 1,
                "new_reqs": [_new_request(index, seed=seed)],
                "ops": [{"req_id": index, "kind": "denoise_gen"}],
            }
        )

    torch.testing.assert_close(models[0].latents[0], models[1].latents[0])
    assert not torch.equal(models[0].latents[0], models[2].latents[0])
    assert models[0].latents[0].device.type == "cpu"


def test_denoise_batch_preserves_wire_order_and_per_request_completion():
    model = RecordingVelocityModel()

    result = _runner(model).execute(
        {
            "step_id": 1,
            "new_reqs": [
                _new_request(42, seed=1),
                _new_request(7, seed=2),
            ],
            "ops": [
                {"req_id": 42, "kind": "denoise_gen"},
                {"req_id": 7, "kind": "denoise_gen"},
            ],
        }
    )

    assert result["per_seq"] == [
        {"req_id": 42, "denoise_done": True, "num_steps_done": 1},
        {"req_id": 7, "denoise_done": True, "num_steps_done": 1},
    ]
    assert model.branches == ["branch_0", "branch_1", "branch_0", "branch_1"]
