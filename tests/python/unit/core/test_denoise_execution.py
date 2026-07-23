"""Observable denoise behavior at the model-runner boundary."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.execution import ExecutorConfig, ModelExecutor

pytestmark = pytest.mark.unit


class RecordingVelocityModel(UniModel):
    supported_ops = ("denoise_gen",)
    device = "cpu"

    def __init__(self) -> None:
        self.branches: list[str] = []
        self.latents: list[torch.Tensor] = []

    fail: bool = False

    def predict_velocity(self, ctx, t, latent, branch):
        del ctx, t
        self.branches.append(branch)
        self.latents.append(latent.detach().clone())
        if self.fail:
            raise RuntimeError("injected denoise failure")
        return torch.full_like(latent, float(branch.rsplit("_", 1)[-1]))


def _runner(model: UniModel) -> ModelExecutor:
    return ModelExecutor(model, config=ExecutorConfig(simulation=True))


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
        seal_batch(
            1,
            [{"req_id": 11, "kind": "denoise_gen"}],
            new_reqs=[_new_request(11, seed=123, steps=2)],
        )
    )
    second = runner.execute(
        seal_batch(
            2,
            [{"req_id": 11, "kind": "denoise_gen"}],
            base_version=1,
        )
    )

    assert first["per_seq"][0]["denoise_done"] is False
    assert first["per_seq"][0]["num_steps_done"] == 1
    assert second["per_seq"][0]["denoise_done"] is True
    assert second["per_seq"][0]["num_steps_done"] == 2
    assert model.branches == ["branch_0", "branch_1", "branch_0", "branch_1"]
    assert not torch.equal(model.latents[0], model.latents[2])


def test_seeded_denoise_requests_expose_reproducible_initial_latents_to_the_model():
    models = [RecordingVelocityModel() for _ in range(3)]
    seeds = [7, 7, 8]

    for index, (model, seed) in enumerate(zip(models, seeds, strict=True), start=1):
        _runner(model).execute(
            seal_batch(
                1,
                [{"req_id": index, "kind": "denoise_gen"}],
                new_reqs=[_new_request(index, seed=seed)],
            )
        )

    torch.testing.assert_close(models[0].latents[0], models[1].latents[0])
    assert not torch.equal(models[0].latents[0], models[2].latents[0])
    assert models[0].latents[0].device.type == "cpu"


def test_denoise_batch_preserves_wire_order_and_per_request_completion():
    model = RecordingVelocityModel()

    result = _runner(model).execute(
        seal_batch(
            1,
            [
                {"req_id": 42, "kind": "denoise_gen"},
                {"req_id": 7, "kind": "denoise_gen"},
            ],
            new_reqs=[
                _new_request(42, seed=1),
                _new_request(7, seed=2),
            ],
        )
    )

    assert [row["req_id"] for row in result["per_seq"]] == [42, 7]
    assert [row["denoise_done"] for row in result["per_seq"]] == [True, True]
    assert [row["num_steps_done"] for row in result["per_seq"]] == [1, 1]
    assert model.branches == ["branch_0", "branch_1", "branch_0", "branch_1"]


def test_generic_denoise_retry_after_rollback_reuses_identical_initial_latent():
    # The model-neutral flow-matching path derives its initial latent from the
    # counter coordinates (session seed, op_id) rather than a stateful stream,
    # so a failed-and-rolled-back step retries to the identical latent with no
    # generator-state snapshot in the transaction. The op id is pinned so the
    # retry re-sends the same operation identity the scheduler would.
    op_id = (1 << 32) + 1
    model = RecordingVelocityModel()
    runner = _runner(model)

    model.fail = True
    with pytest.raises(RuntimeError, match="injected denoise failure"):
        runner.execute(
            seal_batch(
                1,
                [{"req_id": 5, "kind": "denoise_gen", "op_id": op_id}],
                new_reqs=[_new_request(5, seed=17)],
            )
        )
    failed_initial = model.latents[0].clone()
    model.fail = False

    runner.execute(
        seal_batch(
            2,
            [{"req_id": 5, "kind": "denoise_gen", "op_id": op_id}],
            new_reqs=[_new_request(5, seed=17)],
        )
    )
    assert torch.equal(model.latents[1], failed_initial)
