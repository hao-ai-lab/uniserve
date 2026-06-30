"""Conformance for runner-owned denoise execution."""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts.model_protocols import ModelHooks
from uniserve_worker.execution.denoise_driver import DenoiseDriver, TextImageDenoiseStep
from uniserve_worker.execution.runner import ModelRunner
from uniserve_worker.nn.diffusion import combine_text_image_cfg
from uniserve_worker.runtime.request_state import RequestState

pytestmark = pytest.mark.unit


class VelocityOnlyModel(ModelHooks):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict_velocity(self, ctx, t, latent, branch):
        del ctx, t
        self.calls.append(branch)
        return torch.full_like(latent, float(branch.rsplit("_", 1)[-1]))

    def forward(self, batch):  # pragma: no cover - denoise must not route here
        raise AssertionError("denoise should be owned by DenoiseDriver")


def test_runner_denoise_driver_threads_rng_cfg_and_cursor():
    model = VelocityOnlyModel()
    runner = ModelRunner(model)
    batch = {
        "step_id": 1,
        "new_reqs": [
            {
                "req_id": 11,
                "seed": 123,
                "image": {"steps": 2, "latent_shape": [1, 2, 2], "schedule_direction": "ascending"},
                "cfg": {"branch_count": 2, "scales": [2.0], "renorm": "none"},
            }
        ],
        "ops": [{"req_id": 11, "kind": "denoise_gen"}],
    }

    result = runner.execute(batch)
    state = runner.request_states.get(11)
    assert result["per_seq"] == [{"req_id": 11, "denoise_done": False, "num_steps_done": 1}]
    assert model.calls == ["branch_0", "branch_1"]
    assert state.schedule_cursor == 1
    assert tuple(state.latent.shape) == (1, 2, 2)

    before = state.latent.clone()
    result = runner.execute({"step_id": 2, "ops": [{"req_id": 11, "kind": "denoise_gen"}]})
    assert result["per_seq"] == [{"req_id": 11, "denoise_done": True, "num_steps_done": 2}]
    assert runner.request_states.get(11).schedule_cursor == 2
    torch.testing.assert_close(runner.request_states.get(11).latent, before + 1.0)


class TextImageCapabilityModel(ModelHooks):
    def __init__(self, *, text_scale: float, img_scale: float) -> None:
        self.text_scale = text_scale
        self.img_scale = img_scale
        self.calls: list[str] = []
        self.applied: torch.Tensor | None = None
        self.inference_modes: list[bool] = []

    def prepare_denoise(self, state, op):
        return TextImageDenoiseStep(
            req_id=int(op.get("req_id", 1)),
            state=state,
            op=op,
            latent=torch.zeros(1, 1),
            t=torch.tensor(0.5),
            t_next=torch.tensor(1.5),
            step_index=0,
            total_steps=1,
            cfg_text_scale=self.text_scale,
            cfg_img_scale=self.img_scale,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_type="none",
            cfg_renorm_min=0.0,
            image_scale_applies_to_text=True,
        )

    def predict_velocity(self, step, t, latent, branch):
        del t, latent
        self.calls.append(branch)
        self.inference_modes.append(torch.is_inference_mode_enabled())
        values = {"cond": 3.0, "text_uncond": 2.0, "img_uncond": 1.0}
        return torch.full_like(step.latent, values[branch])

    def accept_denoise_update(self, step, latent):
        self.applied = latent


def test_text_image_driver_uses_image_scale_when_only_image_branch_is_needed():
    model = TextImageCapabilityModel(text_scale=1.0, img_scale=3.0)

    DenoiseDriver().step(1, RequestState(), model, {})

    assert model.calls == ["cond", "img_uncond"]
    # Cross-check against the production combiner with the branches the driver
    # actually requested (cond=3, img_uncond=1); text_uncond is absent here.
    # image guidance = 1 + 3 * (3 - 1) = 7
    expected = combine_text_image_cfg(
        torch.full((1, 1), 3.0),
        None,
        torch.full((1, 1), 1.0),
        cfg_text_scale=1.0,
        cfg_img_scale=3.0,
        renorm="none",
        renorm_min=0.0,
        image_scale_applies_to_text=True,
    )
    torch.testing.assert_close(model.applied, expected)
    torch.testing.assert_close(expected, torch.tensor([[7.0]]))


class BatchedTextImageCapabilityModel(ModelHooks):
    def __init__(self) -> None:
        self.batch_calls: list[list[list[str]]] = []
        self.predict_calls: list[str] = []
        self.applied: dict[int, torch.Tensor] = {}

    def prepare_denoise(self, state, op):
        req_id = int(op["req_id"])
        return TextImageDenoiseStep(
            req_id=req_id,
            state=state,
            op=op,
            latent=torch.zeros(1, 1),
            t=torch.tensor(0.5),
            t_next=torch.tensor(1.5),
            step_index=0,
            total_steps=1,
            cfg_text_scale=2.0,
            cfg_img_scale=1.0,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_type="none",
            cfg_renorm_min=0.0,
        )

    def predict_velocity(self, step, t, latent, branch):  # pragma: no cover - batch hook should handle it.
        del step, t, latent
        self.predict_calls.append(branch)
        raise AssertionError("sequential text-image branch predictor should not be called")

    def predict_text_image_velocity_batch(self, steps, branches_by_step):
        self.batch_calls.append([list(branches) for branches in branches_by_step])
        outputs = []
        for step, branches in zip(steps, branches_by_step):
            values = {}
            for branch in branches:
                raw = {"cond": 3.0, "text_uncond": 2.0, "img_uncond": 1.0}[branch]
                values[branch] = torch.full_like(step.latent, raw)
            outputs.append(values)
        return outputs

    def accept_denoise_update(self, step, latent):
        self.applied[int(step.req_id)] = latent


def test_runner_denoise_group_uses_text_image_batch_hook_once():
    model = BatchedTextImageCapabilityModel()
    runner = ModelRunner(model)

    result = runner.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 1, "block_ids": []}, {"req_id": 2, "block_ids": []}],
            "ops": [
                {"req_id": 1, "kind": "denoise_gen"},
                {"req_id": 2, "kind": "denoise_gen"},
            ],
        }
    )

    assert result["per_seq"] == [
        {"req_id": 1, "denoise_done": True, "num_steps_done": 1},
        {"req_id": 2, "denoise_done": True, "num_steps_done": 1},
    ]
    assert model.batch_calls == [[["cond", "text_uncond"], ["cond", "text_uncond"]]]
    assert model.predict_calls == []
    torch.testing.assert_close(model.applied[1], torch.tensor([[4.0]]))
    torch.testing.assert_close(model.applied[2], torch.tensor([[4.0]]))


class IntervalGatedTextImageModel(ModelHooks):
    """Drives the per-step host-sync ``cfg_interval`` gate via ``step.t``."""

    def __init__(self, *, t: float, interval: tuple[float, float]) -> None:
        self.t = t
        self.interval = interval
        self.calls: list[str] = []
        self.applied: torch.Tensor | None = None

    def prepare_denoise(self, state, op):
        return TextImageDenoiseStep(
            req_id=int(op.get("req_id", 1)),
            state=state,
            op=op,
            latent=torch.zeros(1, 1),
            t=torch.tensor(self.t),
            # delta of exactly 1.0 makes euler_step(latent=0) equal the combined
            # velocity, so ``applied`` reads back the post-CFG velocity directly.
            t_next=torch.tensor(self.t + 1.0),
            step_index=0,
            total_steps=1,
            cfg_text_scale=2.0,
            cfg_img_scale=2.0,
            cfg_interval=self.interval,
            cfg_renorm_type="none",
            cfg_renorm_min=0.0,
            image_scale_applies_to_text=True,
        )

    def predict_velocity(self, step, t, latent, branch):
        del t, latent
        self.calls.append(branch)
        return torch.full_like(step.latent, {"cond": 3.0, "text_uncond": 2.0, "img_uncond": 1.0}[branch])

    def accept_denoise_update(self, step, latent):
        self.applied = latent


def test_text_image_driver_skips_cfg_branches_when_t_falls_outside_interval():
    # ``_text_image_branches`` host-syncs ``step.t`` and disables CFG when the
    # timestep is outside ``cfg_interval``; only the conditioned branch should run.
    model = IntervalGatedTextImageModel(t=0.9, interval=(0.2, 0.8))

    DenoiseDriver().step(1, RequestState(), model, {})

    assert model.calls == ["cond"]
    # With CFG gated off the combiner returns ``out_cond`` unchanged.
    torch.testing.assert_close(model.applied, torch.tensor([[3.0]]))


def test_text_image_driver_runs_cfg_branches_when_t_inside_interval():
    # The same model with ``t`` inside the interval must re-enable CFG, proving
    # the gate is driven by the timestep value and not statically disabled.
    model = IntervalGatedTextImageModel(t=0.5, interval=(0.2, 0.8))

    DenoiseDriver().step(1, RequestState(), model, {})

    assert model.calls == ["cond", "text_uncond", "img_uncond"]


def test_generic_latent_init_lands_on_driver_device_and_seeds_from_rng():
    # Covers DenoiseDriver._latent: the lazily-initialised latent must live on
    # the driver's device and be reproducible from the request seed.
    model = VelocityOnlyModel()
    op = {
        "req_id": 11,
        "num_steps": 1,
        "image": {"latent_shape": [1, 2, 2], "seed": 7, "schedule_direction": "ascending"},
        "cfg": {"branch_count": 1, "renorm": "none"},
    }

    state_a = RequestState()
    DenoiseDriver(device="cpu").step(11, state_a, model, op)
    assert isinstance(state_a.latent, torch.Tensor)
    assert state_a.latent.device.type == "cpu"

    # Re-running from a fresh state with the same seed reproduces the same noise,
    # so the generic init_latent path is deterministic under the request rng.
    expected_noise = torch.randn(
        (1, 2, 2), generator=torch.Generator(device="cpu").manual_seed(7), dtype=torch.float32
    )
    state_b = RequestState()
    model_b = VelocityOnlyModel()
    # branch_0 velocity equals 0.0, so euler_step leaves the initial latent intact.
    DenoiseDriver(device="cpu").step(11, state_b, model_b, op)
    torch.testing.assert_close(state_b.latent, expected_noise)
