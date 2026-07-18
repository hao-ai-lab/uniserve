"""Timestep-aware denoise residual reuse: policy math + engine integration."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import uniserve_worker.models.interleaved_image as denoise_mod
from uniserve_worker.execution.engine import TextImageDenoiseStep
from uniserve_worker.models.interleaved_image import (
    DenoiseResidualCacheBinding,
    DenoiseResidualCachePolicy,
    ImageResidualCacheState,
    ImageState,
    TextImageDenoiseOps,
)

IDENTITY_POLY = (1.0, 0.0)  # poly(x) = x


def _state(threshold: float = 0.2) -> ImageResidualCacheState:
    return ImageResidualCacheState(threshold=threshold, coefficients=IDENTITY_POLY)


def test_first_step_always_computes():
    state = _state()
    assert not state.decide_reuse(torch.ones(1, 4, 8), ("cond",))


def test_reuse_until_accumulated_threshold():
    state = _state(threshold=0.15)
    base = torch.ones(1, 4, 8)
    state.decide_reuse(base, ("cond",))
    state.record("cond", torch.zeros(1, 4, 8), torch.ones(1, 4, 8))
    # Per-step rel-L1 is 0.05/(1+0.05*(step-1)): accum ≈ 0.050 / 0.098 / 0.143
    # stays under 0.15 (reuse), then ≈ 0.187 crosses it (recompute + reset).
    reuses = []
    for step in range(1, 5):
        drifted = base * (1.0 + 0.05 * step)
        reuses.append(state.decide_reuse(drifted, ("cond",)))
    assert reuses[:3] == [True, True, True]
    assert reuses[3] is False, "hitting the threshold recomputes and resets"
    assert state.accumulated == 0.0


def test_reuse_requires_recorded_branches():
    state = _state(threshold=1.0)
    base = torch.ones(1, 4, 8)
    state.decide_reuse(base, ("cond",))
    state.record("cond", torch.zeros(1, 4, 8), torch.ones(1, 4, 8))
    assert not state.decide_reuse(base, ("cond", "tu")), "missing branch residual"
    assert state.decide_reuse(base, ("cond",))


def test_replay_adds_residual_to_new_embeds():
    state = _state()
    embeds_then = torch.full((1, 3, 4), 2.0)
    hidden_then = torch.full((1, 3, 4), 5.0)
    state.record("cond", embeds_then, hidden_then)
    embeds_now = torch.full((1, 3, 4), 2.5)
    replayed = state.replay("cond", embeds_now)
    assert torch.allclose(replayed, torch.full((1, 3, 4), 5.5))


def test_invalidate_drops_replay_state():
    state = _state(threshold=1.0)
    base = torch.ones(1, 4, 8)
    state.decide_reuse(base, ("cond",))
    state.record("cond", base, base)
    state.invalidate()
    assert not state.decide_reuse(base, ("cond",)), "post-invalidate step recomputes"


class _FakeOwner(TextImageDenoiseOps):
    """Denoise owner double: non-paged branch caches, counted forwards."""

    device = "cpu"

    def __init__(self) -> None:
        self.forward_calls = 0
        self.finalize_calls = 0
        norm = torch.nn.Identity()

        def _finalize(hidden):
            self.finalize_calls += 1
            return hidden

        self._adapter = DenoiseResidualCacheBinding(
            decision_embedding=lambda embeds: norm(embeds),
            rescale_coefficients=IDENTITY_POLY,
            finalize_hidden=_finalize,
        )

    def denoise_residual_cache_adapter(self):
        return self._adapter

    def _denoise_branch_inputs(self, img, branch):
        return torch.zeros(3, 4), object()  # not a PagedTextCache -> per-row path

    def _wait_gen_cache_ready(self, cache):
        pass

    def interleaved_image_predict_velocity(
        self,
        image_embeds,
        indexes,
        attention_mask,
        cache,
        t,
        z,
        *,
        image_token_num,
        image_size,
        return_hidden=False,
    ):
        self.forward_calls += 1
        velocity = torch.zeros_like(z)
        if return_hidden:
            return velocity, image_embeds + 1.0
        return velocity

    def packed_hidden_to_velocity(self, hidden, t, latent, *, image_token_num, image_size):
        return torch.zeros_like(latent)


class _FakePagedCache:
    pass


class _SingleRowGraphOwner(TextImageDenoiseOps):
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[int, bool, str]] = []
        self.cache = _FakePagedCache()

    def denoise_residual_cache_adapter(self):
        return None

    def _denoise_branch_inputs(self, img, branch):
        return torch.zeros(3, 4), self.cache

    def _predict_v_batched(self, rows, *, return_hidden=False, graph_mode="auto"):
        self.calls.append((len(rows), bool(return_hidden), str(graph_mode)))
        return torch.ones_like(rows[0].step.latent)

    def predict_denoise_velocity(self, step, branch, *, return_hidden=False):  # pragma: no cover
        raise AssertionError("single-row graph-required denoise must use the batched path")


def _image_state() -> ImageState:
    dummy = torch.zeros(1)
    pool = SimpleNamespace(get=lambda handle: dummy, set=lambda handle, value: None)
    return ImageState(
        latent_pool=pool,
        latent_handle=0,
        schedule=None,
        timesteps=torch.zeros(2),
        token_h=2,
        token_w=2,
        grid_h=4,
        grid_w=4,
        grid_hw=torch.tensor([[4, 4]]),
        indexes_cond=torch.zeros(3, 4, dtype=torch.long),
        indexes_tu=None,
        indexes_iu=None,
        cond_cache=object(),
        tu_cache=None,
        iu_cache=None,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.0, 1.0),
        cfg_norm="none",
        cfg_renorm_min=0.0,
        noise_scale=0.0,
        height=64,
        width=64,
    )


def _step(img: ImageState, embeds: torch.Tensor) -> TextImageDenoiseStep:
    return TextImageDenoiseStep(
        req_id=1,
        state=None,
        op={},
        latent=torch.zeros(1, 4, 8),
        t=torch.zeros(1),
        t_next=torch.zeros(1),
        step_index=0,
        total_steps=8,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.0, 1.0),
        cfg_renorm_type="none",
        cfg_renorm_min=0.0,
        extra={"img": img, "image_embeds": embeds},
    )


@pytest.fixture
def enabled_policy(monkeypatch):
    monkeypatch.setattr(
        denoise_mod,
        "_RESIDUAL_CACHE_POLICY",
        DenoiseResidualCachePolicy(enabled=True, threshold=0.5),
    )


def test_engine_replays_stable_steps(enabled_policy):
    owner = _FakeOwner()
    img = _image_state()
    embeds = torch.ones(1, 4, 8)

    owner.predict_text_image_velocity_batch([_step(img, embeds)], [["cond"]])
    assert owner.forward_calls == 1, "first step computes"
    assert img.residual_cache is not None
    assert img.residual_cache.residuals

    owner.predict_text_image_velocity_batch([_step(img, embeds * 1.01)], [["cond"]])
    assert owner.forward_calls == 1, "near-identical step replays the residual"
    assert img.residual_cache.hits == 1
    assert owner.finalize_calls == 1, "replay re-applies the model's final norm"

    # A large drift busts the accumulator and recomputes.
    owner.predict_text_image_velocity_batch([_step(img, embeds * 3.0)], [["cond"]])
    assert owner.forward_calls == 2


def test_engine_disabled_without_policy(monkeypatch):
    monkeypatch.setattr(
        denoise_mod,
        "_RESIDUAL_CACHE_POLICY",
        DenoiseResidualCachePolicy(enabled=False, threshold=0.5),
    )
    owner = _FakeOwner()
    img = _image_state()
    embeds = torch.ones(1, 4, 8)
    owner.predict_text_image_velocity_batch([_step(img, embeds)], [["cond"]])
    owner.predict_text_image_velocity_batch([_step(img, embeds)], [["cond"]])
    assert owner.forward_calls == 2
    assert img.residual_cache is None


def test_single_paged_denoise_row_uses_graph_required_batched_path(monkeypatch):
    monkeypatch.setattr(denoise_mod, "PagedTextCache", _FakePagedCache)
    monkeypatch.setattr(
        denoise_mod,
        "_RESIDUAL_CACHE_POLICY",
        DenoiseResidualCachePolicy(enabled=False, threshold=0.5),
    )
    owner = _SingleRowGraphOwner()
    img = _image_state()
    embeds = torch.ones(1, 4, 8)

    result = owner.predict_text_image_velocity_batch(
        [_step(img, embeds)],
        [["cond"]],
        graph_mode="require",
    )

    assert result is not None
    torch.testing.assert_close(result[0]["cond"], torch.ones(1, 4, 8))
    assert owner.calls == [(1, False, "require")]
