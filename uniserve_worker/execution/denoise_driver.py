"""Shared diffusion denoise driver."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch

from ..contracts.model_protocols import DenoiseContext
from ..contracts.outputs import DenoiseOutput
from ..foundation.errors import invalid_descriptor
from ..nn.diffusion import (
    CfgParams,
    FlowMatchSchedule,
    ScheduleDirection,
    combine_cfg,
    euler_step,
    init_latent,
    x_pred_to_velocity,
)
from ..nn.diffusion.cfg import CfgPlan, CfgRecipe, build_text_image_cfg_plan
from ..runtime.image_params import required_image_height, required_image_width
from ..runtime.request_state import RequestState

if TYPE_CHECKING:
    from ..contracts.model_protocols import DenoiseCapable

__all__ = [
    "DenoiseDriver",
    "TextImageDenoiseStep",
    "combine_text_image_velocity",
    "text_image_branches",
]

# Fallback latent geometry for the model-neutral generic path, used only when an
# op/image descriptor supplies neither an explicit ``latent_shape`` nor the
# per-field overrides. Production models compute their own latent geometry and
# never reach this fallback.
_DEFAULT_LATENT_DOWNSAMPLE = 16
_DEFAULT_LATENT_CHANNELS = 4


@dataclass(frozen=True)
class TextImageDenoiseStep:
    req_id: int
    state: RequestState
    op: Mapping[str, Any]
    latent: torch.Tensor
    t: torch.Tensor
    t_next: torch.Tensor
    step_index: int
    total_steps: int
    cfg_text_scale: float
    cfg_img_scale: float
    cfg_interval: tuple[float, float]
    cfg_renorm_type: str
    cfg_renorm_min: float
    # Names the model's text/image CFG convention. Construction coerces bools
    # and strings to the enum so the execution path always consumes one type.
    image_scale_applies_to_text: CfgRecipe = CfgRecipe.ADDITIVE_DELTAS
    extra: Any = None

    def __post_init__(self) -> None:
        recipe = CfgRecipe.coerce(self.image_scale_applies_to_text)
        if recipe is not self.image_scale_applies_to_text:
            object.__setattr__(self, "image_scale_applies_to_text", recipe)


class DenoiseDriver:
    """Execute one model-neutral flow-matching denoise step."""

    def __init__(self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        # ``device``/``dtype`` take effect only on the model-neutral generic
        # ``DenoiseContext``/``_latent`` path (see ``_finish_prepared_step``).
        # Production diffusion models return a ``TextImageDenoiseStep`` and run
        # on their own device/dtype, and the runner constructs ``DenoiseDriver()``
        # with no arguments -- so these defaults are inert for them.
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def step(self, req_id: int, state: RequestState, model: "DenoiseCapable", op: Mapping[str, Any]) -> DenoiseOutput:
        return self.step_many([(req_id, state, op)], model)[0]

    @torch.inference_mode()
    def step_many(
        self,
        items: Sequence[tuple[int, RequestState, Mapping[str, Any]]],
        model: "DenoiseCapable",
    ) -> list[DenoiseOutput]:
        prepared = [
            (int(req_id), state, op, self._prepare(req_id, state, model, op))
            for req_id, state, op in items
        ]
        if all(isinstance(item[3], TextImageDenoiseStep) for item in prepared):
            return self._text_image_steps(
                model,
                [item[3] for item in prepared if isinstance(item[3], TextImageDenoiseStep)],
            )
        return [
            self._finish_prepared_step(req_id, state, model, op, ctx)
            for req_id, state, op, ctx in prepared
        ]

    def _prepare(
        self,
        req_id: int,
        state: RequestState,
        model: "DenoiseCapable",
        op: Mapping[str, Any],
    ) -> DenoiseContext | TextImageDenoiseStep:
        del req_id
        prepared = _prepare_denoise(model, state, op)
        if isinstance(prepared, TextImageDenoiseStep):
            return prepared
        if not isinstance(prepared, DenoiseContext):
            raise invalid_descriptor("prepare_denoise(state, op) must return DenoiseContext")
        return prepared

    def _finish_prepared_step(
        self,
        req_id: int,
        state: RequestState,
        model: "DenoiseCapable",
        op: Mapping[str, Any],
        prepared: DenoiseContext | TextImageDenoiseStep,
    ) -> DenoiseOutput:
        if isinstance(prepared, TextImageDenoiseStep):
            return self._text_image_steps(model, [prepared])[0]
        # Model-neutral generic flow-matching path. It is the contract for models
        # that return a DenoiseContext (and is exercised by the synthetic
        # velocity-only test model); the production diffusion models instead
        # return a TextImageDenoiseStep above and build their own
        # schedule with model-specific direction/shift-domain defaults. As a
        # consequence the op-level schedule_direction/schedule_shift/flow_shift
        # keys read below (and the equivalent DiffusionConfig fields) are INERT
        # for those production models -- callers must not assume they take effect.
        image = dict(state.image or {})
        image.update(op.get("image") or {})
        steps = int(op.get("num_steps") or image.get("steps") or image.get("num_steps") or 50)
        if steps <= 0:
            raise invalid_descriptor("denoise steps must be positive")
        schedule = FlowMatchSchedule(
            num_steps=steps,
            shift=float(image.get("schedule_shift", image.get("flow_shift", 1.0))),
            direction=ScheduleDirection(str(image.get("schedule_direction", "ascending"))),
        )
        cursor = int(op.get("timestep_idx", state.schedule_cursor) or 0)
        t, t_next = schedule.pair(cursor, device=self.device, dtype=self.dtype)
        latent = self._latent(state, image, op)
        cfg = CfgParams.from_mapping(op.get("cfg") or state.cfg_geometry)
        velocities = []
        for branch_index in range(cfg.branch_count):
            branch = _branch_name(branch_index, cfg.branch_count)
            velocity = model.predict_velocity(prepared, t, latent, branch)
            if not isinstance(velocity, torch.Tensor):
                raise invalid_descriptor("predict_velocity(ctx, t, latent, branch) must return a tensor")
            velocity = _maybe_convert_parameterization(model, velocity, latent, t)
            if velocity.shape != latent.shape:
                raise invalid_descriptor(
                    f"velocity shape {tuple(velocity.shape)} does not match latent {tuple(latent.shape)}"
                )
            velocities.append(velocity)
        velocity = combine_cfg(velocities, cfg)
        state.latent = euler_step(latent, velocity, t, t_next)
        _accept_denoise_update(model, prepared, state.latent)
        done = cursor + 1 >= steps
        return DenoiseOutput(req_id=req_id, denoise_done=done, num_steps_done=cursor + 1)

    def _text_image_steps(
        self,
        model: "DenoiseCapable",
        steps: Sequence[TextImageDenoiseStep],
    ) -> list[DenoiseOutput]:
        branches_by_step = [text_image_branches(step) for step in steps]
        batch_predict = _text_image_batch_predictor(model)
        if batch_predict is not None:
            predicted = batch_predict(steps, branches_by_step)
        else:
            predicted = None
        if predicted is None:
            branch_outputs = [
                {
                    branch: self._predict_text_image_branch(model.predict_velocity, step, branch)
                    for branch in branches
                }
                for step, branches in zip(steps, branches_by_step)
            ]
        else:
            branch_outputs = _validate_batched_text_image_outputs(steps, branches_by_step, predicted)
        outputs = []
        for step, velocities in zip(steps, branch_outputs):
            velocity = combine_text_image_velocity(step, velocities)
            updated = euler_step(step.latent, velocity, step.t, step.t_next)
            _accept_denoise_update(model, step, updated)
            done = step.step_index + 1 >= step.total_steps
            outputs.append(
                DenoiseOutput(
                    req_id=step.req_id,
                    denoise_done=done,
                    num_steps_done=step.step_index + 1,
                )
            )
        return outputs

    def _predict_text_image_branch(self, predict: Any, step: TextImageDenoiseStep, branch: str) -> torch.Tensor:
        velocity = predict(step, step.t, step.latent, branch)
        if not isinstance(velocity, torch.Tensor) or velocity.shape != step.latent.shape:
            raise invalid_descriptor(f"{branch} velocity must be a tensor matching the denoise latent")
        return velocity

    def _latent(self, state: RequestState, image: Mapping[str, Any], op: Mapping[str, Any]) -> torch.Tensor:
        if isinstance(state.latent, torch.Tensor):
            return state.latent.to(device=self.device, dtype=self.dtype)
        shape = op.get("latent_shape") or image.get("latent_shape")
        if shape is None:
            h = required_image_height(image)
            w = required_image_width(image)
            downsample = int(image.get("latent_downsample", _DEFAULT_LATENT_DOWNSAMPLE) or _DEFAULT_LATENT_DOWNSAMPLE)
            channels = int(image.get("latent_channels", _DEFAULT_LATENT_CHANNELS) or _DEFAULT_LATENT_CHANNELS)
            shape = (channels, max(1, h // downsample), max(1, w // downsample))
        shape_tuple = tuple(int(v) for v in shape)
        if state.rng is None:
            # The generator must live on the same device as the sampled latent;
            # ``torch.randn(generator=rng, device=...)`` requires rng.device to
            # match. Building a CPU generator while sampling on CUDA raises.
            state.rng = torch.Generator(device=self.device)
            state.rng.manual_seed(int(image.get("seed", 0) or 0))
        state.latent = init_latent(
            shape_tuple,
            rng=state.rng,
            device=self.device,
            dtype=self.dtype,
            scale=float(image.get("latent_scale", 1.0) or 1.0),
        )
        return state.latent


def _branch_name(branch_index: int, branch_count: int) -> str:
    del branch_count
    return f"branch_{branch_index}"


def _prepare_denoise(
    model: Any,
    state: RequestState,
    op: Mapping[str, Any],
) -> DenoiseContext | TextImageDenoiseStep:
    return model.prepare_denoise(state, op)


def _text_image_batch_predictor(model: Any) -> Any | None:
    return model.predict_text_image_velocity_batch


def _accept_denoise_update(
    model: Any,
    ctx: DenoiseContext | TextImageDenoiseStep,
    latent: torch.Tensor,
) -> None:
    accept = _denoise_update_acceptor(model)
    if accept is not None:
        accept(ctx, latent)
        return
    state = getattr(ctx, "state", None)
    if state is not None:
        state.latent = latent


def _denoise_update_acceptor(model: Any) -> Any | None:
    return model.accept_denoise_update


def _text_image_cfg_plan(step: TextImageDenoiseStep) -> CfgPlan:
    """Single source of truth for a step's branch set and combination weights.

    CFG runs only while the timestep lies within ``cfg_interval`` ``[lo, hi]``
    (inclusive). The default ``(0.0, 1.0)`` enables CFG for the whole trajectory;
    a restricted window such as ``(0.0, 0.5)`` disables CFG above the upper bound
    instead of being short-circuited by ``lo == 0``. The returned plan drives
    both which branches the model evaluates and how they are weighted, so the two
    cannot disagree.
    """
    t_value = float(step.t.detach().float().item())
    lo, hi = step.cfg_interval
    use_cfg = lo <= t_value <= hi
    return build_text_image_cfg_plan(
        cfg_text_scale=step.cfg_text_scale,
        cfg_img_scale=step.cfg_img_scale,
        recipe=step.image_scale_applies_to_text,
        renorm=step.cfg_renorm_type,
        renorm_min=step.cfg_renorm_min,
        use_cfg=use_cfg,
    )


def text_image_branches(step: TextImageDenoiseStep) -> tuple[str, ...]:
    return _text_image_cfg_plan(step).branches


def combine_text_image_velocity(
    step: TextImageDenoiseStep,
    outputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return _text_image_cfg_plan(step).combine(outputs)


def _validate_batched_text_image_outputs(
    steps: Sequence[TextImageDenoiseStep],
    branches_by_step: Sequence[Sequence[str]],
    predicted: Any,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(predicted, Sequence) or isinstance(predicted, (str, bytes, bytearray)):
        raise invalid_descriptor("predict_text_image_velocity_batch must return one mapping per step")
    if len(predicted) != len(steps):
        raise invalid_descriptor("predict_text_image_velocity_batch returned the wrong number of steps")
    out: list[dict[str, torch.Tensor]] = []
    for step, branches, values in zip(steps, branches_by_step, predicted):
        if not isinstance(values, Mapping):
            raise invalid_descriptor("predict_text_image_velocity_batch entries must be mappings")
        checked: dict[str, torch.Tensor] = {}
        for branch in branches:
            velocity = values.get(branch)
            if not isinstance(velocity, torch.Tensor) or velocity.shape != step.latent.shape:
                raise invalid_descriptor(f"{branch} velocity must be a tensor matching the denoise latent")
            checked[branch] = velocity
        out.append(checked)
    return out


def _maybe_convert_parameterization(
    model: Any,
    prediction: torch.Tensor,
    latent: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    parameterization = _velocity_parameterization(model)
    if parameterization == "velocity":
        return prediction
    if parameterization == "x_pred":
        return x_pred_to_velocity(prediction, latent, t)
    raise invalid_descriptor(f"unsupported velocity_parameterization {parameterization!r}")


def _velocity_parameterization(model: Any) -> str:
    return str(model.velocity_parameterization() or "velocity")
