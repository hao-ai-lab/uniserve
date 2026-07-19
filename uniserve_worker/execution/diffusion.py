"""Diffusion preparation and projection internals for ``ModelRunner``."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from uniserve_worker.contracts.forward_batch import (
    DenoiseBranchKey,
    DenoisePostprocessEntry,
    ForwardResult,
)
from uniserve_worker.contracts.model_protocols import FlowContext
from uniserve_worker.contracts.outputs import (
    FlowOutput,
)
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.nn.diffusion import (
    CfgParams,
    FlowMatchSchedule,
    ScheduleDirection,
    combine_cfg,
    euler_step,
    init_latent,
    x_pred_to_velocity,
)
from uniserve_worker.runtime.image_params import required_image_height, required_image_width
from uniserve_worker.runtime.request_state import RequestState

from .flow import PreparedFlowStep, combine_flow_velocity, flow_branches


def _execute_required_denoise(items: Any, model: Any) -> list[FlowOutput]:
    return _DiffusionRuntime().step_many(items, model, graph_mode="require")


if TYPE_CHECKING:
    from uniserve_worker.contracts.model_protocols import UniModel


# Decode token/position relays (device-resident sequence feedback)
# ---------------------
# Flow-step session (single denoise update)
# ---------------------


class _FlowSession:
    """Owns branch prediction and latent update semantics for one denoise step."""

    def __init__(
        self,
        model: Any,
        step: Any,
        *,
        combine_velocity: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor],
        accept_update: Callable[[Any, Any, torch.Tensor], None],
    ) -> None:
        self.model = model
        self.step = step
        self._combine_velocity = combine_velocity
        self._accept_update = accept_update

    def prepare_step(self, op: Mapping[str, Any] | None = None) -> Any:
        del op
        return self.step

    def predict_velocity(self, branch: str) -> torch.Tensor:
        velocity = self.model.predict_velocity(
            self.step,
            self.step.t,
            self.step.latent,
            branch,
        )
        if not isinstance(velocity, torch.Tensor) or velocity.shape != self.step.latent.shape:
            raise invalid_descriptor(
                f"{branch} velocity must be a tensor matching the denoise latent"
            )
        return velocity

    def apply_update(self, velocities: Mapping[str, torch.Tensor]) -> FlowOutput:
        velocity = self._combine_velocity(self.step, velocities)
        updated = euler_step(self.step.latent, velocity, self.step.t, self.step.t_next)
        self._accept_update(self.model, self.step, updated)
        done = self.step.step_index + 1 >= self.step.total_steps
        return FlowOutput(
            req_id=self.step.req_id,
            denoise_done=done,
            num_steps_done=self.step.step_index + 1,
        )

    def release(self) -> None:
        release = getattr(self.step, "release", None)
        if callable(release):
            release()


# ---------------------
# Flow-step execution (denoise)
# ---------------------

# Fallback latent geometry for the model-neutral generic path, used only when an
# op/image descriptor supplies neither an explicit ``latent_shape`` nor the
# per-field overrides. Production models compute their own latent geometry and
# never reach this fallback.
_DEFAULT_LATENT_DOWNSAMPLE = 16
_DEFAULT_LATENT_CHANNELS = 4


class _DiffusionRuntime:
    """Execute one model-neutral flow-matching denoise step."""

    def __init__(
        self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        # ``device``/``dtype`` take effect only on the model-neutral generic
        # ``FlowContext``/``_latent`` path (see ``_finish_prepared_step``).
        # Production diffusion models return a ``PreparedFlowStep`` and run
        # on their own device/dtype, so these defaults are inert for them.
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def step(
        self, req_id: int, state: RequestState, model: "UniModel", op: Mapping[str, Any]
    ) -> FlowOutput:
        return self.step_many([(req_id, state, op)], model)[0]

    @torch.inference_mode()
    def step_many(
        self,
        items: Sequence[tuple[int, RequestState, Mapping[str, Any]]],
        model: "UniModel",
        *,
        graph_mode: str = "auto",
    ) -> list[FlowOutput]:
        step_counts = [_denoise_step_count(op) for _req_id, _state, op in items]
        if any(count > 1 for count in step_counts):
            return self._step_many_burst(items, model, step_counts, graph_mode=graph_mode)
        prepared = [
            (int(req_id), state, op, self._prepare(req_id, state, model, op))
            for req_id, state, op in items
        ]
        if all(isinstance(item[3], PreparedFlowStep) for item in prepared):
            return self._flow_steps(
                model,
                [item[3] for item in prepared if isinstance(item[3], PreparedFlowStep)],
                graph_mode=graph_mode,
            )
        return [
            self._finish_prepared_step(req_id, state, model, op, ctx)
            for req_id, state, op, ctx in prepared
        ]

    @torch.inference_mode()
    def forward_result(
        self,
        items: Sequence[tuple[int, RequestState, Mapping[str, Any]]],
        model: "UniModel",
        *,
        row_indices: Sequence[int] | None = None,
        graph_mode: str = "auto",
    ) -> ForwardResult | None:
        step_counts = [_denoise_step_count(op) for _req_id, _state, op in items]
        if any(count > 1 for count in step_counts):
            return None
        rows = (
            tuple(range(len(items)))
            if row_indices is None
            else tuple(int(row) for row in row_indices)
        )
        if len(rows) != len(items):
            raise invalid_descriptor("denoise row_indices must align with denoise items")
        prepared = [
            (int(row_index), int(req_id), state, op, self._prepare(req_id, state, model, op))
            for row_index, (req_id, state, op) in zip(rows, items, strict=True)
        ]
        flow_items = [
            (row_index, step)
            for row_index, _req_id, _state, _op, step in prepared
            if isinstance(step, PreparedFlowStep)
        ]
        velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        updates: dict[int, DenoisePostprocessEntry] = {}
        if flow_items:
            flow_entries = self._flow_forward_entries(
                model,
                flow_items,
                graph_mode=graph_mode,
            )
            if flow_entries is None:
                return None
            flow_velocities, flow_updates = flow_entries
            velocities.update(flow_velocities)
            updates.update(flow_updates)
        for row_index, req_id, state, op, ctx in prepared:
            if isinstance(ctx, PreparedFlowStep):
                continue
            if graph_mode == "require":
                return None
            entry, branch_velocities = self._generic_forward_entry(
                row_index,
                req_id,
                state,
                model,
                op,
                ctx,
            )
            updates[int(row_index)] = entry
            velocities.update(branch_velocities)
        return ForwardResult(denoise_velocities=velocities, denoise_updates=updates)

    def _step_many_burst(
        self,
        items: Sequence[tuple[int, RequestState, Mapping[str, Any]]],
        model: "UniModel",
        step_counts: Sequence[int],
        *,
        graph_mode: str,
    ) -> list[FlowOutput]:
        outputs: list[FlowOutput | None] = [None] * len(items)
        active: list[dict[str, Any]] = []
        for index, ((req_id, state, op), step_count) in enumerate(
            zip(items, step_counts, strict=True)
        ):
            op_dict = dict(op)
            cursor = int(op_dict.get("timestep_idx", state.schedule_cursor) or 0)
            active.append(
                {
                    "index": index,
                    "req_id": int(req_id),
                    "state": state,
                    "op": op_dict,
                    "cursor": cursor,
                    "remaining": int(step_count),
                }
            )

        while active:
            prepared: list[
                tuple[dict[str, Any], Mapping[str, Any], FlowContext | PreparedFlowStep]
            ] = []
            for item in active:
                op = dict(item["op"])
                op["timestep_idx"] = int(item["cursor"])
                prepared.append(
                    (
                        item,
                        op,
                        self._prepare(int(item["req_id"]), item["state"], model, op),
                    )
                )

            if all(isinstance(ctx, PreparedFlowStep) for _item, _op, ctx in prepared):
                step_outputs = self._flow_steps(
                    model,
                    [ctx for _item, _op, ctx in prepared if isinstance(ctx, PreparedFlowStep)],
                    graph_mode=graph_mode,
                )
            else:
                step_outputs = [
                    self._finish_prepared_step(
                        int(item["req_id"]),
                        item["state"],
                        model,
                        op,
                        ctx,
                    )
                    for item, op, ctx in prepared
                ]

            next_active: list[dict[str, Any]] = []
            for (item, _op, _ctx), output in zip(prepared, step_outputs, strict=True):
                outputs[int(item["index"])] = output
                item["remaining"] = int(item["remaining"]) - 1
                item["cursor"] = int(output.num_steps_done)
                if not output.denoise_done and int(item["remaining"]) > 0:
                    next_active.append(item)
            active = next_active

        if any(output is None for output in outputs):
            raise invalid_descriptor("denoise burst did not produce an output for every op")
        return [output for output in outputs if output is not None]

    def _prepare(
        self,
        req_id: int,
        state: RequestState,
        model: "UniModel",
        op: Mapping[str, Any],
    ) -> FlowContext | PreparedFlowStep:
        del req_id
        prepared = _prepare_flow(model, state, op)
        if isinstance(prepared, PreparedFlowStep):
            return prepared
        if not isinstance(prepared, FlowContext):
            raise invalid_descriptor("prepare_flow(state, op) must return FlowContext")
        return prepared

    def _finish_prepared_step(
        self,
        req_id: int,
        state: RequestState,
        model: "UniModel",
        op: Mapping[str, Any],
        prepared: FlowContext | PreparedFlowStep,
    ) -> FlowOutput:
        if isinstance(prepared, PreparedFlowStep):
            return self._flow_steps(model, [prepared])[0]
        # Model-neutral generic flow-matching path. It is the contract for models
        # that return a FlowContext (and is exercised by the synthetic
        # velocity-only test model); the production diffusion models instead
        # return a PreparedFlowStep above and build their own
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
                raise invalid_descriptor(
                    "predict_velocity(ctx, t, latent, branch) must return a tensor"
                )
            velocity = _maybe_convert_parameterization(model, velocity, latent, t)
            if velocity.shape != latent.shape:
                raise invalid_descriptor(
                    f"velocity shape {tuple(velocity.shape)} does not match latent {tuple(latent.shape)}"
                )
            velocities.append(velocity)
        velocity = combine_cfg(velocities, cfg)
        state.latent = euler_step(latent, velocity, t, t_next)
        _accept_flow_update(model, prepared, state.latent)
        done = cursor + 1 >= steps
        return FlowOutput(req_id=req_id, denoise_done=done, num_steps_done=cursor + 1)

    def _flow_steps(
        self,
        model: "UniModel",
        steps: Sequence[PreparedFlowStep],
        *,
        graph_mode: str = "auto",
    ) -> list[FlowOutput]:
        branches_by_step = [flow_branches(step) for step in steps]
        batch_predict = _flow_batch_predictor(model)
        if batch_predict is not None:
            predicted = _call_flow_batch_predictor(
                batch_predict,
                steps,
                branches_by_step,
                graph_mode=graph_mode,
            )
        else:
            predicted = None
        if predicted is None and graph_mode == "require":
            raise invalid_descriptor("flow graph mode required a graphable batch")
        if predicted is None:
            branch_outputs = [
                {
                    branch: self._predict_flow_branch(model.predict_velocity, step, branch)
                    for branch in branches
                }
                for step, branches in zip(steps, branches_by_step)
            ]
        else:
            branch_outputs = _validate_batched_flow_outputs(steps, branches_by_step, predicted)
        outputs = []
        for step, velocities in zip(steps, branch_outputs):
            session = _FlowSession(
                model,
                step,
                combine_velocity=combine_flow_velocity,
                accept_update=_accept_flow_update,
            )
            outputs.append(session.apply_update(velocities))
        return outputs

    def _predict_flow_branch(
        self, predict: Any, step: PreparedFlowStep, branch: str
    ) -> torch.Tensor:
        velocity = predict(step, step.t, step.latent, branch)
        if not isinstance(velocity, torch.Tensor) or velocity.shape != step.latent.shape:
            raise invalid_descriptor(
                f"{branch} velocity must be a tensor matching the denoise latent"
            )
        return velocity

    def _flow_forward_entries(
        self,
        model: "UniModel",
        items: Sequence[tuple[int, PreparedFlowStep]],
        *,
        graph_mode: str = "auto",
    ) -> tuple[dict[DenoiseBranchKey, torch.Tensor], dict[int, DenoisePostprocessEntry]] | None:
        steps = [step for _row_index, step in items]
        branches_by_step = [flow_branches(step) for step in steps]
        batch_predict = _flow_batch_predictor(model)
        predicted = (
            _call_flow_batch_predictor(
                batch_predict,
                steps,
                branches_by_step,
                graph_mode=graph_mode,
            )
            if batch_predict is not None
            else None
        )
        if predicted is None and graph_mode == "require":
            return None
        if predicted is None:
            branch_outputs = [
                {
                    branch: self._predict_flow_branch(model.predict_velocity, step, branch)
                    for branch in branches
                }
                for step, branches in zip(steps, branches_by_step, strict=True)
            ]
        else:
            branch_outputs = _validate_batched_flow_outputs(steps, branches_by_step, predicted)
        velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        updates: dict[int, DenoisePostprocessEntry] = {}
        for (row_index, step), branches, outputs in zip(
            items,
            branches_by_step,
            branch_outputs,
            strict=True,
        ):
            for branch_id, branch in enumerate(branches):
                velocities[DenoiseBranchKey(int(row_index), int(branch_id))] = outputs[branch]

            def combine_step_velocity(
                values: Mapping[Any, torch.Tensor],
                current_step: PreparedFlowStep = step,
            ) -> torch.Tensor:
                return combine_flow_velocity(current_step, values)

            def accept_step_update(
                latent: torch.Tensor,
                current_model: Any = model,
                current_step: PreparedFlowStep = step,
            ) -> None:
                _accept_flow_update(current_model, current_step, latent)

            updates[int(row_index)] = DenoisePostprocessEntry(
                row_index=int(row_index),
                req_id=int(step.req_id),
                step_index=int(step.step_index),
                total_steps=int(step.total_steps),
                branch_names=tuple(branches),
                latent=step.latent,
                t=step.t,
                t_next=step.t_next,
                combine_velocity=combine_step_velocity,
                accept_update=accept_step_update,
            )
        return velocities, updates

    def _generic_forward_entry(
        self,
        row_index: int,
        req_id: int,
        state: RequestState,
        model: "UniModel",
        op: Mapping[str, Any],
        prepared: FlowContext,
    ) -> tuple[DenoisePostprocessEntry, dict[DenoiseBranchKey, torch.Tensor]]:
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
        branch_names = tuple(
            _branch_name(index, cfg.branch_count) for index in range(cfg.branch_count)
        )
        velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        for branch_id, branch in enumerate(branch_names):
            velocity = model.predict_velocity(prepared, t, latent, branch)
            if not isinstance(velocity, torch.Tensor):
                raise invalid_descriptor(
                    "predict_velocity(ctx, t, latent, branch) must return a tensor"
                )
            velocity = _maybe_convert_parameterization(model, velocity, latent, t)
            if velocity.shape != latent.shape:
                raise invalid_descriptor(
                    f"velocity shape {tuple(velocity.shape)} does not match latent {tuple(latent.shape)}"
                )
            velocities[DenoiseBranchKey(int(row_index), int(branch_id))] = velocity

        def combine_generic_velocity(values: Mapping[Any, torch.Tensor]) -> torch.Tensor:
            return combine_cfg([values[branch] for branch in branch_names], cfg)

        def accept_generic_update(latent_value: torch.Tensor) -> None:
            _accept_flow_update(model, prepared, latent_value)

        entry = DenoisePostprocessEntry(
            row_index=int(row_index),
            req_id=int(req_id),
            step_index=int(cursor),
            total_steps=int(steps),
            branch_names=branch_names,
            latent=latent,
            t=t,
            t_next=t_next,
            combine_velocity=combine_generic_velocity,
            accept_update=accept_generic_update,
        )
        return entry, velocities

    def _latent(
        self, state: RequestState, image: Mapping[str, Any], op: Mapping[str, Any]
    ) -> torch.Tensor:
        if isinstance(state.latent, torch.Tensor):
            return state.latent.to(device=self.device, dtype=self.dtype)
        shape = op.get("latent_shape") or image.get("latent_shape")
        if shape is None:
            h = required_image_height(image)
            w = required_image_width(image)
            downsample = int(
                image.get("latent_downsample", _DEFAULT_LATENT_DOWNSAMPLE)
                or _DEFAULT_LATENT_DOWNSAMPLE
            )
            channels = int(
                image.get("latent_channels", _DEFAULT_LATENT_CHANNELS) or _DEFAULT_LATENT_CHANNELS
            )
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


def _prepare_flow(
    model: Any,
    state: RequestState,
    op: Mapping[str, Any],
) -> FlowContext | PreparedFlowStep:
    return model.prepare_flow(state, op)


def _denoise_step_count(op: Mapping[str, Any]) -> int:
    raw = op.get("denoise_step_count") or 1
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor("denoise_step_count must be a positive integer") from exc
    if value <= 0:
        raise invalid_descriptor("denoise_step_count must be positive")
    return value


def _flow_batch_predictor(model: Any) -> Any | None:
    return model.predict_flow_velocity_batch


def _call_flow_batch_predictor(
    predictor: Any,
    steps: Sequence[PreparedFlowStep],
    branches_by_step: Sequence[Sequence[str]],
    *,
    graph_mode: str,
) -> list[dict[str, torch.Tensor]] | None:
    if graph_mode == "auto":
        return predictor(steps, branches_by_step)
    if not _accepts_graph_mode(predictor):
        return None if graph_mode == "require" else predictor(steps, branches_by_step)
    return predictor(steps, branches_by_step, graph_mode=graph_mode)


def _accepts_graph_mode(callable_obj: Any) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "graph_mode" in signature.parameters


def _accept_flow_update(
    model: Any,
    ctx: FlowContext | PreparedFlowStep,
    latent: torch.Tensor,
) -> None:
    accept = _flow_update_acceptor(model)
    if accept is not None:
        accept(ctx, latent)
        return
    state = getattr(ctx, "state", None)
    if state is not None:
        state.latent = latent


def _flow_update_acceptor(model: Any) -> Any | None:
    return model.accept_flow_update


def _validate_batched_flow_outputs(
    steps: Sequence[PreparedFlowStep],
    branches_by_step: Sequence[Sequence[str]],
    predicted: Any,
) -> list[dict[str, torch.Tensor]]:
    if not isinstance(predicted, Sequence) or isinstance(predicted, (str, bytes, bytearray)):
        raise invalid_descriptor("predict_flow_velocity_batch must return one mapping per step")
    if len(predicted) != len(steps):
        raise invalid_descriptor("predict_flow_velocity_batch returned the wrong number of steps")
    out: list[dict[str, torch.Tensor]] = []
    for step, branches, values in zip(steps, branches_by_step, predicted):
        if not isinstance(values, Mapping):
            raise invalid_descriptor("predict_flow_velocity_batch entries must be mappings")
        checked: dict[str, torch.Tensor] = {}
        for branch in branches:
            velocity = values.get(branch)
            if not isinstance(velocity, torch.Tensor) or velocity.shape != step.latent.shape:
                raise invalid_descriptor(
                    f"{branch} velocity must be a tensor matching the denoise latent"
                )
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
