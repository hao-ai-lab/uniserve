"""Flow prefix, denoise, Euler integration, and latent publication."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import torch

from uniserve_worker.execution.batch import (
    DrawLayout,
    FinishFlags,
    ImageParams,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    RunKind,
    TokenSpan,
)
from uniserve_worker.execution.output import TransferPayload
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.models.generation import BranchSource, LatentLayout
from uniserve_worker.models.inputs import PatchTransform
from uniserve_worker.nn.diffusion.cfg import Branch, build_flow_cfg_plan
from uniserve_worker.nn.diffusion.integrator import euler_step
from uniserve_worker.nn.diffusion.schedule import x_pred_to_velocity
from uniserve_worker.nn.vision import get_flattened_position_ids_extrapolate
from uniserve_worker.runtime.latent_pool import LatentPublication
from uniserve_worker.runtime.request import Request

from . import token
from .forward_batch import FlowPatches, ModelPhase, TokenSelection
from .resources import ExecutionResources
from .rng import flow_noise_seed, normal_noise
from .rows import ForwardRow, LaneState, LatentExecution, OperationState, Outcome


def pack_forward(runtime: ExecutionResources, state: OperationState) -> tuple[object, ...]:
    if state.operation.kind is not RunKind.DIFFUSION_STEP:
        return ()
    if state.phase == "initial":
        _initialize(runtime, state)
    if state.phase == "step":
        _prepare_step(runtime, state)
    if state.phase == "prefix":
        state.phase = "prefix_pending"
        return state.rows
    if state.phase == "denoise":
        _pack_denoise(runtime, state)
        state.phase = "denoise_pending"
        return state.rows
    return ()


def consume_forward(
    runtime: ExecutionResources,
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
) -> None:

    if state.phase == "prefix_pending":
        if len(outputs) != len(state.rows):
            raise RuntimeError("flow prefix result is not aligned")
        for branch, task, output in zip(
            state.data.pop("prefix_branches"), state.rows, outputs, strict=True
        ):
            token.token_logits_or_hidden(output)
            token.commit_kv(
                runtime,
                task,
                task.query_tokens,
                state.lane,
                publish_runtime=False,
            )
            entry = state.data["entries"][branch]
            state.data["entries"][branch] = (
                entry[0],
                entry[1],
                entry[2] + task.query_tokens,
                entry[3],
            )
        state.phase = "denoise"
        state.rows = ()
        return
    if state.phase != "denoise_pending" or len(outputs) != len(state.rows):
        raise RuntimeError("flow denoise result is not aligned")
    state.data["predictions"] = {
        branch: prediction(output)
        for branch, output in zip(state.data["guide"].branches, outputs, strict=True)
    }
    state.phase = "integrate"


def integrate(runtime: ExecutionResources, state: OperationState) -> bool:
    if state.phase != "integrate":
        return False

    flow = state.data["flow"]
    current = state.data["current"]
    timestep = state.data["t"]
    velocity = state.data["guide"].combine(state.data.pop("predictions"))
    if flow.prediction in {"x", "x_prediction", "x_pred"}:
        velocity = x_pred_to_velocity(velocity, current, timestep)
    elif flow.prediction != "velocity":
        raise invalid_descriptor(f"unsupported flow prediction {flow.prediction!r}")
    current.copy_(euler_step(current, velocity, timestep, state.data["t_next"]))
    state.data["request"].flow_step = state.data["step"] + 1
    state.data["step"] += 1
    if state.data["step"] < state.data["end_step"]:
        state.phase = "step"
        state.rows = ()
        return True
    _finish(runtime, state)
    return True


def _initialize(runtime: ExecutionResources, state: OperationState) -> None:

    operation = state.operation
    scope = state.lane
    flow = runtime.generation()
    request_id = operation.request_key.request_id
    conditioning = tuple(
        reference for reference in operation.inputs if reference.kind is ProductKind.KV
    )
    latent_inputs = tuple(
        reference for reference in operation.inputs if reference.kind is ProductKind.LATENT
    )
    latent_outputs = tuple(
        reference for reference in operation.outputs if reference.kind is ProductKind.LATENT
    )
    if len(conditioning) != 1 or len(latent_inputs) != 1 or len(latent_outputs) != 1:
        raise invalid_descriptor(
            "flow operation requires exact conditioning and one latent input/output generation"
        )
    cache = runtime.cache_coordinates(operation, scope)
    request = runtime.request_row(scope, request_id)
    runtime.cache_publications.validate_conditioning(
        request_id,
        conditioning[0],
        request_pool_idx=request.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=scope.cache_publication_inputs.get(conditioning[0]),
    )
    image = request.image
    if image is None:
        raise invalid_descriptor("flow operation has no admitted image parameters")
    if operation.rng is not None:
        raise invalid_descriptor("flow continuation must inherit transition RNG state")
    latent_input = latent_inputs[0]
    latent_output = latent_outputs[0]
    if (
        int(latent_input.generation) < 1
        or int(latent_output.generation) < 1
        or latent_input == latent_output
    ):
        raise invalid_descriptor("flow latent generations are invalid")
    if request.latent_product != latent_input:
        raise invalid_descriptor("flow operation does not name the current latent generation")
    row = runtime.latent_row(operation, scope)
    pool = runtime.require_latent_pool()
    start_step = int(row.placement.start_step)
    end_step = start_step + int(row.placement.step_count)
    current = pool.gather_current(
        row.request_pool_idx,
        row.staging,
        step=start_step,
        generation=int(latent_input.generation),
        latent_units=int(row.placement.latent_units),
        height=int(row.placement.height),
        width=int(row.placement.width),
    )
    state.data.update(
        flow=flow,
        cache=cache,
        request=request,
        image=image,
        latent_input=latent_input,
        latent_output=latent_output,
        row=row,
        pool=pool,
        current=current,
        step=start_step,
        start_step=start_step,
        end_step=end_step,
        conditioning_position=int(request.logical_position),
        image_prompt=image.image_prompts[0] if image.image_prompts else "",
        entries={},
    )
    state.phase = "step"


def _prepare_step(runtime: ExecutionResources, state: OperationState) -> None:

    operation = state.operation
    scope = state.lane
    data = state.data
    image = data["image"]
    step = data["step"]
    host_t, host_t_next = data["flow"].schedule_pair(
        int(image.steps), float(image.timestep_shift), step
    )
    t, t_next = data["pool"].stage_timestep(data["row"].request_pool_idx, host_t, host_t_next)
    guide = build_flow_cfg_plan(
        cfg_text_scale=float(image.cfg_text_scale),
        cfg_img_scale=float(image.cfg_img_scale),
        recipe=data["flow"].cfg_recipe,
        renorm=image.cfg_renorm_type,
        renorm_min=float(image.cfg_renorm_min),
        use_cfg=float(image.cfg_interval[0]) <= host_t <= float(image.cfg_interval[1]),
    )
    if len(guide.branches) > int(data["flow"].max_cfg_branches):
        raise invalid_descriptor("flow CFG plan exceeds the model branch bound")
    prefix_rows = []
    prefix_branches = []
    entries = data["entries"]
    descriptors = scope.forward_rows.get(runtime.operation_identity(operation), ())
    if len(descriptors) < len(guide.branches):
        raise invalid_descriptor("media denoise has incomplete forward-row metadata")
    denoise_descriptors = descriptors[-len(guide.branches) :]
    for branch_index, branch in enumerate(guide.branches):
        if branch in entries:
            continue
        source = branch_source(runtime, branch)
        prefix, copy_conditioning = flow_prefix(
            runtime, source, data["image_prompt"], data["request"]
        )
        descriptor = denoise_descriptors[branch_index]
        if copy_conditioning:
            entry = data["cache"]
        else:
            slot = int(descriptor.request_pool_index)
            capacity = runtime.req_to_token_pool.allocated_length(slot)
            runtime.req_to_token_pool.pages(slot, 0)
            has_prefix_forward = any(
                int(candidate.request_pool_index) == slot
                and int(candidate.seq_len) == 0
                and int(candidate.query_len) == len(prefix)
                for candidate in descriptors[: -len(guide.branches)]
            )
            entry = (slot, 0, 0 if has_prefix_forward else int(descriptor.seq_len), capacity)
        prefix_length = data["cache"][2] if copy_conditioning else len(prefix)
        if prefix_length > entry[3]:
            raise invalid_descriptor("flow prefix exceeds scheduler placement")
        if entry[2] not in {0, prefix_length}:
            raise invalid_descriptor(
                "flow branch prefix disagrees with its initialized physical state"
            )
        initialize_prefix = entry[2] == 0 and prefix_length > 0
        entries[branch] = entry
        if initialize_prefix and prefix:
            prefix_rows.append(prefix_row(runtime, operation, prefix, entry, branch, scope))
            prefix_branches.append(branch)
    data.update(guide=guide, t=t, t_next=t_next)
    if prefix_rows:
        data["prefix_branches"] = tuple(prefix_branches)
        state.rows = tuple(prefix_rows)
        state.phase = "prefix"
    else:
        state.phase = "denoise"


def _pack_denoise(runtime: ExecutionResources, state: OperationState) -> None:

    data = state.data
    row = data["row"]
    state.rows = tuple(
        denoise_row(
            runtime,
            state.operation,
            data["conditioning_position"],
            branch,
            data["entries"][branch],
            data["current"],
            data["t"],
            int(row.placement.height),
            int(row.placement.width),
            state.lane,
        )
        for branch in data["guide"].branches
    )


def _finish(runtime: ExecutionResources, state: OperationState) -> None:

    operation = state.operation
    scope = state.lane
    data = state.data
    row = data["row"]
    final_step = data["end_step"]
    data["pool"].write_inactive(
        row.request_pool_idx,
        row.staging,
        expected_step=data["start_step"],
        expected_generation=int(data["latent_input"].generation),
        latent_units=int(row.placement.latent_units),
        height=int(row.placement.height),
        width=int(row.placement.width),
    )
    scope.latent_publications.append(
        LatentPublication(
            request_pool_idx=row.request_pool_idx,
            page_table=row.placement.page_table,
            expected_generation=int(data["latent_input"].generation),
            expected_step=data["start_step"],
            generation=int(data["latent_output"].generation),
            step=final_step,
            latent_units=int(row.placement.latent_units),
            height=int(row.placement.height),
            width=int(row.placement.width),
        )
    )
    data["request"].latent_product = data["latent_output"]
    products = publish_latent_transfer(
        runtime,
        operation,
        data["latent_output"],
        data["current"],
        row,
        step=final_step,
        scope=scope,
    )
    state.outcome = Outcome(
        status=OpStatus.OK,
        selected_point=1,
        logical_lengths=runtime.logical_lengths(
            operation,
            data["request"],
            data["cache"],
            latent_len=final_step,
        ),
        token_span=TokenSpan(base=data["request"].logical_position, len=0),
        finish_flags=FinishFlags(),
        product_generations=runtime.output_generations(operation),
        products=products,
    )
    main_slot = int(data["request"].request_pool_idx)
    alternative_slots = {
        int(entry[0]) for entry in data["entries"].values() if int(entry[0]) != main_slot
    }
    if alternative_slots and final_step >= int(data["image"].steps):
        for slot in alternative_slots:
            scope.runtime_cache_lengths.pop(slot, None)
        runtime.req_to_token_pool.release(tuple(alternative_slots))
        tracked = runtime._flow_prefix_slots.get(operation.request_key)
        if tracked is not None:
            tracked.difference_update(alternative_slots)
            if not tracked:
                runtime._flow_prefix_slots.pop(operation.request_key, None)
    state.phase = "done"
    state.rows = ()


def publish_latent_transfer(
    runtime: ExecutionResources,
    operation: Operation,
    product: ProductRef,
    value: torch.Tensor,
    row: LatentExecution,
    *,
    step: int,
    scope: LaneState,
) -> tuple[ProductPayload, ...]:
    """Publish a committed-candidate trajectory for an exact staged consumer."""

    transport = runtime.transport
    if (
        transport is None
        or transport.name == "local"
        or (runtime.deployment is not None and int(runtime.deployment.tp_rank) != 0)
    ):
        return ()
    locator = transport.publish_async(value.detach().contiguous())
    metadata = {
        "generation": int(product.generation),
        "height": int(row.placement.height),
        "latent_units": int(row.placement.latent_units),
        "step": int(step),
        "width": int(row.placement.width),
    }
    locator = replace(locator, meta={**locator.meta, **metadata})
    scope.published.append(locator)
    scope.stage_publications[runtime.operation_identity(operation)] = (locator,)
    descriptor = TransferPayload(
        "latent",
        {"locator": locator.to_mapping(), **metadata},
        (locator,),
        transport,
    )
    return (ProductPayload(product=product, payload=cast(bytes, descriptor)),)


def initial_latent(
    runtime: ExecutionResources,
    operation: Operation,
    height: int,
    width: int,
    target: torch.Tensor,
) -> None:
    flow = runtime.generation()
    rng = operation.rng
    assert rng is not None and rng.draw_layout is DrawLayout.FLOW_NOISE
    seed = flow_noise_seed(int(rng.seed), int(rng.semantic_index_base))
    raw = target.reshape(flow.latent_shape(height, width))
    normal_noise(
        tuple(int(value) for value in raw.shape),
        seed=seed,
        device=target.device,
        dtype=target.dtype,
        out=raw,
    )
    raw.mul_(flow.noise_scale(height, width))
    neural = flow.neural_latent(raw)
    if neural.data_ptr() != target.data_ptr() or tuple(neural.shape) != tuple(target.shape):
        target.copy_(neural.reshape_as(target))


def branch_source(runtime: ExecutionResources, branch: Branch) -> BranchSource:
    return runtime.generation().branch_source(branch)


def flow_prefix(
    runtime: ExecutionResources,
    source: BranchSource,
    image_prompt: str,
    request: Request,
) -> tuple[tuple[int, ...], bool]:
    return runtime.generation().prefix(
        source,
        image_prompt=image_prompt,
        negative_prompt=require_image(request).negative_prompt,
        negative_token_ids=request.negative_token_ids,
        tokenizer=runtime.tokenizer,
    )


def prefix_row(
    runtime: ExecutionResources,
    operation: Operation,
    tokens: tuple[int, ...],
    entry: tuple[int, int, int, int],
    branch: Branch,
    scope: LaneState,
) -> ForwardRow:
    request = runtime.request_row(scope, operation.request_key.request_id)
    positions = torch.arange(entry[2], entry[2] + len(tokens), dtype=torch.long)
    return ForwardRow(
        operation=operation,
        request=request,
        weights=runtime.weights,
        phase=ModelPhase.TEXT,
        token_ids=torch.tensor(tokens, dtype=torch.long),
        positions=positions,
        selection=TokenSelection.HIDDEN,
        request_pool_idx=entry[0],
        seq_len=entry[2],
        group_id=entry[1],
        write_kv=True,
        causal=True,
        attention_indexes=torch.stack(
            (positions, torch.zeros_like(positions), torch.zeros_like(positions))
        ),
    )


def denoise_row(
    runtime: ExecutionResources,
    operation: Operation,
    conditioning_position: int,
    branch: Branch,
    entry: tuple[int, int, int, int],
    latent: torch.Tensor,
    timestep: torch.Tensor,
    height: int,
    width: int,
    scope: LaneState,
) -> ForwardRow:
    request = runtime.request_row(scope, operation.request_key.request_id)
    flow = runtime.generation()
    image_tokens = image_token_count(runtime, latent, height, width)
    text_local: tuple[int, ...]
    if flow.latent_layout is LatentLayout.PATCH_TOKENS:
        latent_positions = get_flattened_position_ids_extrapolate(
            height,
            width,
            int(flow.latent_downsample),
            int(math.isqrt(flow.max_latent_tokens)),
        )
        conditioning: FlowPatches | None = None
        query_tokens = image_tokens + int(flow.commit_marker_tokens)
        temporal = _temporal_position(runtime, branch, conditioning_position, entry)
        attention_indexes = torch.stack(
            (
                torch.full((query_tokens,), temporal, dtype=torch.long),
                torch.zeros(query_tokens, dtype=torch.long),
                torch.zeros(query_tokens, dtype=torch.long),
            )
        )
        text_local = (0, query_tokens - 1)
    else:
        latent_positions = _spatial_positions(
            runtime,
            height,
            width,
            int(flow.latent_patch_size),
            _temporal_position(runtime, branch, conditioning_position, entry),
        )
        conditioning = _conditioning(runtime, latent, height, width)
        query_tokens = image_tokens
        attention_indexes = latent_positions
        text_local = ()
    return ForwardRow(
        operation=operation,
        request=request,
        weights=runtime.weights,
        phase=ModelPhase.DENOISE,
        flow_conditioning=conditioning,
        positions=latent_positions,
        timestep=timestep.reshape(1),
        latent=latent,
        image_tokens=query_tokens,
        image_height=height,
        image_width=width,
        request_pool_idx=entry[0],
        seq_len=entry[2],
        group_id=entry[1],
        write_kv=False,
        causal=False,
        attention_indexes=attention_indexes,
        text_local_indices=text_local,
    )


def _temporal_position(
    runtime: ExecutionResources,
    branch: Branch,
    conditioning_position: int,
    entry: tuple[int, int, int, int],
) -> int:
    if branch is Branch.COND:
        return int(conditioning_position)
    return int(entry[2])


def image_token_count(
    runtime: ExecutionResources, latent: torch.Tensor, height: int, width: int
) -> int:
    del latent
    return runtime.generation().image_tokens(height, width)


def require_image(request: Request) -> ImageParams:
    if request.image is None:
        raise invalid_descriptor("flow execution requires admitted image parameters")
    return request.image


def prediction(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("flow route did not return a flow prediction")
    return output


def physical_tokens(runtime: ExecutionResources, height: int, width: int) -> int:
    return runtime.generation().physical_tokens(height, width)


def _conditioning(
    runtime: ExecutionResources,
    latent: torch.Tensor,
    height: int,
    width: int,
) -> FlowPatches | None:
    transform = runtime.image_processor().vit
    generation = runtime.generation()
    return generation.conditioning(
        generation.materialization_latent(latent, height, width),
        height,
        width,
        patch_size=(int(transform.patch_size) if isinstance(transform, PatchTransform) else None),
    )


def _spatial_positions(
    runtime: ExecutionResources,
    height: int,
    width: int,
    patch: int,
    temporal: int,
) -> torch.Tensor:
    grid_height = height // patch
    grid_width = width // patch
    y = torch.arange(grid_height, dtype=torch.long).repeat_interleave(grid_width)
    x = torch.arange(grid_width, dtype=torch.long).repeat(grid_height)
    return torch.stack((torch.full_like(x, int(temporal)), y, x))


__all__ = ["consume_forward", "integrate", "pack_forward"]
