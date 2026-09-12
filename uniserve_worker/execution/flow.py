"""Flow prefix, denoise, Euler integration, and latent publication."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from uniserve_worker.execution.batch import (
    DrawLayout,
    FinishFlags,
    ForwardMode,
    ImageParams,
    OpStatus,
    PipelineStage,
    ScheduledRequest,
    TensorPublication,
    TensorRef,
)
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.models.generation import BranchSource, LatentLayout
from uniserve_worker.models.inputs import PatchTransform
from uniserve_worker.nn.diffusion.cfg import Branch, build_flow_cfg_plan
from uniserve_worker.nn.diffusion.integrator import euler_step
from uniserve_worker.nn.diffusion.schedule import x_pred_to_velocity
from uniserve_worker.nn.vision import get_flattened_position_ids_extrapolate
from uniserve_worker.runtime.latent_pool import LatentPublication
from uniserve_worker.runtime.request import Request

from ..runtime.latent_pool import require_latent_pool
from . import operations as operation_geometry
from . import token
from .forward_batch import FlowPatches, TokenSelection
from .rng import flow_noise_seed, normal_noise
from .rows import ForwardRow, LaneState, LatentExecution, OperationState, Outcome

if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers import PreTrainedTokenizerBase

    from ..bootstrap.worker_info import WorkerInfo
    from ..config import WorkerConfig
    from ..runtime.cache_publications import CachePublications
    from ..runtime.latent_pool import LatentPool
    from ..runtime.req_to_token_pool import ReqToTokenPool
    from ..runtime.runtime_states import RuntimeStates
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner


def pack_forward(
    state: OperationState,
    *,
    cache_registry: CachePublications | None,
    latent_pool: LatentPool | None,
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
    tokenizer: PreTrainedTokenizerBase | None,
) -> tuple[ForwardRow, ...]:
    """Pack diffusion prefix or denoise state into the matching model-forward row."""

    if state.operation.kind is not PipelineStage.DENOISING:
        return ()
    if state.phase == "initial":
        _initialize(
            state,
            cache_registry=cache_registry,
            latent_pool=latent_pool,
            request_tables=request_tables,
            model_runner=model_runner,
        )
    if state.phase == "step":
        _prepare_step(
            state,
            request_tables=request_tables,
            model_runner=model_runner,
            tokenizer=tokenizer,
        )
    if state.phase == "prefix":
        state.phase = "prefix_pending"
        return state.rows
    if state.phase == "denoise":
        _pack_denoise(state, model_runner=model_runner)
        state.phase = "denoise_pending"
        return state.rows
    return ()


def consume_forward(
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
    *,
    request_tables: ReqToTokenPool | None,
    runtime_states: RuntimeStates | None,
) -> None:
    """Publish prefix conditioning or integrate denoise predictions into latent state."""

    if state.phase == "prefix_pending":
        if len(outputs) != len(state.rows):
            raise RuntimeError("flow prefix result is not aligned")
        for branch, task, output in zip(
            state.data.pop("prefix_branches"), state.rows, outputs, strict=True
        ):
            token.token_logits_or_hidden(output)
            token.commit_kv(
                task,
                task.query_tokens,
                state.lane,
                publish_runtime=False,
                request_tables=request_tables,
                runtime_states=runtime_states,
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


def integrate(
    state: OperationState,
    *,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    config: WorkerConfig,
) -> bool:
    """Advance one latent trajectory step and publish its checkpointed state transition."""

    if state.phase != "integrate":
        return False

    request = operation_geometry.request_row(state.lane, state.operation.request_key.request_id)

    flow = state.data["flow"]
    current = state.data["current"]
    timestep = state.data["t"]
    velocity = state.data["guide"].combine(state.data.pop("predictions"))
    if flow.prediction in {"x", "x_prediction", "x_pred"}:
        velocity = x_pred_to_velocity(velocity, current, timestep)
    elif flow.prediction != "velocity":
        raise invalid_descriptor(f"unsupported flow prediction {flow.prediction!r}")
    current.copy_(euler_step(current, velocity, timestep, state.data["t_next"]))
    request.flow_step = state.data["step"] + 1
    state.data["step"] += 1
    if state.data["step"] < state.data["end_step"]:
        state.phase = "step"
        state.rows = ()
        return True
    _finish(
        state,
        worker_info=worker_info,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
        request_tables=request_tables,
        config=config,
    )
    return True


def _initialize(
    state: OperationState,
    *,
    cache_registry: CachePublications | None,
    latent_pool: LatentPool | None,
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
) -> None:
    """Create a diffusion trajectory from deterministic noise and publish its initial state."""

    operation = state.operation
    scope = state.lane
    flow = model_runner.generation()
    request_id = operation.request_key.request_id
    conditioning = operation.kv_input
    latent_input = operation.latent_input
    latent_output = operation.latent_output
    if conditioning is None or latent_input is None or latent_output is None:
        raise invalid_descriptor(
            "flow operation requires exact conditioning and one latent input/output generation"
        )
    cache = operation_geometry.cache_coordinates(operation, scope, tables=request_tables)
    request = operation_geometry.request_row(scope, request_id)
    publications = cache_registry
    if publications is None:
        raise invalid_descriptor("flow conditioning requires cache publication storage")
    publications.validate_conditioning(
        operation.request_key,
        conditioning,
        request_pool_idx=request.request.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=scope.cache_publication_inputs.get(conditioning),
    )
    image = request.request.image
    if image is None:
        raise invalid_descriptor("flow operation has no admitted image parameters")
    if operation.rng is not None:
        raise invalid_descriptor("flow continuation must inherit transition RNG state")
    if (
        int(latent_input.generation) < 1
        or int(latent_output.generation) < 1
        or latent_input == latent_output
    ):
        raise invalid_descriptor("flow latent generations are invalid")
    if request.latent_product != latent_input:
        raise invalid_descriptor("flow operation does not name the current latent generation")
    row = operation_geometry.latent_row(operation, scope)
    pool = require_latent_pool(latent_pool)
    start_step = int(row.params.start_step)
    end_step = start_step + int(row.params.step_count)
    current = pool.gather_current(
        row.request_pool_idx,
        row.staging,
        step=start_step,
        generation=int(latent_input.generation),
        latent_units=int(row.params.latent_units),
        height=int(row.params.height),
        width=int(row.params.width),
    )
    state.data.update(
        flow=flow,
        cache=cache,
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


def _prepare_step(
    state: OperationState,
    *,
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    """Gather current latent pages and construct one guided diffusion-step batch."""

    request = operation_geometry.request_row(state.lane, state.operation.request_key.request_id)

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
    descriptors = scope.forward_indices.get(operation_geometry.operation_identity(operation), ())
    if len(descriptors) < len(guide.branches):
        raise invalid_descriptor("media denoise has incomplete forward-row metadata")
    denoise_descriptors = descriptors[-len(guide.branches) :]
    for branch_index, branch in enumerate(guide.branches):
        if branch in entries:
            continue
        source = branch_source(branch, model_runner=model_runner)
        prefix, copy_conditioning = flow_prefix(
            source,
            data["image_prompt"],
            request.request,
            model_runner=model_runner,
            tokenizer=tokenizer,
        )
        descriptor = denoise_descriptors[branch_index]
        if copy_conditioning:
            entry = data["cache"]
        else:
            slot = scope.lane.request_pool_indices[descriptor]
            page_tables = request_tables
            if page_tables is None:
                raise invalid_descriptor("flow prefixes require request page tables")
            capacity = page_tables.allocated_length(slot)
            page_tables.pages(slot, 0)
            has_prefix_forward = any(
                scope.lane.request_pool_indices[candidate] == slot
                and (scope.lane.seq_lens[candidate] - scope.lane.query_lens[candidate]) == 0
                and scope.lane.query_lens[candidate] == len(prefix)
                for candidate in descriptors[: -len(guide.branches)]
            )
            entry = (
                slot,
                0,
                0
                if has_prefix_forward
                else (scope.lane.seq_lens[descriptor] - scope.lane.query_lens[descriptor]),
                capacity,
            )
        prefix_length = data["cache"][2] if copy_conditioning else len(prefix)
        if prefix_length > entry[3]:
            raise invalid_descriptor("flow prefix exceeds scheduler params")
        if entry[2] not in {0, prefix_length}:
            raise invalid_descriptor(
                "flow branch prefix disagrees with its initialized physical state"
            )
        initialize_prefix = entry[2] == 0 and prefix_length > 0
        entries[branch] = entry
        if initialize_prefix and prefix:
            prefix_rows.append(prefix_row(operation, prefix, entry, branch, scope))
            prefix_branches.append(branch)
    data.update(guide=guide, t=t, t_next=t_next)
    if prefix_rows:
        data["prefix_branches"] = tuple(prefix_branches)
        state.rows = tuple(prefix_rows)
        state.phase = "prefix"
    else:
        state.phase = "denoise"


def _pack_denoise(state: OperationState, *, model_runner: ModelRunner) -> None:
    """Assemble denoising rows, branch weights, positions, and timestep conditioning."""

    data = state.data
    row = data["row"]
    state.rows = tuple(
        denoise_row(
            state.operation,
            data["conditioning_position"],
            branch,
            data["entries"][branch],
            data["current"],
            data["t"],
            int(row.params.height),
            int(row.params.width),
            state.lane,
            model_runner=model_runner,
        )
        for branch in data["guide"].branches
    )


def _finish(
    state: OperationState,
    *,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    config: WorkerConfig,
) -> None:
    """Integrate predicted velocity, write the next latent bank, and prepare publication."""

    request = operation_geometry.request_row(state.lane, state.operation.request_key.request_id)

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
        latent_units=int(row.params.latent_units),
        height=int(row.params.height),
        width=int(row.params.width),
    )
    scope.latent_publications.append(
        LatentPublication(
            request_pool_idx=row.request_pool_idx,
            page_table=row.params.page_table,
            expected_generation=int(data["latent_input"].generation),
            expected_step=data["start_step"],
            generation=int(data["latent_output"].generation),
            step=final_step,
            latent_units=int(row.params.latent_units),
            height=int(row.params.height),
            width=int(row.params.width),
        )
    )
    request.latent_product = data["latent_output"]
    products = publish_latent_transfer(
        operation,
        data["latent_output"],
        row,
        step=final_step,
        scope=scope,
        worker_info=worker_info,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
        config=config,
    )
    state.outcome = Outcome(
        status=OpStatus.OK,
        runtime=operation_geometry.execution_runtime(
            request,
            data["cache"],
            flow_step=final_step,
        ),
        finish_flags=FinishFlags(),
        product_generations=operation_geometry.output_generations(operation),
        products=products,
    )
    main_slot = int(request.request.request_pool_idx)
    alternative_slots = {
        int(entry[0]) for entry in data["entries"].values() if int(entry[0]) != main_slot
    }
    if alternative_slots and final_step >= int(data["image"].steps):
        for slot in alternative_slots:
            scope.runtime_cache_lengths.pop(slot, None)
        page_tables = request_tables
        if page_tables is None:
            raise RuntimeError("flow prefix retirement lost its request page tables")
        page_tables.release_prefixes(operation.request_key, tuple(alternative_slots))
    state.phase = "done"
    state.rows = ()


def publish_latent_transfer(
    operation: ScheduledRequest,
    product: TensorRef,
    row: LatentExecution,
    *,
    step: int,
    scope: LaneState,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
    config: WorkerConfig,
) -> tuple[TensorPublication, ...]:
    """Publish a committed-candidate trajectory for an exact staged consumer."""

    transports = publication_transports
    if not any(name != "local" for name in transports) or (
        config.rank != worker_info.output_rank(operation.entry)
    ):
        return ()
    pool = require_latent_pool(latent_pool)
    source = pool.reserve_publication(
        product,
        request_pool_idx=row.request_pool_idx,
        page_table=row.params.page_table,
        latent_units=row.params.latent_units,
    )
    from .transfer import publish_latent_source

    return (
        publish_latent_source(
            product,
            source,
            row,
            step=step,
            scope=scope,
            latent_pool=latent_pool,
            publication_transports=publication_transports,
        ),
    )


def initial_latent(
    operation: ScheduledRequest,
    height: int,
    width: int,
    target: torch.Tensor,
    *,
    model_runner: ModelRunner,
) -> None:
    """Create deterministic bounded latent noise or reuse the request’s staged image latent."""

    flow = model_runner.generation()
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


def branch_source(branch: Branch, *, model_runner: ModelRunner) -> BranchSource:
    """Resolve a guidance branch to conditioning, negative/start, or start-state input."""

    return model_runner.generation().branch_source(branch)


def flow_prefix(
    source: BranchSource,
    image_prompt: str,
    request: Request,
    *,
    model_runner: ModelRunner,
    tokenizer: PreTrainedTokenizerBase | None,
) -> tuple[tuple[int, ...], bool]:
    """Tokenize and embed the prompt source used to construct diffusion conditioning."""

    return model_runner.generation().prefix(
        source,
        image_prompt=image_prompt,
        negative_prompt=require_image(request).negative_prompt,
        negative_token_ids=request.negative_token_ids,
        tokenizer=tokenizer,
    )


def prefix_row(
    operation: ScheduledRequest,
    tokens: tuple[int, ...],
    entry: tuple[int, int, int, int],
    branch: Branch,
    scope: LaneState,
) -> ForwardRow:
    """Build the model-forward row that materializes one diffusion conditioning prefix."""

    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    positions = torch.arange(entry[2], entry[2] + len(tokens), dtype=torch.long)
    return ForwardRow(
        operation=operation,
        request=request,
        forward_mode=ForwardMode.PREFILL,
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
    operation: ScheduledRequest,
    conditioning_position: int,
    branch: Branch,
    entry: tuple[int, int, int, int],
    latent: torch.Tensor,
    timestep: torch.Tensor,
    height: int,
    width: int,
    scope: LaneState,
    *,
    model_runner: ModelRunner,
) -> ForwardRow:
    """Build one guided denoise row with latent, timestep, conditioning, and spatial positions."""

    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    flow = model_runner.generation()
    transform = (
        model_runner.image_processor().vit
        if flow.latent_layout is not LatentLayout.PATCH_TOKENS
        else None
    )
    latent_positions, attention_indexes, conditioning, query_tokens, text_local = denoise_geometry(
        flow,
        latent,
        height,
        width,
        temporal=_temporal_position(branch, conditioning_position, entry),
        patch_size=int(transform.patch_size) if isinstance(transform, PatchTransform) else None,
    )
    return ForwardRow(
        operation=operation,
        request=request,
        forward_mode=PipelineStage.DENOISING,
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


def denoise_geometry(
    flow, latent: torch.Tensor, height: int, width: int, *, temporal: int, patch_size: int | None
):
    """Construct identical numerical denoising geometry for startup and requests."""

    image_tokens = flow.image_tokens(height, width)
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
            height,
            width,
            int(flow.latent_patch_size),
            temporal,
        )
        conditioning = flow.conditioning(
            flow.materialization_latent(latent, height, width), height, width, patch_size=patch_size
        )
        query_tokens = image_tokens
        attention_indexes = latent_positions
        text_local = ()
    return latent_positions, attention_indexes, conditioning, query_tokens, text_local


def _temporal_position(
    branch: Branch,
    conditioning_position: int,
    entry: tuple[int, int, int, int],
) -> int:
    """Resolve a branch temporal coordinate from conditioning and latent params."""

    if branch is Branch.COND:
        return int(conditioning_position)
    return int(entry[2])


def image_token_count(
    latent: torch.Tensor, height: int, width: int, *, model_runner: ModelRunner
) -> int:
    """Validate latent geometry and return its model-visible patch-token count."""

    del latent
    return model_runner.generation().image_tokens(height, width)


def require_image(request: Request) -> ImageParams:
    """Return the request image input required by image-conditioned diffusion."""

    if request.image is None:
        raise invalid_descriptor("flow execution requires admitted image parameters")
    return request.image


def prediction(output: torch.Tensor) -> torch.Tensor:
    """Extract a denoise prediction tensor from the supported model output wrapper."""

    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("flow route did not return a flow prediction")
    return output


def physical_tokens(height: int, width: int, *, model_runner: ModelRunner) -> int:
    """Return the padded image-token capacity for the requested raster geometry."""

    return model_runner.generation().physical_tokens(height, width)


def _spatial_positions(
    height: int,
    width: int,
    patch: int,
    temporal: int,
) -> torch.Tensor:
    """Build flattened temporal-height-width coordinates for latent patches."""

    grid_height = height // patch
    grid_width = width // patch
    y = torch.arange(grid_height, dtype=torch.long).repeat_interleave(grid_width)
    x = torch.arange(grid_width, dtype=torch.long).repeat(grid_height)
    return torch.stack((torch.full_like(x, int(temporal)), y, x))


__all__ = ["consume_forward", "integrate", "pack_forward"]
