"""Flow prefix, denoise, Euler integration, and latent publication."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.model.batch import DiffusionBatch
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.media import ImageSize
from uniserve.model.tensors import TokenSelection
from uniserve.nn.diffusion.cfg import Branch, CfgPlan, RenormKind
from uniserve.nn.diffusion.config import DiffusionConfig
from uniserve.nn.rng import flow_noise_seed, normal_noise
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.protocol.batch import (
    DrawLayout,
    FinishFlags,
    ForwardMode,
    ImageParams,
    KvTransfer,
    OpStatus,
    ScheduledRequest,
    TensorPublication,
    TensorRef,
)
from uniserve_worker.runtime.request import RequestState

from . import operations
from .batch_state import BatchState
from .diffusion_state import DiffusionState, resolve_prefix
from .output import PendingOutput
from .rows import ForwardRow

if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers import PreTrainedTokenizerBase

    from ..bootstrap.worker_info import WorkerInfo
    from ..config import WorkerConfig
    from ..runtime.block_tables import BlockTables
    from ..runtime.cache_manager import CacheManager
    from ..runtime.latent_pool import LatentPool
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner


def diffusion_config(image: ImageParams) -> DiffusionConfig:
    """Normalize admitted wire options at the numerical execution boundary."""

    return DiffusionConfig(
        steps=image.steps,
        timestep_shift=image.timestep_shift if image.timestep_shift > 0 else None,
        cfg_text_scale=image.cfg_text_scale,
        cfg_img_scale=image.cfg_img_scale,
        cfg_interval=image.cfg_interval,
        cfg_renorm=RenormKind(image.cfg_renorm_type),
        cfg_renorm_min=image.cfg_renorm_min,
        seed=image.seed or 0,
    )


def prepare_latent(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    worker_info: WorkerInfo,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> PendingOutput:
    """Seed and publish the initial latent trajectory for one diffusion request."""

    model_runner.generation()
    request_id = operation.request_key.request_id

    # Media preparation joins one visible conditioning publication to one new
    # latent product; accepting any other arity would make ownership ambiguous.
    conditioning = operation.kv_input
    output = operation.latent_output
    if conditioning is None or output is None:
        raise invalid_descriptor(
            "media preparation requires one exact conditioning input and latent output"
        )
    request = state.pending_output(completion_group, request_id)
    cache = operations.cache_coordinates(request, tables=request_tables)
    publications = kv_cache
    if publications is None:
        raise invalid_descriptor("media preparation requires KV publication storage")
    publication = next((value for value in state.kv_inputs if value.source == conditioning), None)
    publications.validate_conditioning(
        operation.request_key,
        conditioning,
        request_pool_idx=request.request.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=publication if isinstance(publication, KvTransfer) else None,
    )
    image = request.request.image
    if image is None:
        raise invalid_descriptor("media preparation has no admitted image parameters")
    if (
        operations.require_progress(request).latent_product is not None
        or operations.require_progress(request).flow_step != 0
    ):
        raise invalid_descriptor("media preparation repeats an active latent trajectory")
    rng = operation.rng
    if rng is None or rng.draw_layout is not DrawLayout.FLOW_NOISE:
        raise invalid_descriptor("media preparation requires semantic flow-noise RNG coordinates")
    if int(rng.seed) != int(image.seed or 0):
        raise invalid_descriptor("media preparation seed disagrees with admitted image seed")
    if int(rng.semantic_index_base) < 1:
        raise invalid_descriptor("flow-noise semantic image index must be positive")
    if int(output.generation) < 1:
        raise invalid_descriptor("media preparation latent has no logical generation")

    # Noise is generated directly into request-owned staging, then installed in
    # the pool before its generation becomes visible to downstream operations.
    row = state.pending_output(completion_group, operation.request_key.request_id)
    params = row.input_latent_params
    staging = row.latent_staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory operation has no staged latent inputs")
    pool = latent_pool
    staging.value.zero_()
    initial = staging.value[: int(params.latent_units)]
    diffusion = model_runner.diffusion
    if diffusion is None:
        raise invalid_descriptor("image initialization requires a diffusion execution owner")
    numerical_config = diffusion_config(image)
    trajectory = diffusion.initialize(
        ImageSize(int(params.height), int(params.width)), numerical_config
    )
    row.request.diffusion = trajectory
    initial_latent(
        operation,
        int(params.height),
        int(params.width),
        initial,
        seed=numerical_config.seed,
        model_runner=model_runner,
    )
    pool.initialize(
        row.request.request_pool_idx,
        staging,
        latent_units=int(params.latent_units),
    )

    # Publication is deferred with the completion group commit so a failed completion group cannot
    # expose a partially initialized trajectory.
    request.latent_params = params
    request.latent_expected_generation = 0
    request.latent_expected_step = 0
    request.latent_generation = int(output.generation)
    request.latent_step = 0
    request.projected_progress = replace(
        operations.require_progress(request), latent_product=output
    )
    products = publish_latent_transfer(
        operation,
        output,
        row,
        step=0,
        completion_group=completion_group,
        worker_info=worker_info,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
        config=config,
        state=state,
    )
    request.status = OpStatus.OK
    request.projected_progress = operations.execution_runtime(request, cache, flow_step=0)
    request.finish_flags = FinishFlags()
    request.product_generations = operations.output_generations(operation)
    request.products = products
    return request


def initialize(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    latent_pool: LatentPool,
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
) -> DiffusionState:
    """Bind reusable request geometry and refill the current operation from its exact latent version."""

    model_runner.generation()
    request_id = operation.request_key.request_id
    conditioning = operation.kv_input
    latent_input = operation.latent_input
    latent_output = operation.latent_output
    if conditioning is None or latent_input is None or latent_output is None:
        raise invalid_descriptor(
            "flow operation requires exact conditioning and one latent input/output generation"
        )
    request = state.pending_output(completion_group, request_id)
    cache = operations.cache_coordinates(request, tables=request_tables)
    publications = kv_cache
    if publications is None:
        raise invalid_descriptor("flow conditioning requires cache publication storage")
    publication = next((value for value in state.kv_inputs if value.source == conditioning), None)
    publications.validate_conditioning(
        operation.request_key,
        conditioning,
        request_pool_idx=request.request.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=publication if isinstance(publication, KvTransfer) else None,
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
    if operations.require_progress(request).latent_product != latent_input:
        raise invalid_descriptor("flow operation does not name the current latent generation")
    row = state.pending_output(completion_group, operation.request_key.request_id)
    params = row.input_latent_params
    staging = row.latent_staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory operation has no staged latent inputs")
    pool = latent_pool
    start_step = int(params.start_step)
    pool.gather_current(
        row.request.request_pool_idx,
        staging,
        step=start_step,
        generation=int(latent_input.generation),
        latent_units=int(params.latent_units),
        height=int(params.height),
        width=int(params.width),
    )
    diffusion = model_runner.diffusion
    if diffusion is None:
        raise invalid_descriptor("flow operation has no diffusion execution owner")
    size = ImageSize(int(params.height), int(params.width))
    trajectory = row.request.diffusion
    if trajectory is None or trajectory.size != size:
        trajectory = diffusion.initialize(size, diffusion_config(image))
        row.request.diffusion = trajectory
    # Prefix initialization follows this submission's descriptors, including
    # retries after a failed operation; retained metadata is not accepted state.
    trajectory.cache = cache
    trajectory.entries.clear()
    return trajectory


def prepare_step(
    operation: ScheduledRequest,
    completion_group: int,
    trajectory: DiffusionState,
    step_index: int,
    *,
    state: BatchState,
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
    latent_pool: LatentPool,
    tokenizer: PreTrainedTokenizerBase | None,
) -> tuple[CfgPlan, torch.Tensor, torch.Tensor, tuple[tuple[Branch, ForwardRow], ...]]:
    """Gather current latent pages and construct one guided diffusion-step batch."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    image = request.request.image
    if image is None:
        raise invalid_descriptor("flow step requires admitted image parameters")
    generation = model_runner.generation()
    row = state.pending_output(completion_group, operation.request_key.request_id)
    params = row.input_latent_params
    staging = row.latent_staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory operation has no staged latent inputs")
    schedule, config = trajectory.schedule, trajectory.config
    if schedule is None or config is None:
        raise invalid_descriptor("image trajectory requires numerical configuration and schedule")
    if not 0 <= step_index < config.steps:
        raise IndexError(step_index)
    times = schedule.timesteps[0]
    t, t_next = latent_pool.stage_timestep(
        row.request.request_pool_idx, float(times[step_index]), float(times[step_index + 1])
    )
    guide = generation.guidance(config, step_index)
    prefix_rows = []
    prefix_branches = []
    entries = trajectory.entries
    descriptors = state.group_forward_indices[completion_group].get(
        operations.operation_identity(operation), ()
    )
    if len(descriptors) < len(guide.branches):
        raise invalid_descriptor("media denoise has incomplete forward-row metadata")
    denoise_descriptors = descriptors[-len(guide.branches) :]
    for branch_index, branch in enumerate(guide.branches):
        if branch in entries:
            continue
        source = generation.branch_source(branch)
        if source not in trajectory.prefixes:
            trajectory.prefixes[source] = resolve_prefix(
                model_runner.flow_prompt,
                source,
                image_prompt=image.image_prompts[0] if image.image_prompts else "",
                negative_prompt=image.negative_prompt,
                negative_token_ids=request.request.negative_token_ids,
                tokenizer=tokenizer,
            )
        prefix, copy_conditioning = trajectory.prefixes[source]
        descriptor = denoise_descriptors[branch_index]
        if copy_conditioning:
            entry = trajectory.cache
        else:
            slot = state.batch.request_pool_indices[descriptor]
            page_tables = request_tables
            if page_tables is None:
                raise invalid_descriptor("flow prefixes require request page tables")
            capacity = page_tables.allocated_length(slot)
            page_tables.pages(slot, 0)
            has_prefix_forward = any(
                state.batch.request_pool_indices[candidate] == slot
                and (state.batch.seq_lens[candidate] - state.batch.query_lens[candidate]) == 0
                and state.batch.query_lens[candidate] == len(prefix)
                for candidate in descriptors[: -len(guide.branches)]
            )
            entry = (
                slot,
                0,
                0
                if has_prefix_forward
                else (state.batch.seq_lens[descriptor] - state.batch.query_lens[descriptor]),
                capacity,
            )
        prefix_length = trajectory.cache[2] if copy_conditioning else len(prefix)
        if prefix_length > entry[3]:
            raise invalid_descriptor("flow prefix exceeds scheduler params")
        if entry[2] not in {0, prefix_length}:
            raise invalid_descriptor(
                "flow branch prefix disagrees with its initialized physical state"
            )
        initialize_prefix = entry[2] == 0 and prefix_length > 0
        entries[branch] = entry
        if initialize_prefix and prefix:
            prefix_rows.append(prefix_row(prefix, entry))
            prefix_branches.append(branch)
    return guide, t, t_next, tuple(zip(prefix_branches, prefix_rows, strict=True))


def finish(
    operation: ScheduledRequest,
    completion_group: int,
    trajectory: DiffusionState,
    *,
    state: BatchState,
    worker_info: WorkerInfo,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    config: WorkerConfig,
) -> PendingOutput:
    """Integrate predicted velocity, write the next latent bank, and prepare publication."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    row = state.pending_output(completion_group, operation.request_key.request_id)
    params = row.input_latent_params
    staging = row.latent_staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory operation has no staged latent inputs")
    latent_input, latent_output = operation.latent_input, operation.latent_output
    if latent_input is None or latent_output is None or request.request.image is None:
        raise invalid_descriptor("flow completion lost its trajectory contract")
    start_step = int(params.start_step)
    final_step = start_step + int(params.step_count)
    latent_pool.write_inactive(
        row.request.request_pool_idx,
        staging,
        expected_step=start_step,
        expected_generation=int(latent_input.generation),
        latent_units=int(params.latent_units),
        height=int(params.height),
        width=int(params.width),
    )
    request.latent_params = params
    request.latent_expected_generation = int(latent_input.generation)
    request.latent_expected_step = start_step
    request.latent_generation = int(latent_output.generation)
    request.latent_step = final_step
    request.projected_progress = replace(
        operations.require_progress(request), latent_product=latent_output
    )
    products = publish_latent_transfer(
        operation,
        latent_output,
        row,
        step=final_step,
        completion_group=completion_group,
        worker_info=worker_info,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
        config=config,
        state=state,
    )
    request.status = OpStatus.OK
    request.projected_progress = operations.execution_runtime(
        request,
        trajectory.cache,
        flow_step=final_step,
    )
    request.finish_flags = FinishFlags()
    request.product_generations = operations.output_generations(operation)
    request.products = products
    main_slot = int(request.request.request_pool_idx)
    alternative_slots = {
        int(entry[0]) for entry in trajectory.entries.values() if int(entry[0]) != main_slot
    }
    if alternative_slots and final_step >= int(request.request.image.steps):
        page_tables = request_tables
        if page_tables is None:
            raise RuntimeError("flow prefix retirement lost its request page tables")
        page_tables.release_prefixes(operation.request_key, tuple(alternative_slots))
    return request


def publish_latent_transfer(
    operation: ScheduledRequest,
    product: TensorRef,
    row: PendingOutput,
    *,
    state: BatchState,
    step: int,
    completion_group: int,
    worker_info: WorkerInfo,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
    config: WorkerConfig,
) -> tuple[TensorPublication, ...]:
    """Publish a committed-candidate trajectory for an exact staged consumer."""

    params = row.input_latent_params
    if params is None:
        raise invalid_descriptor("latent publication has no staged parameters")
    transports = publication_transports
    if not any(name != "local" for name in transports) or (
        config.rank != worker_info.output_rank(operation.entry)
    ):
        return ()
    pool = latent_pool
    source = pool.reserve_publication(
        product,
        request_pool_idx=row.request.request_pool_idx,
        page_table=params.page_table,
        latent_units=params.latent_units,
    )
    from .transfer import publish_latent_source

    return (
        publish_latent_source(
            product,
            source,
            row,
            step=step,
            completion_group=completion_group,
            latent_pool=latent_pool,
            publication_transports=publication_transports,
            state=state,
        ),
    )


def initial_latent(
    operation: ScheduledRequest,
    height: int,
    width: int,
    target: torch.Tensor,
    *,
    seed: int,
    model_runner: ModelRunner,
) -> None:
    """Create deterministic bounded latent noise or reuse the request’s staged image latent."""

    model = model_runner.model
    if not isinstance(model, DiffusionMixin):
        raise invalid_descriptor("latent preparation requires the diffusion capability")
    rng = operation.rng
    assert rng is not None and rng.draw_layout is DrawLayout.FLOW_NOISE
    shape = ImageSize(height, width)
    raw = target.reshape(model.noise_shape("image", shape)).unsqueeze(0)
    noise = {"image": raw}
    normal_noise(
        (flow_noise_seed(seed, int(rng.semantic_index_base)),),
        tuple(noise.values()),
    )
    model.prepare_latents(
        DiffusionBatch(latents={"image": (target,)}, sizes=(shape,)),
        noise=noise,
        state={"image": target.unsqueeze(0)},
        constants={},
        scratch={},
    )


def prefix_row(
    tokens: tuple[int, ...],
    entry: tuple[int, int, int, int],
) -> ForwardRow:
    """Build the model-forward row that materializes one diffusion conditioning prefix."""

    positions = torch.arange(entry[2], entry[2] + len(tokens), dtype=torch.long)
    return ForwardRow(
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


def require_image(request: RequestState) -> ImageParams:
    """Return the request image input required by image-conditioned diffusion."""

    if request.image is None:
        raise invalid_descriptor("flow execution requires admitted image parameters")
    return request.image
