"""Flow prefix, denoise, Euler integration, and latent publication."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from uniserve.diffusion import Branch, Renorm
from uniserve.media import image as media_image
from uniserve.nn.rng import flow_noise_seed
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution import calls
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.diffusion_state import ImageState
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.model_executor.diffusion_inputs import (
    DiffusionRow,
    resolve_prefix,
)
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    DrawLayout,
    ForwardMode,
    ImageParams,
)
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.protocol.tensor import TensorRef
from uniserve_worker.protocol.transfer import KvTransfer
from uniserve_worker.sampling.metadata import TokenSelection

if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers import PreTrainedTokenizerBase

    from uniserve_worker.config.execution import WorkerConfig
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.protocol.worker_info import WorkerInfo
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.transport.interface import Transport


def _to_device(
    value: torch.Tensor, device: torch.device | None
) -> torch.Tensor:
    """Borrow a local tensor or copy it to the device on the caller's stream."""
    if device is None or value.device == device:
        return value
    # Host consumers need a completed D2H result; device consumers retain the
    # stream dependency and can overlap the copy with independent work.
    return value.to(device, non_blocking=device.type != "cpu")


def image_state(builder, size, image: ImageParams) -> ImageState:
    """Bind admitted sampling choices to the denoiser's mathematical recipes."""
    schedule = builder.denoiser.make_schedules(
        image.steps,
        shift=image.timestep_shift if image.timestep_shift > 0 else None,
        device="cpu",
    )["image"]
    guidance = builder.denoiser.make_guidance(
        text_scale=image.cfg_text_scale,
        image_scale=image.cfg_img_scale,
        interval=image.cfg_interval,
        renorm=Renorm(image.cfg_renorm_type),
        renorm_min=image.cfg_renorm_min,
    )
    return ImageState(size, schedule, guidance)


def require_inputs(runner):
    if runner.image_builder is None:
        raise invalid_descriptor(
            "image computation requires its denoiser input builder"
        )
    return runner.image_builder


def prepare_latent(
    call: Call,
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    worker_info: WorkerInfo,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
    config: WorkerConfig,
) -> PendingOutput:
    """Seed and publish the initial latent trajectory for one diffusion.

    request.
    """
    require_inputs(model_runner)
    request_id = call.request_key.request_id

    # Media preparation joins one visible conditioning publication to one new
    # latent product; accepting any other arity would make ownership ambiguous.
    conditioning = call.kv_input
    output = call.latent_output
    if conditioning is None or output is None:
        raise invalid_descriptor(
            "media preparation requires one exact conditioning input and "
            "latent output"
        )
    request = state.pending_output(request_id)
    cache = calls.cache_coordinates(request, tables=request_tables)
    publications = kv_cache
    if publications is None:
        raise invalid_descriptor(
            "media preparation requires KV publication storage"
        )
    publication = next(
        (value for value in state.kv_inputs if value.source == conditioning),
        None,
    )
    publications.validate_conditioning(
        call.request_key,
        conditioning,
        request_pool_idx=request.request.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=publication
        if isinstance(publication, KvTransfer)
        else None,
    )

    image = request.request.image
    if image is None:
        raise invalid_descriptor(
            "media preparation has no admitted image parameters"
        )
    if calls.require_progress(request).flow_step != 0:
        raise invalid_descriptor(
            "media preparation repeats an active latent trajectory"
        )

    rng = call.rng
    if rng is None or rng.draw_layout is not DrawLayout.FLOW_NOISE:
        raise invalid_descriptor(
            "media preparation requires semantic flow-noise RNG coordinates"
        )
    if int(rng.seed) != int(image.seed or 0):
        raise invalid_descriptor(
            "media preparation seed disagrees with admitted image seed"
        )
    if int(rng.semantic_index_base) < 1:
        raise invalid_descriptor(
            "flow-noise semantic image index must be positive"
        )
    if int(output.generation) < 1:
        raise invalid_descriptor(
            "media preparation latent has no logical generation"
        )

    # Noise is generated directly into request-owned staging, then installed in
    # the pool before its generation becomes visible to downstream calls.
    row = state.pending_output(call.request_key.request_id)
    params = row.latent.input_params
    staging = row.latent.staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory call has no staged latent inputs")

    pool = latent_pool
    staging.value.zero_()
    initial = staging.value[: int(params.latent_units)]
    trajectory = image_state(
        require_inputs(model_runner),
        media_image.Config(int(params.height), int(params.width)),
        image,
    )
    row.request.diffusion = trajectory

    initial_latent(
        call,
        int(params.height),
        int(params.width),
        initial,
        seed=image.seed or 0,
        model_runner=model_runner,
    )
    pool.initialize(
        row.request.request_pool_idx,
        staging,
        latent_units=int(params.latent_units),
    )

    # Publication is deferred with the batch commit so a failed
    # batch cannot expose a partially initialized trajectory.
    request.latent.update.params = params
    request.latent.update.expected_generation = 0
    request.latent.update.expected_step = 0
    request.latent.update.generation = int(output.generation)
    request.latent.update.step = 0

    products = publish_latent_transfer(
        call,
        output,
        row,
        step=0,
        worker_info=worker_info,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
        config=config,
        state=state,
    )

    request.status = CallStatus.OK
    request.progress = calls.execution_runtime(request, cache, flow_step=0)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.products = products
    return request


def initialize(
    call: Call,
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    latent_pool: LatentPool,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> ImageState:
    """Bind reusable request state and refill the call from its exact.

    latent version.
    """
    require_inputs(model_runner)
    request_id = call.request_key.request_id
    conditioning = call.kv_input
    latent_input = call.latent_input
    latent_output = call.latent_output
    if conditioning is None or latent_input is None or latent_output is None:
        raise invalid_descriptor(
            "flow call requires exact conditioning and one latent "
            "input/output generation"
        )
    request = state.pending_output(request_id)
    cache = calls.cache_coordinates(request, tables=request_tables)
    publications = kv_cache
    if publications is None:
        raise invalid_descriptor(
            "flow conditioning requires cache publication storage"
        )
    publication = next(
        (value for value in state.kv_inputs if value.source == conditioning),
        None,
    )
    publications.validate_conditioning(
        call.request_key,
        conditioning,
        request_pool_idx=request.request.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=publication
        if isinstance(publication, KvTransfer)
        else None,
    )

    image = request.request.image
    if image is None:
        raise invalid_descriptor("flow call has no admitted image parameters")
    if call.rng is not None:
        raise invalid_descriptor(
            "flow continuation must inherit transition RNG state"
        )
    if (
        int(latent_input.generation) < 1
        or int(latent_output.generation) < 1
        or latent_input == latent_output
    ):
        raise invalid_descriptor("flow latent generations are invalid")

    row = state.pending_output(call.request_key.request_id)
    params = row.latent.input_params
    staging = row.latent.staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory call has no staged latent inputs")

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

    size = media_image.Config(int(params.height), int(params.width))
    trajectory = row.request.diffusion
    if not isinstance(trajectory, ImageState) or trajectory.size != size:
        trajectory = image_state(require_inputs(model_runner), size, image)
        row.request.diffusion = trajectory

    # Prefix initialization follows this submission's descriptors, including
    # retries after a failed call; retained metadata is not accepted state.
    trajectory.cache = cache
    trajectory.entries.clear()
    return trajectory


def prepare_step(
    call: Call,
    trajectory: ImageState,
    step_index: int,
    *,
    state: BatchState,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
    latent_pool: LatentPool,
    tokenizer: PreTrainedTokenizerBase | None,
) -> tuple[
    tuple[Branch, ...],
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[Branch, TokenRow], ...],
]:
    """Gather current latent pages and construct one guided diffusion-step.

    batch.
    """
    request = state.pending_output(call.request_key.request_id)
    image = request.request.image
    if image is None:
        raise invalid_descriptor("flow step requires admitted image parameters")

    builder = require_inputs(model_runner)
    row = state.pending_output(call.request_key.request_id)
    params = row.latent.input_params
    staging = row.latent.staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory call has no staged latent inputs")

    schedule = trajectory.schedule
    if not 0 <= step_index < schedule.num_steps:
        raise IndexError(step_index)
    times = schedule.timesteps
    t, t_next = latent_pool.stage_timestep(
        row.request.request_pool_idx,
        float(times[step_index]),
        float(times[step_index + 1]),
    )

    branches = trajectory.guidance.branches(schedule, step_index)
    prefix_rows = []
    prefix_branches = []
    entries = trajectory.entries
    descriptors = state.forward_indices.get(calls.call_identity(call), ())
    if len(descriptors) < len(branches):
        raise invalid_descriptor(
            "media denoise has incomplete forward-row metadata"
        )

    # The last descriptors of this call belong to the denoise branches;
    # any earlier ones cover prefix forwards emitted on a first visit.
    denoise_descriptors = descriptors[-len(branches) :]
    for branch_index, branch in enumerate(branches):
        if branch in entries:
            continue

        source = builder.branch_source(branch)
        if source not in trajectory.prefixes:
            trajectory.prefixes[source] = resolve_prefix(
                model_runner.flow_prompt,
                source,
                image_prompt=image.image_prompts[0]
                if image.image_prompts
                else "",
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
                raise invalid_descriptor(
                    "flow prefixes require request page tables"
                )
            capacity = page_tables.allocated_length(slot)
            page_tables.pages(slot, 0)

            # A sibling forward in this same submission may already write the
            # prefix; otherwise the branch row carries its own prefix extent.
            has_prefix_forward = any(
                state.batch.request_pool_indices[candidate] == slot
                and (
                    state.batch.seq_lens[candidate]
                    - state.batch.query_lens[candidate]
                )
                == 0
                and state.batch.query_lens[candidate] == len(prefix)
                for candidate in descriptors[: -len(branches)]
            )

            # entry = (pool slot, KV group, materialized prefix length, token
            # capacity).
            entry = (
                slot,
                0,
                0
                if has_prefix_forward
                else (
                    state.batch.seq_lens[descriptor]
                    - state.batch.query_lens[descriptor]
                ),
                capacity,
            )

        prefix_length = (
            trajectory.cache[2] if copy_conditioning else len(prefix)
        )
        if prefix_length > entry[3]:
            raise invalid_descriptor("flow prefix exceeds scheduler params")
        if entry[2] not in {0, prefix_length}:
            raise invalid_descriptor(
                "flow branch prefix disagrees with its initialized physical "
                "state"
            )

        initialize_prefix = entry[2] == 0 and prefix_length > 0
        entries[branch] = entry
        if initialize_prefix and prefix:
            prefix_rows.append(prefix_row(prefix, entry))
            prefix_branches.append(branch)

    return (
        branches,
        t,
        t_next,
        tuple(zip(prefix_branches, prefix_rows, strict=True)),
    )


def finish(
    call: Call,
    trajectory: ImageState,
    *,
    state: BatchState,
    worker_info: WorkerInfo,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    config: WorkerConfig,
) -> PendingOutput:
    """Integrate predicted velocity, write the next latent bank.

    and prepare publication.
    """
    request = state.pending_output(call.request_key.request_id)
    row = state.pending_output(call.request_key.request_id)
    params = row.latent.input_params
    staging = row.latent.staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory call has no staged latent inputs")

    latent_input, latent_output = (
        call.latent_input,
        call.latent_output,
    )
    if (
        latent_input is None
        or latent_output is None
        or request.request.image is None
    ):
        raise invalid_descriptor("flow completion lost its trajectory state")

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
    request.latent.update.params = params
    request.latent.update.expected_generation = int(latent_input.generation)
    request.latent.update.expected_step = start_step
    request.latent.update.generation = int(latent_output.generation)
    request.latent.update.step = final_step

    products = publish_latent_transfer(
        call,
        latent_output,
        row,
        step=final_step,
        worker_info=worker_info,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
        config=config,
        state=state,
    )

    request.status = CallStatus.OK
    request.progress = calls.execution_runtime(
        request,
        trajectory.cache,
        flow_step=final_step,
    )
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.products = products

    # Branch prefixes live in pool slots separate from the request's own KV;
    # they are retired once the trajectory has written its final step.
    main_slot = int(request.request.request_pool_idx)
    alternative_slots = {
        int(entry[0])
        for entry in trajectory.entries.values()
        if int(entry[0]) != main_slot
    }
    if alternative_slots and final_step >= int(request.request.image.steps):
        page_tables = request_tables
        if page_tables is None:
            raise RuntimeError(
                "flow prefix retirement lost its request page tables"
            )
        page_tables.release_prefixes(call.request_key, tuple(alternative_slots))
    return request


def publish_latent_transfer(
    call: Call,
    product: TensorRef,
    row: PendingOutput,
    *,
    state: BatchState,
    step: int,
    worker_info: WorkerInfo,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
    config: WorkerConfig,
) -> tuple[TensorPublication, ...]:
    """Publish a committed-candidate trajectory for an exact staged consumer."""
    params = row.latent.input_params
    if params is None:
        raise invalid_descriptor("latent publication has no staged parameters")

    transports = publication_transports
    if not any(name != "local" for name in transports) or (
        config.rank != worker_info.output_rank(call.component)
    ):
        return ()

    pool = latent_pool
    source = pool.reserve_publication(
        product,
        request_pool_idx=row.request.request_pool_idx,
        page_table=params.page_table,
        latent_units=params.latent_units,
    )
    from uniserve_worker.execution.transfer import publish_latent_source

    return (
        publish_latent_source(
            product,
            source,
            row,
            step=step,
            latent_pool=latent_pool,
            publication_transports=publication_transports,
            state=state,
        ),
    )


def initial_latent(
    call: Call,
    height: int,
    width: int,
    target: torch.Tensor,
    *,
    seed: int,
    model_runner: ModelExecutor,
) -> None:
    """Create deterministic bounded latent noise or reuse the request’s staged.

    image latent.
    """
    rng = call.rng
    assert rng is not None and rng.draw_layout is DrawLayout.FLOW_NOISE
    require_inputs(model_runner).initialize(
        media_image.Config(height, width),
        seed=flow_noise_seed(seed, int(rng.semantic_index_base)),
        out=target,
    )


def prefix_row(
    tokens: tuple[int, ...],
    entry: tuple[int, int, int, int],
) -> TokenRow:
    """Build the model-forward row that materializes one diffusion conditioning.

    prefix.
    """
    positions = torch.arange(entry[2], entry[2] + len(tokens), dtype=torch.long)
    return TokenRow(
        forward_mode=ForwardMode.PREFILL,
        token_ids=torch.tensor(tokens, dtype=torch.long),
        positions=positions,
        selection=TokenSelection.HIDDEN,
        request_pool_idx=entry[0],
        seq_len=entry[2],
        group_id=entry[1],
        write_kv=True,
        causal=True,
    )


def flow_rows(
    builder,
    trajectory,
    current,
    branches,
    timestep,
    *,
    conditioning_position,
    device,
):
    """Borrow one learned sample copy for every active guidance branch."""
    from uniserve_worker.protocol.call import MediaCall

    current = _to_device(current, device)
    timestep = _to_device(timestep, device)

    size, rows = trajectory.size, []
    for branch in branches:
        entry = trajectory.entries[branch]
        temporal = (
            conditioning_position if branch is Branch.CONDITIONED else entry[2]
        )
        if temporal not in trajectory.positions:
            trajectory.positions[temporal] = builder.positions(
                size, temporal, device=device
            )
        rows.append(
            DiffusionRow(
                forward_mode=MediaCall.DENOISING,
                positions=trajectory.positions[temporal],
                timestep=timestep.reshape(1),
                latent=current,
                image_tokens=builder.sequence_length(size),
                image_height=size.height,
                image_width=size.width,
                request_pool_idx=entry[0],
                seq_len=entry[2],
                group_id=entry[1],
                write_kv=False,
                causal=False,
            )
        )
    return tuple(rows)


def integrate(
    builder, trajectory, current, outputs, index, timestep, next_timestep
):
    """Apply guidance and the model's public solver to the call's.

    sample.
    """
    schedule, guidance = trajectory.schedule, trajectory.guidance
    branches = guidance.branches(schedule, index)
    velocity = guidance.combine(
        {
            branch: _to_device(output, current.device)
            for branch, output in zip(branches, outputs, strict=True)
        },
        schedule,
        index,
    )
    builder.denoiser.solver.step_(
        velocity,
        current,
        timestep,
        next_timestep,
        sigma=schedule.sigmas[index],
        next_sigma=schedule.sigmas[index + 1],
    )
