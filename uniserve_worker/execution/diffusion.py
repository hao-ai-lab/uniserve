"""Image diffusion calls conditioned on a request's KV prefixes.

These functions serve a worker whose ``ModelExecutor`` has an
``image_builder``. ``uniserve_worker.execution.schedule`` sends latent
preparation to ``prepare_latent``, and ``uniserve_worker.execution.forward``
drives each denoising call through ``initialize``, then ``prepare_step`` and
``flow_rows`` once per solver step, then ``finish`` after the last step of
the call's declared interval. The solver update between steps runs in
``forward.integrate_predictions``. A standalone video denoiser's calls go to
``uniserve_worker.execution.media`` instead.

The solver sample lives in the worker's ``LatentPool``. Preparation writes
the seeded noise to bank one of the request's pages; each denoising call
gathers the committed bank into its buffer and ``finish`` scatters the
successor to the inactive bank. Neither becomes visible until the batch
commit applies the ``LatentUpdate`` left on the call's ``PendingOutput``.
Each guidance branch attends to a KV prefix tracked in the request's
``KVConditioning``; a prefix that is not yet materialized gets a prefill row
from ``prefix_row``, forwarded before the step's denoising rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from uniserve.diffusion import Branch, Renorm
from uniserve.media import image as media_image
from uniserve.nn.rng import flow_noise_seed
from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution.diffusion_state import (
    DiffusionState,
    KVConditioning,
)
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.model_executor.diffusion_inputs import (
    DiffusionRow,
    resolve_prefix,
)
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.call import (
    Call,
    DrawLayout,
    ForwardMode,
    ImageParams,
)
from uniserve_worker.sampling.metadata import TokenSelection

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool


def _to_device(
    value: torch.Tensor, device: torch.device | None
) -> torch.Tensor:
    """Borrow a local tensor or copy it to the device on the caller's stream."""
    if device is None or value.device == device:
        return value
    # Host consumers need a completed D2H result; device consumers retain the
    # stream dependency and can overlap the copy with independent work.
    return value.to(device, non_blocking=device.type != "cpu")


def image_state(builder, size, image: ImageParams) -> DiffusionState:
    """Open an admitted image's diffusion state from its sampling choices.

    The schedules are built on the host; ``prepare_step`` reads each step's
    time as a Python float. A non-positive ``timestep_shift`` is passed as
    ``None``, leaving the shift to the denoiser's ``make_schedules``.
    """
    denoiser = builder.denoiser
    return DiffusionState.open(
        denoiser,
        size,
        steps=image.steps,
        shift=image.timestep_shift if image.timestep_shift > 0 else None,
        device="cpu",
        guidance=denoiser.make_guidance(
            text_scale=image.cfg_text_scale,
            image_scale=image.cfg_img_scale,
            interval=image.cfg_interval,
            renorm=Renorm(image.cfg_renorm_type),
            renorm_min=image.cfg_renorm_min,
        ),
    )


def kv_conditioning(trajectory: DiffusionState) -> KVConditioning:
    """Return the KV conditioning of an image request's diffusion state."""
    if trajectory.kv is None:
        raise invalid_descriptor(
            "image diffusion requires KV-conditioned request state"
        )
    return trajectory.kv


def require_inputs(runner):
    """Return the executor's ``ImageBuilder``.

    Raises ``invalid_descriptor`` when the model has no ``ImageDenoiser``
    capability, so the executor has no builder.
    """
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
    latent_pool: LatentPool,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Seed a diffusion request's trajectory in its latent buffer.

    Validates the call's conditioning export, flow-noise RNG coordinates
    and output generation, opens the request's ``DiffusionState``, draws the
    seeded noise into the call's buffer and writes it to bank one of the
    request's ``LatentPool`` pages. The ``LatentUpdate`` that publishes it as
    ``output.generation`` at step zero is applied by the batch commit.

    Returns:
        The call's ``PendingOutput`` with its completed numerical state.
        The native executor supplies its latent export before batch commit.

    Raises:
        WorkerError: For example when the worker has no image builder or KV
            storage, the conditioning export, image parameters, RNG
            coordinates, output generation or latent inputs do not match the
            call, or the request's trajectory has already started.
    """
    require_inputs(model_runner)
    request_id = call.request_key.request_id

    # Media preparation joins one visible conditioning export to one new
    # latent product; accepting any other arity would make ownership ambiguous.
    conditioning = call.kv_input
    output = call.latent_output
    if conditioning is None or output is None:
        raise invalid_descriptor(
            "media preparation requires one exact conditioning input and "
            "latent output"
        )
    request = state.pending_output(request_id)
    cache = request.cache_coordinates(request_tables)
    exports = kv_cache
    if exports is None:
        raise invalid_descriptor("media preparation requires KV export storage")
    exports.validate_conditioning(
        call.request_key,
        conditioning,
        request_pool_idx=request.request.request_pool_idx,
        visible_length=cache[1],
    )

    image = request.request.image
    if image is None:
        raise invalid_descriptor(
            "media preparation has no admitted image parameters"
        )
    if request.progress.flow_step != 0:
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

    # Noise is generated directly into the request-owned buffer, then installed
    # in the pool before its generation becomes visible to downstream calls.
    params = request.latent_params
    buffer = request.latent_buffer
    if params is None or buffer is None:
        raise invalid_descriptor("trajectory call has no bound latent inputs")

    pool = latent_pool
    # ``LatentPool.initialize`` writes the whole buffer, so the page padding
    # past ``latent_units`` is zeroed rather than left stale.
    buffer.value.zero_()
    initial = buffer.value[: int(params.latent_units)]
    trajectory = image_state(
        require_inputs(model_runner),
        media_image.Config(int(params.height), int(params.width)),
        image,
    )
    request.request.diffusion = trajectory

    initial_latent(
        call,
        int(params.height),
        int(params.width),
        initial,
        seed=image.seed or 0,
        model_runner=model_runner,
    )
    pool.initialize(
        request.request.request_pool_idx,
        buffer,
        latent_units=int(params.latent_units),
    )

    # Export is deferred with the batch commit so a failed
    # batch cannot expose a partially initialized trajectory.
    state.complete_latent(call.request_key.request_id)

    request.set_cache_length(cache[1])
    return request


def initialize(
    call: Call,
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    latent_pool: LatentPool,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> DiffusionState:
    """Open a denoising call's trajectory from its exact input generation.

    Validates the call's conditioning export and latent generations,
    gathers the committed bank at ``params.start_step`` into the call's
    buffer, and returns the request's ``DiffusionState``, reopening it when
    it is absent or has a different size. ``KVConditioning.cache`` is set
    from this call's descriptors and ``entries`` is cleared for
    ``prepare_step`` to fill again.
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
    cache = request.cache_coordinates(request_tables)
    exports = kv_cache
    if exports is None:
        raise invalid_descriptor(
            "flow conditioning requires cache export storage"
        )
    exports.validate_conditioning(
        call.request_key,
        conditioning,
        request_pool_idx=request.request.request_pool_idx,
        visible_length=cache[1],
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
    params = row.latent_params
    buffer = row.latent_buffer
    if params is None or buffer is None:
        raise invalid_descriptor("trajectory call has no bound latent inputs")

    pool = latent_pool
    start_step = int(params.start_step)
    pool.gather_current(
        row.request.request_pool_idx,
        buffer,
        step=start_step,
        generation=int(latent_input.generation),
        latent_units=int(params.latent_units),
        height=int(params.height),
        width=int(params.width),
    )

    size = media_image.Config(int(params.height), int(params.width))
    trajectory = row.request.diffusion
    if not isinstance(trajectory, DiffusionState) or trajectory.size != size:
        trajectory = image_state(require_inputs(model_runner), size, image)
        row.request.diffusion = trajectory

    # Prefix initialization follows this submission's descriptors, including
    # retries after a failed call; retained metadata is not accepted state.
    kv = kv_conditioning(trajectory)
    kv.cache = cache
    kv.entries.clear()
    return trajectory


def prepare_step(
    call: Call,
    trajectory: DiffusionState,
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
    tuple[tuple[Branch, TokenRow], ...],
]:
    """Fill one solver step's time and resolve each branch's KV prefix.

    Prefix tokens are resolved once per branch source and retained in
    ``KVConditioning.prefixes``; each branch's ``(slot, group, materialized
    prefix length, token capacity)`` coordinate is recorded in
    ``KVConditioning.entries`` on its first step of this call.

    Returns:
        The guidance branches active at ``step_index``, the one-element
        timestep view in the slot's ``LatentPool`` storage, and
        ``(branch, row)`` pairs for prefixes that a prefill forward must
        materialize before the step's denoising rows.

    Raises:
        IndexError: ``step_index`` is outside the image schedule.
        WorkerError: For example when image parameters, input latents,
            guidance or forward-row metadata are missing, or a prefix exceeds
            its slot's capacity or disagrees with its materialized extent.
    """
    request = state.pending_output(call.request_key.request_id)
    image = request.request.image
    if image is None:
        raise invalid_descriptor("flow step requires admitted image parameters")

    builder = require_inputs(model_runner)
    row = state.pending_output(call.request_key.request_id)
    params = row.latent_params
    buffer = row.latent_buffer
    if params is None or buffer is None:
        raise invalid_descriptor("trajectory call has no bound latent inputs")

    schedule = trajectory.schedules["image"]
    if not 0 <= step_index < schedule.num_steps:
        raise IndexError(step_index)
    times = schedule.timesteps
    t = latent_pool.fill_timestep(
        row.request.request_pool_idx, float(times[step_index])
    )

    guidance = trajectory.guidance
    if guidance is None:
        raise invalid_descriptor("image diffusion requires guidance")
    branches = guidance.branches(schedule, step_index)
    prefix_rows = []
    prefix_branches = []
    kv = kv_conditioning(trajectory)
    entries = kv.entries
    descriptors = state.forward_rows(call.request_key.request_id)
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
        if source not in kv.prefixes:
            kv.prefixes[source] = resolve_prefix(
                model_runner.flow_prompt,
                source,
                image_prompt=image.image_prompts[0]
                if image.image_prompts
                else "",
                negative_prompt=image.negative_prompt,
                negative_token_ids=request.request.negative_token_ids,
                tokenizer=tokenizer,
            )
        prefix, copy_conditioning = kv.prefixes[source]
        descriptor = denoise_descriptors[branch_index]

        # ``resolve_prefix`` sets ``copy_conditioning`` for the conditioning
        # source when the image prompt is blank; the branch then reads the
        # request's own conditioning KV in place.
        if copy_conditioning:
            entry = kv.cache
        else:
            slot = state.batch.request_pool_indices[descriptor]
            page_tables = request_tables
            if page_tables is None:
                raise invalid_descriptor(
                    "flow prefixes require request page tables"
                )
            capacity = page_tables.allocated_length(slot)
            # Called only for its check, which raises when the slot lacks the
            # block table of some cache group.
            for group in range(len(page_tables.groups)):
                page_tables.table(slot, group)

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

            # entry = (pool slot, materialized prefix length, token
            # capacity).
            entry = (
                slot,
                0
                if has_prefix_forward
                else (
                    state.batch.seq_lens[descriptor]
                    - state.batch.query_lens[descriptor]
                ),
                capacity,
            )

        prefix_length = kv.cache[1] if copy_conditioning else len(prefix)
        if prefix_length > entry[2]:
            raise invalid_descriptor("flow prefix exceeds scheduler params")
        if entry[1] not in {0, prefix_length}:
            raise invalid_descriptor(
                "flow branch prefix disagrees with its initialized physical "
                "state"
            )

        # ``forward.prepare_diffusion_step`` forwards the prefill rows and
        # advances each entry's materialized length by the rows it wrote.
        initialize_prefix = entry[1] == 0 and prefix_length > 0
        entries[branch] = entry
        if initialize_prefix and prefix:
            prefix_rows.append(prefix_row(prefix, entry))
            prefix_branches.append(branch)

    return (
        branches,
        t,
        tuple(zip(prefix_branches, prefix_rows, strict=True)),
    )


def finish(
    call: Call,
    trajectory: DiffusionState,
    *,
    state: BatchState,
    latent_pool: LatentPool,
) -> PendingOutput:
    """Write a denoising call's integrated latent and report completion.

    The native executor selects completed intervals after their final
    prediction has been integrated. This scatters the numerical values to
    the inactive bank and records the ``LatentUpdate`` that
    advances the request from ``params.start_step`` by ``params.step_count``
    steps at the batch commit. Rust exports the bank and retires completed
    guidance prefixes after all numerical writes have been submitted.
    """
    request = state.pending_output(call.request_key.request_id)
    params = request.latent_params
    buffer = request.latent_buffer
    if params is None or buffer is None:
        raise invalid_descriptor("trajectory call has no bound latent inputs")

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
    latent_pool.write_inactive(
        request.request.request_pool_idx,
        buffer,
        expected_step=start_step,
        expected_generation=int(latent_input.generation),
        latent_units=int(params.latent_units),
        height=int(params.height),
        width=int(params.width),
    )
    state.complete_latent(call.request_key.request_id)

    kv = kv_conditioning(trajectory)
    request.set_cache_length(kv.cache[1])

    return request


def initial_latent(
    call: Call,
    height: int,
    width: int,
    target: torch.Tensor,
    *,
    seed: int,
    model_runner: ModelExecutor,
) -> None:
    """Draw a request's seeded initial latent into ``target``.

    The seed is ``flow_noise_seed(seed, rng.semantic_index_base)``.
    ``prepare_latent`` validates ``call.rng`` before calling this.
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
    entry: tuple[int, int, int],
) -> TokenRow:
    """Build the prefill row that writes one guidance branch's KV prefix.

    ``entry`` is the branch's ``(slot, materialized prefix length, token
    capacity)`` coordinate; ``tokens`` are written causally from the
    materialized length onward, selecting hidden states rather than logits.
    """
    positions = torch.arange(entry[1], entry[1] + len(tokens), dtype=torch.long)
    return TokenRow(
        forward_mode=ForwardMode.PREFILL,
        token_ids=torch.tensor(tokens, dtype=torch.long),
        positions=positions,
        selection=TokenSelection.HIDDEN,
        request_pool_idx=entry[0],
        seq_len=entry[1],
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
    """Build one denoising row per guidance branch over one shared sample.

    Every row borrows ``current`` (copied to ``device`` first when it lives
    elsewhere) and attends to its branch's KV entry without writing KV. The
    conditioned branch takes its temporal position from
    ``conditioning_position``; every other branch takes its entry's
    materialized prefix length. Position tensors are cached by temporal
    position in ``KVConditioning.positions``.
    """
    from uniserve_worker.protocol.call import MediaCall

    current = _to_device(current, device)
    timestep = _to_device(timestep, device)

    size, rows, kv = trajectory.size, [], kv_conditioning(trajectory)
    for branch in branches:
        entry = kv.entries[branch]
        temporal = (
            conditioning_position if branch is Branch.CONDITIONED else entry[1]
        )
        if temporal not in kv.positions:
            kv.positions[temporal] = builder.positions(
                size, temporal, device=device
            )
        rows.append(
            DiffusionRow(
                forward_mode=MediaCall.DENOISING,
                positions=kv.positions[temporal],
                timestep=timestep.reshape(1),
                latent=current,
                image_tokens=builder.sequence_length(size),
                image_height=size.height,
                image_width=size.width,
                request_pool_idx=entry[0],
                seq_len=entry[1],
                write_kv=False,
                causal=False,
            )
        )
    return tuple(rows)
