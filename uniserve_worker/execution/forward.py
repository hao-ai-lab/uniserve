"""Build homogeneous model calls and publish their results.

The native executor selects active calls. Token, canvas and encoder calls
run one forward; image diffusion runs each declared solver interval, with
guidance prefixes before denoising and integration after each prediction.

Call indexes address the original batch, including gaps left by predicates.
Rust selects the active calls and absolute solver steps; numerical helpers
consume those selections without maintaining a second completion set.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve.diffusion import Branch
from uniserve.tensors import OutputLayout
from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution import canvas
from uniserve_worker.execution.diffusion_state import DiffusionState
from uniserve_worker.execution.output import PendingOutput, capture_samples
from uniserve_worker.model_executor.diffusion_inputs import DiffusionRow
from uniserve_worker.model_executor.image_inputs import DecodeRow, PreparedImage
from uniserve_worker.model_executor.input_batch import InputRow, TokenRow
from uniserve_worker.profiling import record_component
from uniserve_worker.protocol.call import Call, ForwardMode, MediaCall
from uniserve_worker.sampling.metadata import SamplingMetadata
from uniserve_worker.sampling.result import SamplerRow
from uniserve_worker.sampling.sampler import broadcast_selection
from uniserve_worker.sampling.sampler import sample as _sample_task_batch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.tensor_store import TensorStore


ForwardValue = tuple[
    torch.Tensor,
    torch.Tensor,
    SamplerRow | None,
    OutputLayout | None,
]
SampleCandidate = tuple[
    int,
    TokenRow | DiffusionRow,
    torch.Tensor,
    SamplingMetadata | None,
    SamplerRow | None,
]


def forward_step(
    scheduled: tuple[Call, ...],
    steps: Sequence[tuple[int, int]],
    trajectories: Mapping[int, DiffusionState],
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
) -> tuple[int, ...]:
    """Evaluate one step and return the integrated trajectory indexes."""
    if trajectories:
        assert latent_pool is not None
        step_inputs = prepare_diffusion_step(
            steps,
            trajectories,
            scheduled,
            state=state,
            kv_cache=kv_cache,
            latent_pool=latent_pool,
            request_tables=request_tables,
            model_runner=model_runner,
            decode_state=decode_state,
            sampling_group=sampling_group,
            tokenizer=tokenizer,
        )
    else:
        step_inputs = {}

    forward, prepared_images = prepare_forward_rows(
        steps,
        step_inputs,
        trajectories,
        scheduled,
        state=state,
        tensor_store=tensor_store,
        latent_pool=latent_pool,
        request_tables=request_tables,
        model_runner=model_runner,
        decode_state=decode_state,
    )

    values = (
        forward_values(
            model_runner,
            tuple((task, scheduled[index]) for index, task in forward),
            cache=kv_cache,
            tables=request_tables,
            states=decode_state,
            sampling_group=sampling_group,
            state=state,
        )
        if forward
        else ()
    )

    predictions = publish_forward_values(
        forward,
        values,
        prepared_images,
        trajectories,
        scheduled,
        state=state,
        tensor_store=tensor_store,
        model_runner=model_runner,
        request_tables=request_tables,
        decode_state=decode_state,
        sampling_group=sampling_group,
    )

    if predictions:
        integrate_predictions(
            predictions,
            step_inputs,
            trajectories,
            scheduled,
            state=state,
            model_runner=model_runner,
        )

    return tuple(predictions)


def forward_values(
    model_runner: ModelExecutor,
    inputs: tuple[tuple[InputRow, Call], ...],
    *,
    state: BatchState,
    cache: KVCacheManager | None,
    tables: BlockTables | None,
    states: DecodeState | None,
    sampling_group: Communicator | None,
) -> tuple[ForwardValue | None, ...]:
    """Bind numerical outputs to calls and attribute execution statistics.

    Runs ``inputs`` through ``ModelExecutor.forward`` and returns one
    ``(value, request_pool_index, graph_sample, layout)`` tuple per input row,
    in input order. ``graph_sample`` is the row's graph-replayed greedy
    selection when ``token.graph_decode_samples`` accepts it for the whole
    output group, and None when the group is sampled eagerly; eager groups
    are materialized first (waiting on the forward's output event and
    gathering vocabulary shards). Each group's ``ForwardStats`` is appended
    to ``state.forward_stats``.

    Raises:
        BaseException: The error that failed any output group, re-raised as
            yielded by ``ModelExecutor.forward``.
        RuntimeError: When an output lacks statistics or request slot views.
    """
    # A call's CUDA input device must be one the batch's output buffer
    # declares; ``register_device`` also fails once the buffer is sealed.
    for row, _call in inputs:
        state.output_buffer.register_device(model_runner.call_devices(_call)[1])

    outputs = model_runner.forward(
        inputs,
        cache=cache,
        tables=tables,
        states=states,
    )
    values: list[ForwardValue | None] = [None] * len(inputs)
    from uniserve_worker.execution.token import graph_decode_samples

    for indexes, output in outputs:
        if isinstance(output, BaseException):
            raise output

        if output.stats is None or output.request_pool_indices is None:
            raise RuntimeError(
                "numerical forward lost statistics or request slot views"
            )
        stats = output.stats
        request_pool_indices = output.request_pool_indices

        selected = graph_decode_samples(
            tuple(inputs[index][1] for index in indexes),
            tuple(
                state.pending_output(
                    inputs[index][1].request_key.request_id,
                )
                for index in indexes
            ),
            tuple(inputs[index][0] for index in indexes),
            output.greedy,
            sampling_group=sampling_group,
            request_pool_indices=output.request_pool_indices,
        )
        if selected is None:
            output = output.materialize()

        # One forward output group maps back to its input rows; bind each
        # row's value, request slot view, graph sample, and output layout.
        state.record_forward(stats)
        for local, (index, value) in enumerate(
            zip(indexes, output.values, strict=True)
        ):
            values[index] = (
                value,
                request_pool_indices[local : local + 1],
                None if selected is None else selected[local],
                output.layouts[local],
            )
    return tuple(values)


def _publish_samples(
    candidates: list[SampleCandidate],
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
    request_tables: BlockTables | None,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
) -> None:
    """Sample token rows and publish their request-visible results.

    Each candidate is ``(index, row, logits, sampling_metadata,
    graph_sample)``; exactly one of the last two is set. Rows without a graph
    selection are sampled together in one ``sample`` call, the selections are
    captured into the batch's output buffer and token products are written.
    """
    from uniserve_worker.execution import token

    if not candidates:
        return

    # Rows already sampled inside a graph replay carry a selection;
    # sample only the rows that still need one, preserving order.
    sample_started = time.perf_counter_ns()
    sampling_inputs = tuple(
        work
        for _index, _task, _logits, work, _selected in candidates
        if work is not None
    )
    sampled_values = iter(
        _sample_task_batch(
            sampling_inputs,
            selection_broadcast=partial(broadcast_selection, sampling_group),
        )
    )
    sampled = tuple(
        selected if selected is not None else next(sampled_values)
        for _index, _task, _logits, _work, selected in candidates
    )
    capture_samples(
        sampled,
        tuple(
            state.pending_output(
                scheduled[index].request_key.request_id,
            )
            for index, _task, _logits, _work, _selected in candidates
        ),
        state.output_buffer,
    )
    record_component(
        state.component_us,
        "text_sample",
        sample_started,
    )

    finalize_started = time.perf_counter_ns()
    token.publish_token_products(
        tuple(
            scheduled[index]
            for index, _task, _logits, _work, _selected in candidates
        ),
        sampled,
        tensor_store=tensor_store,
        state=state,
    )
    for (index, task, logits, work, _captured), selected in zip(
        candidates, sampled, strict=True
    ):
        token.publish_sample(
            scheduled[index],
            task,
            logits,
            work,
            selected,
            image_builder=model_runner.image_builder,
            request_tables=request_tables,
            decode_state=decode_state,
            state=state,
        )
    record_component(
        state.component_us,
        "text_finalize",
        finalize_started,
    )


def initialize_trajectories(
    scheduled: tuple[Call, ...],
    indices: Sequence[int],
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    latent_pool: LatentPool,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> dict[int, DiffusionState]:
    """Bind numerical trajectories for the selected denoising calls."""
    from uniserve_worker.execution import diffusion

    return {
        index: diffusion.initialize(
            scheduled[index],
            kv_cache=kv_cache,
            latent_pool=latent_pool,
            request_tables=request_tables,
            model_runner=model_runner,
            state=state,
        )
        for index in indices
    }


def prepare_diffusion_step(
    steps: Sequence[tuple[int, int]],
    trajectories: Mapping[int, DiffusionState],
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    latent_pool: LatentPool,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
) -> dict[int, tuple[tuple[Branch, ...], torch.Tensor, int]]:
    """Prepare one solver step and materialize any missing guidance prefixes.

    ``steps`` names each selected call and its absolute solver step. The
    guidance branches, timestep and step are returned by call index. A branch
    whose KV prefix is not yet materialized comes back from
    ``diffusion.prepare_step`` as a prefix row; those rows run through one
    ``forward_values`` call here, before the denoiser rows that read them.
    """
    from uniserve_worker.execution import diffusion, token

    step_inputs: dict[int, tuple[tuple[Branch, ...], torch.Tensor, int]] = {}
    prefixes: list[tuple[int, Branch, TokenRow]] = []

    for index, step in steps:
        call = scheduled[index]
        trajectory = trajectories[index]

        guide, timestep, prefix_rows = diffusion.prepare_step(
            call,
            trajectory,
            step,
            request_tables=request_tables,
            model_runner=model_runner,
            latent_pool=latent_pool,
            tokenizer=tokenizer,
            state=state,
        )
        step_inputs[index] = guide, timestep, step
        prefixes.extend((index, branch, task) for branch, task in prefix_rows)

    if not prefixes:
        return step_inputs

    values = forward_values(
        model_runner,
        tuple((task, scheduled[index]) for index, _branch, task in prefixes),
        cache=kv_cache,
        tables=request_tables,
        states=decode_state,
        sampling_group=sampling_group,
        state=state,
    )
    for (index, branch, task), numerical_result in zip(
        prefixes, values, strict=True
    ):
        call = scheduled[index]
        if numerical_result is None:
            continue

        value, _sampling_index, _selection, _layout = numerical_result
        # The prefix fills a guidance branch's KV slot rather than extending
        # the request's own sequence, so its extent is validated without
        # preparing a runtime cache length for the request.
        token.commit_kv(
            task,
            task.query_tokens,
            state.pending_output(call.request_key.request_id),
            publish_runtime=False,
            request_tables=request_tables,
            decode_state=decode_state,
        )
        # Prefix coordinates are (slot, materialized length, token capacity).
        kv = diffusion.kv_conditioning(trajectories[index])
        prefix = kv.entries[branch]
        kv.entries[branch] = (
            prefix[0],
            prefix[1] + task.query_tokens,
            prefix[2],
        )

    return step_inputs


def prepare_forward_rows(
    steps: Sequence[tuple[int, int]],
    step_inputs: Mapping[int, tuple[tuple[Branch, ...], torch.Tensor, int]],
    trajectories: Mapping[int, DiffusionState],
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
    decode_state: DecodeState | None,
) -> tuple[list[tuple[int, InputRow]], dict[int, PreparedImage]]:
    """Build homogeneous numerical rows for the current solver step.

    Returns ``(index, row)`` pairs for the forward and the prepared encoder
    images by index. ``steps`` contains only the calls selected for this
    invocation. A trajectory contributes one
    denoiser row per guidance branch and a token-denoising call one row per
    canvas, so an index can appear more than once.
    An image-decoding call without a latent input finishes here without a
    forward (``image.diffusion_finalize_frames``) and contributes no row.
    """
    from uniserve_worker.execution import diffusion, image, token

    forward: list[tuple[int, InputRow]] = []
    images: dict[int, PreparedImage] = {}

    for index, _step in steps:
        call = scheduled[index]

        if index in trajectories:
            guide, timestep, _step = step_inputs[index]
            row = state.pending_output(call.request_key.request_id)
            params = row.latent_params
            buffer = row.latent_buffer
            if params is None or buffer is None:
                raise invalid_descriptor(
                    "trajectory call has no bound latent inputs"
                )

            # The model sees only the first ``latent_units`` rows of the
            # page-sized buffer.
            rows = diffusion.flow_rows(
                diffusion.require_inputs(model_runner),
                trajectories[index],
                buffer.value[: int(params.latent_units)],
                guide,
                timestep,
                conditioning_position=int(row.progress.logical_position),
                device=model_runner.call_devices(call)[1],
            )
            forward.extend((index, task) for task in rows)
        elif (
            call.kind is ForwardMode.TOKEN_DENOISING and call.canvas is not None
        ):
            forward.append(
                (
                    index,
                    canvas.prepare_step(
                        call,
                        state=state,
                        request_tables=request_tables,
                        canvas_slots=model_runner.canvas_slots,
                    ),
                )
            )
        elif call.kind is ForwardMode.TOKEN_DENOISING:
            forward.extend(
                (index, task)
                for task in canvas.prepare_rows(
                    call, request_tables=request_tables, state=state
                )
            )
        elif isinstance(call.kind, ForwardMode):
            build_started = time.perf_counter_ns()
            rows = (
                token.prepare_context(
                    call,
                    tensor_store=tensor_store,
                    request_tables=request_tables,
                    model_runner=model_runner,
                    state=state,
                )
                if token.writes_context(call)
                else (
                    token.prepare_forward(
                        call,
                        tensor_store=tensor_store,
                        request_tables=request_tables,
                        model_runner=model_runner,
                        decode_state=decode_state,
                        state=state,
                    ),
                )
            )
            record_component(
                state.component_us,
                "text_build_batch",
                build_started,
            )
            forward.extend((index, task) for task in rows)
        elif call.kind in {
            MediaCall.VISION_ENCODING,
            MediaCall.LATENT_ENCODING,
        }:
            prepared = image.prepare_features(
                call,
                tensor_store=tensor_store,
                model_runner=model_runner,
                state=state,
            )
            images[index] = prepared
            forward.append(
                (
                    index,
                    image.encode_row(cast(MediaCall, call.kind), prepared),
                )
            )
        # The remaining calls are image decodes. Without a latent input the
        # decoded image is already a resident product and only needs encoding.
        elif call.latent_input is None:
            image.diffusion_finalize_frames(
                call,
                tensor_store=tensor_store,
                model_runner=model_runner,
                state=state,
            )
        else:
            if latent_pool is None:
                raise RuntimeError("image decoding requires a latent pool")
            latent = image.materialization_latent(
                call,
                latent_pool=latent_pool,
                model_runner=model_runner,
                state=state,
            )
            row = state.pending_output(call.request_key.request_id)
            params = row.latent_params
            buffer = row.latent_buffer
            if params is None or buffer is None:
                raise invalid_descriptor(
                    "trajectory call has no bound latent inputs"
                )

            forward.append(
                (
                    index,
                    DecodeRow(
                        forward_mode=MediaCall.IMAGE_DECODING,
                        latent=latent,
                        image_height=int(params.height),
                        image_width=int(params.width),
                    ),
                )
            )

    return forward, images


def publish_forward_values(
    forward: list[tuple[int, InputRow]],
    values: tuple[ForwardValue | None, ...],
    images: Mapping[int, PreparedImage],
    trajectories: Mapping[int, DiffusionState],
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
) -> dict[int, list[torch.Tensor]]:
    """Publish completed numerical values and retain diffusion predictions.

    Dispatches each forward value by its call: denoiser predictions are
    collected per trajectory index and returned for
    ``integrate_predictions``; canvas readouts are collected per call and
    published by ``canvas.publish``, and a batch's canvas steps together
    by ``canvas.publish_steps``; sequence rows become sampling
    candidates (or finish directly when ``token.prepare_sampling`` returns an
    outcome);
    encoder values publish features; image-decoder values publish images.
    All sampling candidates are then sampled and published together.

    Raises:
        ValueError: When an image decoder's output layout declares no value
            range.
    """
    from uniserve_worker.execution import image, token

    predictions: dict[int, list[torch.Tensor]] = defaultdict(list)
    readouts: dict[int, list[torch.Tensor]] = defaultdict(list)
    contexts: dict[int, list[tuple[TokenRow, torch.Tensor, torch.Tensor]]] = (
        defaultdict(list)
    )
    # Canvas steps publish together, in row order, as one block.
    steps: list[tuple[int, torch.Tensor]] = []
    samples: list[SampleCandidate] = []

    for (index, task), numerical_result in zip(forward, values, strict=True):
        call = scheduled[index]
        if numerical_result is None:
            continue

        value, sampling_index, graph_sample, layout = numerical_result
        if index in trajectories:
            predictions[index].append(value)
        elif (
            call.kind is ForwardMode.TOKEN_DENOISING and call.canvas is not None
        ):
            steps.append((index, value))
        elif call.kind is ForwardMode.TOKEN_DENOISING:
            # Canvas rows of one call arrive in row order.
            readouts[index].append(value)
        elif token.writes_context(call):
            # Commit a context once all its text and vision rows are ready.
            # Publishing a row alone would hide the remaining segments.
            assert isinstance(task, TokenRow)
            contexts[index].append((task, value, sampling_index))
        elif isinstance(call.kind, ForwardMode):
            # A sequence call's row comes from token.prepare_forward.
            assert isinstance(task, (TokenRow, DiffusionRow))
            if graph_sample is not None:
                # A graph-sampled decode commits its one token as
                # ``token.prepare_sampling`` does for an eager decode: without
                # preparing a runtime cache length.
                request = state.pending_output(call.request_key.request_id)
                token.commit_kv(
                    task,
                    1,
                    request,
                    publish_runtime=False,
                    request_tables=request_tables,
                    decode_state=decode_state,
                )
                samples.append((index, task, value, None, graph_sample))
            else:
                selection = token.prepare_sampling(
                    call,
                    task,
                    value,
                    request_pool_index=sampling_index,
                    tensor_store=tensor_store,
                    image_builder=model_runner.image_builder,
                    request_tables=request_tables,
                    decode_state=decode_state,
                    state=state,
                )
                if not isinstance(selection, PendingOutput):
                    samples.append((index, task, value, selection, None))
        elif index in images:
            image.write_features(
                call,
                images[index],
                value,
                tensor_store=tensor_store,
                state=state,
            )
        else:
            if layout is None or layout.value_range is None:
                raise ValueError(
                    "image decoder must declare its numerical range"
                )
            image.export_image(
                call,
                value.detach(),
                layout.value_range,
                tensor_store=tensor_store,
                state=state,
            )

    for index, context_rows in contexts.items():
        context_result = token.finish_context(
            scheduled[index],
            tuple(context_rows),
            request_tables=request_tables,
            decode_state=decode_state,
            state=state,
        )
        if not isinstance(context_result, PendingOutput):
            context_row, logits, sampling = context_result
            samples.append((index, context_row, logits, sampling, None))

    canvas.publish_steps(
        tuple((scheduled[index], value) for index, value in steps),
        request_tables=request_tables,
        state=state,
        tensor_store=tensor_store,
    )

    for index, readout_values in readouts.items():
        canvas.capture_readout(
            scheduled[index],
            readout_values,
            request_tables=request_tables,
            state=state,
        )

    _publish_samples(
        samples,
        scheduled,
        state=state,
        tensor_store=tensor_store,
        model_runner=model_runner,
        request_tables=request_tables,
        decode_state=decode_state,
        sampling_group=sampling_group,
    )
    return predictions


def integrate_predictions(
    predictions: Mapping[int, list[torch.Tensor]],
    step_inputs: Mapping[int, tuple[tuple[Branch, ...], torch.Tensor, int]],
    trajectories: Mapping[int, DiffusionState],
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    model_runner: ModelExecutor,
) -> None:
    """Apply one mathematical solver step to each denoising prediction.

    ``predictions`` holds one denoiser output per guidance branch, in the
    branch order ``prepare_diffusion_step`` prepared. The solver integrates them
    into the model-visible part of the latent view, preserving page padding.
    The native executor selects which trajectories finish after this step.
    """
    for index, values in predictions.items():
        call = scheduled[index]
        _, timestep, step = step_inputs[index]
        row = state.pending_output(call.request_key.request_id)
        params = row.latent_params
        buffer = row.latent_buffer
        if params is None or buffer is None:
            raise invalid_descriptor(
                "trajectory call has no bound latent inputs"
            )

        # The solver updates only the model-visible portion of this
        # call's buffer, preserving page padding.
        model_runner.diffusion_entry(call).integrate(
            trajectories[index],
            buffer.value[: int(params.latent_units)],
            timestep,
            tuple(values),
            step,
        )
