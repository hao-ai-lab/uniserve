"""Build homogeneous model calls and publish their results."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve.diffusion import Branch
from uniserve.tensors import OutputLayout
from uniserve_worker.execution import operations
from uniserve_worker.execution.batch_state import BatchState
from uniserve_worker.execution.image_input import PreparedImage
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.execution.rows import ForwardRow
from uniserve_worker.execution.sample import broadcast_selection
from uniserve_worker.execution.sample import sample as _sample_task_batch
from uniserve_worker.foundation.errors import classify, invalid_descriptor
from uniserve_worker.profiling import record_component
from uniserve_worker.protocol.operation import (
    ForwardMode,
    PipelineStage,
    ScheduledRequest,
)

from .diffusion_state import ImageState
from .output import capture_samples
from .sampling import SamplerRow, SamplingMetadata

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.runtime.block_tables import BlockTables
    from uniserve_worker.runtime.cache_manager import CacheManager
    from uniserve_worker.runtime.decode_state import DecodeState
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.tensor_store import TensorStore
    from uniserve_worker.transfer.tickets import Transport


ForwardValue = tuple[
    torch.Tensor,
    torch.Tensor,
    SamplerRow | None,
    OutputLayout | None,
]
SampleCandidate = tuple[
    int,
    ForwardRow,
    torch.Tensor,
    SamplingMetadata | None,
    SamplerRow | None,
]


def forward_values(
    model_runner: ModelRunner,
    inputs: tuple[tuple[ForwardRow, ScheduledRequest, int], ...],
    *,
    state: BatchState,
    errors: dict[int, BaseException],
    retain_sampling: bool = False,
    cache: CacheManager | None,
    tables: BlockTables | None,
    states: DecodeState | None,
    sampling_group: Communicator | None,
) -> tuple[ForwardValue | None, ...]:
    """Bind numerical outputs to their completion owners and attribute group statistics."""

    for row, _operation, completion_group in inputs:
        state.group_buffers[completion_group].register_device(
            model_runner.operation_devices(_operation)[1]
        )

    outputs = model_runner.forward(
        tuple((row, operation) for row, operation, _scope in inputs),
        cache=cache,
        tables=tables,
        states=states,
    )
    values: list[ForwardValue | None] = [None] * len(inputs)
    from .token import graph_decode_samples

    for indexes, output in outputs:
        if isinstance(output, BaseException):
            for index in indexes:
                errors[inputs[index][2]] = output
            continue

        try:
            if output.stats is None or output.request_pool_indices is None:
                raise RuntimeError("numerical forward lost statistics or request slot views")
            stats = output.stats
            request_pool_indices = output.request_pool_indices

            selected = graph_decode_samples(
                tuple(inputs[index][1] for index in indexes),
                tuple(
                    state.pending_output(inputs[index][2], inputs[index][1].request_key.request_id)
                    for index in indexes
                ),
                tuple(inputs[index][0] for index in indexes),
                output.greedy.clone()
                if retain_sampling and output.greedy is not None
                else output.greedy,
                sampling_group=sampling_group,
                request_pool_indices=output.request_pool_indices,
            )
            if selected is None:
                output = output.materialize()

            # One forward output group maps back to its input rows; bind each
            # row's value, request slot view, graph sample, and output layout.
            state.group_forward_stats[inputs[indexes[0]][2]].append(stats)
            for local, (index, value) in enumerate(zip(indexes, output.values, strict=True)):
                values[index] = (
                    value,
                    request_pool_indices[local : local + 1],
                    None if selected is None else selected[local],
                    output.layouts[local],
                )
        except BaseException as error:
            if classify(error).fatal:
                raise
            for index in indexes:
                errors[inputs[index][2]] = error
    return tuple(values)


def _publish_sample_groups(
    samples: Mapping[int, list[SampleCandidate]],
    scheduled: tuple[tuple[ScheduledRequest, int], ...],
    outcomes: dict[int, PendingOutput],
    errors: dict[int, BaseException],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelRunner,
    request_tables: BlockTables | None,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
) -> None:
    """Sample compatible token rows and publish their request-visible results."""

    from . import token

    for group_id, candidates in samples.items():
        if group_id in errors:
            continue

        completion_group = scheduled[candidates[0][0]][1]
        try:
            # Rows already sampled inside a graph replay carry a selection;
            # sample only the rows that still need one, preserving order.
            sample_started = time.perf_counter_ns()
            sampling_inputs = tuple(
                work for _index, _task, _logits, work, _selected in candidates if work is not None
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
                        completion_group, scheduled[index][0].request_key.request_id
                    )
                    for index, _task, _logits, _work, _selected in candidates
                ),
                state.group_buffers[completion_group],
            )
            record_component(state.group_component_us[group_id], "text_sample", sample_started)

            finalize_started = time.perf_counter_ns()
            token.publish_token_products(
                tuple(
                    scheduled[index][0] for index, _task, _logits, _work, _selected in candidates
                ),
                sampled,
                completion_group,
                tensor_store=tensor_store,
                state=state,
            )
            for (index, task, logits, work, _captured), selected in zip(
                candidates, sampled, strict=True
            ):
                outcomes[index] = token.publish_sample(
                    scheduled[index][0],
                    completion_group,
                    task,
                    logits,
                    work,
                    selected,
                    image_builder=model_runner.image_builder,
                    request_tables=request_tables,
                    decode_state=decode_state,
                    state=state,
                )
            record_component(state.group_component_us[group_id], "text_finalize", finalize_started)
        except BaseException as error:
            errors[group_id] = error


def initialize_trajectories(
    numerical: tuple[int, ...],
    scheduled: tuple[tuple[ScheduledRequest, int], ...],
    outcomes: dict[int, PendingOutput],
    errors: dict[int, BaseException],
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    latent_pool: LatentPool,
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
) -> tuple[dict[int, ImageState], int]:
    """Open diffusion trajectories and return the longest declared interval."""

    from . import flow

    trajectories: dict[int, ImageState] = {}
    for index in numerical:
        operation, completion_group = scheduled[index]
        if (
            operation.kind is not PipelineStage.DENOISING
            or index in outcomes
            or completion_group in errors
        ):
            continue

        try:
            trajectories[index] = flow.initialize(
                operation,
                completion_group,
                kv_cache=kv_cache,
                latent_pool=latent_pool,
                request_tables=request_tables,
                model_runner=model_runner,
                state=state,
            )
        except BaseException as error:
            errors[completion_group] = error

    step_count = 1
    for index in trajectories:
        operation, completion_group = scheduled[index]
        request = state.pending_output(completion_group, operation.request_key.request_id)
        params = request.input_latent_params
        if params is None:
            raise invalid_descriptor("diffusion operation has no staged latent parameters")
        step_count = max(step_count, int(params.step_count))

    return trajectories, step_count


def prepare_diffusion_step(
    offset: int,
    trajectories: dict[int, ImageState],
    scheduled: tuple[tuple[ScheduledRequest, int], ...],
    outcomes: dict[int, PendingOutput],
    errors: dict[int, BaseException],
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    latent_pool: LatentPool,
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
) -> dict[int, tuple[tuple[Branch, ...], torch.Tensor, torch.Tensor]]:
    """Prepare one solver step and materialize any missing CFG prefixes."""

    from . import flow, token

    step_inputs: dict[int, tuple[tuple[Branch, ...], torch.Tensor, torch.Tensor]] = {}
    prefixes: list[tuple[int, Branch, ForwardRow]] = []

    for index, trajectory in trajectories.items():
        operation, completion_group = scheduled[index]
        if index in outcomes or completion_group in errors:
            continue

        row = state.pending_output(completion_group, operation.request_key.request_id)
        params = row.input_latent_params
        staging = row.latent_staging
        if params is None or staging is None:
            raise invalid_descriptor("trajectory operation has no staged latent inputs")
        if offset >= int(params.step_count):
            continue

        try:
            guide, timestep, next_timestep, prefix_rows = flow.prepare_step(
                operation,
                completion_group,
                trajectory,
                int(params.start_step) + offset,
                request_tables=request_tables,
                model_runner=model_runner,
                latent_pool=latent_pool,
                tokenizer=tokenizer,
                state=state,
            )
            step_inputs[index] = guide, timestep, next_timestep
            prefixes.extend((index, branch, task) for branch, task in prefix_rows)
        except BaseException as error:
            errors[completion_group] = error

    active_prefixes = tuple(
        item for item in prefixes if item[0] not in outcomes and scheduled[item[0]][1] not in errors
    )
    if not active_prefixes:
        return step_inputs

    values = forward_values(
        model_runner,
        tuple(
            (task, scheduled[index][0], scheduled[index][1])
            for index, _branch, task in active_prefixes
        ),
        cache=kv_cache,
        tables=request_tables,
        states=decode_state,
        sampling_group=sampling_group,
        state=state,
        errors=errors,
    )
    for (index, branch, task), numerical_result in zip(active_prefixes, values, strict=True):
        operation, completion_group = scheduled[index]
        if index in outcomes or completion_group in errors or numerical_result is None:
            continue

        value, _sampling_index, _selection, _layout = numerical_result
        try:
            token.commit_kv(
                task,
                task.query_tokens,
                state.pending_output(completion_group, operation.request_key.request_id),
                publish_runtime=False,
                request_tables=request_tables,
                decode_state=decode_state,
            )
            entry = trajectories[index].entries[branch]
            trajectories[index].entries[branch] = (
                entry[0],
                entry[1],
                entry[2] + task.query_tokens,
                entry[3],
            )
        except BaseException as error:
            errors[completion_group] = error

    return step_inputs


def prepare_forward_rows(
    numerical: tuple[int, ...],
    offset: int,
    step_inputs: Mapping[int, tuple[tuple[Branch, ...], torch.Tensor, torch.Tensor]],
    trajectories: Mapping[int, ImageState],
    scheduled: tuple[tuple[ScheduledRequest, int], ...],
    outcomes: dict[int, PendingOutput],
    errors: dict[int, BaseException],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
) -> tuple[list[tuple[int, ForwardRow]], dict[int, PreparedImage]]:
    """Build homogeneous numerical rows for the current dependency frontier."""

    from . import encode, flow, token

    forward: list[tuple[int, ForwardRow]] = []
    images: dict[int, PreparedImage] = {}

    for index in numerical:
        operation, completion_group = scheduled[index]
        if (
            index in outcomes
            or completion_group in errors
            or (offset > 0 and index not in step_inputs)
        ):
            continue

        try:
            if index in trajectories:
                if index not in step_inputs:
                    continue
                guide, timestep, _next_timestep = step_inputs[index]
                row = state.pending_output(completion_group, operation.request_key.request_id)
                params = row.input_latent_params
                staging = row.latent_staging
                if params is None or staging is None:
                    raise invalid_descriptor("trajectory operation has no staged latent inputs")

                rows = flow.flow_rows(
                    flow.require_inputs(model_runner),
                    trajectories[index],
                    staging.value[: int(params.latent_units)],
                    guide,
                    timestep,
                    conditioning_position=int(operations.require_progress(row).logical_position),
                    device=model_runner.operation_devices(operation)[1],
                )
                forward.extend((index, task) for task in rows)
            elif isinstance(operation.kind, ForwardMode):
                build_started = time.perf_counter_ns()
                task = token.prepare_forward(
                    operation,
                    completion_group,
                    tensor_store=tensor_store,
                    request_tables=request_tables,
                    model_runner=model_runner,
                    decode_state=decode_state,
                    state=state,
                )
                record_component(
                    state.group_component_us[completion_group],
                    "text_build_batch",
                    build_started,
                )
                forward.append((index, task))
            elif operation.kind in {
                PipelineStage.VISION_ENCODING,
                PipelineStage.LATENT_ENCODING,
            }:
                prepared = encode.prepare_features(
                    operation,
                    completion_group,
                    tensor_store=tensor_store,
                    model_runner=model_runner,
                    state=state,
                )
                images[index] = prepared
                forward.append(
                    (
                        index,
                        encode.encode_row(cast(PipelineStage, operation.kind), prepared),
                    )
                )
            elif operation.latent_input is None:
                outcomes[index] = encode.diffusion_finalize_frames(
                    operation,
                    completion_group,
                    tensor_store=tensor_store,
                    model_runner=model_runner,
                    state=state,
                )
            else:
                if latent_pool is None:
                    raise RuntimeError("image decoding requires a latent pool")
                latent = encode.materialization_latent(
                    operation,
                    completion_group,
                    latent_pool=latent_pool,
                    model_runner=model_runner,
                    state=state,
                )
                row = state.pending_output(completion_group, operation.request_key.request_id)
                params = row.input_latent_params
                staging = row.latent_staging
                if params is None or staging is None:
                    raise invalid_descriptor("trajectory operation has no staged latent inputs")

                forward.append(
                    (
                        index,
                        ForwardRow(
                            forward_mode=PipelineStage.IMAGE_DECODING,
                            latent=latent,
                            image_height=int(params.height),
                            image_width=int(params.width),
                        ),
                    )
                )
        except BaseException as error:
            errors[completion_group] = error

    return (
        [
            item
            for item in forward
            if item[0] not in outcomes and scheduled[item[0]][1] not in errors
        ],
        images,
    )


def publish_forward_values(
    forward: list[tuple[int, ForwardRow]],
    values: tuple[ForwardValue | None, ...],
    images: Mapping[int, PreparedImage],
    trajectories: Mapping[int, ImageState],
    scheduled: tuple[tuple[ScheduledRequest, int], ...],
    outcomes: dict[int, PendingOutput],
    errors: dict[int, BaseException],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    config: WorkerConfig,
) -> dict[int, list[torch.Tensor]]:
    """Publish completed numerical values and retain diffusion predictions."""

    from . import encode, token

    predictions: dict[int, list[torch.Tensor]] = defaultdict(list)
    samples: dict[int, list[SampleCandidate]] = defaultdict(list)

    for (index, task), numerical_result in zip(forward, values, strict=True):
        operation, completion_group = scheduled[index]
        if index in outcomes or completion_group in errors or numerical_result is None:
            continue

        value, sampling_index, graph_sample, layout = numerical_result
        try:
            if index in trajectories:
                predictions[index].append(value)
            elif isinstance(operation.kind, ForwardMode):
                if graph_sample is not None:
                    request = state.pending_output(
                        completion_group, operation.request_key.request_id
                    )
                    token.commit_kv(
                        task,
                        1,
                        request,
                        publish_runtime=False,
                        request_tables=request_tables,
                        decode_state=decode_state,
                    )
                    samples[completion_group].append((index, task, value, None, graph_sample))
                else:
                    selection = token.prepare_sampling(
                        operation,
                        completion_group,
                        task,
                        value,
                        request_pool_index=sampling_index,
                        tensor_store=tensor_store,
                        image_builder=model_runner.image_builder,
                        request_tables=request_tables,
                        decode_state=decode_state,
                        state=state,
                    )
                    if isinstance(selection, PendingOutput):
                        outcomes[index] = selection
                    else:
                        samples[completion_group].append((index, task, value, selection, None))
            elif index in images:
                outcomes[index] = encode.publish_features(
                    operation,
                    completion_group,
                    images[index],
                    value,
                    tensor_store=tensor_store,
                    worker_info=worker_info,
                    publication_transports=publication_transports,
                    config=config,
                    state=state,
                )
            else:
                if layout is None or layout.value_range is None:
                    raise ValueError("image decoder must declare its numerical range")
                outcomes[index] = encode.publish_image(
                    operation,
                    completion_group,
                    value.detach(),
                    layout.value_range,
                    tensor_store=tensor_store,
                    state=state,
                )
        except BaseException as error:
            errors[completion_group] = error

    _publish_sample_groups(
        samples,
        scheduled,
        outcomes,
        errors,
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
    step_inputs: Mapping[int, tuple[tuple[Branch, ...], torch.Tensor, torch.Tensor]],
    trajectories: Mapping[int, ImageState],
    offset: int,
    scheduled: tuple[tuple[ScheduledRequest, int], ...],
    outcomes: dict[int, PendingOutput],
    errors: dict[int, BaseException],
    *,
    state: BatchState,
    worker_info: WorkerInfo,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> None:
    """Advance diffusion solvers and publish trajectories at their final step."""

    from . import flow

    for index, values in predictions.items():
        operation, completion_group = scheduled[index]
        if index in outcomes or completion_group in errors:
            continue

        try:
            guide, timestep, next_timestep = step_inputs[index]
            row = state.pending_output(completion_group, operation.request_key.request_id)
            params = row.input_latent_params
            staging = row.latent_staging
            if params is None or staging is None:
                raise invalid_descriptor("trajectory operation has no staged latent inputs")

            # The solver updates only the model-visible portion of this
            # operation's staging, preserving page padding.
            flow.integrate(
                flow.require_inputs(model_runner),
                trajectories[index],
                staging.value[: int(params.latent_units)],
                tuple(values),
                int(params.start_step) + offset,
                timestep,
                next_timestep,
            )
            if offset + 1 == int(params.step_count):
                outcomes[index] = flow.finish(
                    operation,
                    completion_group,
                    trajectories[index],
                    worker_info=worker_info,
                    latent_pool=latent_pool,
                    publication_transports=publication_transports,
                    request_tables=request_tables,
                    config=config,
                    state=state,
                )
        except BaseException as error:
            errors[completion_group] = error
