"""Advance logical execution groups through computation and state publication."""

from __future__ import annotations

import logging
import time
import traceback
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker.execution import operations as operation_geometry
from uniserve_worker.execution.batch_state import BatchState
from uniserve_worker.execution.commit import _commit_group, _discard_group
from uniserve_worker.execution.image_input import ImageInputs
from uniserve_worker.execution.operations import _predicated_outcome
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.execution.prepare import _open_group
from uniserve_worker.execution.rows import ForwardRow
from uniserve_worker.execution.sample import broadcast_selection
from uniserve_worker.execution.sample import sample as _sample_task_batch
from uniserve_worker.foundation.errors import (
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
    should_capture_trace,
)
from uniserve_worker.models.generation import Materialization
from uniserve_worker.models.video import VideoModel
from uniserve_worker.nn.diffusion.cfg import Branch, CfgPlan
from uniserve_worker.profiling import _forward_stats, record_component
from uniserve_worker.protocol.batch import (
    DType,
    ErrorCode,
    FinishFlags,
    ForwardMode,
    ForwardStats,
    OpStatus,
    PipelineStage,
    RequestOutput,
    ScheduledRequest,
    TimingCounters,
    TransferMode,
)
from uniserve_worker.runtime.tensor_store import ImageRange

from ..models.inputs import PatchTransform
from .diffusion_state import DiffusionState
from .output import capture_samples
from .sampling import SamplerRow, SamplingMetadata

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.execution.output import OutputPool
    from uniserve_worker.media.buffers import MediaBuffers
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.models.runtime import ExecutionModel
    from uniserve_worker.nn.mesh import Communicator
    from uniserve_worker.runtime.block_tables import BlockTables
    from uniserve_worker.runtime.cpu import CpuPool
    from uniserve_worker.runtime.decode_state import DecodeState
    from uniserve_worker.runtime.kv_cache import KVCache
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.tensor_store import TensorStore
    from uniserve_worker.transfer.tickets import Transport


logger = logging.getLogger(__name__)

SAMPLING_COMPLETION_FIELDS = 4
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1


def _completion_error_code(code: WorkerErrorCode) -> ErrorCode:
    """Map internal failure classes to their completion-wire error codes."""

    if code == WorkerErrorCode.RESOURCE_ERROR:
        return ErrorCode.RESOURCE_EXHAUSTED
    if code == WorkerErrorCode.COMPUTE_ERROR:
        return ErrorCode.COMPUTE_ERROR
    if code in {WorkerErrorCode.INVARIANT_VIOLATION, WorkerErrorCode.FATAL_WORKER_FAILURE}:
        return ErrorCode.INTERNAL
    return ErrorCode.INVALID_OPERATION


def execute_batch(
    state: BatchState,
    *,
    propagate_errors: bool = False,
    kv_cache: KVCache | None,
    cpu_tasks: CpuPool,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    media_buffers: MediaBuffers | None,
    execution_model: ExecutionModel,
    output_pool: OutputPool,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> None:
    """Execute a run, reusing staged inputs when present.

    Startup propagates computation errors; service execution reports nonfatal
    errors per completion group so independent work can still complete.
    """

    batch = state.batch
    if not state.inputs_ready():
        raise RuntimeError("prepared execution was observed before transfer readiness")
    predicate_values = state.predicate_values()
    started = time.perf_counter_ns()

    required_predicates = {
        operation_geometry.operation_identity(operation)
        for operation in batch.operations
        if operation.predicate is not None and operation.predicate.dtype is DType.U8
    }
    if required_predicates != set(predicate_values):
        raise invalid_descriptor(
            "completion-predicated operations require exact prepared predicate values"
        )
    if not batch.operations:
        state.launched = True
        return

    groups = tuple(state.output_groups)
    completion_groups: list[int] = []
    for completion_group in groups:
        try:
            completion_groups.append(
                _open_group(
                    batch,
                    completion_group,
                    predicate_values,
                    kv_cache=kv_cache,
                    cpu_tasks=cpu_tasks,
                    tensor_store=tensor_store,
                    worker_info=worker_info,
                    latent_pool=latent_pool,
                    media_mux=media_mux,
                    media_buffers=media_buffers,
                    execution_model=execution_model,
                    output_pool=output_pool,
                    request_tables=request_tables,
                    request_pool=request_pool,
                    model_runner=model_runner,
                    transfer_backends=transfer_backends,
                    config=config,
                    state=state,
                )
            )
        except BaseException as error:
            classified = _classify_group_failure(
                completion_group,
                error,
                phase="completion group registration",
                state=state,
            )
            if propagate_errors or classified.fatal:
                for completion_group in completion_groups:
                    _discard_group(
                        completion_group,
                        classified,
                        kv_cache=kv_cache,
                        tensor_store=tensor_store,
                        latent_pool=latent_pool,
                        media_mux=media_mux,
                        transfer_backends=transfer_backends,
                        state=state,
                    )
                raise classified
            _error_outputs(
                state,
                completion_group,
                classified,
                started,
                registration_visible=False,
                forward_stats=ForwardStats(),
                request_pool=request_pool,
            )

    try:
        outcomes, execution_errors = _execute_groups(
            tuple(completion_groups),
            kv_cache=kv_cache,
            tensor_store=tensor_store,
            worker_info=worker_info,
            latent_pool=latent_pool,
            media_mux=media_mux,
            execution_model=execution_model,
            publication_transports=publication_transports,
            request_tables=request_tables,
            request_pool=request_pool,
            model_runner=model_runner,
            decode_state=decode_state,
            sampling_group=sampling_group,
            tokenizer=tokenizer,
            config=config,
            state=state,
        )
    except BaseException as error:
        classified = _classify_group_failure(
            groups[0],
            error,
            phase="completion group execution",
            state=state,
        )
        for completion_group in completion_groups:
            _discard_group(
                completion_group,
                classified,
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                latent_pool=latent_pool,
                media_mux=media_mux,
                transfer_backends=transfer_backends,
                state=state,
            )
        if propagate_errors or classified.fatal:
            raise classified
        for completion_group in completion_groups:
            _error_outputs(
                state,
                completion_group,
                classified,
                state.group_started_ns[completion_group],
                registration_visible=state.group_registered[completion_group],
                forward_stats=_forward_stats(
                    state.group_forward_stats[completion_group],
                    state.group_component_us[completion_group],
                ),
                request_pool=request_pool,
            )
        completion_groups.clear()
        outcomes, execution_errors = {}, {}

    if propagate_errors and execution_errors:
        first_group = next(
            completion_group for completion_group in groups if completion_group in execution_errors
        )
        classified = _classify_group_failure(
            first_group,
            execution_errors[first_group],
            phase="completion group execution",
            state=state,
        )
        for completion_group in completion_groups:
            _discard_group(
                completion_group,
                classified,
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                latent_pool=latent_pool,
                media_mux=media_mux,
                transfer_backends=transfer_backends,
                state=state,
            )
        raise classified

    for completion_group in completion_groups:
        group_error = execution_errors.get(completion_group)
        if group_error is not None:
            classified = _classify_group_failure(
                completion_group,
                group_error,
                phase="completion group execution",
                state=state,
            )
            _discard_group(
                completion_group,
                classified,
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                latent_pool=latent_pool,
                media_mux=media_mux,
                transfer_backends=transfer_backends,
                state=state,
            )
            if propagate_errors or classified.fatal:
                raise classified
            _error_outputs(
                state,
                completion_group,
                classified,
                state.group_started_ns[completion_group],
                registration_visible=state.group_registered[completion_group],
                forward_stats=_forward_stats(
                    state.group_forward_stats[completion_group],
                    state.group_component_us[completion_group],
                ),
                request_pool=request_pool,
            )
            continue
        group_outcomes = outcomes[completion_group]
        try:
            _commit_group(
                batch.run_id,
                completion_group,
                group_outcomes,
                started,
                state=state,
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                worker_info=worker_info,
                latent_pool=latent_pool,
                request_pool=request_pool,
                decode_state=decode_state,
                config=config,
            )
        except BaseException as error:
            if state.group_published[completion_group]:
                classified = _published_group_failure(
                    completion_group,
                    error,
                    state=state,
                )
            else:
                classified = _classify_group_failure(
                    completion_group,
                    error,
                    phase="completion group commit",
                    state=state,
                )
                _discard_group(
                    completion_group,
                    classified,
                    kv_cache=kv_cache,
                    tensor_store=tensor_store,
                    latent_pool=latent_pool,
                    media_mux=media_mux,
                    transfer_backends=transfer_backends,
                    state=state,
                )
            if propagate_errors or classified.fatal:
                raise classified
            _error_outputs(
                state,
                completion_group,
                classified,
                state.group_started_ns[completion_group],
                registration_visible=state.group_registered[completion_group],
                forward_stats=_forward_stats(
                    state.group_forward_stats[completion_group],
                    state.group_component_us[completion_group],
                ),
                request_pool=request_pool,
            )


def _classify_group_failure(
    completion_group: int,
    error: BaseException,
    *,
    phase: str,
    state: BatchState,
) -> WorkerError:
    """Classify a pre-publication completion group failure with complete operation and route context."""

    operations = tuple(
        (
            int(operation.request_key.engine_id),
            int(operation.request_key.request_id),
            int(operation.request_key.request_epoch),
            operation.op_id,
        )
        for operation in state.group_operations(completion_group)
    )
    sole = (
        state.group_operations(completion_group)[0]
        if len(state.group_operations(completion_group)) == 1
        else None
    )
    classified = classify(
        error,
        context=phase,
        phase=phase,
        operations=operations,
        req_id=None if sole is None else int(sole.request_key.request_id),
        op_id=None if sole is None else sole.op_id,
        op_kind=None if sole is None else sole.kind.value,
        route=str(0),
    )
    _log_group_failure(completion_group, classified, cause=error)
    return classified


def _published_group_failure(
    completion_group: int,
    error: BaseException,
    *,
    state: BatchState,
) -> WorkerError:
    """Classify a post-visibility publication failure as a fatal invariant violation."""

    operations = tuple(
        (
            int(operation.request_key.engine_id),
            int(operation.request_key.request_id),
            int(operation.request_key.request_epoch),
            operation.op_id,
        )
        for operation in state.group_operations(completion_group)
    )
    classified = WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=f"completion_group publication failed after visibility began: {error}",
        fatal=True,
        phase="completion group publication",
        route=str(0),
        operations=operations,
    )
    _log_group_failure(completion_group, classified, cause=error)
    return classified


def _log_group_failure(
    completion_group: int,
    error: WorkerError,
    *,
    cause: BaseException | None = None,
) -> None:
    """Log a classified completion group failure with traceback only for diagnostic error classes."""

    capture_trace = should_capture_trace(error.code)
    log = logger.error if capture_trace else logger.warning
    log(
        "completion group failed: %s [code=%s group_id=%s route=%s operations=%s]",
        error.message,
        error.code,
        completion_group,
        0,
        error.operations,
        exc_info=(type(cause), cause, cause.__traceback__)
        if capture_trace and cause is not None
        else None,
    )
    # A queued log record may outlive the worker. Keep the traceback locations
    # and exception chain, but do not let diagnostic frames retain borrowed
    # staging tensors after their CUDA stream has been closed.
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        traceback.clear_frames(cause.__traceback__)
        cause = cause.__cause__ or cause.__context__


def _execute_groups(
    completion_groups: tuple[int, ...],
    *,
    state: BatchState,
    kv_cache: KVCache | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    execution_model: ExecutionModel,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> tuple[dict[int, tuple[PendingOutput, ...]], dict[int, BaseException]]:
    """Execute active operations across completion groups and align outcomes with original completion group order."""

    for completion_group in completion_groups:
        active = tuple(
            operation
            for operation in state.group_operations(completion_group)
            if operation_geometry.request_row(
                completion_group, operation.request_key.request_id, state=state
            ).status
            is not OpStatus.PREDICATED
        )
        for device in dict.fromkeys(
            device for operation in active for device in model_runner.operation_devices(operation)
        ):
            state.group_buffers[completion_group].begin_device(device)
    grouped: list[list[PendingOutput | None]] = [
        [None] * len(state.group_operations(completion_group))
        for completion_group in completion_groups
    ]
    operations: list[tuple[ScheduledRequest, int]] = []
    locations: list[tuple[int, int]] = []
    for group_index, completion_group in enumerate(completion_groups):
        for operation_index, operation in enumerate(state.group_operations(completion_group)):
            if (
                operation_geometry.request_row(
                    completion_group, operation.request_key.request_id, state=state
                ).status
                is OpStatus.PREDICATED
            ):
                grouped[group_index][operation_index] = _predicated_outcome(
                    operation, completion_group, state=state
                )
                continue
            locations.append((group_index, operation_index))
            operations.append((operation, completion_group))
    completed, errors = _execute_operations(
        tuple(operations),
        kv_cache=kv_cache,
        tensor_store=tensor_store,
        worker_info=worker_info,
        latent_pool=latent_pool,
        media_mux=media_mux,
        execution_model=execution_model,
        publication_transports=publication_transports,
        request_tables=request_tables,
        request_pool=request_pool,
        model_runner=model_runner,
        decode_state=decode_state,
        sampling_group=sampling_group,
        tokenizer=tokenizer,
        config=config,
        state=state,
    )
    for index, outcome in completed.items():
        group_index, operation_index = locations[index]
        grouped[group_index][operation_index] = outcome
    outcomes: dict[int, tuple[PendingOutput, ...]] = {}
    for completion_group, group_outcomes in zip(completion_groups, grouped, strict=True):
        group_id = completion_group
        if group_id in errors:
            continue
        if any(outcome is None for outcome in group_outcomes):
            raise RuntimeError("successful completion group did not resolve every operation")
        outcomes[group_id] = tuple(cast(PendingOutput, outcome) for outcome in group_outcomes)
    return outcomes, errors


def _forward_values(
    model_runner: ModelRunner,
    inputs: tuple[tuple[ForwardRow, ScheduledRequest, int], ...],
    *,
    state: BatchState,
    errors: dict[int, BaseException],
    retain_sampling: bool = False,
    cache: KVCache | None,
    tables: BlockTables | None,
    states: DecodeState | None,
    sampling_group: Communicator | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, SamplerRow | None] | None, ...]:
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
    values: list[tuple[torch.Tensor, torch.Tensor, SamplerRow | None] | None] = [None] * len(inputs)
    from .token import graph_decode_samples

    for indexes, output in outputs:
        if isinstance(output, BaseException):
            for index in indexes:
                errors[inputs[index][2]] = output
            continue
        try:
            if output.stats is None or output.request_pool_indices is None:
                raise RuntimeError("numerical forward lost statistics or request slot views")
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

            state.group_forward_stats[inputs[indexes[0]][2]].append(output.stats)
            for local, (index, value) in enumerate(zip(indexes, output.values, strict=True)):
                values[index] = (
                    value,
                    output.request_pool_indices[local : local + 1],
                    None if selected is None else selected[local],
                )
        except BaseException as error:
            if classify(error).fatal:
                raise
            for index in indexes:
                errors[inputs[index][2]] = error
    return tuple(values)


def _execute_operations(
    operations: tuple[tuple[ScheduledRequest, int], ...],
    *,
    state: BatchState,
    kv_cache: KVCache | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    execution_model: ExecutionModel,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> tuple[dict[int, PendingOutput], dict[int, BaseException]]:
    """Execute product dependency frontiers with direct numerical algorithms.

    Each index addresses an original operation. Only completed products unlock
    successors; an error suppresses its completion group while independent
    groups continue. CFG prefixes precede the mixed token/denoise forward.
    """

    from . import encode, flow, token, transfer, video

    producers = {
        buffer: index
        for index, (operation, _scope) in enumerate(operations)
        for buffer in (
            *(output.buffer_id for output in operation.tensor_outputs()),
            *((operation.kv_output,) if operation.kv_output is not None else ()),
        )
    }
    outcomes: dict[int, PendingOutput] = {}
    errors: dict[int, BaseException] = {}

    def live(index: int) -> bool:
        return index not in outcomes and operations[index][1] not in errors

    def ready(index: int) -> bool:
        operation = operations[index][0]
        return all(
            producer in outcomes
            for buffer in (
                *(reference.buffer_id for reference in operation.tensor_inputs()),
                *((operation.kv_input,) if operation.kv_input is not None else ()),
            )
            if (producer := producers.get(buffer)) is not None
        )

    while any(live(index) for index in range(len(operations))):
        frontier = tuple(index for index in range(len(operations)) if live(index) and ready(index))
        if not frontier:
            blocked = tuple(
                operation_geometry.operation_identity(operation)
                for index, (operation, _scope) in enumerate(operations)
                if live(index)
            )
            raise RuntimeError(f"operation products contain an unresolved dependency: {blocked!r}")
        numerical = tuple(
            index
            for index in frontier
            if isinstance(operations[index][0].kind, ForwardMode)
            or operations[index][0].kind
            in {PipelineStage.VISION_ENCODING, PipelineStage.LATENT_ENCODING}
            or (
                latent_pool is not None
                and operations[index][0].kind
                in {
                    PipelineStage.DENOISING,
                    PipelineStage.IMAGE_DECODING,
                }
            )
        )
        if not numerical:
            for index in frontier:
                operation, completion_group = operations[index]
                if not live(index):
                    continue
                try:
                    if isinstance(operation.kind, TransferMode):
                        result = transfer.execute(
                            operation,
                            completion_group,
                            kv_cache=kv_cache,
                            tensor_store=tensor_store,
                            latent_pool=latent_pool,
                            publication_transports=publication_transports,
                            request_tables=request_tables,
                            model_runner=model_runner,
                            state=state,
                        )
                    elif (
                        operation.kind is PipelineStage.LATENT_PREPARATION
                        and latent_pool is not None
                    ):
                        result = flow.prepare_latent(
                            operation,
                            completion_group,
                            kv_cache=kv_cache,
                            worker_info=worker_info,
                            latent_pool=latent_pool,
                            publication_transports=publication_transports,
                            request_tables=request_tables,
                            model_runner=model_runner,
                            config=config,
                            state=state,
                        )
                    elif operation.kind is PipelineStage.TEXT_ENCODING:
                        result = encode.text(
                            operation,
                            completion_group,
                            tensor_store=tensor_store,
                            publication_transports=publication_transports,
                            model_runner=model_runner,
                            state=state,
                        )
                    elif isinstance(execution_model, VideoModel):
                        result = video.execute(
                            operation,
                            completion_group,
                            tensor_store=tensor_store,
                            media_mux=media_mux,
                            execution_model=execution_model,
                            publication_transports=publication_transports,
                            request_pool=request_pool,
                            model_runner=model_runner,
                            state=state,
                        )
                    else:
                        raise invalid_descriptor(f"unsupported operation {operation.kind!r}")
                    outcomes[index] = result
                except BaseException as error:
                    errors[completion_group] = error
            continue

        trajectories: dict[int, DiffusionState] = {}
        for index in numerical:
            operation, completion_group = operations[index]
            if operation.kind is not PipelineStage.DENOISING or not live(index):
                continue
            try:
                assert latent_pool is not None
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
            operation, completion_group = operations[index]
            request = operation_geometry.request_row(
                completion_group, operation.request_key.request_id, state=state
            )
            params = request.input_latent_params
            if params is None:
                raise invalid_descriptor("diffusion operation has no staged latent parameters")
            step_count = max(step_count, int(params.step_count))
        for offset in range(step_count):
            # The numerical schedule is local to this loop. Accepted request
            # progress is published only after the complete declared interval.
            step_inputs: dict[int, tuple[CfgPlan, torch.Tensor, torch.Tensor]] = {}
            prefixes: list[tuple[int, Branch, ForwardRow]] = []
            for index, trajectory in trajectories.items():
                if not live(index):
                    continue
                operation, completion_group = operations[index]
                row = operation_geometry.request_row(
                    completion_group, operation.request_key.request_id, state=state
                )
                params = row.input_latent_params
                staging = row.latent_staging
                if params is None or staging is None:
                    raise invalid_descriptor("trajectory operation has no staged latent inputs")
                if offset >= int(params.step_count):
                    continue
                try:
                    if latent_pool is None:
                        raise RuntimeError("diffusion trajectory lost its latent pool")
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
            prefixes = [item for item in prefixes if live(item[0])]
            if prefixes:
                values = _forward_values(
                    model_runner,
                    tuple(
                        (task, operations[index][0], operations[index][1])
                        for index, _branch, task in prefixes
                    ),
                    cache=kv_cache,
                    tables=request_tables,
                    states=decode_state,
                    sampling_group=sampling_group,
                    state=state,
                    errors=errors,
                )
                for (index, branch, task), numerical_result in zip(prefixes, values, strict=True):
                    operation, completion_group = operations[index]
                    if not live(index) or numerical_result is None:
                        continue
                    value, _sampling_index, _selection = numerical_result
                    try:
                        token.commit_kv(
                            task,
                            task.query_tokens,
                            operation_geometry.request_row(
                                completion_group, operation.request_key.request_id, state=state
                            ),
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

            forward: list[tuple[int, ForwardRow]] = []
            images: dict[int, ImageInputs] = {}
            for index in numerical:
                operation, completion_group = operations[index]
                if not live(index) or (offset > 0 and index not in step_inputs):
                    continue
                try:
                    if index in trajectories:
                        if index not in step_inputs:
                            continue
                        guide, timestep, _next_timestep = step_inputs[index]
                        row = operation_geometry.request_row(
                            completion_group, operation.request_key.request_id, state=state
                        )
                        params = row.input_latent_params
                        staging = row.latent_staging
                        if params is None or staging is None:
                            raise invalid_descriptor(
                                "trajectory operation has no staged latent inputs"
                            )
                        request = operation_geometry.request_row(
                            completion_group, operation.request_key.request_id, state=state
                        )
                        diffusion = model_runner.diffusion
                        if diffusion is None:
                            raise RuntimeError("flow execution lost its diffusion owner")
                        processor = model_runner.model.image_processor
                        transform = None if processor is None else processor.vit
                        rows = diffusion.flow_rows(
                            trajectories[index],
                            staging.value[: int(params.latent_units)],
                            guide,
                            timestep,
                            conditioning_position=int(
                                operation_geometry.require_progress(request).logical_position
                            ),
                            height=int(params.height),
                            width=int(params.width),
                            patch_size=int(transform.patch_size)
                            if isinstance(transform, PatchTransform)
                            else None,
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
                            tokenizer=tokenizer,
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
                        assert latent_pool is not None
                        latent = encode.materialization_latent(
                            operation,
                            completion_group,
                            latent_pool=latent_pool,
                            model_runner=model_runner,
                            state=state,
                        )
                        if model_runner.generation().materialization is Materialization.RGB_LATENT:
                            outcomes[index] = encode.publish_image(
                                operation,
                                completion_group,
                                latent.detach(),
                                ImageRange.SIGNED_UNIT,
                                tensor_store=tensor_store,
                                state=state,
                            )
                        else:
                            row = operation_geometry.request_row(
                                completion_group, operation.request_key.request_id, state=state
                            )
                            params = row.input_latent_params
                            staging = row.latent_staging
                            if params is None or staging is None:
                                raise invalid_descriptor(
                                    "trajectory operation has no staged latent inputs"
                                )
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
            forward = [item for item in forward if live(item[0])]
            values = (
                _forward_values(
                    model_runner,
                    tuple(
                        (task, operations[index][0], operations[index][1])
                        for index, task in forward
                    ),
                    cache=kv_cache,
                    tables=request_tables,
                    states=decode_state,
                    sampling_group=sampling_group,
                    state=state,
                    errors=errors,
                    retain_sampling=offset + 1 < step_count,
                )
                if forward
                else ()
            )
            predictions: dict[int, list[torch.Tensor]] = defaultdict(list)
            samples: dict[
                int,
                list[
                    tuple[int, ForwardRow, torch.Tensor, SamplingMetadata | None, SamplerRow | None]
                ],
            ] = defaultdict(list)
            for (index, task), numerical_result in zip(forward, values, strict=True):
                operation, completion_group = operations[index]
                if not live(index) or numerical_result is None:
                    continue
                value, sampling_index, graph_sample = numerical_result
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
                            samples[completion_group].append(
                                (index, task, value, None, graph_sample)
                            )
                        else:
                            selection = token.prepare_sampling(
                                operation,
                                completion_group,
                                task,
                                value,
                                request_pool_index=sampling_index,
                                tensor_store=tensor_store,
                                execution_model=execution_model,
                                request_tables=request_tables,
                                decode_state=decode_state,
                                state=state,
                            )
                            if isinstance(selection, PendingOutput):
                                outcomes[index] = selection
                            else:
                                samples[completion_group].append(
                                    (index, task, value, selection, None)
                                )
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
                        outcomes[index] = encode.publish_image(
                            operation,
                            completion_group,
                            value.detach(),
                            ImageRange.UNIT,
                            tensor_store=tensor_store,
                            state=state,
                        )
                except BaseException as error:
                    errors[completion_group] = error
            for group_id, candidates in samples.items():
                if group_id in errors:
                    continue
                completion_group = operations[candidates[0][0]][1]
                try:
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
                            operation_geometry.request_row(
                                completion_group,
                                operations[index][0].request_key.request_id,
                                state=state,
                            )
                            for index, _task, _logits, _work, _selected in candidates
                        ),
                        state.group_buffers[completion_group],
                    )
                    record_component(
                        state.group_component_us[group_id], "text_sample", sample_started
                    )
                    finalize_started = time.perf_counter_ns()
                    token.publish_token_products(
                        tuple(
                            operations[index][0]
                            for index, _task, _logits, _work, _selected in candidates
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
                            operations[index][0],
                            completion_group,
                            task,
                            logits,
                            work,
                            selected,
                            execution_model=execution_model,
                            request_tables=request_tables,
                            decode_state=decode_state,
                            state=state,
                        )
                    record_component(
                        state.group_component_us[group_id], "text_finalize", finalize_started
                    )
                except BaseException as error:
                    errors[group_id] = error
            for index, outputs in predictions.items():
                if not live(index):
                    continue
                operation, completion_group = operations[index]
                try:
                    guide, timestep, next_timestep = step_inputs[index]
                    row = operation_geometry.request_row(
                        completion_group, operation.request_key.request_id, state=state
                    )
                    params = row.input_latent_params
                    staging = row.latent_staging
                    if params is None or staging is None:
                        raise invalid_descriptor("trajectory operation has no staged latent inputs")
                    diffusion = model_runner.diffusion
                    if diffusion is None:
                        raise RuntimeError("flow execution lost its diffusion owner")
                    # Page staging includes allocation padding; solver updates
                    # only this operation's model-visible latent units.
                    diffusion.integrate(
                        staging.value[: int(params.latent_units)],
                        tuple(outputs),
                        guide,
                        timestep,
                        next_timestep,
                    )
                    if offset + 1 == int(params.step_count):
                        assert latent_pool is not None
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
    return outcomes, errors


def _error_outputs(
    state: BatchState,
    completion_group: int,
    error: WorkerError,
    started: int,
    *,
    registration_visible: bool,
    forward_stats: ForwardStats,
    request_pool: RequestPool,
) -> None:
    """Record final errors at the owning completion boundary without accepting progress."""

    completion_code = _completion_error_code(error.code)
    records: list[RequestOutput] = []
    for operation in state.group_operations(completion_group):
        # Report execution coordinates only for the matching admitted epoch;
        # a stale descriptor cannot observe a replacement request slot.
        request = request_pool.peek(operation.request_key.request_id)
        if (
            request is None
            or request.request_key != operation.request_key
            or operation.predecessor is None
        ):
            runtime = None
        else:
            runtime = request.accepted_progress
        placeholder = RequestOutput(
            request_key=operation.request_key,
            op_id=operation.op_id,
            status=OpStatus.ERROR,
            product_generations=(),
            error_code=completion_code,
            timing_counters=TimingCounters(),
            kind=operation.kind,
            position=(0 if runtime is None else int(runtime.logical_position)),
            kv_visible_len=(0 if runtime is None else int(runtime.kv_visible_len)),
            kv_computed_len=(0 if runtime is None else int(runtime.kv_computed_len)),
            num_completed_steps=(0 if runtime is None else int(runtime.flow_step)),
            committed_tokens=(),
            finish_flags=FinishFlags(),
        )
        records.append(placeholder)
    state.record_outputs(
        completion_group,
        tuple(records),
        visible=registration_visible,
        execution_us=(time.perf_counter_ns() - started) // 1000,
        stats=forward_stats,
    )
