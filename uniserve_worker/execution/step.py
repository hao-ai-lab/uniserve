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
from uniserve_worker.execution.batch import (
    DType,
    ErrorCode,
    FinishFlags,
    ForwardMode,
    LaneResult,
    ModelOutput,
    OpStatus,
    PipelineStage,
    RegistrationAck,
    Run,
    RunLane,
    RunResult,
    TimingCounters,
    WorkerForwardStats,
)
from uniserve_worker.execution.commit import _commit_lane, _discard_lane
from uniserve_worker.execution.operations import _completion_devices, _predicated_outcome
from uniserve_worker.execution.prepare import _apply_batch_controls, _open_lane
from uniserve_worker.execution.retirement import _apply_release_controls
from uniserve_worker.execution.rows import (
    ForwardRow,
    LaneState,
    OperationState,
    Outcome,
    PreparedExecution,
    SampleWork,
    dependencies_ready,
)
from uniserve_worker.execution.sample import broadcast_selection
from uniserve_worker.execution.sample import sample as _sample_task_batch
from uniserve_worker.execution.video import run_action as run_video_action
from uniserve_worker.foundation.errors import (
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
    should_capture_trace,
)
from uniserve_worker.profiling import _forward_stats
from uniserve_worker.runtime.request import RequestRuntime

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.execution.output import OutputPool
    from uniserve_worker.execution.video import VideoMuxCoordinator, VideoOutputRing
    from uniserve_worker.models.runtime import ExecutionModel
    from uniserve_worker.nn.mesh import Communicator
    from uniserve_worker.runtime.cache_pool import CachePool
    from uniserve_worker.runtime.cache_publications import CachePublications
    from uniserve_worker.runtime.cpu import CpuPool
    from uniserve_worker.runtime.device_products import DeviceProducts
    from uniserve_worker.runtime.encoder_cache import EncoderCache
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.req_to_token_pool import ReqToTokenPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.runtime_states import RuntimeStates
    from uniserve_worker.transfer.publications import TransferPublications
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
    batch: Run,
    *,
    prepared: PreparedExecution | None = None,
    propagate_errors: bool = False,
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    cpu_tasks: CpuPool,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: VideoMuxCoordinator | None,
    media_output_ring: VideoOutputRing | None,
    execution_model: ExecutionModel,
    output_pool: OutputPool,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    runtime_states: RuntimeStates | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    transfer_publications: TransferPublications,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> RunResult:
    """Execute a run, reusing staged inputs when present.

    Startup propagates computation errors; service execution reports nonfatal
    errors per lane so independent work can still complete.
    """

    if prepared is not None and not prepared.ready():
        raise RuntimeError("prepared execution was observed before transfer readiness")
    transfers = () if prepared is None else prepared.transfers
    predicate_values = {} if prepared is None else prepared.predicate_values()

    started = time.perf_counter_ns()
    # Preparation has already installed admissions and applied batch controls.
    if prepared is None:
        _apply_batch_controls(
            batch,
            cache_pool=cache_pool,
            cache_registry=cache_registry,
            device_products=device_products,
            encoder_cache=encoder_cache,
            worker_info=worker_info,
            latent_pool=latent_pool,
            execution_model=execution_model,
            request_pool=request_pool,
            model_runner=model_runner,
            runtime_states=runtime_states,
            transfer_publications=transfer_publications,
            config=config,
        )

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
        return RunResult(
            batch_id=batch.batch_id,
            run_id=batch.run_id,
            lanes=(),
        )

    reports: dict[int, LaneResult] = {}
    groups: dict[int, list[RunLane]] = {}
    for lane in batch.lanes:
        groups.setdefault(lane.launch_id, []).append(lane)

    for lanes in groups.values():
        scopes: list[LaneState] = []
        for lane in lanes:
            try:
                scopes.append(
                    _open_lane(
                        batch,
                        lane,
                        transfers,
                        predicate_values,
                        graph_eligible=True,
                        cache_pool=cache_pool,
                        cache_registry=cache_registry,
                        cpu_tasks=cpu_tasks,
                        device_products=device_products,
                        encoder_cache=encoder_cache,
                        worker_info=worker_info,
                        latent_pool=latent_pool,
                        media_mux=media_mux,
                        media_output_ring=media_output_ring,
                        execution_model=execution_model,
                        output_pool=output_pool,
                        request_tables=request_tables,
                        request_pool=request_pool,
                        model_runner=model_runner,
                        transfer_backends=transfer_backends,
                        config=config,
                    )
                )
            except BaseException as error:
                classified = _classify_lane_failure(
                    lane,
                    error,
                    phase="lane registration",
                )
                if propagate_errors or classified.fatal:
                    for scope in scopes:
                        _discard_lane(
                            scope,
                            classified,
                            cache_pool=cache_pool,
                            device_products=device_products,
                            encoder_cache=encoder_cache,
                            latent_pool=latent_pool,
                            media_mux=media_mux,
                            transfer_backends=transfer_backends,
                        )
                    raise classified
                reports[lane.lane_id] = _registration_error_lane(
                    batch.run_id, lane, classified, started, request_pool=request_pool
                )

        if not scopes:
            continue
        try:
            outcomes, execution_errors = _execute_lane_group(
                tuple(scopes),
                cache_pool=cache_pool,
                cache_registry=cache_registry,
                device_products=device_products,
                encoder_cache=encoder_cache,
                worker_info=worker_info,
                latent_pool=latent_pool,
                media_mux=media_mux,
                execution_model=execution_model,
                publication_transports=publication_transports,
                request_tables=request_tables,
                request_pool=request_pool,
                model_runner=model_runner,
                runtime_states=runtime_states,
                sampling_group=sampling_group,
                tokenizer=tokenizer,
                config=config,
            )
        except BaseException as error:
            classified = _classify_lane_failure(
                lanes[0],
                error,
                phase="lane execution",
            )
            for scope in scopes:
                _discard_lane(
                    scope,
                    classified,
                    cache_pool=cache_pool,
                    device_products=device_products,
                    encoder_cache=encoder_cache,
                    latent_pool=latent_pool,
                    media_mux=media_mux,
                    transfer_backends=transfer_backends,
                )
            if propagate_errors or classified.fatal:
                raise classified
            for scope in scopes:
                reports[scope.lane.lane_id] = _error_lane(
                    batch.run_id, scope, classified, started, request_pool=request_pool
                )
            continue

        if propagate_errors and execution_errors:
            first_lane = next(lane for lane in lanes if lane.lane_id in execution_errors)
            classified = _classify_lane_failure(
                first_lane,
                execution_errors[first_lane.lane_id],
                phase="lane execution",
            )
            for scope in scopes:
                _discard_lane(
                    scope,
                    classified,
                    cache_pool=cache_pool,
                    device_products=device_products,
                    encoder_cache=encoder_cache,
                    latent_pool=latent_pool,
                    media_mux=media_mux,
                    transfer_backends=transfer_backends,
                )
            raise classified

        for scope in scopes:
            lane_error = execution_errors.get(scope.lane.lane_id)
            if lane_error is not None:
                classified = _classify_lane_failure(
                    scope.lane,
                    lane_error,
                    phase="lane execution",
                )
                _discard_lane(
                    scope,
                    classified,
                    cache_pool=cache_pool,
                    device_products=device_products,
                    encoder_cache=encoder_cache,
                    latent_pool=latent_pool,
                    media_mux=media_mux,
                    transfer_backends=transfer_backends,
                )
                if propagate_errors or classified.fatal:
                    raise classified
                reports[scope.lane.lane_id] = _error_lane(
                    batch.run_id, scope, classified, started, request_pool=request_pool
                )
                continue
            lane_outcomes = outcomes[scope.lane.lane_id]
            try:
                reports[scope.lane.lane_id] = _commit_lane(
                    batch.run_id,
                    scope,
                    lane_outcomes,
                    started,
                    cache_registry=cache_registry,
                    device_products=device_products,
                    encoder_cache=encoder_cache,
                    worker_info=worker_info,
                    latent_pool=latent_pool,
                    request_pool=request_pool,
                    runtime_states=runtime_states,
                    transfer_publications=transfer_publications,
                    config=config,
                )
            except BaseException as error:
                if scope.publication_started:
                    classified = _published_lane_failure(
                        scope.lane,
                        error,
                    )
                else:
                    classified = _classify_lane_failure(
                        scope.lane,
                        error,
                        phase="lane commit",
                    )
                    _discard_lane(
                        scope,
                        classified,
                        cache_pool=cache_pool,
                        device_products=device_products,
                        encoder_cache=encoder_cache,
                        latent_pool=latent_pool,
                        media_mux=media_mux,
                        transfer_backends=transfer_backends,
                    )
                if propagate_errors or classified.fatal:
                    raise classified
                reports[scope.lane.lane_id] = _error_lane(
                    batch.run_id, scope, classified, started, request_pool=request_pool
                )

    _apply_release_controls(
        batch,
        before_execution=False,
        cache_pool=cache_pool,
        cache_registry=cache_registry,
        device_products=device_products,
        encoder_cache=encoder_cache,
        latent_pool=latent_pool,
        transfer_publications=transfer_publications,
    )
    report = RunResult(
        batch_id=batch.batch_id,
        run_id=batch.run_id,
        lanes=tuple(reports[lane.lane_id] for lane in batch.lanes),
    )
    return report


def _classify_lane_failure(
    lane: RunLane,
    error: BaseException,
    *,
    phase: str,
) -> WorkerError:
    """Classify a pre-publication lane failure with complete operation and route context."""

    operations = tuple(
        (
            int(operation.request_key.engine_id),
            int(operation.request_key.request_id),
            int(operation.request_key.request_epoch),
            operation.op_id,
        )
        for operation in lane.operations
    )
    sole = lane.operations[0] if len(lane.operations) == 1 else None
    classified = classify(
        error,
        context=phase,
        phase=phase,
        operations=operations,
        req_id=None if sole is None else int(sole.request_key.request_id),
        op_id=None if sole is None else sole.op_id,
        op_kind=None if sole is None else sole.kind.value,
        route=str(lane.route),
    )
    _log_lane_failure(lane, classified, cause=error)
    return classified


def _published_lane_failure(
    lane: RunLane,
    error: BaseException,
) -> WorkerError:
    """Classify a post-visibility publication failure as a fatal invariant violation."""

    operations = tuple(
        (
            int(operation.request_key.engine_id),
            int(operation.request_key.request_id),
            int(operation.request_key.request_epoch),
            operation.op_id,
        )
        for operation in lane.operations
    )
    classified = WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=f"lane publication failed after visibility began: {error}",
        fatal=True,
        phase="lane publication",
        route=str(lane.route),
        operations=operations,
    )
    _log_lane_failure(lane, classified, cause=error)
    return classified


def _log_lane_failure(
    lane: RunLane,
    error: WorkerError,
    *,
    cause: BaseException | None = None,
) -> None:
    """Log a classified lane failure with traceback only for diagnostic error classes."""

    capture_trace = should_capture_trace(error.code)
    log = logger.error if capture_trace else logger.warning
    log(
        "lane failed: %s [code=%s lane_id=%s route=%s operations=%s]",
        error.message,
        error.code,
        lane.lane_id,
        lane.route,
        error.operations,
        exc_info=(type(cause), cause, cause.__traceback__)
        if capture_trace and cause is not None
        else None,
    )
    # A queued log record may outlive the worker. Keep the traceback locations
    # and exception chain, but do not let diagnostic frames retain borrowed
    # staging tensors after their CUDA lane has been closed.
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        traceback.clear_frames(cause.__traceback__)
        cause = cause.__cause__ or cause.__context__


def _execute_lane_group(
    scopes: tuple[LaneState, ...],
    *,
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: VideoMuxCoordinator | None,
    execution_model: ExecutionModel,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    runtime_states: RuntimeStates | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> tuple[dict[int, tuple[Outcome, ...]], dict[int, BaseException]]:
    """Execute active operations across lanes and align outcomes with original lane order."""

    from . import token

    for scope in scopes:
        active = tuple(
            operation
            for operation in scope.lane.operations
            if operation_geometry.operation_identity(operation) not in scope.predicated_operations
        )
        for device in _completion_devices(active, config=config):
            scope.completion.begin_device(device)
    grouped: list[list[Outcome | None]] = [[None] * len(scope.lane.operations) for scope in scopes]
    group_active = tuple(
        operation
        for scope in scopes
        for operation in scope.lane.operations
        if operation_geometry.operation_identity(operation) not in scope.predicated_operations
    )
    homogeneous_decode = bool(group_active) and all(
        operation.kind is ForwardMode.DECODE for operation in group_active
    )
    states: list[OperationState] = []
    locations: dict[int, tuple[int, int]] = {}
    for scope_index, scope in enumerate(scopes):
        active = tuple(
            operation
            for operation in scope.lane.operations
            if operation_geometry.operation_identity(operation) not in scope.predicated_operations
        )
        if homogeneous_decode:
            decoded = token.decode_batch(
                active,
                scope,
                cache_pool=cache_pool,
                device_products=device_products,
                request_tables=request_tables,
                model_runner=model_runner,
                runtime_states=runtime_states,
                sampling_group=sampling_group,
            )
            decoded_by_identity = dict(
                zip(
                    (operation_geometry.operation_identity(operation) for operation in active),
                    decoded,
                    strict=True,
                )
            )
        else:
            decoded_by_identity = {}
        for operation_index, operation in enumerate(scope.lane.operations):
            if operation_geometry.operation_identity(operation) in scope.predicated_operations:
                grouped[scope_index][operation_index] = _predicated_outcome(
                    operation,
                    scope,
                )
                continue
            identity = operation_geometry.operation_identity(operation)
            if identity in decoded_by_identity:
                grouped[scope_index][operation_index] = decoded_by_identity[identity]
                continue
            state = OperationState(operation=operation, lane=scope)
            locations[id(state)] = (scope_index, operation_index)
            states.append(state)
    errors = _run_ready_set(
        states,
        cache_pool=cache_pool,
        cache_registry=cache_registry,
        device_products=device_products,
        encoder_cache=encoder_cache,
        worker_info=worker_info,
        latent_pool=latent_pool,
        media_mux=media_mux,
        execution_model=execution_model,
        publication_transports=publication_transports,
        request_tables=request_tables,
        request_pool=request_pool,
        model_runner=model_runner,
        runtime_states=runtime_states,
        sampling_group=sampling_group,
        tokenizer=tokenizer,
        config=config,
    )
    for state in states:
        scope_index, operation_index = locations[id(state)]
        if state.outcome is not None:
            grouped[scope_index][operation_index] = state.outcome
    outcomes: dict[int, tuple[Outcome, ...]] = {}
    for scope, lane_outcomes in zip(scopes, grouped, strict=True):
        lane_id = scope.lane.lane_id
        if lane_id in errors:
            continue
        if any(outcome is None for outcome in lane_outcomes):
            raise RuntimeError("successful lane did not resolve every operation")
        outcomes[lane_id] = tuple(cast(Outcome, outcome) for outcome in lane_outcomes)
    return outcomes, errors


def _run_ready_set(
    states: list[OperationState],
    *,
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: VideoMuxCoordinator | None,
    execution_model: ExecutionModel,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    runtime_states: RuntimeStates | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> dict[int, BaseException]:
    """Advance dependency-ready operations through forward, sampling, integration, and actions."""

    from . import encode, flow, token, transfer

    # Products define the in-lane dependency graph; failures suppress only the
    # affected lane while independent lanes continue through the ready set.
    producers = {
        buffer: state
        for state in states
        for buffer in (
            *(output.buffer_id for output in state.operation.tensor_outputs()),
            *((state.operation.kv_output,) if state.operation.kv_output is not None else ()),
        )
    }
    errors: dict[int, BaseException] = {}

    def live(state: OperationState) -> bool:
        """Select unresolved operations whose lane has not recorded a failure."""

        return state.outcome is None and state.lane.lane.lane_id not in errors

    while any(live(state) for state in states):
        # Pack all ready neural work first. Flow prefix preparation forms an
        # exclusive wave because it may change the rows available to peers.
        forward: list[tuple[OperationState, object]] = []
        ready = tuple(
            state for state in states if live(state) and dependencies_ready(state, producers)
        )
        flow_ready = tuple(
            state for state in ready if state.operation.kind is PipelineStage.DENOISING
        )
        flow_ready_ids = {id(state) for state in flow_ready}
        for state in flow_ready:
            try:
                rows = _pack_state_forward(
                    state,
                    cache_registry=cache_registry,
                    device_products=device_products,
                    encoder_cache=encoder_cache,
                    latent_pool=latent_pool,
                    request_tables=request_tables,
                    model_runner=model_runner,
                    runtime_states=runtime_states,
                    tokenizer=tokenizer,
                    config=config,
                )
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
                continue
            forward.extend((state, row) for row in rows)
        preparing_flow_prefix = any(
            live(state) and state.phase == "prefix_pending" for state in flow_ready
        )
        if not preparing_flow_prefix:
            for state in ready:
                if id(state) in flow_ready_ids or not live(state):
                    continue
                try:
                    rows = _pack_state_forward(
                        state,
                        cache_registry=cache_registry,
                        device_products=device_products,
                        encoder_cache=encoder_cache,
                        latent_pool=latent_pool,
                        request_tables=request_tables,
                        model_runner=model_runner,
                        runtime_states=runtime_states,
                        tokenizer=tokenizer,
                        config=config,
                    )
                except BaseException as error:
                    errors[state.lane.lane.lane_id] = error
                    continue
                forward.extend((state, row) for row in rows)
        if forward:
            outputs = model_runner.run_wave(
                tuple((cast(ForwardRow, row), state.lane) for state, row in forward),
                cache=cache_pool,
                tables=request_tables,
                states=runtime_states,
            )
            aligned: dict[int, list[torch.Tensor]] = defaultdict(list)
            ordered: list[OperationState] = []
            for (state, _row), output in zip(forward, outputs, strict=True):
                if id(state) not in aligned:
                    ordered.append(state)
                aligned[id(state)].append(output)
            for state in ordered:
                try:
                    _consume_state_forward(
                        state,
                        tuple(aligned[id(state)]),
                        device_products=device_products,
                        encoder_cache=encoder_cache,
                        worker_info=worker_info,
                        latent_pool=latent_pool,
                        execution_model=execution_model,
                        publication_transports=publication_transports,
                        request_tables=request_tables,
                        runtime_states=runtime_states,
                        config=config,
                    )
                except BaseException as error:
                    errors[state.lane.lane.lane_id] = error
            continue

        # Sampling runs after its logits dependencies land and publishes token
        # products that can unlock later operations in the same lane.
        samples: dict[int, list[tuple[OperationState, SampleWork]]] = defaultdict(list)
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            sample = token.pack_sample(state)
            if sample is not None:
                samples[state.lane.lane.lane_id].append((state, cast(SampleWork, sample)))
        if samples:
            for lane_id, candidates in samples.items():
                state = candidates[0][0]
                try:
                    values = _sample_task_batch(
                        tuple(sample for _state, sample in candidates),
                        state.lane.completion,
                        device_products=device_products,
                        device_reads=tuple(state.lane.device_reads),
                        selection_broadcast=partial(broadcast_selection, sampling_group),
                    )
                    for (candidate, _sample), value in zip(candidates, values, strict=True):
                        token.consume_sample(
                            candidate,
                            value,
                            device_products=device_products,
                            execution_model=execution_model,
                            request_tables=request_tables,
                            model_runner=model_runner,
                            runtime_states=runtime_states,
                        )
                except BaseException as error:
                    errors[lane_id] = error
            continue

        # Integration consumes velocity outputs without another model call.
        # Other host/device actions run only when no forward or sample is ready.
        progressed = False
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            try:
                progressed = (
                    flow.integrate(
                        state,
                        worker_info=worker_info,
                        latent_pool=latent_pool,
                        publication_transports=publication_transports,
                        request_tables=request_tables,
                        config=config,
                    )
                    or progressed
                )
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
        if progressed:
            continue
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            try:
                progressed = (
                    encode.run_action(
                        state,
                        device_products=device_products,
                        encoder_cache=encoder_cache,
                        publication_transports=publication_transports,
                        model_runner=model_runner,
                    )
                    or progressed
                )
                progressed = (
                    transfer.run_action(
                        state,
                        cache_registry=cache_registry,
                        device_products=device_products,
                        encoder_cache=encoder_cache,
                        worker_info=worker_info,
                        latent_pool=latent_pool,
                        publication_transports=publication_transports,
                        request_tables=request_tables,
                        model_runner=model_runner,
                        config=config,
                    )
                    or progressed
                )
                progressed = (
                    run_video_action(
                        state,
                        device_products=device_products,
                        encoder_cache=encoder_cache,
                        media_mux=media_mux,
                        execution_model=execution_model,
                        publication_transports=publication_transports,
                        request_pool=request_pool,
                        model_runner=model_runner,
                    )
                    or progressed
                )
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
        if progressed:
            continue

        # Reaching a live fixed point indicates a dependency or state-machine
        # invariant violation rather than ordinary asynchronous waiting.
        if not any(live(state) for state in states):
            break
        blocked = tuple(
            operation_geometry.operation_identity(state.operation)
            for state in states
            if live(state)
        )
        raise RuntimeError(f"execution ready set made no progress: {blocked!r}")
    return errors


def _pack_state_forward(
    state: OperationState,
    *,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    latent_pool: LatentPool | None,
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
    runtime_states: RuntimeStates | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> tuple[object, ...]:
    """Dispatch an operation state to its token, flow, or encoder forward packer."""

    from . import encode, flow, token

    operation = state.operation
    if isinstance(operation.kind, ForwardMode):
        return token.pack_forward(
            state,
            encoder_cache=encoder_cache,
            request_tables=request_tables,
            model_runner=model_runner,
            runtime_states=runtime_states,
            tokenizer=tokenizer,
        )
    if operation.kind is PipelineStage.DENOISING and latent_pool is not None:
        return flow.pack_forward(
            state,
            cache_registry=cache_registry,
            latent_pool=latent_pool,
            request_tables=request_tables,
            model_runner=model_runner,
            tokenizer=tokenizer,
        )
    if operation.kind in {PipelineStage.VISION_ENCODING, PipelineStage.LATENT_ENCODING} or (
        operation.kind is PipelineStage.IMAGE_DECODING and latent_pool is not None
    ):
        return encode.pack_forward(
            state,
            device_products=device_products,
            latent_pool=latent_pool,
            model_runner=model_runner,
            config=config,
        )
    return ()


def _consume_state_forward(
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    execution_model: ExecutionModel,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    runtime_states: RuntimeStates | None,
    config: WorkerConfig,
) -> None:
    """Dispatch aligned model outputs to the operation family's consumer."""

    from . import encode, flow, token

    operation = state.operation
    if isinstance(operation.kind, ForwardMode):
        token.consume_forward(
            state,
            outputs,
            device_products=device_products,
            execution_model=execution_model,
            request_tables=request_tables,
            runtime_states=runtime_states,
        )
    elif operation.kind is PipelineStage.DENOISING and latent_pool is not None:
        flow.consume_forward(
            state, outputs, request_tables=request_tables, runtime_states=runtime_states
        )
    elif operation.kind in {PipelineStage.VISION_ENCODING, PipelineStage.LATENT_ENCODING} or (
        operation.kind is PipelineStage.IMAGE_DECODING and latent_pool is not None
    ):
        encode.consume_forward(
            state,
            outputs,
            device_products=device_products,
            encoder_cache=encoder_cache,
            worker_info=worker_info,
            publication_transports=publication_transports,
            config=config,
        )
    else:
        raise RuntimeError("model output has no operation consumer")


def _registration_error_lane(
    run_id: int, lane: RunLane, error: WorkerError, started: int, *, request_pool: RequestPool
) -> LaneResult:
    """Build aligned error outputs without committing candidate request state."""

    report = _build_error_lane(
        lane, False, error, started, WorkerForwardStats(), request_pool=request_pool
    )
    return report


def _error_lane(
    run_id: int, scope: LaneState, error: WorkerError, started: int, *, request_pool: RequestPool
) -> LaneResult:
    """Build a failed lane report and release its unpublished resources."""

    report = _build_error_lane(
        scope.lane,
        scope.registration_visible,
        error,
        scope.started_ns,
        _forward_stats(scope.observations, scope.component_us),
        request_pool=request_pool,
    )
    return report


def _build_error_lane(
    lane: RunLane,
    registration_visible: bool,
    error: WorkerError,
    started: int,
    forward_stats: WorkerForwardStats,
    *,
    request_pool: RequestPool,
) -> LaneResult:
    """Materialize one error completion per lane operation without mutating request state."""

    completion_code = _completion_error_code(error.code)
    records: list[ModelOutput] = []
    for operation in lane.operations:
        # Report execution coordinates only for the matching admitted epoch;
        # a stale descriptor cannot observe a replacement request slot.
        request = request_pool.peek(operation.request_key.request_id)
        if (
            request is None
            or request.request_key != operation.request_key
            or operation.predecessor is None
        ):
            runtime = RequestRuntime()
        else:
            runtime = request.current.runtime
        placeholder = ModelOutput(
            request_key=operation.request_key,
            op_id=operation.op_id,
            status=OpStatus.ERROR,
            product_generations=(),
            error_code=completion_code,
            timing_counters=TimingCounters(),
            kind=operation.kind,
            position=int(runtime.logical_position),
            kv_visible_len=int(runtime.kv_visible_len),
            kv_computed_len=int(runtime.kv_computed_len),
            num_completed_steps=int(runtime.flow_step),
            committed_tokens=(),
            finish_flags=FinishFlags(),
        )
        records.append(placeholder)
    return LaneResult(
        lane_id=lane.lane_id,
        completions=tuple(records),
        registration=RegistrationAck(visible=registration_visible),
        worker_exec_us=(time.perf_counter_ns() - started) // 1000,
        forward_stats=forward_stats,
    )
