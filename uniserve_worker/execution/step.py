"""Execute and commit worker batches at completion-group boundaries."""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING

from uniserve_worker.execution import operations
from uniserve_worker.execution.batch_state import BatchState
from uniserve_worker.execution.commit import _commit_group, _discard_group
from uniserve_worker.execution.prepare import _open_group
from uniserve_worker.foundation.errors import (
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
    should_capture_trace,
)
from uniserve_worker.profiling import _forward_stats
from uniserve_worker.protocol.operation import ErrorCode, OpStatus
from uniserve_worker.protocol.output import (
    FinishFlags,
    ForwardStats,
    RequestOutput,
    TimingCounters,
)
from uniserve_worker.protocol.tensor import DType

from .schedule import execute_groups

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.execution.output import OutputPool
    from uniserve_worker.media.buffers import MediaBuffers
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.runtime.block_tables import BlockTables
    from uniserve_worker.runtime.cache_manager import CacheManager
    from uniserve_worker.runtime.decode_state import DecodeState
    from uniserve_worker.runtime.host_lane import HostLane
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.tensor_store import TensorStore
    from uniserve_worker.transfer.tickets import Transport


logger = logging.getLogger(__name__)


def _completion_error_code(code: WorkerErrorCode) -> ErrorCode:
    """Map internal failure classes to their completion-wire error codes."""
    if code == WorkerErrorCode.RESOURCE_ERROR:
        return ErrorCode.RESOURCE_EXHAUSTED
    if code == WorkerErrorCode.COMPUTE_ERROR:
        return ErrorCode.COMPUTE_ERROR
    if code in {
        WorkerErrorCode.INVARIANT_VIOLATION,
        WorkerErrorCode.FATAL_WORKER_FAILURE,
    }:
        return ErrorCode.INTERNAL
    return ErrorCode.INVALID_OPERATION


def execute_batch(
    state: BatchState,
    *,
    propagate_errors: bool = False,
    kv_cache: CacheManager | None,
    host_tasks: HostLane,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    media_buffers: MediaBuffers | None,
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
        raise RuntimeError(
            "prepared execution was observed before transfer readiness"
        )

    predicate_values = state.predicate_values()
    started = time.perf_counter_ns()

    required_predicates = {
        operations.operation_identity(operation)
        for operation in batch.operations
        if operation.predicate is not None
        and operation.predicate.dtype is DType.U8
    }
    if required_predicates != set(predicate_values):
        raise invalid_descriptor(
            "completion-predicated operations require exact prepared predicate "
            "values"
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
                    host_tasks=host_tasks,
                    tensor_store=tensor_store,
                    worker_info=worker_info,
                    latent_pool=latent_pool,
                    media_mux=media_mux,
                    media_buffers=media_buffers,
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
        outcomes, execution_errors = execute_groups(
            tuple(completion_groups),
            kv_cache=kv_cache,
            tensor_store=tensor_store,
            worker_info=worker_info,
            latent_pool=latent_pool,
            media_mux=media_mux,
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
            completion_group
            for completion_group in groups
            if completion_group in execution_errors
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
                batch.batch_id,
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
    """Classify a pre-publication completion group failure with complete.

    operation and route context.
    """
    scheduled = tuple(
        (
            int(operation.request_key.engine_id),
            int(operation.request_key.request_id),
            int(operation.request_key.request_epoch),
            operation.op_id,
        )
        for operation in state.group_operations(completion_group)
    )

    # Attach request coordinates when the group holds exactly one operation.
    sole = (
        state.group_operations(completion_group)[0]
        if len(state.group_operations(completion_group)) == 1
        else None
    )

    classified = classify(
        error,
        context=phase,
        phase=phase,
        operations=scheduled,
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
    """Classify a post-visibility publication failure as a fatal invariant.

    violation.
    """
    scheduled = tuple(
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
        message=(
            f"completion_group publication failed after visibility "
            f"began: {error}"
        ),
        fatal=True,
        phase="completion group publication",
        route=str(0),
        operations=scheduled,
    )
    _log_group_failure(completion_group, classified, cause=error)
    return classified


def _log_group_failure(
    completion_group: int,
    error: WorkerError,
    *,
    cause: BaseException | None = None,
) -> None:
    """Log a classified completion group failure with traceback only for.

    diagnostic error classes.
    """
    capture_trace = should_capture_trace(error.code)
    log = logger.error if capture_trace else logger.warning
    log(
        "completion group failed: %s [code=%s group_id=%s route=%s "
        "operations=%s]",
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
    """Record final errors at the owning completion boundary without accepting.

    progress.
    """
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
            kv_visible_len=(
                0 if runtime is None else int(runtime.kv_visible_len)
            ),
            kv_computed_len=(
                0 if runtime is None else int(runtime.kv_computed_len)
            ),
            num_completed_steps=(
                0 if runtime is None else int(runtime.flow_step)
            ),
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
