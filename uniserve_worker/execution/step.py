"""Execute and commit worker batches at their completion boundary."""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING

from uniserve_worker.execution import calls
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
from uniserve_worker.protocol.call import CallStatus, ErrorCode
from uniserve_worker.protocol.output import (
    FinishFlags,
    ForwardStats,
    RequestOutput,
    TimingCounters,
)
from uniserve_worker.protocol.tensor import DType

from .schedule import execute_completion

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.execution.output import OutputPool
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
    return ErrorCode.INVALID_CALL


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
    errors as the batch's completion so the worker can keep serving.
    """
    batch = state.batch
    if not state.inputs_ready():
        raise RuntimeError(
            "prepared execution was observed before transfer readiness"
        )

    predicate_values = state.predicate_values()
    started = time.perf_counter_ns()

    required_predicates = {
        calls.call_identity(call)
        for call in batch.calls
        if call.predicate is not None and call.predicate.dtype is DType.U8
    }
    if required_predicates != set(predicate_values):
        raise invalid_descriptor(
            "completion-predicated calls require exact prepared predicate "
            "values"
        )

    if not batch.calls:
        state.launched = True
        return

    completion = BatchState.COMPLETION
    try:
        _open_group(
            batch,
            completion,
            predicate_values,
            kv_cache=kv_cache,
            host_tasks=host_tasks,
            tensor_store=tensor_store,
            worker_info=worker_info,
            latent_pool=latent_pool,
            media_mux=media_mux,
            output_pool=output_pool,
            request_tables=request_tables,
            request_pool=request_pool,
            model_runner=model_runner,
            transfer_backends=transfer_backends,
            config=config,
            state=state,
        )
    except BaseException as error:
        classified = _classify_group_failure(
            completion,
            error,
            phase="completion group registration",
            state=state,
        )
        if propagate_errors or classified.fatal:
            raise classified

        _error_outputs(
            state,
            completion,
            classified,
            started,
            registration_visible=False,
            forward_stats=ForwardStats(),
            request_pool=request_pool,
        )
        return

    try:
        outcomes, execution_error = execute_completion(
            completion,
            kv_cache=kv_cache,
            tensor_store=tensor_store,
            worker_info=worker_info,
            latent_pool=latent_pool,
            media_mux=media_mux,
            publication_transports=publication_transports,
            transports=transfer_backends,
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
        outcomes, execution_error = None, error

    if outcomes is None:
        assert execution_error is not None
        classified = _classify_group_failure(
            completion,
            execution_error,
            phase="completion group execution",
            state=state,
        )
        _discard_group(
            completion,
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
            completion,
            classified,
            state.group_started_ns[completion],
            registration_visible=state.group_registered[completion],
            forward_stats=_forward_stats(
                state.group_forward_stats[completion],
                state.group_component_us[completion],
            ),
            request_pool=request_pool,
        )
        return

    try:
        _commit_group(
            batch.batch_id,
            completion,
            outcomes,
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
        if state.group_published[completion]:
            classified = _published_group_failure(
                completion,
                error,
                state=state,
            )
        else:
            classified = _classify_group_failure(
                completion,
                error,
                phase="completion group commit",
                state=state,
            )
            _discard_group(
                completion,
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
            completion,
            classified,
            state.group_started_ns[completion],
            registration_visible=state.group_registered[completion],
            forward_stats=_forward_stats(
                state.group_forward_stats[completion],
                state.group_component_us[completion],
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

    call and route context.
    """
    scheduled = tuple(
        (
            int(call.request_key.engine_id),
            int(call.request_key.request_id),
            int(call.request_key.request_epoch),
            call.call_id,
        )
        for call in state.batch.calls
    )

    # Attach request coordinates when the batch holds exactly one call.
    sole = state.batch.calls[0] if len(state.batch.calls) == 1 else None

    classified = classify(
        error,
        context=phase,
        phase=phase,
        calls=scheduled,
        req_id=None if sole is None else int(sole.request_key.request_id),
        call_id=None if sole is None else sole.call_id,
        call_kind=None if sole is None else sole.kind.value,
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
            int(call.request_key.engine_id),
            int(call.request_key.request_id),
            int(call.request_key.request_epoch),
            call.call_id,
        )
        for call in state.batch.calls
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
        calls=scheduled,
    )
    _log_group_failure(completion_group, classified, cause=error)
    return classified


def _log_group_failure(
    completion_group: int,
    error: WorkerError,
    *,
    cause: BaseException | None = None,
) -> None:
    """Log a completion-group failure, adding a diagnostic traceback."""
    capture_trace = should_capture_trace(error.code)
    log = logger.error if capture_trace else logger.warning
    log(
        "completion group failed: %s [code=%s group_id=%s route=%s calls=%s]",
        error.message,
        error.code,
        completion_group,
        0,
        error.calls,
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
    for call in state.batch.calls:
        # Report execution coordinates only for the matching admitted epoch;
        # a stale descriptor cannot observe a replacement request slot.
        request = request_pool.peek(call.request_key.request_id)
        if (
            request is None
            or request.request_key != call.request_key
            or state.predecessor(call) is None
        ):
            runtime = None
        else:
            runtime = request.accepted_progress

        placeholder = RequestOutput(
            request_key=call.request_key,
            call_id=call.call_id,
            status=CallStatus.ERROR,
            product_generations=(),
            error_code=completion_code,
            timing_counters=TimingCounters(),
            kind=call.kind,
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
