"""Execute and commit worker batches at their completion boundary.

``execute_batch`` is the worker executor's entry point for one prepared batch.
It runs three phases in order: ``prepare.reserve_outputs`` binds completion
storage and provisional resources, ``schedule.dispatch_batch`` runs the calls,
and ``commit.commit_batch`` makes resources and request progress visible.
Every call of a batch has the same kind and component, and a failure in any
phase fails the whole batch. Nonfatal failures become ``CallStatus.ERROR``
outputs for every call; fatal failures, and every failure when the caller asks
for propagation, are raised as a classified ``WorkerError``.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING

from uniserve_worker.errors import (
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
    should_capture_trace,
)
from uniserve_worker.execution import calls
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.commit import commit_batch, discard_batch
from uniserve_worker.execution.prepare import reserve_outputs
from uniserve_worker.execution.schedule import dispatch_batch
from uniserve_worker.profiling import _forward_stats
from uniserve_worker.protocol.call import CallStatus, ErrorCode
from uniserve_worker.protocol.output import (
    FinishFlags,
    ForwardStats,
    RequestOutput,
    TimingCounters,
)
from uniserve_worker.protocol.tensor import DType

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.config.execution import WorkerConfig
    from uniserve_worker.execution.host import HostLane
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.execution.request import RequestPool
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.protocol.worker_info import WorkerInfo
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.output import OutputPool
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport


logger = logging.getLogger(__name__)


def _completion_error_code(code: WorkerErrorCode) -> ErrorCode:
    """Map internal failure classes to their completion-wire error codes.

    Every class without its own case, including descriptor, setup, input and
    scheduler errors, maps to ``ErrorCode.INVALID_CALL``.
    """
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
    kv_cache: KVCacheManager | None,
    host_tasks: HostLane,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    output_pool: OutputPool,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelExecutor,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    transfer_backends: Mapping[str, Transport],
    config: WorkerConfig,
) -> None:
    """Reserve, execute and commit one prepared batch.

    The caller must have observed ``state.inputs.ready()``; a batch whose
    inputs are not ready raises ``RuntimeError``. The prepared predicate
    values must cover exactly the calls with a U8 completion predicate, or
    ``invalid_descriptor`` is raised. A batch without calls is only marked
    launched and records no outputs.

    On success the batch's outputs are recorded by ``commit_batch``. On a
    nonfatal failure every call receives an error output through
    ``_error_outputs`` and the function returns normally, so the worker keeps
    serving. With ``propagate_errors`` (as warmup submits) or a fatal
    classification, the classified ``WorkerError`` is raised instead.
    Provisional resources are released before either outcome unless the batch
    had already begun publication.
    """
    batch = state.batch
    if not state.inputs.ready():
        raise RuntimeError(
            "prepared execution was observed before transfer readiness"
        )

    predicate_values = state.predicate_values()
    started = time.perf_counter_ns()

    required_predicates = {
        calls.call_identity(call)
        for call in batch.calls
        if call.predicate is not None
        and call.predicate.dtype is DType.U8
        and not calls.device_gated(call)
    }
    if required_predicates != set(predicate_values):
        raise invalid_descriptor(
            "completion-predicated calls require exact prepared predicate "
            "values"
        )

    if not batch.calls:
        return

    # reserve_outputs releases what it bound before re-raising (see its
    # docstring), so this failure path only classifies and reports.
    try:
        reserve_outputs(
            batch,
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
        classified = _classify_failure(
            error,
            phase="batch registration",
            state=state,
        )
        if propagate_errors or classified.fatal:
            raise classified

        # state.started_ns is set only by BatchState.bind_outputs, which a
        # registration failure may precede, so this path times from `started`.
        _error_outputs(
            state,
            classified,
            started,
            forward_stats=ForwardStats(),
            request_pool=request_pool,
        )
        return

    try:
        outcomes = dispatch_batch(
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
        classified = _classify_failure(
            execution_error,
            phase="batch execution",
            state=state,
        )
        discard_batch(
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
            classified,
            state.started_ns,
            forward_stats=_forward_stats(
                state.forward_stats,
                state.component_us,
            ),
            request_pool=request_pool,
        )
        return

    try:
        commit_batch(
            batch.batch_id,
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
        # Once commit_batch sets state.published, resources may already be
        # visible and cannot be discarded; the failure is always fatal, so
        # the error outputs below are recorded only for unpublished batches.
        if state.published:
            classified = _publication_failure(
                error,
                state=state,
            )
        else:
            classified = _classify_failure(
                error,
                phase="batch commit",
                state=state,
            )
            discard_batch(
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
            classified,
            state.started_ns,
            forward_stats=_forward_stats(
                state.forward_stats,
                state.component_us,
            ),
            request_pool=request_pool,
        )


def _classify_failure(
    error: BaseException,
    *,
    phase: str,
    state: BatchState,
) -> WorkerError:
    """Classify and log a pre-publication batch failure.

    An error built from any other exception carries the coordinates of every
    call of the batch, plus request and call identity when the batch holds
    exactly one call. Its route is the batch's call kind (`BatchState.route`).
    ``classify`` returns an existing ``WorkerError`` itself and fills only
    its None-valued fields, so a route the raiser set (a model forward's
    mode), ``calls`` and ``fatal`` stay as raised.
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

    sole = state.batch.calls[0] if len(state.batch.calls) == 1 else None

    classified = classify(
        error,
        context=phase,
        phase=phase,
        calls=scheduled,
        req_id=None if sole is None else int(sole.request_key.request_id),
        call_id=None if sole is None else sole.call_id,
        call_kind=None if sole is None else sole.kind.value,
        route=state.route,
    )
    _log_failure(classified, cause=error)
    return classified


def _publication_failure(
    error: BaseException,
    *,
    state: BatchState,
) -> WorkerError:
    """Classify a failure after publication began as a fatal invariant error.

    Part of the batch's state may already be visible to successors, so it
    cannot be discarded and the error is always fatal. It carries the
    batch's call kind as its route and every call's coordinates.
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
        message=(f"batch publication failed after visibility began: {error}"),
        fatal=True,
        phase="batch publication",
        route=state.route,
        calls=scheduled,
    )
    _log_failure(classified, cause=error)
    return classified


def _log_failure(
    error: WorkerError,
    *,
    cause: BaseException | None = None,
) -> None:
    """Log a batch failure and clear the frames of its exception chain.

    Error classes for which ``should_capture_trace`` holds log at error level
    with the cause's traceback; the others log a warning without it.
    """
    capture_trace = should_capture_trace(error.code)
    log = logger.error if capture_trace else logger.warning
    log(
        "batch failed: %s [code=%s route=%s calls=%s]",
        error.message,
        error.code,
        error.route,
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
    error: WorkerError,
    started: int,
    *,
    forward_stats: ForwardStats,
    request_pool: RequestPool,
) -> None:
    """Record an error output for every call without accepting progress.

    Each output reports the request's already accepted progress, or zero
    coordinates when the call has no predecessor or the request pool no
    longer holds the call's exact request key.
    """
    completion_code = _completion_error_code(error.code)
    records: list[RequestOutput] = []
    for call in state.batch.calls:
        # Report accepted coordinates only for the matching admitted epoch;
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
        tuple(records),
        execution_us=(time.perf_counter_ns() - started) // 1000,
        stats=forward_stats,
    )
