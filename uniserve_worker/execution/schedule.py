"""Schedule dependency frontiers and dispatch ready worker calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from uniserve_worker.execution import calls
from uniserve_worker.execution.batch_state import BatchState
from uniserve_worker.execution.calls import _predicated_outcome
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    ForwardMode,
    MediaCall,
    TransferMode,
)

from .forward import (
    forward_values,
    initialize_trajectories,
    integrate_predictions,
    prepare_diffusion_step,
    prepare_forward_rows,
    publish_forward_values,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.runtime.block_tables import BlockTables
    from uniserve_worker.runtime.cache_manager import CacheManager
    from uniserve_worker.runtime.decode_state import DecodeState
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.tensor_store import TensorStore
    from uniserve_worker.transfer.tickets import Transport


def execute_completion(
    completion_group: int,
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> tuple[tuple[PendingOutput, ...] | None, BaseException | None]:
    """Execute the batch's active calls and align outcomes with its call order.

    Returns the pending outputs in call order and no error, or no outputs and
    the error that failed the completion.
    """
    batch_calls = state.batch.calls
    active = tuple(
        call
        for call in batch_calls
        if state.pending_output(
            completion_group, call.request_key.request_id
        ).status
        is not CallStatus.PREDICATED
    )
    with state.group_scope(completion_group):
        for device in dict.fromkeys(
            device
            for call in active
            for device in model_runner.call_devices(call)
        ):
            state.group_buffers[completion_group].begin_device(device)

    outcomes: list[PendingOutput | None] = [None] * len(batch_calls)
    scheduled: list[tuple[Call, int]] = []
    locations: list[int] = []

    for call_index, call in enumerate(batch_calls):
        if (
            state.pending_output(
                completion_group, call.request_key.request_id
            ).status
            is CallStatus.PREDICATED
        ):
            outcomes[call_index] = _predicated_outcome(
                call, completion_group, state=state
            )
            continue
        locations.append(call_index)
        scheduled.append((call, completion_group))

    completed, errors = _execute_calls(
        tuple(scheduled),
        kv_cache=kv_cache,
        tensor_store=tensor_store,
        worker_info=worker_info,
        latent_pool=latent_pool,
        media_mux=media_mux,
        publication_transports=publication_transports,
        transports=transports,
        request_tables=request_tables,
        request_pool=request_pool,
        model_runner=model_runner,
        decode_state=decode_state,
        sampling_group=sampling_group,
        tokenizer=tokenizer,
        config=config,
        state=state,
    )

    if (error := errors.get(completion_group)) is not None:
        return None, error

    for index, outcome in completed.items():
        outcomes[locations[index]] = outcome
    if any(outcome is None for outcome in outcomes):
        raise RuntimeError(
            "successful completion group did not resolve every call"
        )
    return tuple(cast(PendingOutput, outcome) for outcome in outcomes), None


def _execute_ready_actions(
    frontier: tuple[int, ...],
    scheduled: tuple[tuple[Call, int], ...],
    outcomes: dict[int, PendingOutput],
    errors: dict[int, BaseException],
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> None:
    """Execute a dependency frontier that contains no numerical model calls."""
    from . import encode, flow, host_media, transfer, video
    from .host_media import HOST_MEDIA_CALLS

    for index in frontier:
        call, completion_group = scheduled[index]
        if index in outcomes or completion_group in errors:
            continue

        try:
            with state.group_scope(completion_group):
                if isinstance(call.kind, TransferMode):
                    result = transfer.execute(
                        call,
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
                    call.kind is MediaCall.LATENT_PREPARATION
                    and latent_pool is not None
                ):
                    result = flow.prepare_latent(
                        call,
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
                elif call.kind is MediaCall.TEXT_ENCODING:
                    result = encode.text(
                        call,
                        completion_group,
                        tensor_store=tensor_store,
                        publication_transports=publication_transports,
                        model_runner=model_runner,
                        state=state,
                    )
                elif call.kind in HOST_MEDIA_CALLS:
                    result = host_media.execute(
                        call,
                        completion_group,
                        tensor_store=tensor_store,
                        media_mux=media_mux,
                        publication_transports=publication_transports,
                        transports=transports,
                        model_runner=model_runner,
                        state=state,
                    )
                elif model_runner.video_postprocessor is not None:
                    result = video.execute(
                        call,
                        completion_group,
                        tensor_store=tensor_store,
                        publication_transports=publication_transports,
                        request_pool=request_pool,
                        model_runner=model_runner,
                        state=state,
                    )
                else:
                    raise invalid_descriptor(f"unsupported call {call.kind!r}")
            outcomes[index] = result
        except BaseException as error:
            errors[completion_group] = error


def _execute_calls(
    scheduled: tuple[tuple[Call, int], ...],
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> tuple[dict[int, PendingOutput], dict[int, BaseException]]:
    """Execute product dependency frontiers with direct numerical algorithms.

    Each index addresses an original call. Only completed products unlock
    successors; an error suppresses the rest of the completion. CFG prefixes
    precede their homogeneous denoiser calls.
    """
    producers = {
        buffer: index
        for index, (call, _scope) in enumerate(scheduled)
        for buffer in (
            *(output.buffer_id for output in call.tensor_outputs()),
            *((call.kv_output,) if call.kv_output is not None else ()),
        )
    }

    outcomes: dict[int, PendingOutput] = {}
    errors: dict[int, BaseException] = {}

    def live(index: int) -> bool:
        return index not in outcomes and scheduled[index][1] not in errors

    def ready(index: int) -> bool:
        call = scheduled[index][0]
        return all(
            producer in outcomes
            for buffer in (
                *(reference.buffer_id for reference in call.tensor_inputs()),
                *((call.kv_input,) if call.kv_input is not None else ()),
            )
            if (producer := producers.get(buffer)) is not None
        )

    while any(live(index) for index in range(len(scheduled))):
        frontier = tuple(
            index
            for index in range(len(scheduled))
            if live(index) and ready(index)
        )
        if not frontier:
            blocked = tuple(
                calls.call_identity(call)
                for index, (call, _scope) in enumerate(scheduled)
                if live(index)
            )
            raise RuntimeError(
                f"call products contain an unresolved dependency: {blocked!r}"
            )

        numerical = tuple(
            index
            for index in frontier
            if isinstance(scheduled[index][0].kind, ForwardMode)
            or scheduled[index][0].kind
            in {MediaCall.VISION_ENCODING, MediaCall.LATENT_ENCODING}
            or (
                latent_pool is not None
                and scheduled[index][0].kind
                in {
                    MediaCall.DENOISING,
                    MediaCall.IMAGE_DECODING,
                }
            )
        )

        if not numerical:
            _execute_ready_actions(
                frontier,
                scheduled,
                outcomes,
                errors,
                state=state,
                kv_cache=kv_cache,
                tensor_store=tensor_store,
                worker_info=worker_info,
                latent_pool=latent_pool,
                media_mux=media_mux,
                publication_transports=publication_transports,
                transports=transports,
                request_tables=request_tables,
                request_pool=request_pool,
                model_runner=model_runner,
                config=config,
            )
            continue

        trajectories, step_count = (
            initialize_trajectories(
                numerical,
                scheduled,
                outcomes,
                errors,
                state=state,
                kv_cache=kv_cache,
                latent_pool=latent_pool,
                request_tables=request_tables,
                model_runner=model_runner,
            )
            if latent_pool is not None
            else ({}, 1)
        )
        for offset in range(step_count):
            # The numerical schedule is local to this loop. Accepted request
            # progress is published only after the complete declared interval.
            if trajectories:
                assert latent_pool is not None
                step_inputs = prepare_diffusion_step(
                    offset,
                    trajectories,
                    scheduled,
                    outcomes,
                    errors,
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

            forward, images = prepare_forward_rows(
                numerical,
                offset,
                step_inputs,
                trajectories,
                scheduled,
                outcomes,
                errors,
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
                    tuple(
                        (task, scheduled[index][0], scheduled[index][1])
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

            predictions = publish_forward_values(
                forward,
                values,
                images,
                trajectories,
                scheduled,
                outcomes,
                errors,
                state=state,
                tensor_store=tensor_store,
                worker_info=worker_info,
                publication_transports=publication_transports,
                model_runner=model_runner,
                request_tables=request_tables,
                decode_state=decode_state,
                sampling_group=sampling_group,
                config=config,
            )

            if predictions:
                assert latent_pool is not None
                integrate_predictions(
                    predictions,
                    step_inputs,
                    trajectories,
                    offset,
                    scheduled,
                    outcomes,
                    errors,
                    state=state,
                    worker_info=worker_info,
                    latent_pool=latent_pool,
                    publication_transports=publication_transports,
                    request_tables=request_tables,
                    model_runner=model_runner,
                    config=config,
                )
    return outcomes, errors
