"""Schedule dependency frontiers and dispatch ready worker calls.

``step.execute_batch`` calls ``dispatch_batch`` after ``reserve_outputs`` has
bound every call's pending output. Calls of one batch may consume products
that other calls of the same batch produce, so execution proceeds in
dependency frontiers: each round selects every live call whose in-batch
producers have completed. A frontier containing model-forward work runs
through the numerical helpers in ``forward``; any other frontier dispatches
each call to its owning module (``transfer``, ``diffusion``, ``image``,
``media_reader``, ``conditions``, ``host_media`` or ``media``). Outcomes stay
provisional until ``commit.commit_batch`` publishes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution import calls
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.calls import _predicated_outcome
from uniserve_worker.execution.forward import (
    forward_values,
    initialize_trajectories,
    integrate_predictions,
    prepare_diffusion_step,
    prepare_forward_rows,
    publish_forward_values,
)
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    ErrorCode,
    ForwardMode,
    MediaCall,
    TransferMode,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from uniserve.distributed.mesh import Communicator
    from uniserve_worker.config.execution import WorkerConfig
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.execution.request import RequestPool
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.protocol.worker_info import WorkerInfo
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport


def dispatch_batch(
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelExecutor,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> tuple[PendingOutput, ...]:
    """Execute the batch's active calls and align outcomes with its call order.

    Calls whose output ``reserve_outputs`` marked ``CallStatus.PREDICATED``
    are not executed; ``_predicated_outcome`` stages their outcome. A call
    of a request this rank refused (``RequestState.refusal``) is not
    executed either; ``refused_output`` reports the refusal. The returned
    tuple holds one outcome per call of ``state.batch.calls``, in that
    order.

    A failure raises and aborts the homogeneous batch; its owner discards all
    provisional outputs before reporting the error.
    """
    batch_calls = state.batch.calls

    # Mark device execution start on every device an active call uses.
    active = tuple(
        call
        for call in batch_calls
        if state.pending_output(call.request_key.request_id).status
        is not CallStatus.PREDICATED
    )
    with state.scope():
        for device in dict.fromkeys(
            device
            for call in active
            for device in model_runner.call_devices(call)
        ):
            state.output_buffer.begin_device(device)

    outcomes: list[PendingOutput | None] = [None] * len(batch_calls)
    scheduled: list[Call] = []
    locations: list[int] = []

    for call_index, call in enumerate(batch_calls):
        pending = state.pending_output(call.request_key.request_id)
        if pending.status is CallStatus.PREDICATED:
            outcomes[call_index] = _predicated_outcome(call, state=state)
            continue
        if pending.request.refusal is not None:
            outcomes[call_index] = refused_output(pending, tensor_store)
            continue
        locations.append(call_index)
        scheduled.append(call)

    # _execute_calls indexes outcomes by position in `scheduled`; `locations`
    # maps each position back to its index in the batch.
    completed = _execute_calls(
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

    for index, outcome in completed.items():
        outcomes[locations[index]] = outcome
    if any(outcome is None for outcome in outcomes):
        raise RuntimeError("successful batch did not resolve every call")
    return tuple(cast(PendingOutput, outcome) for outcome in outcomes)


def refused_output(
    pending: PendingOutput, tensor_store: TensorStore
) -> PendingOutput:
    """Report a call of a refused request without running it.

    The call fills none of its reserved writes, so they are abandoned, and
    its output reports ``ErrorCode.INVALID_REQUEST`` with the refusal's
    reason, which the engine returns to the client.
    """
    tensor_store.abandon_writes(tuple(pending.writes))
    pending.writes.clear()
    pending.status = CallStatus.ERROR
    pending.error_code = ErrorCode.INVALID_REQUEST
    pending.error_message = pending.request.refusal
    return pending


def _execute_ready_actions(
    frontier: tuple[int, ...],
    scheduled: tuple[Call, ...],
    outcomes: dict[int, PendingOutput],
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelExecutor,
    config: WorkerConfig,
) -> None:
    """Execute a frontier that contains no model-forward rows.

    Each call dispatches by kind to its owning module and runs inside
    ``state.scope()``, which makes the batch stream current when the batch
    has one. On a video worker (``video_postprocessor`` set), device
    computation also lands here: a request's vision and condition latent
    encodings through ``conditions``, and denoising and decode rounds
    through ``media.execute``. Calls that already have an outcome are
    skipped.

    Raises:
        WorkerError: ``invalid_descriptor`` when no module serves the call's
            kind on this worker.
    """
    from uniserve_worker.execution import (
        conditions,
        diffusion,
        host_media,
        image,
        media,
        media_reader,
        transfer,
    )
    from uniserve_worker.execution.host_media import HOST_MEDIA_CALLS

    for index in frontier:
        call = scheduled[index]
        if index in outcomes:
            continue

        with state.scope():
            if isinstance(call.kind, TransferMode):
                result = transfer.execute(
                    call,
                    kv_cache=kv_cache,
                    tensor_store=tensor_store,
                    latent_pool=latent_pool,
                    publication_transports=publication_transports,
                    request_tables=request_tables,
                    model_runner=model_runner,
                    state=state,
                )
            # Latent preparation belongs to diffusion only on image workers;
            # a video worker's latent preparation falls through to media.
            elif (
                call.kind is MediaCall.LATENT_PREPARATION
                and model_runner.image_builder is not None
            ):
                assert latent_pool is not None
                result = diffusion.prepare_latent(
                    call,
                    kv_cache=kv_cache,
                    worker_info=worker_info,
                    latent_pool=latent_pool,
                    publication_transports=publication_transports,
                    request_tables=request_tables,
                    model_runner=model_runner,
                    config=config,
                    state=state,
                )
            elif call.kind is MediaCall.MEDIA_READING:
                result = media_reader.execute(
                    call,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif (
                call.kind is MediaCall.VISION_ENCODING
                and model_runner.video_postprocessor is not None
            ):
                result = conditions.encode_vision(
                    call,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif (
                call.kind is MediaCall.LATENT_ENCODING
                and model_runner.video_postprocessor is not None
            ):
                result = conditions.encode_latents(
                    call,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif call.kind is MediaCall.TEXT_ENCODING:
                result = image.text(
                    call,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif call.kind in HOST_MEDIA_CALLS:
                result = host_media.execute(
                    call,
                    tensor_store=tensor_store,
                    media_mux=media_mux,
                    publication_transports=publication_transports,
                    transports=transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif model_runner.video_postprocessor is not None:
                result = media.execute(
                    call,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    request_pool=request_pool,
                    model_runner=model_runner,
                    state=state,
                )
            else:
                raise invalid_descriptor(f"unsupported call {call.kind!r}")
        outcomes[index] = result


def _execute_calls(
    scheduled: tuple[Call, ...],
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelExecutor,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> dict[int, PendingOutput]:
    """Execute product dependency frontiers with direct numerical algorithms.

    Each index addresses a position in ``scheduled``. A call is ready once
    every tensor or KV input produced by a call of ``scheduled`` has an
    outcome; inputs with no producer in ``scheduled`` do not gate readiness.
    An error raises and suppresses the rest of the batch. CFG prefixes
    precede their homogeneous denoiser calls.

    Returns:
        The outcome of every call, keyed by its position in ``scheduled``.

    Raises:
        RuntimeError: When live calls remain but none is ready.
    """
    producers = {
        buffer: index
        for index, call in enumerate(scheduled)
        for buffer in (
            *(output.buffer_id for output in call.tensor_outputs()),
            *((call.kv_output,) if call.kv_output is not None else ()),
        )
    }

    outcomes: dict[int, PendingOutput] = {}

    def live(index: int) -> bool:
        return index not in outcomes

    def ready(index: int) -> bool:
        call = scheduled[index]
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
                for index, call in enumerate(scheduled)
                if live(index)
            )
            raise RuntimeError(
                f"call products contain an unresolved dependency: {blocked!r}"
            )

        # KV-conditioned image denoising and decoding, and image encodings,
        # run as forward rows; a standalone denoiser's calls, a video
        # request's condition encodings included, are media actions. When
        # the frontier holds forward work, only those calls run in this
        # round; the other ready calls stay live and run in a later round.
        images = model_runner.image_builder is not None
        videos = model_runner.video_postprocessor is not None
        numerical = tuple(
            index
            for index in frontier
            if isinstance(scheduled[index].kind, ForwardMode)
            or (
                not videos
                and scheduled[index].kind
                in {MediaCall.VISION_ENCODING, MediaCall.LATENT_ENCODING}
            )
            or (
                images
                and scheduled[index].kind
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

        if images:
            # An image worker always holds its latent pool.
            assert latent_pool is not None
            trajectories, step_count = initialize_trajectories(
                numerical,
                scheduled,
                outcomes,
                state=state,
                kv_cache=kv_cache,
                latent_pool=latent_pool,
                request_tables=request_tables,
                model_runner=model_runner,
            )
        else:
            trajectories, step_count = {}, 1

        # step_count is the longest declared solver interval among the opened
        # trajectories; a trajectory with fewer steps drops out once its
        # interval ends.
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
                numerical,
                offset,
                step_inputs,
                trajectories,
                scheduled,
                outcomes,
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
                    retain_sampling=offset + 1 < step_count,
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
                outcomes,
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
                    state=state,
                    worker_info=worker_info,
                    latent_pool=latent_pool,
                    publication_transports=publication_transports,
                    request_tables=request_tables,
                    model_runner=model_runner,
                    config=config,
                )
    return outcomes
