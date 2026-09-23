"""Schedule dependency frontiers and dispatch ready worker calls."""

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

    A failure aborts the homogeneous batch; its owner discards all provisional
    outputs before reporting the error.
    """
    batch_calls = state.batch.calls
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
        if (
            state.pending_output(call.request_key.request_id).status
            is CallStatus.PREDICATED
        ):
            outcomes[call_index] = _predicated_outcome(call, state=state)
            continue
        locations.append(call_index)
        scheduled.append(call)

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
    """Execute a dependency frontier that contains no numerical model calls."""
    from uniserve_worker.execution import (
        diffusion,
        host_media,
        image,
        media,
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

    Each index addresses an original call. Only completed products unlock
    successors; an error suppresses the rest of the completion. CFG prefixes
    precede their homogeneous denoiser calls.
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

        # KV-conditioned image denoising and decoding run as forward rows; a
        # standalone denoiser's calls are media actions.
        images = model_runner.image_builder is not None
        numerical = tuple(
            index
            for index in frontier
            if isinstance(scheduled[index].kind, ForwardMode)
            or scheduled[index].kind
            in {MediaCall.VISION_ENCODING, MediaCall.LATENT_ENCODING}
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
