"""Schedule dependency frontiers and dispatch ready worker operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from uniserve_worker.execution import operations
from uniserve_worker.execution.batch_state import BatchState
from uniserve_worker.execution.operations import _predicated_outcome
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.protocol.operation import (
    ForwardMode,
    OpStatus,
    PipelineStage,
    ScheduledRequest,
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


def execute_groups(
    completion_groups: tuple[int, ...],
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    decode_state: DecodeState | None,
    sampling_group: Communicator | None,
    tokenizer: PreTrainedTokenizerBase | None,
    config: WorkerConfig,
) -> tuple[dict[int, tuple[PendingOutput, ...]], dict[int, BaseException]]:
    """Execute active operations across completion groups and align outcomes.

    with original completion group order.
    """
    for completion_group in completion_groups:
        active = tuple(
            operation
            for operation in state.group_operations(completion_group)
            if state.pending_output(
                completion_group, operation.request_key.request_id
            ).status
            is not OpStatus.PREDICATED
        )
        with state.group_scope(completion_group):
            for device in dict.fromkeys(
                device
                for operation in active
                for device in model_runner.operation_devices(operation)
            ):
                state.group_buffers[completion_group].begin_device(device)

    grouped: list[list[PendingOutput | None]] = [
        [None] * len(state.group_operations(completion_group))
        for completion_group in completion_groups
    ]
    scheduled: list[tuple[ScheduledRequest, int]] = []
    locations: list[tuple[int, int]] = []

    for group_index, completion_group in enumerate(completion_groups):
        for operation_index, operation in enumerate(
            state.group_operations(completion_group)
        ):
            if (
                state.pending_output(
                    completion_group, operation.request_key.request_id
                ).status
                is OpStatus.PREDICATED
            ):
                grouped[group_index][operation_index] = _predicated_outcome(
                    operation, completion_group, state=state
                )
                continue
            locations.append((group_index, operation_index))
            scheduled.append((operation, completion_group))

    completed, errors = _execute_operations(
        tuple(scheduled),
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

    for index, outcome in completed.items():
        group_index, operation_index = locations[index]
        grouped[group_index][operation_index] = outcome

    outcomes: dict[int, tuple[PendingOutput, ...]] = {}
    for completion_group, group_outcomes in zip(
        completion_groups, grouped, strict=True
    ):
        group_id = completion_group
        if group_id in errors:
            continue
        if any(outcome is None for outcome in group_outcomes):
            raise RuntimeError(
                "successful completion group did not resolve every operation"
            )
        outcomes[group_id] = tuple(
            cast(PendingOutput, outcome) for outcome in group_outcomes
        )
    return outcomes, errors


def _execute_ready_actions(
    frontier: tuple[int, ...],
    scheduled: tuple[tuple[ScheduledRequest, int], ...],
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
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> None:
    """Execute a dependency frontier that contains no numerical model calls."""
    from . import encode, flow, transfer, video

    for index in frontier:
        operation, completion_group = scheduled[index]
        if index in outcomes or completion_group in errors:
            continue

        try:
            with state.group_scope(completion_group):
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
                elif model_runner.video_postprocessor is not None:
                    result = video.execute(
                        operation,
                        completion_group,
                        tensor_store=tensor_store,
                        media_mux=media_mux,
                        publication_transports=publication_transports,
                        request_pool=request_pool,
                        model_runner=model_runner,
                        state=state,
                    )
                else:
                    raise invalid_descriptor(
                        f"unsupported operation {operation.kind!r}"
                    )
            outcomes[index] = result
        except BaseException as error:
            errors[completion_group] = error


def _execute_operations(
    scheduled: tuple[tuple[ScheduledRequest, int], ...],
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
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
    groups continue. CFG prefixes precede their homogeneous denoiser calls.
    """
    producers = {
        buffer: index
        for index, (operation, _scope) in enumerate(scheduled)
        for buffer in (
            *(output.buffer_id for output in operation.tensor_outputs()),
            *(
                (operation.kv_output,)
                if operation.kv_output is not None
                else ()
            ),
        )
    }

    outcomes: dict[int, PendingOutput] = {}
    errors: dict[int, BaseException] = {}

    def live(index: int) -> bool:
        return index not in outcomes and scheduled[index][1] not in errors

    def ready(index: int) -> bool:
        operation = scheduled[index][0]
        return all(
            producer in outcomes
            for buffer in (
                *(
                    reference.buffer_id
                    for reference in operation.tensor_inputs()
                ),
                *(
                    (operation.kv_input,)
                    if operation.kv_input is not None
                    else ()
                ),
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
                operations.operation_identity(operation)
                for index, (operation, _scope) in enumerate(scheduled)
                if live(index)
            )
            raise RuntimeError(
                f"operation products contain an unresolved dependency: "
                f"{blocked!r}"
            )

        numerical = tuple(
            index
            for index in frontier
            if isinstance(scheduled[index][0].kind, ForwardMode)
            or scheduled[index][0].kind
            in {PipelineStage.VISION_ENCODING, PipelineStage.LATENT_ENCODING}
            or (
                latent_pool is not None
                and scheduled[index][0].kind
                in {
                    PipelineStage.DENOISING,
                    PipelineStage.IMAGE_DECODING,
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
