"""Dispatch selected homogeneous calls to their numerical operations.

Rust selects active calls after their input batches have launched. Results
stay in the batch's reserved output rows until the native executor commits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.forward import execute_forward
from uniserve_worker.protocol.call import (
    Call,
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


def execute_calls(
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
) -> None:
    """Dispatch the homogeneous calls selected by the native executor."""
    kind = scheduled[0].kind
    images = model_runner.image_builder is not None
    videos = model_runner.video_postprocessor is not None
    if (
        isinstance(kind, ForwardMode)
        or (
            not videos
            and kind in {MediaCall.VISION_ENCODING, MediaCall.LATENT_ENCODING}
        )
        or (images and kind in {MediaCall.DENOISING, MediaCall.IMAGE_DECODING})
    ):
        execute_forward(
            scheduled,
            state=state,
            kv_cache=kv_cache,
            tensor_store=tensor_store,
            worker_info=worker_info,
            latent_pool=latent_pool,
            publication_transports=publication_transports,
            request_tables=request_tables,
            model_runner=model_runner,
            decode_state=decode_state,
            sampling_group=sampling_group,
            tokenizer=tokenizer,
            config=config,
        )
    else:
        _execute_actions(
            scheduled,
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


def _execute_actions(
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
    config: WorkerConfig,
) -> None:
    """Run transfer, encoding and media operations on the batch stream."""
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

    for call in scheduled:
        with state.scope():
            if isinstance(call.kind, TransferMode):
                transfer.execute(
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
                diffusion.prepare_latent(
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
                media_reader.execute(
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
                conditions.encode_vision(
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
                conditions.encode_latents(
                    call,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif call.kind is MediaCall.TEXT_ENCODING:
                image.text(
                    call,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif call.kind in HOST_MEDIA_CALLS:
                host_media.execute(
                    call,
                    tensor_store=tensor_store,
                    media_mux=media_mux,
                    publication_transports=publication_transports,
                    transports=transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif model_runner.video_postprocessor is not None:
                media.execute(
                    call,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    request_pool=request_pool,
                    model_runner=model_runner,
                    state=state,
                )
            else:
                raise invalid_descriptor(f"unsupported call {call.kind!r}")
