"""Dispatch selected homogeneous calls to their numerical operations.

Rust selects active calls after their input batches have launched. Results
stay in the batch's reserved output rows until the native executor commits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.call import (
    Call,
    MediaCall,
    TransferMode,
)

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.execution.request import RequestPool
    from uniserve_worker.media.mux import MediaMux
    from uniserve_worker.storage.block_tables import BlockTables
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
    latent_pool: LatentPool | None,
    media_mux: MediaMux | None,
    export_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    request_pool: RequestPool,
    model_runner: ModelExecutor,
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
                    tensor_store=tensor_store,
                    export_transports=export_transports,
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
                    latent_pool=latent_pool,
                    request_tables=request_tables,
                    model_runner=model_runner,
                    state=state,
                )
            elif call.kind is MediaCall.MEDIA_READING:
                media_reader.execute(
                    call,
                    tensor_store=tensor_store,
                    export_transports=export_transports,
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
                    export_transports=export_transports,
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
                    export_transports=export_transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif call.kind is MediaCall.TEXT_ENCODING:
                image.text(
                    call,
                    tensor_store=tensor_store,
                    export_transports=export_transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif call.kind in HOST_MEDIA_CALLS:
                host_media.execute(
                    call,
                    tensor_store=tensor_store,
                    media_mux=media_mux,
                    export_transports=export_transports,
                    transports=transports,
                    model_runner=model_runner,
                    state=state,
                )
            elif model_runner.video_postprocessor is not None:
                media.execute(
                    call,
                    tensor_store=tensor_store,
                    export_transports=export_transports,
                    request_pool=request_pool,
                    model_runner=model_runner,
                    state=state,
                )
            else:
                raise invalid_descriptor(f"unsupported call {call.kind!r}")
