"""Cross-resource retirement of explicitly released logical buffers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from uniserve_worker.execution.batch import BufferId, Finish, Free, Retire, Run

if TYPE_CHECKING:
    from uniserve_worker.runtime.cache_pool import CachePool
    from uniserve_worker.runtime.cache_publications import CachePublications
    from uniserve_worker.runtime.device_products import DeviceProducts
    from uniserve_worker.runtime.encoder_cache import EncoderCache
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.req_to_token_pool import ReqToTokenPool
    from uniserve_worker.runtime.request import RequestPool
    from uniserve_worker.runtime.runtime_states import RuntimeStates
    from uniserve_worker.transfer.publications import TransferPublications
    from uniserve_worker.transfer.tickets import Transport


def release_buffers(
    buffers: Sequence[BufferId],
    *,
    cache_pool: CachePool | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    latent_pool: LatentPool | None,
    transfer_publications: TransferPublications,
) -> None:
    """Revoke product acquisition without waiting for existing physical readers.

    Releases can arrive while earlier computation waits to reuse storage.
    They do not advance request state or acknowledge physical retirement;
    each owning store retains reader fences and transport registrations.
    """

    device_products.release_buffers(buffers)
    encoder_cache.release_buffers(buffers)
    if cache_pool is not None:
        cache_pool.release_buffers(buffers)
    if latent_pool is not None:
        latent_pool.release_buffers(buffers)
    transfer_publications.release(buffers)


def _apply_release_controls(
    batch: Run,
    *,
    before_execution: bool,
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    latent_pool: LatentPool | None,
    transfer_publications: TransferPublications,
) -> None:
    """Apply lifecycle releases in the phase required by their ownership contract."""

    consumed = {
        (reference.request_key, int(reference.producer_op_id))
        for operation in batch.operations
        for reference in (
            *operation.inputs,
            *(() if operation.predicate is None else (operation.predicate,)),
        )
    }
    releases = tuple(
        (operation.request_key, operation.parent.op_id)
        for operation in batch.operations
        if operation.parent is not None
        and operation.parent.op_id > 0
        and (
            ((operation.request_key, int(operation.parent.op_id)) not in consumed)
            == before_execution
        )
    )
    device_products.release_operations(releases)
    encoder_cache.release_operations(releases)
    if cache_registry is not None:
        released = cache_registry.release_operations(releases)
        if cache_pool is not None:
            cache_pool.release_buffers(released)
        for buffer in released:
            # Keep the registration until Free/Finish can observe its physical
            # retirement. Semantic release only revokes acquisition by new readers.
            transfer_publications.release((buffer,))
    if before_execution:
        freed = {command.buffer for command in batch.commands if isinstance(command, Free)}
        closed = {
            command.request_key: frozenset(command.retained_buffers) - freed
            for command in batch.commands
            if isinstance(command, (Finish, Retire))
        }
        closing_publications = tuple(
            buffer
            for request_key, retained in closed.items()
            for buffer in transfer_publications.retiring(
                requests=frozenset((request_key,)),
                retained=retained | freed,
            )
        )
        buffers = (*freed, *closing_publications)
        release_buffers(
            buffers,
            cache_pool=cache_pool,
            device_products=device_products,
            encoder_cache=encoder_cache,
            latent_pool=latent_pool,
            transfer_publications=transfer_publications,
        )
        if cache_pool is not None:
            for request_key, retained in closed.items():
                cache_pool.imports.cancel_requests(frozenset((request_key,)), retained=retained)
    if not before_execution:
        consumed_predicates = tuple(
            predicate.buffer_id
            for operation in batch.operations
            if (predicate := operation.predicate) is not None
            and (operation.parent is None or predicate.producer_op_id != operation.parent.op_id)
        )
        device_products.release_buffers(consumed_predicates)


def drop_request(
    request_id: int,
    *,
    retained: frozenset[BufferId] = frozenset(),
    cache_pool: CachePool | None,
    cache_registry: CachePublications | None,
    request_tables: ReqToTokenPool | None,
    request_pool: RequestPool,
    runtime_states: RuntimeStates | None,
    transfer_publications: TransferPublications,
    transfer_backends: Mapping[str, Transport],
) -> None:
    """Release request state and publications whose allocation ownership ends with it."""

    request = request_pool.peek(int(request_id))
    if request is not None:
        if cache_pool is not None:
            cache_pool.imports.cancel_requests(frozenset((request.request_key,)), retained=retained)
        if runtime_states is not None:
            runtime_states.release((int(request.request_pool_idx),))
        if request_tables is not None:
            request_tables.release((int(request.request_pool_idx),))
    if cache_registry is not None:
        cache_registry.drop(request_id)
    if request is not None and request_tables is not None:
        request_tables.release_prefixes(request.request_key)
    if not transfer_backends:
        return
    selected = transfer_publications.drop_request(request_id, retained)
    if cache_pool is not None:
        cache_pool.release_buffers(selected)
