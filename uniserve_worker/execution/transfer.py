"""Product, KV, and latent movement with no model call."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch

from uniserve import _slices
from uniserve_worker.execution import calls as calls
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    unsupported_setup,
)
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.call import Call, TransferMode
from uniserve_worker.protocol.tensor import TensorRef
from uniserve_worker.protocol.transfer import (
    DeviceProductTransferValue,
    EncoderTransferValue,
    LatentTransferValue,
    Locator,
    TensorTransfer,
    TransferValue,
)
from uniserve_worker.runtime.latent_pool import LatentExport
from uniserve_worker.runtime.tensor_store import (
    FeatureMetadata,
    ImageMetadata,
    device_product_storage,
)
from uniserve_worker.transfer.tickets import publish_tensor

from .batch_state import BatchState

if TYPE_CHECKING:
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.runtime.block_tables import BlockTables
    from uniserve_worker.runtime.cache_manager import CacheManager
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.tensor_store import TensorStore
    from uniserve_worker.transfer.tickets import Transport


def execute(
    call: Call,
    completion_group: int,
    *,
    state: BatchState,
    kv_cache: CacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
) -> PendingOutput:
    """Execute a tensor transfer or KV publication/install call and stage.

    its result.
    """
    from . import encode

    transports = publication_transports
    if not transports:
        raise unsupported_setup(
            "product transfer requires a configured transport"
        )
    request_id = call.request_key.request_id
    mode = call.kind
    if mode is TransferMode.KV_PUBLISH:
        publications = kv_cache
        if publications is None:
            raise invalid_descriptor("KV publication requires cache storage")
        output = call.kv_output
        if output is None:
            raise invalid_descriptor(
                "KV publication requires a cache output identity"
            )
        request = state.pending_output(completion_group, request_id)
        cache = calls.cache_coordinates(request, tables=request_tables)
        expected_base = publications.destination_base(call.request_key, "gen")
        snapshot = publications.publish(
            request_pool_idx=request.request.request_pool_idx,
            group_id=cache[1],
            visible_length=cache[2],
            destination="gen",
            expected_base=expected_base,
            buffer=output,
            transports=transports,
            consumers=call.consumer_slots,
        )
        request.cache_publication = (output, snapshot)

        for tensor in snapshot.tensors:
            for locator in tensor.locations:
                request.exported_locators.append(locator)

        request.cache_exports[output] = tuple(
            (transports[location.backend], location)
            for tensor in snapshot.tensors
            for location in tensor.locations
        )

        outcome = encode.non_state_outcome(call, completion_group, state=state)
        outcome.kv_output = snapshot
    elif mode is TransferMode.KV_INSTALL:
        publications = kv_cache
        if publications is None:
            raise invalid_descriptor("KV installation requires cache storage")
        source = call.kv_input
        output = call.kv_output
        if source is None or output is None:
            raise invalid_descriptor(
                "KV installation requires source and output identities"
            )
        request = state.pending_output(completion_group, request_id)
        cache = calls.cache_coordinates(request, tables=request_tables)
        write = state.cache_imports.get(source)
        if write is None:
            raise invalid_descriptor(
                "KV installation has no reserved physical input"
            )
        installed = publications.install(
            request_pool_idx=request.request.request_pool_idx,
            group_id=cache[1],
            request_key=call.request_key,
            source=source,
            installed_buffer=output,
            write=write,
        )
        request.cache_installation = (source, output, installed)

        outcome = encode.non_state_outcome(call, completion_group, state=state)
        if outcome.progress is not None:
            outcome.progress = replace(
                outcome.progress,
                kv_visible_len=int(installed.published_extent),
                kv_computed_len=int(installed.published_extent),
            )
    else:
        inputs = call.tensor_inputs()
        outputs = call.tensor_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise invalid_descriptor(
                "product transfer requires one physical input and one output"
            )
        if call.latent_input is not None:
            if latent_pool is None:
                raise unsupported_setup(
                    "latent transfer requires a physical latent pool"
                )
            tensor_publication = _publish_current_latent(
                call,
                inputs[0],
                outputs[0],
                completion_group,
                latent_pool=latent_pool,
                publication_transports=publication_transports,
                state=state,
            )
        else:
            value, metadata = fetch_product(
                call,
                completion_group,
                tensor_store=tensor_store,
                model_runner=model_runner,
                state=state,
            )
            tensor_publication = publish_product(
                outputs[0],
                value,
                metadata,
                completion_group,
                tensor_store=tensor_store,
                publication_transports=publication_transports,
                state=state,
            )
        outcome = encode.non_state_outcome(
            call,
            completion_group,
            products=(tensor_publication,),
            state=state,
        )
    return outcome


def _publish_current_latent(
    call: Call,
    reference: TensorRef,
    product: TensorRef,
    completion_group: int,
    *,
    state: BatchState,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
) -> TensorPublication:
    """Publish the committed trajectory's current latent pages as a product."""
    if product != call.latent_output:
        raise invalid_descriptor(
            "product transfer changes the physical product kind"
        )

    row = state.pending_output(completion_group, call.request_key.request_id)
    params = row.input_latent_params
    staging = row.latent_staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory call has no staged latent inputs")

    source = latent_pool.reserve_current_publication(
        product,
        request_pool_idx=row.request.request_pool_idx,
        page_table=params.page_table,
        generation=reference.generation,
        step=params.start_step,
        latent_units=params.latent_units,
        height=params.height,
        width=params.width,
    )

    return publish_latent_source(
        product,
        source,
        row,
        step=params.start_step,
        completion_group=completion_group,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
        state=state,
    )


def publish_latent_source(
    product: TensorRef,
    source: LatentExport,
    row: PendingOutput,
    *,
    state: BatchState,
    step: int,
    completion_group: int,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
) -> TensorPublication:
    """Register exact latent page spans and retain their bank for every.

    reader.
    """
    params = row.input_latent_params
    if params is None:
        raise invalid_descriptor("latent publication has no staged parameters")

    request = state.pending_output(
        completion_group, product.request_key.request_id
    )
    transports = publication_transports
    if not transports:
        raise unsupported_setup(
            "latent publication requires a configured transport"
        )

    pool = latent_pool
    shape = (params.latent_units, pool.latent_width)
    assert shape is not None
    if not _representation_matches_product(
        shape,
        str(pool.storage.dtype).removeprefix("torch."),
        math.prod(shape) * pool.storage.element_size(),
        product,
    ):
        raise invalid_descriptor(
            "latent publication disagrees with its declared representation"
        )

    locations = publish_tensor(
        transports,
        source.spans,
        retain=partial(pool.retain_publication, source),
        consumers=row.call.consumer_slots,
    )
    request.exported_locators.extend(locations)
    request.latent_exports[product.buffer_id] = tuple(
        (transports[location.backend], location) for location in locations
    )

    descriptor = LatentTransferValue(
        height=params.height,
        width=params.width,
        latent_units=params.latent_units,
        step=step,
        tensor=TensorTransfer(shape=shape, locations=locations),
    )
    return TensorPublication(product=product, value=descriptor)


def publish_tensors(
    call: Call,
    values: tuple[torch.Tensor, ...],
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    host: bool = False,
) -> tuple[TensorPublication, ...]:
    """Publish each numerical result from the rank owning its assigned region.

    A product published as `host` travels as host bytes over the host
    mechanism of the rank's edges, whatever device produced it.
    """
    outputs = call.outputs
    if len(outputs) != len(values):
        raise invalid_descriptor(
            "numerical results disagree with declared Tensor outputs"
        )
    request = state.pending_output(
        completion_group, call.request_key.request_id
    )
    owned = {write.reference for write in request.writes if not write.feature}
    return tuple(
        publish_product(
            output,
            value,
            None,
            completion_group,
            tensor_store=tensor_store,
            publication_transports=publication_transports,
            state=state,
            consumers=call.consumer_slots,
            host=host,
        )
        for output, value in zip(outputs, values, strict=True)
        if output in owned
    )


def publish_product(
    product: TensorRef,
    value: torch.Tensor,
    source_metadata: ImageMetadata | FeatureMetadata | None,
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    consumers: Sequence[int] = (),
    host: bool = False,
) -> TensorPublication:
    """Publish a typed device, encoder or artifact product.

    `consumers` are the acknowledgment slots the producing call names; a
    `host` product is published as host bytes.
    """
    from .encode import bound_device_write

    transports = publication_transports
    if not transports:
        raise unsupported_setup(
            "product publication requires a configured transport"
        )

    request = state.pending_output(
        completion_group, product.request_key.request_id
    )
    encoder_write = next(
        (
            write
            for write in request.writes
            if write.reference == product and write.feature
        ),
        None,
    )
    device_write = (
        None
        if encoder_write is not None
        else bound_device_write(completion_group, product, state=state)
    )

    height = 0 if source_metadata is None else source_metadata.height
    width = 0 if source_metadata is None else source_metadata.width
    value_range = (
        (
            "signed_unit"
            if source_metadata.value_range == (-1.0, 1.0)
            else "unit"
        )
        if isinstance(source_metadata, ImageMetadata)
        and source_metadata.value_range is not None
        else ""
    )

    source_kind = ""
    if encoder_write is not None:
        if not isinstance(source_metadata, FeatureMetadata):
            raise invalid_descriptor(
                "encoder transfer requires feature dimensions"
            )
        source_call = request.call
        if source_call.vision_input is not None:
            source_kind = "vision_feature"
        elif source_call.latent_feature_input is not None:
            source_kind = "latent_feature"
        else:
            raise invalid_descriptor("encoder transfer has no feature source")
    elif isinstance(source_metadata, FeatureMetadata):
        raise invalid_descriptor(
            "device tensor transfer cannot carry feature dimensions"
        )
    elif height == 0 and value_range:
        raise invalid_descriptor("non-image tensor carries an image range")

    region = None if device_write is None else device_write.region
    if region is not None and tuple(value.shape) != _slices.shape(region):
        raise invalid_descriptor(
            "product tensor disagrees with its assigned region"
        )

    shape = (
        device_write.logical_shape
        if device_write is not None and region is not None
        else tuple(value.shape)
    )
    if shape is None:
        raise invalid_descriptor(
            "tensor region publication has no logical shape"
        )

    if not _representation_matches_product(
        shape,
        str(value.dtype).removeprefix("torch."),
        math.prod(shape) * value.element_size(),
        product,
    ):
        raise invalid_descriptor(
            "product transfer changes its declared representation"
        )
    if encoder_write is not None:
        value = tensor_store.publish_write(
            encoder_write, value, metadata=source_metadata
        )
    else:
        assert device_write is not None
        value = tensor_store.publish_write(
            device_write,
            value,
            metadata=source_metadata,
        )

    if encoder_write is not None:
        retain = partial(tensor_store.retain_publication, encoder_write)
    else:
        assert device_write is not None
        retain = partial(tensor_store.retain_publication, device_write)

    locations = publish_tensor(
        transports,
        value,
        retain=retain,
        offset=None if region is None else _slices.offset(region),
        consumers=consumers,
        host=host,
    )
    request.exported_locators.extend(locations)
    request.tensor_exports[product.buffer_id] = tuple(
        (transports[location.backend], location) for location in locations
    )

    if encoder_write is not None:
        descriptor: TransferValue = EncoderTransferValue(
            height=height,
            width=width,
            payload_kind=source_kind,
            tensor=TensorTransfer(shape=shape, locations=locations),
        )
    else:
        descriptor = DeviceProductTransferValue(
            height=height,
            width=width,
            value_range=value_range,
            tensor=TensorTransfer(shape=shape, locations=locations),
        )

    return TensorPublication(product=product, value=descriptor)


def fetch_product(
    call: Call,
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelRunner,
) -> tuple[torch.Tensor, ImageMetadata | FeatureMetadata | None]:
    """Fetch a transfer handle and stage its typed value for the consuming.

    call.
    """
    request = state.pending_output(
        completion_group, call.request_key.request_id
    )
    for reference in (call.vision_input, call.latent_feature_input):
        if reference is None:
            continue
        read = tensor_store.consume(
            reference,
            consumer_call_id=call.call_id,
            device=model_runner.call_devices(call)[0],
        )
        request.feature_reads.append(read)
        metadata = read.metadata
        if not isinstance(metadata, FeatureMetadata):
            raise invalid_descriptor(
                "feature transfer requires spatial metadata"
            )
        return read.tensor, metadata
    references = (
        *call.inputs,
        *(
            value
            for value in (call.token_input, call.image_input)
            if value is not None
        ),
    )
    if len(references) != 1:
        raise invalid_descriptor("tensor transfer requires one resident source")
    device_read = tensor_store.consume(
        references[0],
        consumer_call_id=call.call_id,
        device=model_runner.call_devices(call)[0],
    )
    request.device_reads.append(device_read)
    return device_read.tensor, device_read.metadata


def tensor_matches_product(tensor: TensorTransfer, product: TensorRef) -> bool:
    """Verify that a transfer locator's byte size, dtype.

    and shape match its product.
    """
    return _representation_matches_product(
        tensor.shape, tensor.dtype, tensor.nbytes, product
    )


def _representation_matches_product(
    shape: tuple[int, ...], physical_dtype: str, nbytes: int, product: TensorRef
) -> bool:
    """Match physical storage to the immutable declared value without.

    converting it.
    """
    elements = math.prod(shape)
    shape_matches = product.shape_bound.contains_shape(shape)
    dtype, element_bytes = device_product_storage(product.dtype)
    return (
        all(value > 0 for value in shape)
        and shape_matches
        and physical_dtype == dtype
        and nbytes == elements * element_bytes
    )


__all__ = ["execute"]


def _release_locators(
    locators: Iterable[Locator], *, transfer_backends: Mapping[str, Transport]
) -> None:
    """Release transfer locators through the runtime transport owner."""
    if not transfer_backends:
        return
    for locator in locators:
        transfer_backends[locator.backend].release(locator)


def reserved_unit_row(
    call: Call,
    completion_group: int,
    *,
    state: BatchState,
):
    """Return the product row reserved for one encoded media unit.

    The row is published with its batch and filled when the host task that
    encodes the unit completes. Nothing reads it before then: the muxer's call
    is scheduled only once every encode round has completed.
    """
    from .encode import bound_device_write

    outputs = call.outputs
    if len(outputs) != 1:
        raise invalid_descriptor(
            "encoded media unit requires exactly one declared product"
        )
    write = bound_device_write(completion_group, outputs[0], state=state)
    row = write.tensor
    if row.ndim != 2 or row.shape[0] != 1:
        raise invalid_descriptor(
            "encoded media unit product must reserve one row"
        )
    return row
