"""Product, KV, and latent movement with no model call.

``execute`` runs the ``TransferMode`` calls that ``schedule`` dispatches: KV
publication and installation through ``KVCacheManager``, and product
transfers that republish one resident tensor or the committed trajectory's
current latent pages. The publication helpers are shared with the image,
media, diffusion and host-media call paths.

Every tensor publication checks the physical tensor against the product's
declared shape bound, dtype and byte size before exporting it. Publications made
during execution record their locators on the call's ``PendingOutput``:
``commit.commit_batch`` makes the exports visible and
``commit.discard_batch`` releases the locators of a failed batch.
``publish_deferred_product`` runs after its call committed and registers and
commits its export with ``TensorStore`` directly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch

from uniserve import _slices
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution import calls as calls
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.output import PendingOutput
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
from uniserve_worker.storage.latent_pool import LatentExport
from uniserve_worker.storage.tensor_store import (
    FeatureMetadata,
    ImageMetadata,
    device_product_storage,
)
from uniserve_worker.transport.publication import publish_tensor

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.tensor_store import Buffer, TensorStore
    from uniserve_worker.transport.interface import Transport


def execute(
    call: Call,
    *,
    state: BatchState,
    kv_cache: KVCacheManager | None,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Execute one transfer call and stage its outcome.

    ``KV_PUBLISH`` exports the request's visible KV extent under the call's
    KV output identity. ``KV_INSTALL`` adopts the physical import reserved
    for the call's KV input and sets the staged visible and computed KV
    lengths to its published extent. Any other mode republishes the call's
    single tensor input as its single output.

    Raises:
        WorkerError: ``unsupported_setup`` when no publication transport or,
            for a latent transfer, no latent pool is configured;
            ``invalid_descriptor`` when, for example, KV storage, a declared
            identity or a reserved input is missing.
    """
    from uniserve_worker.execution import image

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
        request = state.pending_output(request_id)

        # (request slot, accepted visible length, capacity).
        cache = calls.cache_coordinates(request, tables=request_tables)
        snapshot = publications.publish(
            request_pool_idx=request.request.request_pool_idx,
            visible_length=cache[1],
            destination="gen",
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

        outcome = image.non_state_outcome(call, state=state)
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
        request = state.pending_output(request_id)
        write = state.inputs.cache(source)
        if write is None:
            raise invalid_descriptor(
                "KV installation has no reserved physical input"
            )
        installed = publications.install(
            installed_buffer=output,
            write=write,
        )
        request.cache_installation = (source, output, installed)

        outcome = image.non_state_outcome(call, state=state)
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
                latent_pool=latent_pool,
                publication_transports=publication_transports,
                state=state,
            )
        else:
            value, metadata = fetch_product(
                call,
                tensor_store=tensor_store,
                model_runner=model_runner,
                state=state,
            )
            tensor_publication = publish_product(
                outputs[0],
                value,
                metadata,
                tensor_store=tensor_store,
                publication_transports=publication_transports,
                state=state,
            )
        outcome = image.non_state_outcome(
            call,
            products=(tensor_publication,),
            state=state,
        )
    return outcome


def _publish_current_latent(
    call: Call,
    reference: TensorRef,
    product: TensorRef,
    *,
    state: BatchState,
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
) -> TensorPublication:
    """Publish the committed trajectory's current latent pages as a product.

    ``LatentPool.reserve_current_export`` requires the staged start
    step, the input's generation, units, raster and pages to match the
    slot's committed trajectory. The product must be the call's declared
    latent output, and publication leaves the request's generation and step
    unchanged.
    """
    if product != call.latent_output:
        raise invalid_descriptor(
            "product transfer changes the physical product kind"
        )

    row = state.pending_output(call.request_key.request_id)
    params = row.latent.input_params
    staging = row.latent.staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory call has no staged latent inputs")

    source = latent_pool.reserve_current_export(
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
    latent_pool: LatentPool,
    publication_transports: Mapping[str, Transport],
) -> TensorPublication:
    """Publish reserved latent page spans as one latent product.

    The spans are published as a ``[latent_units, latent_width]`` tensor in
    the pool's storage dtype. Each export's completion is
    attached through ``LatentPool.retain_export``, which keeps the page
    bank until its readers retire. The locators are recorded on the
    product's pending output.
    """
    params = row.latent.input_params
    if params is None:
        raise invalid_descriptor("latent publication has no staged parameters")

    request = state.pending_output(product.request_key.request_id)
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
        retain=partial(pool.retain_export, source),
        consumers=row.call.consumer_slots,
    )
    request.exported_locators.extend(locations)
    request.latent.exports[product.buffer_id] = tuple(
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
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    host: bool = False,
) -> tuple[TensorPublication, ...]:
    """Publish each numerical result from the rank owning its assigned region.

    Only outputs for which this rank holds a non-feature write are published;
    the others are skipped, so the result may hold fewer publications than
    ``values``. A product published as `host` travels as host bytes over the
    host mechanism of the rank's edges, whatever device produced it.
    """
    outputs = call.outputs
    if len(outputs) != len(values):
        raise invalid_descriptor(
            "numerical results disagree with declared Tensor outputs"
        )
    request = state.pending_output(call.request_key.request_id)
    owned = {write.reference for write in request.writes if not write.feature}
    return tuple(
        publish_product(
            output,
            value,
            None,
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
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    consumers: Sequence[int] = (),
    host: bool = False,
) -> TensorPublication:
    """Publish one encoder-feature or device product from ``value``.

    A product the call reserved as a feature write carries its spatial
    dimensions from ``FeatureMetadata`` and its source kind from the
    producing call's vision or latent feature input. Any other product
    fills its bound device write, possibly as a region of a larger logical
    tensor. The value is validated against the declared representation,
    written through ``TensorStore.publish_write`` and exported; the locators
    are recorded on the product's pending output.

    `consumers` are the acknowledgment slots the producing call names; a
    `host` product is published as host bytes.

    Raises:
        WorkerError: ``unsupported_setup`` without a publication transport;
            ``invalid_descriptor`` when the metadata, region or
            representation disagrees with the product.
    """
    from uniserve_worker.execution.image import bound_device_write

    transports = publication_transports
    if not transports:
        raise unsupported_setup(
            "product publication requires a configured transport"
        )

    request = state.pending_output(product.request_key.request_id)
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
        else bound_device_write(product, state=state)
    )

    # A value with `ImageMetadata` that declares a range names it: [-1, 1] is
    # "signed_unit" and any other range "unit". Otherwise the range is empty.
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
        if source_call.vision_inputs:
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

    # A region write publishes `value` at its offset within the write's
    # logical shape, which is what the descriptor reports.
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
        retain = partial(tensor_store.retain_export, encoder_write)
    else:
        assert device_write is not None
        retain = partial(tensor_store.retain_export, device_write)

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


def publish_deferred_product(
    product: TensorRef,
    write: Buffer,
    value: torch.Tensor,
    *,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    consumers: Sequence[int],
    regions: Sequence[tuple[slice, ...]] | None = None,
) -> TensorPublication:
    """Publish a product whose write host work filled after its call committed.

    The committed call released its execution references, so the caller
    hands over the write it retained. The product is published as host
    bytes, registered for retirement, and committed here; the caller reports
    the publication with the completion the work belongs to. If only regions
    are initialized, publish those views at their logical offsets. Consumers
    must read these regions rather than the uninitialized reserved capacity.
    If publishing any view fails, the views already published are released
    before the error propagates.
    """
    if not publication_transports:
        raise unsupported_setup(
            "product publication requires a configured transport"
        )
    region = write.region
    if region is not None and tuple(value.shape) != _slices.shape(region):
        raise invalid_descriptor(
            "product tensor disagrees with its assigned region"
        )
    shape = write.logical_shape if region is not None else tuple(value.shape)
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
    value = tensor_store.publish_write(write, value, metadata=None)
    views = (
        regions
        if regions is not None
        else (tuple(slice(0, n) for n in value.shape),)
    )
    origin = (0,) * value.ndim if region is None else _slices.offset(region)
    locations: list[Locator] = []
    try:
        for view in views:
            if not _slices.within(view, value.shape):
                raise invalid_descriptor(
                    "publication exceeds its product region"
                )
            locations.extend(
                publish_tensor(
                    publication_transports,
                    value[view],
                    retain=partial(tensor_store.retain_export, write),
                    offset=tuple(
                        a + b
                        for a, b in zip(
                            origin, _slices.offset(view), strict=True
                        )
                    ),
                    consumers=consumers,
                    host=True,
                )
            )
    except BaseException:
        for location in locations:
            publication_transports[location.backend].release(location)
        raise
    tensor_store.exports[product.buffer_id] = tuple(
        (publication_transports[location.backend], location)
        for location in locations
    )
    tensor_store.commit_writes((write,))
    return TensorPublication(
        product=product,
        value=DeviceProductTransferValue(
            height=0,
            width=0,
            value_range="",
            tensor=TensorTransfer(shape=shape, locations=tuple(locations)),
        ),
    )


def fetch_product(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
) -> tuple[torch.Tensor, ImageMetadata | FeatureMetadata | None]:
    """Consume the call's single resident source with its metadata.

    A vision or latent feature input takes precedence and must carry
    ``FeatureMetadata``; otherwise the call must name exactly one source
    among its tensor, token and image inputs. The read lands on the call's
    first device and is recorded on its pending output, where commit or
    discard completes it.
    """
    request = state.pending_output(call.request_key.request_id)
    features = (
        *(block.feature for block in call.vision_inputs),
        *((call.latent_feature_input,) if call.latent_feature_input else ()),
    )
    if features:
        if len(features) != 1:
            raise invalid_descriptor("feature transfer requires one source")
        read = tensor_store.consume(
            features[0],
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
    """Check a transfer tensor's shape, dtype and size against its product."""
    return _representation_matches_product(
        tensor.shape, tensor.dtype, tensor.nbytes, product
    )


def _representation_matches_product(
    shape: tuple[int, ...], physical_dtype: str, nbytes: int, product: TensorRef
) -> bool:
    """Match physical storage to its declaration without conversion.

    Every dimension must be positive and within the product's shape bound,
    ``physical_dtype`` must name the product dtype's storage dtype, and
    ``nbytes`` must be exactly the element count times its element size.
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
    """Release transfer locators through the runtime transport owner.

    Does nothing when no transport is configured.
    """
    if not transfer_backends:
        return
    for locator in locators:
        transfer_backends[locator.backend].release(locator)


def reserved_unit_rows(
    call: Call,
    *,
    state: BatchState,
    count: int,
):
    """Return the product rows reserved for this rank's encoded media units.

    The rows are filled and published when the host tasks that encode the
    units complete. Nothing reads them before then: the muxer's call is
    scheduled only once every encode round has completed.
    """
    from uniserve_worker.execution.image import bound_device_write

    outputs = call.outputs
    if len(outputs) != 1:
        raise invalid_descriptor(
            "encoded media units require exactly one declared product"
        )
    write = bound_device_write(outputs[0], state=state)
    rows = write.tensor
    if rows.ndim != 2 or rows.shape[0] != count:
        raise invalid_descriptor(
            "encoded media unit product must reserve one row per unit"
        )
    return write, rows
