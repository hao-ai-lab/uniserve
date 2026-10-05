"""Tensor and latent movement with no model call.

``execute`` transfers one resident tensor or the committed trajectory's
current latent pages. The native executor handles KV export and installation.
The export helpers are shared with image, media, diffusion and host-media calls.

Every tensor export checks the physical tensor against the product's
declared shape bound, dtype and byte size before exporting it. Exports made
during execution record their locators on the call's ``PendingOutput``:
The native executor commits the exports or releases them after a failed batch.
``export_deferred_product`` runs after its call committed and registers and
commits its export with ``TensorStore`` directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING

import torch

from uniserve import _slices
from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution import calls as calls
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.protocol.batch import TensorExport
from uniserve_worker.protocol.call import Call
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
from uniserve_worker.transport.exports import export_tensor

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.tensor_store import Buffer, TensorStore
    from uniserve_worker.transport.interface import Transport


def execute(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    latent_pool: LatentPool | None,
    export_transports: Mapping[str, Transport],
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Transfer one tensor or latent value through its numerical storage."""
    from uniserve_worker.execution import image

    transports = export_transports
    if not transports:
        raise unsupported_setup(
            "product transfer requires a configured transport"
        )
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
        tensor_export = _publish_current_latent(
            call,
            inputs[0],
            outputs[0],
            latent_pool=latent_pool,
            export_transports=export_transports,
            state=state,
        )
    else:
        value, metadata = fetch_product(
            call,
            tensor_store=tensor_store,
            model_runner=model_runner,
            state=state,
        )
        tensor_export = export_product(
            outputs[0],
            value,
            metadata,
            tensor_store=tensor_store,
            export_transports=export_transports,
            state=state,
        )
    outcome = image.non_state_outcome(
        call,
        products=(tensor_export,),
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
    export_transports: Mapping[str, Transport],
) -> TensorExport:
    """Publish the committed trajectory's current latent pages as a product.

    ``LatentPool.reserve_current_export`` requires the bound start
    step, the input's generation, units, raster and pages to match the
    slot's committed trajectory. The product must be the call's declared
    latent output, and export leaves the request's generation and step
    unchanged.
    """
    if product != call.latent_output:
        raise invalid_descriptor(
            "product transfer changes the physical product kind"
        )

    row = state.pending_output(call.request_key.request_id)
    params = row.latent_params
    buffer = row.latent_buffer
    if params is None or buffer is None:
        raise invalid_descriptor("trajectory call has no bound latent inputs")

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

    return export_latent_source(
        product,
        source,
        row,
        step=params.start_step,
        latent_pool=latent_pool,
        export_transports=export_transports,
        state=state,
    )


def export_latent_source(
    product: TensorRef,
    source: LatentExport,
    row: PendingOutput,
    *,
    state: BatchState,
    step: int,
    latent_pool: LatentPool,
    export_transports: Mapping[str, Transport],
) -> TensorExport:
    """Publish reserved latent page spans as one latent product.

    The spans are published as a ``[latent_units, latent_width]`` tensor in
    the pool's storage dtype. Each export's completion is
    attached through ``LatentPool.retain_export``, which keeps the page
    bank until its readers retire. The locators are recorded on the
    product's pending output.
    """
    params = row.latent_params
    if params is None:
        raise invalid_descriptor("latent export has no bound parameters")

    request = state.pending_output(product.request_key.request_id)
    transports = export_transports
    if not transports:
        raise unsupported_setup("latent export requires a configured transport")

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
            "latent export disagrees with its declared representation"
        )

    locations = export_tensor(
        transports,
        source.spans,
        retain=partial(pool.retain_export, source),
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
    return TensorExport(product=product, value=descriptor)


def export_tensors(
    call: Call,
    values: tuple[torch.Tensor, ...],
    *,
    state: BatchState,
    tensor_store: TensorStore,
    export_transports: Mapping[str, Transport],
    host: bool = False,
) -> tuple[TensorExport, ...]:
    """Publish each numerical result from the rank owning its assigned region.

    Only outputs for which this rank holds a non-feature write are published;
    the others are skipped, so the result may hold fewer exports than
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
        export_product(
            output,
            value,
            None,
            tensor_store=tensor_store,
            export_transports=export_transports,
            state=state,
            consumers=call.consumer_slots,
            host=host,
        )
        for output, value in zip(outputs, values, strict=True)
        if output in owned
    )


def export_product(
    product: TensorRef,
    value: torch.Tensor,
    source_metadata: ImageMetadata | FeatureMetadata | None,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    export_transports: Mapping[str, Transport],
    consumers: Sequence[int] = (),
    host: bool = False,
) -> TensorExport:
    """Publish one encoder-feature or device product from ``value``.

    A product the call reserved as a feature write carries its spatial
    dimensions from ``FeatureMetadata`` and its source kind from the
    producing call's vision or latent feature input. Any other product
    fills its bound device write, possibly as a region of a larger logical
    tensor. The value is validated against the declared representation,
    written through ``TensorStore.write`` and exported; the locators
    are recorded on the product's pending output.

    `consumers` are the acknowledgment slots the producing call names; a
    `host` product is published as host bytes.

    Raises:
        WorkerError: ``unsupported_setup`` without an export transport;
            ``invalid_descriptor`` when the metadata, region or
            representation disagrees with the product.
    """
    from uniserve_worker.execution.image import bound_device_write

    transports = export_transports
    if not transports:
        raise unsupported_setup(
            "product export requires a configured transport"
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
        raise invalid_descriptor("tensor region export has no logical shape")

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
        value = tensor_store.write(
            encoder_write, value, metadata=source_metadata
        )
    else:
        assert device_write is not None
        value = tensor_store.write(
            device_write,
            value,
            metadata=source_metadata,
        )

    if encoder_write is not None:
        retain = partial(tensor_store.retain_export, encoder_write)
    else:
        assert device_write is not None
        retain = partial(tensor_store.retain_export, device_write)

    locations = export_tensor(
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

    return TensorExport(product=product, value=descriptor)


def export_deferred_product(
    product: TensorRef,
    write: Buffer,
    value: torch.Tensor,
    *,
    tensor_store: TensorStore,
    export_transports: Mapping[str, Transport],
    consumers: Sequence[int],
    regions: Sequence[tuple[slice, ...]] | None = None,
) -> TensorExport:
    """Publish a product whose write host work filled after its call committed.

    The committed call released its execution references, so the caller
    hands over the write it retained. The product is published as host
    bytes, registered for retirement, and committed here; the caller reports
    the export with the completion the work belongs to. If only regions
    are initialized, publish those views at their logical offsets. Consumers
    must read these regions rather than the uninitialized reserved capacity.
    If publishing any view fails, the views already published are released
    before the error propagates.
    """
    if not export_transports:
        raise unsupported_setup(
            "product export requires a configured transport"
        )
    region = write.region
    if region is not None and tuple(value.shape) != _slices.shape(region):
        raise invalid_descriptor(
            "product tensor disagrees with its assigned region"
        )
    shape = write.logical_shape if region is not None else tuple(value.shape)
    if shape is None:
        raise invalid_descriptor("tensor region export has no logical shape")
    if not _representation_matches_product(
        shape,
        str(value.dtype).removeprefix("torch."),
        math.prod(shape) * value.element_size(),
        product,
    ):
        raise invalid_descriptor(
            "product transfer changes its declared representation"
        )
    value = tensor_store.write(write, value, metadata=None)
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
                raise invalid_descriptor("export exceeds its product region")
            locations.extend(
                export_tensor(
                    export_transports,
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
            export_transports[location.backend].release(location)
        raise
    tensor_store.exports[product.buffer_id] = tuple(
        (export_transports[location.backend], location)
        for location in locations
    )
    tensor_store.commit_writes((write,))
    return TensorExport(
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
