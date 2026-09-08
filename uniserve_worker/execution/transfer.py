"""Product, KV, and latent movement with no model call."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker.execution.batch import (
    DeviceDim,
    DeviceProductTransferValue,
    DrawLayout,
    EncoderTransferValue,
    FinishFlags,
    LatentTransferValue,
    OpCode,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    StaticDim,
    StorageClass,
    TensorTransfer,
    TokenSpan,
    TransferHandle,
    TransferMode,
)
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.runtime.cache_transfer import CacheWrite
from uniserve_worker.runtime.device_products import (
    DeviceProductMetadata,
    ImageRange,
    device_product_storage,
)
from uniserve_worker.runtime.encoder_cache import EncoderMetadata
from uniserve_worker.runtime.latent_pool import LatentPublication, LatentSource
from uniserve_worker.transfer.tickets import publish_tensor

from . import flow
from .rows import LaneState, LatentExecution, OperationState, Outcome

if TYPE_CHECKING:
    from ..worker.worker import Worker


def run_action(runtime: Worker, state: OperationState) -> bool:
    """Execute product transfer, KV publication, or KV installation without a model call."""

    if state.phase != "initial":
        return False
    work = state.operation.kind
    if work is OpCode.DIFFUSION_PREPARE and runtime.latent_pool is not None:
        _prepare_media(runtime, state)
        return True
    if work.transfer_mode is not None:
        _transfer(runtime, state)
        return True
    return False


def _prepare_media(runtime: Worker, state: OperationState) -> None:
    """Seed and publish the initial latent trajectory for one diffusion request."""

    operation = state.operation
    scope = state.lane
    runtime.generation()
    request_id = operation.request_key.request_id

    # Media preparation joins one visible conditioning publication to one new
    # latent product; accepting any other arity would make ownership ambiguous.
    conditioning = tuple(
        reference for reference in operation.inputs if reference.kind is ProductKind.KV
    )
    latent_outputs = tuple(
        reference for reference in operation.outputs if reference.kind is ProductKind.LATENT
    )
    if len(conditioning) != 1 or len(latent_outputs) != 1:
        raise invalid_descriptor(
            "media preparation requires one exact conditioning input and latent output"
        )
    cache = runtime.cache_coordinates(operation, scope)
    request = runtime.request_row(scope, request_id)
    publications = runtime.cache_publications
    if publications is None:
        raise invalid_descriptor("media preparation requires KV publication storage")
    publications.validate_conditioning(
        request_id,
        conditioning[0],
        request_pool_idx=request.request.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=scope.cache_publication_inputs.get(conditioning[0]),
    )
    image = request.request.image
    if image is None:
        raise invalid_descriptor("media preparation has no admitted image parameters")
    if request.latent_product is not None or request.flow_step != 0:
        raise invalid_descriptor("media preparation repeats an active latent trajectory")
    rng = operation.rng
    if rng is None or rng.draw_layout is not DrawLayout.FLOW_NOISE:
        raise invalid_descriptor("media preparation requires semantic flow-noise RNG coordinates")
    if int(rng.seed) != int(image.seed or 0):
        raise invalid_descriptor("media preparation seed disagrees with admitted image seed")
    if int(rng.semantic_index_base) < 1:
        raise invalid_descriptor("flow-noise semantic image index must be positive")
    output = latent_outputs[0]
    if int(output.generation) < 1:
        raise invalid_descriptor("media preparation latent has no logical generation")

    # Noise is generated directly into request-owned staging, then installed in
    # the pool before its generation becomes visible to downstream operations.
    row = runtime.latent_row(operation, scope)
    pool = runtime.require_latent_pool()
    row.staging.value.zero_()
    initial = row.staging.value[: int(row.params.latent_units)]
    flow.initial_latent(
        runtime,
        operation,
        int(row.params.height),
        int(row.params.width),
        initial,
    )
    pool.initialize(
        row.request_pool_idx,
        row.staging,
        latent_units=int(row.params.latent_units),
    )

    # Publication is deferred with the lane commit so a failed lane cannot
    # expose a partially initialized trajectory.
    scope.latent_publications.append(
        LatentPublication(
            request_pool_idx=row.request_pool_idx,
            page_table=row.params.page_table,
            expected_generation=0,
            expected_step=0,
            generation=int(output.generation),
            step=0,
            latent_units=int(row.params.latent_units),
            height=int(row.params.height),
            width=int(row.params.width),
        )
    )
    request.latent_product = output
    products = flow.publish_latent_transfer(runtime, output, row, step=0, scope=scope)
    state.outcome = Outcome(
        status=OpStatus.OK,
        selected_point=1,
        logical_lengths=runtime.logical_lengths(operation, request, cache, latent_len=0),
        token_span=TokenSpan(base=request.logical_position, len=0),
        finish_flags=FinishFlags(),
        product_generations=runtime.output_generations(operation),
        products=products,
    )
    # The operation itself is complete once all state and product publications
    # have been staged; lane commit establishes their external visibility.
    state.phase = "done"


def _transfer(runtime: Worker, state: OperationState) -> None:
    """Execute a tensor transfer or KV publication/install operation and stage its result."""

    from . import encode

    operation = state.operation
    scope = state.lane
    transports = runtime.publication_transports
    if not transports:
        raise unsupported_setup("product transfer requires a configured transport")
    request_id = operation.request_key.request_id
    mode = operation.kind.transfer_mode
    if mode is TransferMode.KV_PUBLISH:
        publications = runtime.cache_publications
        if publications is None:
            raise invalid_descriptor("KV publication requires cache storage")
        runtime.fixed_parent(operation)
        outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.KV)
        if len(outputs) != 1:
            raise invalid_descriptor("KV publication requires one KV output product")
        if any(reference.kind is ProductKind.KV for reference in operation.inputs):
            raise invalid_descriptor("KV publication is rooted only by its fixed parent")
        cache = runtime.cache_coordinates(operation, scope)
        request = runtime.request_row(scope, request_id)
        expected_base = publications.destination_base(request_id, "gen")
        snapshot = publications.publish(
            request_pool_idx=request.request.request_pool_idx,
            group_id=cache[1],
            visible_length=cache[2],
            source_version=operation.state_parent,
            destination="gen",
            expected_base=expected_base,
            product=outputs[0],
            transports=transports,
        )
        scope.cache_publications.append((outputs[0], snapshot))
        for tensor in snapshot.tensors:
            for locator in tensor.locations:
                scope.published.append(locator)
        scope.stage_publications[outputs[0].buffer_id] = tuple(
            location for tensor in snapshot.tensors for location in tensor.locations
        )
        payload = TransferHandle(snapshot)
        state.outcome = encode.non_state_outcome(
            runtime,
            operation,
            scope,
            products=(ProductPayload(product=outputs[0], payload=payload),),
        )
    elif mode is TransferMode.KV_INSTALL:
        publications = runtime.cache_publications
        if publications is None:
            raise invalid_descriptor("KV installation requires cache storage")
        inputs = tuple(
            reference for reference in operation.inputs if reference.kind is ProductKind.KV
        )
        outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.KV)
        if len(inputs) != 1 or len(outputs) != 1:
            raise invalid_descriptor("KV installation requires one input and one output")
        cache = runtime.cache_coordinates(operation, scope)
        request = runtime.request_row(scope, request_id)
        prepared = scope.prepared_transfers.get(inputs[0])
        if prepared is None:
            raise invalid_descriptor("KV installation has no prepared physical inputs")
        if not isinstance(prepared.destination, CacheWrite):
            raise invalid_descriptor("KV installation lost its physical destination")
        installed = publications.install(
            request_pool_idx=request.request.request_pool_idx,
            group_id=cache[1],
            request_id=request_id,
            source=inputs[0],
            installed_product=outputs[0],
            write=prepared.destination,
        )
        prepared.adopt_destination()
        scope.cache_installations.append((inputs[0], outputs[0], installed))
        outcome = encode.non_state_outcome(
            runtime,
            operation,
            scope,
            products=(ProductPayload(product=outputs[0], payload=b""),),
        )
        state.outcome = replace(
            outcome,
            logical_lengths=replace(
                outcome.logical_lengths,
                kv_visible_len=int(installed.published_extent),
                kv_computed_len=int(installed.published_extent),
            ),
        )
    else:
        inputs = tuple(reference for reference in operation.inputs if transferable(reference))
        outputs = tuple(reference for reference in operation.outputs if transferable(reference))
        if len(inputs) != 1 or len(outputs) != 1:
            raise invalid_descriptor("product transfer requires one physical input and one output")
        if inputs[0].kind is ProductKind.LATENT:
            product_payload = _publish_current_latent(
                runtime, operation, inputs[0], outputs[0], scope
            )
        else:
            value, metadata = fetch_product(runtime, operation, scope)
            product_payload = publish_product(runtime, outputs[0], value, metadata, scope)
        state.outcome = encode.non_state_outcome(
            runtime, operation, scope, products=(product_payload,)
        )
    state.phase = "done"


def _publish_current_latent(
    runtime: Worker,
    operation: Operation,
    reference: ProductRef,
    product: ProductRef,
    scope: LaneState,
) -> ProductPayload:
    request = runtime.request_row(scope, operation.request_key.request_id)
    if request.latent_product != reference:
        raise invalid_descriptor("latent transfer does not name the committed trajectory")
    if product.kind is not ProductKind.LATENT:
        raise invalid_descriptor("product transfer changes the physical product kind")
    row = runtime.latent_row(operation, scope)
    source = runtime.require_latent_pool().reserve_current_publication(
        product,
        request_pool_idx=row.request_pool_idx,
        page_table=row.params.page_table,
        generation=reference.generation,
        step=row.params.start_step,
        latent_units=row.params.latent_units,
        height=row.params.height,
        width=row.params.width,
    )
    return publish_latent_source(
        runtime, product, source, row, step=row.params.start_step, scope=scope
    )


def publish_latent_source(
    runtime: Worker,
    product: ProductRef,
    source: LatentSource,
    row: LatentExecution,
    *,
    step: int,
    scope: LaneState,
) -> ProductPayload:
    """Register exact latent page spans and retain their bank for every reader."""

    transports = runtime.publication_transports
    if not transports:
        raise unsupported_setup("latent publication requires a configured transport")
    pool = runtime.require_latent_pool()
    shape = (row.params.latent_units, pool.latent_width)
    assert shape is not None
    if not _representation_matches_product(
        shape,
        str(pool.dtype).removeprefix("torch."),
        math.prod(shape) * pool.storage.element_size(),
        product,
    ):
        raise invalid_descriptor("latent publication disagrees with its declared representation")
    locations = publish_tensor(
        transports, source.spans, retain=partial(pool.retain_publication, source)
    )
    scope.published.extend(locations)
    scope.stage_publications[product.buffer_id] = locations
    descriptor = TransferHandle(
        LatentTransferValue(
            generation=product.generation,
            height=row.params.height,
            width=row.params.width,
            latent_units=row.params.latent_units,
            step=step,
            tensor=TensorTransfer(shape=shape, locations=locations),
        )
    )
    return ProductPayload(product=product, payload=descriptor)


def publish_tensors(
    runtime: Worker,
    operation: Operation,
    values: tuple[torch.Tensor, ...],
    scope: LaneState,
) -> tuple[ProductPayload, ...]:
    """Publish each numerical result from the rank owning its assigned region."""

    outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.TENSOR)
    if len(outputs) != len(values):
        raise invalid_descriptor("numerical results disagree with declared Tensor outputs")
    owned = {write.reference for write in scope.device_writes}
    return tuple(
        publish_product(runtime, output, value, {"payload_kind": "tensor"}, scope)
        for output, value in zip(outputs, values, strict=True)
        if output in owned
    )


def publish_product(
    runtime: Worker,
    product: ProductRef,
    value: torch.Tensor,
    source_metadata: Mapping[str, object],
    scope: LaneState,
) -> ProductPayload:
    """Publish a typed device, encoder, or artifact product through the selected transport."""

    from .encode import bound_device_write, bound_encoder_write

    transports = runtime.publication_transports
    if not transports:
        raise unsupported_setup("product publication requires a configured transport")
    source_kind = metadata_string(source_metadata, "payload_kind", "")
    if source_kind != product.kind.value and not (
        product.kind is ProductKind.ARTIFACT and source_kind == "image_nchw"
    ):
        raise invalid_descriptor("product transfer changes the physical product kind")
    generation = int(product.generation)
    height = metadata_uint(source_metadata, "height", 0)
    width = metadata_uint(source_metadata, "width", 0)
    if product.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}:
        if min(height, width, generation) < 1:
            raise invalid_descriptor("encoder transfer has incomplete geometry")
    elif requires_device_product_binding(product):
        value_range = metadata_string(source_metadata, "value_range", "")
        if (height == 0) != (width == 0):
            raise invalid_descriptor("device-product transfer has incomplete geometry")
        if value_range not in {"", *(member.value for member in ImageRange)}:
            raise invalid_descriptor("device-product transfer has an invalid value range")
        if height == 0 and value_range:
            raise invalid_descriptor("non-image device product carries an image range")
    else:
        raise invalid_descriptor("product transfer output has no concrete physical owner")
    device_write = (
        bound_device_write(scope, product) if requires_device_product_binding(product) else None
    )
    region = None if device_write is None else device_write.region
    if region is not None and tuple(value.shape) != region.shape:
        raise invalid_descriptor("product tensor disagrees with its assigned region")
    shape = (
        device_write.logical_shape
        if device_write is not None and region is not None
        else tuple(value.shape)
    )
    if shape is None:
        raise invalid_descriptor("tensor region publication has no logical shape")
    if not _representation_matches_product(
        shape,
        str(value.dtype).removeprefix("torch."),
        math.prod(shape) * value.element_size(),
        product,
    ):
        raise invalid_descriptor("product transfer changes its declared representation")
    encoder_write = None
    if product.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}:
        encoder_write = bound_encoder_write(scope, product)
        value = runtime.encoder_cache.publish(
            encoder_write, value, EncoderMetadata(height=height, width=width)
        )
    elif requires_device_product_binding(product):
        assert device_write is not None
        value = runtime.device_products.publish_write(
            device_write,
            value,
            metadata=(
                None
                if height == 0
                else DeviceProductMetadata(
                    height=height,
                    width=width,
                    value_range=None if not value_range else ImageRange(value_range),
                )
            ),
        )
    if encoder_write is not None:
        retain = partial(runtime.encoder_cache.retain_publication, encoder_write)
    else:
        assert device_write is not None
        retain = partial(runtime.device_products.retain_publication, device_write)
    locations = publish_tensor(
        transports, value, retain=retain, offset=None if region is None else region.offset
    )
    scope.published.extend(locations)
    scope.stage_publications[product.buffer_id] = locations
    if product.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}:
        descriptor = TransferHandle(
            EncoderTransferValue(
                generation=generation,
                height=height,
                width=width,
                payload_kind=product.kind.value,
                tensor=TensorTransfer(shape=shape, locations=locations),
            )
        )
    else:
        descriptor = TransferHandle(
            DeviceProductTransferValue(
                generation=generation,
                height=height,
                width=width,
                value_range=value_range,
                tensor=TensorTransfer(shape=shape, locations=locations),
            )
        )
    return ProductPayload(product=product, payload=descriptor)


def fetch_product(
    runtime: Worker,
    operation: Operation,
    scope: LaneState,
) -> tuple[torch.Tensor, Mapping[str, object]]:
    """Fetch a transfer handle and stage its typed value for the consuming operation."""

    for reference in operation.inputs:
        if reference.storage_class in {
            StorageClass.DEVICE_TENSOR,
            StorageClass.REQUEST_RELAY,
        }:
            device_read = runtime.device_products.consume(
                reference,
                consumer_op_id=operation.op_id,
                device=runtime.operation_device(operation),
            )
            scope.device_reads.append(device_read)
            metadata = device_read.metadata
            values: dict[str, object] = {"payload_kind": reference.kind.value}
            if metadata is not None and metadata.height > 0:
                values.update(
                    {
                        "payload_kind": "image_nchw",
                        "height": metadata.height,
                        "width": metadata.width,
                        "value_range": (
                            "" if metadata.value_range is None else metadata.value_range.value
                        ),
                    }
                )
            return device_read.tensor, values
        if reference.kind in {
            ProductKind.VISION_FEATURE,
            ProductKind.LATENT_FEATURE,
        }:
            encoder_read = runtime.encoder_cache.consume(
                reference,
                consumer_op_id=operation.op_id,
                device=runtime.operation_device(operation),
            )
            scope.encoder_reads.append(encoder_read)
            return encoder_read.tensor, {
                "payload_kind": reference.kind.value,
                "height": encoder_read.metadata.height,
                "width": encoder_read.metadata.width,
            }
    raise invalid_descriptor("transfer product is not resident or transport-addressable")


def metadata_uint(metadata: Mapping[str, object], name: str, default: int) -> int:
    """Read a nonnegative integer from transfer metadata with a validated default."""

    value = metadata.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"product metadata field {name!r} must be a non-negative integer")
    return value


def requires_device_product_binding(reference: ProductRef) -> bool:
    """Return whether receiving this product requires a destination device slot."""

    return reference.storage_class in {
        StorageClass.DEVICE_TENSOR,
        StorageClass.REQUEST_RELAY,
    } or (
        reference.storage_class is StorageClass.LATENT_ARENA
        and reference.kind is ProductKind.ARTIFACT
    )


def transferable(reference: ProductRef) -> bool:
    """Return whether a product storage class supports transport publication."""

    return (
        reference.kind is ProductKind.LATENT
        or reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
        or reference.storage_class
        in {
            StorageClass.DEVICE_TENSOR,
            StorageClass.REQUEST_RELAY,
        }
        or (
            reference.storage_class is StorageClass.LATENT_ARENA
            and reference.kind is ProductKind.ARTIFACT
        )
    )


def metadata_string(metadata: Mapping[str, object], name: str, default: str) -> str:
    """Read a nonempty string from transfer metadata with a validated default."""

    value = metadata.get(name, default)
    if not isinstance(value, str):
        raise invalid_descriptor(f"product metadata field {name!r} must be a string")
    return value


def tensor_matches_product(tensor: TensorTransfer, product: ProductRef) -> bool:
    """Verify that a transfer locator’s byte size, dtype, and shape match a product contract."""

    return _representation_matches_product(tensor.shape, tensor.dtype, tensor.nbytes, product)


def _representation_matches_product(
    shape: tuple[int, ...], physical_dtype: str, nbytes: int, product: ProductRef
) -> bool:
    """Match physical storage to the immutable declared value without converting it."""

    elements = math.prod(shape)
    bounds = product.shape_bound.dims
    if (
        any(isinstance(bound, DeviceDim) for bound in bounds)
        and product.kind is not ProductKind.TENSOR
    ):
        shape_matches = 0 < elements <= product.shape_bound.max_elements
    elif any(isinstance(bound, DeviceDim) for bound in bounds):
        shape_matches = len(shape) == len(bounds) and all(
            extent == bound.extent if isinstance(bound, StaticDim) else 0 < extent <= bound.bound
            for extent, bound in zip(shape, bounds, strict=True)
        )
    elif bounds:
        shape_matches = shape == tuple(cast(StaticDim, bound).extent for bound in bounds)
    else:
        shape_matches = elements == 1
    dtype, element_bytes = device_product_storage(product.dtype)
    return (
        all(value > 0 for value in shape)
        and shape_matches
        and physical_dtype == dtype
        and nbytes == elements * element_bytes
    )


__all__ = ["run_action"]
