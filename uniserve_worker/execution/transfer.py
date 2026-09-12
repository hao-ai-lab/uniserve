"""Product, KV, and latent movement with no model call."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch

from uniserve_worker.execution import flow
from uniserve_worker.execution import operations as operation_geometry
from uniserve_worker.execution.batch import (
    DeviceProductTransferValue,
    DrawLayout,
    EncoderTransferValue,
    FinishFlags,
    LatentTransferValue,
    Locator,
    OpStatus,
    PipelineStage,
    ScheduledRequest,
    TensorPublication,
    TensorRef,
    TensorTransfer,
    TransferMode,
    TransferValue,
)
from uniserve_worker.execution.rows import LaneState, LatentExecution, OperationState, Outcome
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.runtime.cache_transfer import CacheWrite
from uniserve_worker.runtime.device_products import (
    DeviceProductMetadata,
    ImageRange,
    device_product_storage,
)
from uniserve_worker.runtime.encoder_cache import EncoderMetadata
from uniserve_worker.runtime.latent_pool import LatentPublication, LatentSource, require_latent_pool
from uniserve_worker.transfer.tickets import publish_tensor

if TYPE_CHECKING:
    from uniserve_worker.bootstrap.worker_info import WorkerInfo
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.runtime.cache_publications import CachePublications
    from uniserve_worker.runtime.device_products import DeviceProducts
    from uniserve_worker.runtime.encoder_cache import EncoderCache
    from uniserve_worker.runtime.latent_pool import LatentPool
    from uniserve_worker.runtime.req_to_token_pool import ReqToTokenPool
    from uniserve_worker.transfer.tickets import Transport


def run_action(
    state: OperationState,
    *,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> bool:
    """Execute product transfer, KV publication, or KV installation without a model call."""

    if state.phase != "initial":
        return False
    work = state.operation.kind
    if work is PipelineStage.LATENT_PREPARATION and latent_pool is not None:
        _prepare_media(
            state,
            cache_registry=cache_registry,
            worker_info=worker_info,
            latent_pool=latent_pool,
            publication_transports=publication_transports,
            request_tables=request_tables,
            model_runner=model_runner,
            config=config,
        )
        return True
    if isinstance(work, TransferMode):
        _transfer(
            state,
            cache_registry=cache_registry,
            device_products=device_products,
            encoder_cache=encoder_cache,
            latent_pool=latent_pool,
            publication_transports=publication_transports,
            request_tables=request_tables,
            model_runner=model_runner,
        )
        return True
    return False


def _prepare_media(
    state: OperationState,
    *,
    cache_registry: CachePublications | None,
    worker_info: WorkerInfo,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> None:
    """Seed and publish the initial latent trajectory for one diffusion request."""

    operation = state.operation
    scope = state.lane
    model_runner.generation()
    request_id = operation.request_key.request_id

    # Media preparation joins one visible conditioning publication to one new
    # latent product; accepting any other arity would make ownership ambiguous.
    conditioning = operation.kv_input
    output = operation.latent_output
    if conditioning is None or output is None:
        raise invalid_descriptor(
            "media preparation requires one exact conditioning input and latent output"
        )
    cache = operation_geometry.cache_coordinates(operation, scope, tables=request_tables)
    request = operation_geometry.request_row(scope, request_id)
    publications = cache_registry
    if publications is None:
        raise invalid_descriptor("media preparation requires KV publication storage")
    publications.validate_conditioning(
        operation.request_key,
        conditioning,
        request_pool_idx=request.request.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=scope.cache_publication_inputs.get(conditioning),
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
    if int(output.generation) < 1:
        raise invalid_descriptor("media preparation latent has no logical generation")

    # Noise is generated directly into request-owned staging, then installed in
    # the pool before its generation becomes visible to downstream operations.
    row = operation_geometry.latent_row(operation, scope)
    pool = require_latent_pool(latent_pool)
    row.staging.value.zero_()
    initial = row.staging.value[: int(row.params.latent_units)]
    flow.initial_latent(
        operation, int(row.params.height), int(row.params.width), initial, model_runner=model_runner
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
    products = flow.publish_latent_transfer(
        operation,
        output,
        row,
        step=0,
        scope=scope,
        worker_info=worker_info,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
        config=config,
    )
    state.outcome = Outcome(
        status=OpStatus.OK,
        runtime=operation_geometry.execution_runtime(request, cache, flow_step=0),
        finish_flags=FinishFlags(),
        product_generations=operation_geometry.output_generations(operation),
        products=products,
    )
    # The operation itself is complete once all state and product publications
    # have been staged; lane commit establishes their external visibility.
    state.phase = "done"


def _transfer(
    state: OperationState,
    *,
    cache_registry: CachePublications | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
) -> None:
    """Execute a tensor transfer or KV publication/install operation and stage its result."""

    from . import encode

    operation = state.operation
    scope = state.lane
    transports = publication_transports
    if not transports:
        raise unsupported_setup("product transfer requires a configured transport")
    request_id = operation.request_key.request_id
    mode = operation.kind
    if mode is TransferMode.KV_PUBLISH:
        publications = cache_registry
        if publications is None:
            raise invalid_descriptor("KV publication requires cache storage")
        output = operation.kv_output
        if output is None:
            raise invalid_descriptor("KV publication requires a cache output identity")
        cache = operation_geometry.cache_coordinates(operation, scope, tables=request_tables)
        request = operation_geometry.request_row(scope, request_id)
        expected_base = publications.destination_base(operation.request_key, "gen")
        snapshot = publications.publish(
            request_pool_idx=request.request.request_pool_idx,
            group_id=cache[1],
            visible_length=cache[2],
            destination="gen",
            expected_base=expected_base,
            buffer=output,
            transports=transports,
        )
        scope.cache_publications.append((output, snapshot))
        for tensor in snapshot.tensors:
            for locator in tensor.locations:
                scope.published.append(locator)
        scope.stage_publications[output] = tuple(
            location for tensor in snapshot.tensors for location in tensor.locations
        )
        state.outcome = replace(encode.non_state_outcome(operation, scope), kv_output=snapshot)
    elif mode is TransferMode.KV_INSTALL:
        publications = cache_registry
        if publications is None:
            raise invalid_descriptor("KV installation requires cache storage")
        source = operation.kv_input
        output = operation.kv_output
        if source is None or output is None:
            raise invalid_descriptor("KV installation requires source and output identities")
        cache = operation_geometry.cache_coordinates(operation, scope, tables=request_tables)
        request = operation_geometry.request_row(scope, request_id)
        prepared = scope.prepared_transfers.get(source)
        if prepared is None:
            raise invalid_descriptor("KV installation has no prepared physical inputs")
        if not isinstance(prepared.destination, CacheWrite):
            raise invalid_descriptor("KV installation lost its physical destination")
        installed = publications.install(
            request_pool_idx=request.request.request_pool_idx,
            group_id=cache[1],
            request_key=operation.request_key,
            source=source,
            installed_buffer=output,
            write=prepared.destination,
        )
        prepared.adopt_destination()
        scope.cache_installations.append((source, output, installed))
        outcome = encode.non_state_outcome(
            operation,
            scope,
        )
        state.outcome = replace(
            outcome,
            runtime=replace(
                outcome.runtime,
                kv_visible_len=int(installed.published_extent),
                kv_computed_len=int(installed.published_extent),
            ),
        )
    else:
        inputs = operation.tensor_inputs()
        outputs = operation.tensor_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise invalid_descriptor("product transfer requires one physical input and one output")
        if operation.latent_input is not None:
            tensor_publication = _publish_current_latent(
                operation,
                inputs[0],
                outputs[0],
                scope,
                latent_pool=latent_pool,
                publication_transports=publication_transports,
            )
        else:
            value, metadata = fetch_product(
                operation,
                scope,
                device_products=device_products,
                encoder_cache=encoder_cache,
                model_runner=model_runner,
            )
            tensor_publication = publish_product(
                outputs[0],
                value,
                metadata,
                scope,
                device_products=device_products,
                encoder_cache=encoder_cache,
                publication_transports=publication_transports,
            )
        state.outcome = encode.non_state_outcome(operation, scope, products=(tensor_publication,))
    state.phase = "done"


def _publish_current_latent(
    operation: ScheduledRequest,
    reference: TensorRef,
    product: TensorRef,
    scope: LaneState,
    *,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
) -> TensorPublication:
    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    if request.latent_product != reference:
        raise invalid_descriptor("latent transfer does not name the committed trajectory")
    if product != operation.latent_output:
        raise invalid_descriptor("product transfer changes the physical product kind")
    row = operation_geometry.latent_row(operation, scope)
    source = require_latent_pool(latent_pool).reserve_current_publication(
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
        product,
        source,
        row,
        step=row.params.start_step,
        scope=scope,
        latent_pool=latent_pool,
        publication_transports=publication_transports,
    )


def publish_latent_source(
    product: TensorRef,
    source: LatentSource,
    row: LatentExecution,
    *,
    step: int,
    scope: LaneState,
    latent_pool: LatentPool | None,
    publication_transports: Mapping[str, Transport],
) -> TensorPublication:
    """Register exact latent page spans and retain their bank for every reader."""

    transports = publication_transports
    if not transports:
        raise unsupported_setup("latent publication requires a configured transport")
    pool = require_latent_pool(latent_pool)
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
    descriptor = LatentTransferValue(
        height=row.params.height,
        width=row.params.width,
        latent_units=row.params.latent_units,
        step=step,
        tensor=TensorTransfer(shape=shape, locations=locations),
    )
    return TensorPublication(product=product, value=descriptor)


def publish_tensors(
    operation: ScheduledRequest,
    values: tuple[torch.Tensor, ...],
    scope: LaneState,
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    publication_transports: Mapping[str, Transport],
) -> tuple[TensorPublication, ...]:
    """Publish each numerical result from the rank owning its assigned region."""

    outputs = operation.outputs
    if len(outputs) != len(values):
        raise invalid_descriptor("numerical results disagree with declared Tensor outputs")
    owned = {write.reference for write in scope.device_writes}
    return tuple(
        publish_product(
            output,
            value,
            {"payload_kind": "tensor"},
            scope,
            device_products=device_products,
            encoder_cache=encoder_cache,
            publication_transports=publication_transports,
        )
        for output, value in zip(outputs, values, strict=True)
        if output in owned
    )


def publish_product(
    product: TensorRef,
    value: torch.Tensor,
    source_metadata: Mapping[str, object],
    scope: LaneState,
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    publication_transports: Mapping[str, Transport],
) -> TensorPublication:
    """Publish a typed device, encoder, or artifact product through the selected transport."""

    from .encode import bound_device_write

    transports = publication_transports
    if not transports:
        raise unsupported_setup("product publication requires a configured transport")
    encoder_write = next(
        (write for write in scope.encoder_writes if write.reference == product), None
    )
    device_write = None if encoder_write is not None else bound_device_write(scope, product)
    height = metadata_uint(source_metadata, "height", 0)
    width = metadata_uint(source_metadata, "width", 0)
    source_kind = metadata_string(source_metadata, "payload_kind", "")
    value_range = metadata_string(source_metadata, "value_range", "")
    if encoder_write is not None:
        if min(height, width) < 1 or source_kind not in {"vision_feature", "latent_feature"}:
            raise invalid_descriptor("encoder transfer has incomplete geometry or encoding")
    else:
        if (height == 0) != (width == 0):
            raise invalid_descriptor("device tensor transfer has incomplete image geometry")
        if value_range not in {"", *(member.value for member in ImageRange)}:
            raise invalid_descriptor("device tensor transfer has an invalid value range")
        if height == 0 and value_range:
            raise invalid_descriptor("non-image tensor carries an image range")
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
    if encoder_write is not None:
        value = encoder_cache.publish(
            encoder_write, value, EncoderMetadata(height=height, width=width)
        )
    else:
        assert device_write is not None
        value = device_products.publish_write(
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
        retain = partial(encoder_cache.retain_publication, encoder_write)
    else:
        assert device_write is not None
        retain = partial(device_products.retain_publication, device_write)
    locations = publish_tensor(
        transports, value, retain=retain, offset=None if region is None else region.offset
    )
    scope.published.extend(locations)
    scope.stage_publications[product.buffer_id] = locations
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
    operation: ScheduledRequest,
    scope: LaneState,
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    model_runner: ModelRunner,
) -> tuple[torch.Tensor, Mapping[str, object]]:
    """Fetch a transfer handle and stage its typed value for the consuming operation."""

    for reference, encoding in (
        (operation.vision_input, "vision_feature"),
        (operation.latent_feature_input, "latent_feature"),
    ):
        if reference is None:
            continue
        read = encoder_cache.consume(
            reference,
            consumer_op_id=operation.op_id,
            device=model_runner.operation_device(operation),
        )
        scope.encoder_reads.append(read)
        return read.tensor, {
            "payload_kind": encoding,
            "height": read.metadata.height,
            "width": read.metadata.width,
        }
    references = (
        *operation.inputs,
        *(value for value in (operation.token_input, operation.image_input) if value is not None),
    )
    if len(references) != 1:
        raise invalid_descriptor("tensor transfer requires one resident source")
    device_read = device_products.consume(
        references[0],
        consumer_op_id=operation.op_id,
        device=model_runner.operation_device(operation),
    )
    scope.device_reads.append(device_read)
    image_metadata = device_read.metadata
    values: dict[str, object] = {}
    if image_metadata is not None and image_metadata.height > 0:
        values.update(
            height=image_metadata.height,
            width=image_metadata.width,
            value_range=""
            if image_metadata.value_range is None
            else image_metadata.value_range.value,
        )
    return device_read.tensor, values


def metadata_uint(metadata: Mapping[str, object], name: str, default: int) -> int:
    """Read a nonnegative integer from transfer metadata with a validated default."""

    value = metadata.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"product metadata field {name!r} must be a non-negative integer")
    return value


def metadata_string(metadata: Mapping[str, object], name: str, default: str) -> str:
    """Read a nonempty string from transfer metadata with a validated default."""

    value = metadata.get(name, default)
    if not isinstance(value, str):
        raise invalid_descriptor(f"product metadata field {name!r} must be a string")
    return value


def tensor_matches_product(tensor: TensorTransfer, product: TensorRef) -> bool:
    """Verify that a transfer locator’s byte size, dtype, and shape match a product contract."""

    return _representation_matches_product(tensor.shape, tensor.dtype, tensor.nbytes, product)


def _representation_matches_product(
    shape: tuple[int, ...], physical_dtype: str, nbytes: int, product: TensorRef
) -> bool:
    """Match physical storage to the immutable declared value without converting it."""

    elements = math.prod(shape)
    shape_matches = product.shape_bound.contains_shape(shape)
    dtype, element_bytes = device_product_storage(product.dtype)
    return (
        all(value > 0 for value in shape)
        and shape_matches
        and physical_dtype == dtype
        and nbytes == elements * element_bytes
    )


__all__ = ["run_action"]


def _release_locators(
    locators: Iterable[Locator], *, transfer_backends: Mapping[str, Transport]
) -> None:
    """Release transfer locators through the runtime transport owner."""

    if not transfer_backends:
        return
    for locator in locators:
        transfer_backends[locator.backend].release(locator)
