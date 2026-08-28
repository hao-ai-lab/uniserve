"""Product, KV, and latent movement with no model call."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import torch

from uniserve_worker.execution.batch import (
    DeviceDim,
    DrawLayout,
    FinishFlags,
    ForwardMode,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    StaticDim,
    StorageClass,
    TokenSpan,
    TransferMode,
)
from uniserve_worker.foundation.errors import unsupported_setup, invalid_descriptor
from uniserve_worker.runtime.device_products import ImageRange, device_product_storage
from uniserve_worker.runtime.latent_pool import LatentPublication
from uniserve_worker.server.completion import DeferredTransferPayload
from uniserve_worker.transfer.tickets import Locator

from . import flow
from .resources import ExecutionResources
from .rows import OperationState, Outcome, PartitionState


def run_action(runtime: ExecutionResources, state: OperationState) -> bool:
    if state.phase != "initial":
        return False
    work = state.operation.work
    if work is ForwardMode.GEN_TRANSITION and runtime.latent_pool is not None:
        _transition(runtime, state)
        return True
    if work.transfer_mode is not None or work is ForwardMode.DRAFT:
        _transfer(runtime, state)
        return True
    return False


def _transition(runtime: ExecutionResources, state: OperationState) -> None:

    operation = state.operation
    partition = state.partition
    runtime.generation()
    session_id = operation.request_key.session_id
    conditioning = tuple(
        reference for reference in operation.inputs if reference.kind is ProductKind.KV
    )
    latent_outputs = tuple(
        reference for reference in operation.outputs if reference.kind is ProductKind.LATENT
    )
    if len(conditioning) != 1 or len(latent_outputs) != 1:
        raise invalid_descriptor(
            "generation transition requires one exact conditioning input and latent output"
        )
    cache = runtime.cache_coordinates(operation, partition)
    session = runtime.request_row(partition, session_id)
    runtime.cache_publications.validate_conditioning(
        session_id,
        conditioning[0],
        request_pool_idx=session.request_pool_idx,
        group_id=cache[1],
        visible_length=cache[2],
        publication=partition.cache_publication_inputs.get(conditioning[0]),
    )
    image = session.image
    if image is None:
        raise invalid_descriptor("generation transition has no admitted image parameters")
    if session.latent_product is not None or session.flow_step != 0:
        raise invalid_descriptor("generation transition repeats an active latent trajectory")
    rng = operation.rng
    if rng is None or rng.draw_layout is not DrawLayout.FLOW_NOISE:
        raise invalid_descriptor(
            "generation transition requires semantic flow-noise RNG coordinates"
        )
    if int(rng.seed) != int(image.seed or 0):
        raise invalid_descriptor("generation transition seed disagrees with admitted image seed")
    if int(rng.semantic_index_base) < 1:
        raise invalid_descriptor("flow-noise semantic image index must be positive")
    output = latent_outputs[0]
    if int(output.generation) < 1:
        raise invalid_descriptor("generation transition latent has no logical generation")
    row = runtime.latent_row(operation, partition)
    pool = runtime.require_latent_pool()
    row.staging.value.zero_()
    initial = row.staging.value[: int(row.placement.latent_units)]
    flow.initial_latent(
        runtime,
        operation,
        int(row.placement.height),
        int(row.placement.width),
        initial,
    )
    pool.initialize(
        row.request_pool_idx,
        row.staging,
        latent_units=int(row.placement.latent_units),
    )
    partition.latent_publications.append(
        LatentPublication(
            request_pool_idx=row.request_pool_idx,
            page_table=row.placement.page_table,
            expected_generation=0,
            expected_step=0,
            generation=int(output.generation),
            step=0,
            latent_units=int(row.placement.latent_units),
            height=int(row.placement.height),
            width=int(row.placement.width),
        )
    )
    session.latent_product = output
    products = flow.publish_latent_transfer(
        runtime, operation, output, initial, row, step=0, scope=partition
    )
    state.outcome = Outcome(
        status=OpStatus.OK,
        selected_point=1,
        logical_lengths=runtime.logical_lengths(operation, session, cache, latent_len=0),
        token_span=TokenSpan(base=session.logical_position, len=0),
        finish_flags=FinishFlags(),
        product_generations=runtime.output_generations(operation),
        products=products,
    )
    state.phase = "done"


def _transfer(runtime: ExecutionResources, state: OperationState) -> None:
    from . import encode

    operation = state.operation
    partition = state.partition
    transport = runtime.transport
    if transport is None:
        raise unsupported_setup("product transfer requires a configured transport")
    session_id = operation.request_key.session_id
    mode = operation.work.transfer_mode
    if mode is TransferMode.KV_PUBLISH:
        point = runtime.fixed_parent(operation)
        outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.KV)
        if len(outputs) != 1:
            raise invalid_descriptor("KV publication requires one KV output product")
        if any(reference.kind is ProductKind.KV for reference in operation.inputs):
            raise invalid_descriptor("KV publication is rooted only by its fixed parent")
        cache = runtime.cache_coordinates(operation, partition)
        request = runtime.request_row(partition, session_id)
        expected_base = runtime.cache_publications.destination_base(session_id, "gen")
        snapshot = runtime.cache_publications.publish(
            request_pool_idx=request.request_pool_idx,
            group_id=cache[1],
            visible_length=cache[2],
            source_version=operation.parent,
            destination="gen",
            expected_base=expected_base,
            product=outputs[0],
            transport=transport,
        )
        partition.cache_publications.append((outputs[0], snapshot))
        for encoded in snapshot.locators:
            partition.published.append(Locator.from_json(encoded))
        payload = DeferredTransferPayload(
            "kv",
            {"generation": int(outputs[0].generation), "snapshot": snapshot.to_mapping()},
            tuple(Locator.from_json(encoded) for encoded in snapshot.locators),
            transport,
        )
        state.outcome = encode.non_state_outcome(
            runtime,
            operation,
            partition,
            products=(ProductPayload(product=outputs[0], payload=cast(bytes, payload)),),
        )
    elif mode is TransferMode.KV_INSTALL:
        inputs = tuple(
            reference for reference in operation.inputs if reference.kind is ProductKind.KV
        )
        outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.KV)
        if len(inputs) != 1 or len(outputs) != 1:
            raise invalid_descriptor("KV installation requires one input and one output")
        cache = runtime.cache_coordinates(operation, partition)
        request = runtime.request_row(partition, session_id)
        prepared = partition.prepared_transfers.get(inputs[0])
        installed = runtime.cache_publications.install(
            request_pool_idx=request.request_pool_idx,
            group_id=cache[1],
            session_id=session_id,
            source=inputs[0],
            installed_product=outputs[0],
            transport=transport,
            transferred_tensors=None if prepared is None else prepared.tensors(),
            publication=partition.cache_publication_inputs.get(inputs[0]),
        )
        partition.cache_installations.append((inputs[0], outputs[0], installed))
        outcome = encode.non_state_outcome(
            runtime,
            operation,
            partition,
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
        value, metadata = fetch_product(runtime, operation, partition)
        product_payload = publish_product(
            runtime, operation, outputs[0], value, metadata, partition
        )
        state.outcome = encode.non_state_outcome(
            runtime, operation, partition, products=(product_payload,)
        )
    state.phase = "done"


def publish_product(
    runtime: ExecutionResources,
    operation: Operation,
    product: ProductRef,
    value: torch.Tensor,
    source_metadata: Mapping[str, object],
    scope: PartitionState,
) -> ProductPayload:
    transport = runtime.transport
    if transport is None:
        raise unsupported_setup("product publication requires a configured transport")
    source_kind = metadata_string(source_metadata, "payload_kind", "")
    if source_kind != product.kind.value and not (
        product.kind is ProductKind.ARTIFACT and source_kind == "image_nchw"
    ):
        raise invalid_descriptor("product transfer changes the physical product kind")
    generation = int(product.generation)
    height = metadata_uint(source_metadata, "height", 0)
    width = metadata_uint(source_metadata, "width", 0)
    locator_metadata: dict[str, object]
    descriptor_value: dict[str, object]
    if product.kind is ProductKind.LATENT:
        latent_units = metadata_uint(source_metadata, "latent_units", 0)
        step = metadata_uint(source_metadata, "step", 0)
        if min(height, width, latent_units, generation) < 1:
            raise invalid_descriptor("latent transfer has incomplete physical metadata")
        locator_metadata = {
            "generation": generation,
            "height": height,
            "latent_units": latent_units,
            "step": step,
            "width": width,
        }
        descriptor_kind = "latent"
        descriptor_value = dict(locator_metadata)
    elif product.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}:
        if min(height, width, generation) < 1:
            raise invalid_descriptor("encoder transfer has incomplete geometry")
        locator_metadata = {
            "generation": generation,
            "height": height,
            "payload_kind": product.kind.value,
            "width": width,
        }
        descriptor_kind = "encoder"
        descriptor_value = dict(locator_metadata)
    elif requires_device_product_binding(product):
        value_range = metadata_string(source_metadata, "value_range", "")
        if (height == 0) != (width == 0):
            raise invalid_descriptor("device-product transfer has incomplete geometry")
        if value_range not in {"", *(member.value for member in ImageRange)}:
            raise invalid_descriptor("device-product transfer has an invalid value range")
        if height == 0 and value_range:
            raise invalid_descriptor("non-image device product carries an image range")
        locator_metadata = {
            "generation": generation,
            "height": height,
            "value_range": value_range,
            "width": width,
        }
        descriptor_kind = "device_product"
        descriptor_value = dict(locator_metadata)
    else:
        raise invalid_descriptor("product transfer output has no concrete physical owner")
    locator = transport.publish_async(value.detach().contiguous())
    locator = replace(locator, meta={**locator.meta, **locator_metadata})
    if not locator_matches_product(locator, product):
        transport.release(locator)
        raise invalid_descriptor("product transfer value disagrees with its output bound")
    scope.published.append(locator)
    scope.stage_publications[runtime.operation_identity(operation)] = (locator,)
    descriptor_value["locator"] = locator.to_mapping()
    descriptor = DeferredTransferPayload(
        descriptor_kind,
        descriptor_value,
        (locator,),
        transport,
    )
    return ProductPayload(product=product, payload=cast(bytes, descriptor))


def fetch_product(
    runtime: ExecutionResources,
    operation: Operation,
    scope: PartitionState,
) -> tuple[torch.Tensor, Mapping[str, object]]:
    for reference in operation.inputs:
        if reference.kind is ProductKind.LATENT:
            session = runtime.request_row(scope, operation.request_key.session_id)
            if session.latent_product != reference:
                raise invalid_descriptor("latent transfer does not name the committed trajectory")
            row = runtime.latent_row(operation, scope)
            value = runtime.require_latent_pool().gather_current(
                row.request_pool_idx,
                row.staging,
                step=int(row.placement.start_step),
                generation=int(reference.generation),
                latent_units=int(row.placement.latent_units),
                height=int(row.placement.height),
                width=int(row.placement.width),
            )
            return value, {
                "payload_kind": ProductKind.LATENT.value,
                "height": int(row.placement.height),
                "latent_units": int(row.placement.latent_units),
                "width": int(row.placement.width),
                "step": int(row.placement.start_step),
                "generation": int(reference.generation),
            }
        if reference.storage_class is StorageClass.DEVICE_TENSOR:
            device_read = runtime.consume_device_product(
                reference,
                scope,
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
            encoder_read = runtime.consume_encoder_feature(
                reference,
                scope,
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
    value = metadata.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"product metadata field {name!r} must be a non-negative integer")
    return value


def requires_device_product_binding(reference: ProductRef) -> bool:
    return reference.storage_class is StorageClass.DEVICE_TENSOR or (
        reference.storage_class is StorageClass.LATENT_ARENA
        and reference.kind is ProductKind.ARTIFACT
    )


def transferable(reference: ProductRef) -> bool:
    return (
        reference.kind is ProductKind.LATENT
        or reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
        or requires_device_product_binding(reference)
    )


def metadata_string(metadata: Mapping[str, object], name: str, default: str) -> str:
    value = metadata.get(name, default)
    if not isinstance(value, str):
        raise invalid_descriptor(f"product metadata field {name!r} must be a string")
    return value


def locator_matches_product(locator: Locator, product: ProductRef) -> bool:
    shape = tuple(int(value) for value in locator.shape)
    elements = math.prod(shape)
    bounds = product.shape_bound.dims
    if any(isinstance(bound, DeviceDim) for bound in bounds):
        shape_matches = 0 < elements <= product.shape_bound.max_elements
    elif bounds:
        shape_matches = shape == tuple(cast(StaticDim, bound).extent for bound in bounds)
    else:
        shape_matches = elements == 1
    dtype, element_bytes = device_product_storage(product.dtype)
    return (
        all(value > 0 for value in shape)
        and shape_matches
        and locator.dtype == dtype
        and int(locator.nbytes) == elements * element_bytes
    )


__all__ = ["run_action"]
