"""Image encode, latent decode, and materialization transformations."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import torch

from uniserve_worker.batch import (
    EncodeMode,
    FinishFlags,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    StorageClass,
    TokenSpan,
    ForwardMode,
)
from uniserve_worker.foundation.errors import capability_mismatch, invalid_descriptor
from uniserve_worker.models.generation import LatentLayout, Materialization
from uniserve_worker.models.inputs import FeatureLayout, PatchTransform
from uniserve_worker.models.runtime import PositionLayout
from uniserve_worker.nn.vision import get_flattened_position_ids_extrapolate
from uniserve_worker.runtime.device_products import (
    DeviceProductMetadata,
    DeviceProductWrite,
    ImageRange,
)
from uniserve_worker.runtime.encoder_cache import EncoderMetadata, EncoderWrite
from uniserve_worker.runtime.latent_pool import LatentRelease
from uniserve_worker.server.completion import (
    _CompletionImagePayload,
    _CompletionTransferPayload,
)
from uniserve_worker.server.image_codec import quantize_image_hwc

from . import step as ops
from . import transfer
from ._inputs import PreparedImage, patch_grid_shape, prepare_image, prepare_tensor_image
from .forward_batch import ModelPhase, TokenSelection
from .rows import (
    ForwardRow,
    OperationState,
    Outcome,
    PartitionState,
    StateOutcome,
)


def pack_forward(runtime: object, state: OperationState) -> tuple[object, ...]:
    if state.phase != "initial":
        return ()
    if state.operation.work.encode_mode is not None:
        return _pack_encode(runtime, state)
    if state.operation.work is ForwardMode.MATERIALIZE:
        return _pack_materialize(runtime, state)
    return ()


def consume_forward(
    runtime: object,
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    if state.phase != "forward_pending" or len(outputs) != 1:
        raise RuntimeError("encode forward result is not aligned")
    if state.data["mode"] == "encode":
        _consume_encode(runtime, state, outputs[0])
    else:
        state.data["image_tensor"] = decoded_tensor(outputs[0]).detach()
        state.data["image_range"] = ImageRange.UNIT
        _finish_materialize(runtime, state)


def run_action(runtime: object, state: OperationState) -> bool:
    if state.phase != "action":
        return False
    if state.data["mode"] == "frames":
        state.outcome = materialize_frames(runtime, state.operation, state.partition)
        state.phase = "done"
        return True
    _finish_materialize(runtime, state)
    return True


def _pack_encode(runtime: object, state: OperationState) -> tuple[object, ...]:
    from . import step as ops

    operation = state.operation
    partition = state.partition
    image_spec = ops._image_processor(runtime)
    mode = operation.work.encode_mode
    if mode is None:
        raise invalid_descriptor("encode operation is missing an encode mode")
    feature_outputs = tuple(
        output
        for output in operation.outputs
        if output.storage_class is StorageClass.LATENT_ARENA
        and output.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
    )
    if len(feature_outputs) != 1:
        raise invalid_descriptor("encode operation requires one resident feature output")
    feature_output = feature_outputs[0]
    if int(feature_output.generation) < 1:
        raise invalid_descriptor("encode feature output requires a positive generation")
    source = encode_source(runtime, operation, partition)
    target_device = runtime._generation_device if mode is EncodeMode.LATENT else runtime._device
    if isinstance(source, tuple):
        source_tensor, source_metadata = source
        prepared = prepare_tensor_image(
            image_spec,
            mode,
            source_tensor,
            device=target_device,
            signed_unit=source_metadata.value_range is ImageRange.SIGNED_UNIT,
        )
    else:
        prepared = prepare_image(image_spec, mode, source, device=target_device)
    task = encode_row(runtime, operation, mode, prepared, partition)
    state.data.update(
        mode="encode",
        prepared=prepared,
        feature_output=feature_output,
        task=task,
    )
    state.rows = (task,)
    state.phase = "forward_pending"
    return state.rows


def _consume_encode(runtime: object, state: OperationState, output: torch.Tensor) -> None:
    from . import step as ops

    operation = state.operation
    partition = state.partition
    prepared = state.data["prepared"]
    feature_output = state.data["feature_output"]
    features = encode_features(output).detach()
    write = bound_encoder_write(partition, feature_output)
    resident = runtime.encoder_cache.publish(
        write,
        features,
        EncoderMetadata(height=prepared.height, width=prepared.width),
    )
    products: tuple[ProductPayload, ...] = ()
    if (
        runtime.transport is not None
        and runtime.transport.name != "local"
        and int(runtime.deployment.tp_rank) == 0
    ):
        locator = runtime.transport.publish_async(resident)
        locator = replace(
            locator,
            meta={
                **locator.meta,
                "generation": int(feature_output.generation),
                "height": int(prepared.height),
                "payload_kind": feature_output.kind.value,
                "width": int(prepared.width),
            },
        )
        partition.published.append(locator)
        partition.stage_publications[ops._operation_identity(operation)] = (locator,)
        descriptor = _CompletionTransferPayload(
            "encoder",
            {
                "generation": int(feature_output.generation),
                "locator": locator.to_mapping(),
                "payload_kind": feature_output.kind.value,
                "height": prepared.height,
                "width": prepared.width,
            },
            (locator,),
            operation.plan_digest,
            runtime.transport,
        )
        products = (ProductPayload(product=feature_output, payload=cast(bytes, descriptor)),)
    state.outcome = non_state_outcome(runtime, operation, partition, products=products)
    state.phase = "done"


def _pack_materialize(runtime: object, state: OperationState) -> tuple[object, ...]:
    from . import step as ops

    operation = state.operation
    partition = state.partition
    session = ops._request_row(runtime, partition, operation.request_key.session_id)
    latent_inputs = tuple(
        reference for reference in operation.inputs if reference.kind is ProductKind.LATENT
    )
    if not latent_inputs:
        state.data["mode"] = "frames"
        state.phase = "action"
        return ()
    if len(latent_inputs) != 1:
        raise invalid_descriptor("materialization requires one exact latent generation")
    latent_input = latent_inputs[0]
    if int(latent_input.generation) < 1 or session.latent_product != latent_input:
        raise invalid_descriptor("materialization does not name the current latent generation")
    flow = ops._generation(runtime)
    image_params = session.image
    if image_params is None:
        raise invalid_descriptor("image materialization has no admitted image parameters")
    row = ops._latent_row(runtime, operation, partition)
    if int(row.placement.start_step) != int(image_params.steps):
        raise invalid_descriptor("image materialization requires a completed latent trajectory")
    current = ops._latent_pool(runtime).gather_current(
        row.request_pool_idx,
        row.staging,
        step=int(row.placement.start_step),
        generation=int(latent_input.generation),
        latent_units=int(row.placement.latent_units),
        height=int(row.placement.height),
        width=int(row.placement.width),
    )
    latent = flow.materialization_latent(
        current, int(row.placement.height), int(row.placement.width)
    )
    state.data.update(
        mode="materialize",
        session=session,
        latent_input=latent_input,
        row=row,
    )
    if flow.materialization is Materialization.DECODE_ROUTE:
        task = ops.ForwardRow(
            operation=operation,
            request=session,
            weights=ops._weights(runtime),
            phase=ModelPhase.DECODE_LATENT,
            latent=latent,
            image_height=int(row.placement.height),
            image_width=int(row.placement.width),
        )
        state.data["task"] = task
        state.rows = (task,)
        state.phase = "forward_pending"
        return state.rows
    if flow.materialization is not Materialization.RGB_LATENT:
        raise invalid_descriptor("model declares an unknown image materialization kind")
    state.data["image_tensor"] = latent.detach()
    state.data["image_range"] = ImageRange.SIGNED_UNIT
    state.phase = "action"
    return ()


def _finish_materialize(runtime: object, state: OperationState) -> None:
    from . import step as ops

    operation = state.operation
    partition = state.partition
    session = state.data["session"]
    row = state.data["row"]
    latent_input = state.data["latent_input"]
    image_tensor = state.data["image_tensor"]
    image_range = state.data["image_range"]
    artifact = artifact_product(operation)
    resident_outputs = tuple(
        output
        for output in operation.outputs
        if output.storage_class is StorageClass.LATENT_ARENA and output.kind is ProductKind.ARTIFACT
    )
    if len(resident_outputs) > 1:
        raise invalid_descriptor("materialize operation repeats its resident image product")
    if resident_outputs:
        resident_output = resident_outputs[0]
        if int(resident_output.generation) < 1:
            raise invalid_descriptor("materialized resident image requires a positive generation")
        write = bound_device_write(partition, resident_output)
        runtime.device_products.publish_write(
            write,
            image_tensor,
            metadata=DeviceProductMetadata(
                height=int(row.placement.height),
                width=int(row.placement.width),
                value_range=image_range,
            ),
        )
        partition.operation_writes.setdefault(ops._operation_identity(operation), write)
    image_task = defer_image_encoding(
        runtime,
        operation,
        image_tensor,
        image_range,
        partition,
        max_bytes=int(artifact.max_bytes),
    )
    session.latent_product = None
    session.flow_step = 0
    partition.latent_releases.append(
        LatentRelease(
            request_pool_idx=row.request_pool_idx,
            page_table=row.placement.page_table,
            generation=int(latent_input.generation),
            step=int(row.placement.start_step),
            latent_units=int(row.placement.latent_units),
            height=int(row.placement.height),
            width=int(row.placement.width),
        )
    )
    products = (ProductPayload(product=artifact, payload=cast(bytes, image_task)),)
    state.outcome = non_state_outcome(
        runtime,
        operation,
        partition,
        products=products,
        completion_tasks=(image_task,),
    )
    state.phase = "done"


def state_outcome(
    runtime,
    operation: Operation,
    outcome: StateOutcome,
    scope: PartitionState,
    *,
    base: int | None = None,
    products: tuple[ProductPayload, ...] = (),
) -> Outcome:
    session = ops._request_row(runtime, scope, operation.request_key.session_id)
    cache = ops._cache_coordinates(runtime, operation, scope)
    selected = scope.runtime_cache_lengths.get(cache[0], cache[2])
    if not isinstance(selected, int):
        raise RuntimeError("visual state completion has a dynamic KV length")
    cache = (cache[0], cache[1], selected, cache[3])
    span_base = session.logical_position if base is None else int(base)
    return Outcome(
        status=OpStatus.OK,
        selected_point=1 if operation.advances_state else 0,
        logical_lengths=ops._logical_lengths(runtime, operation, session, cache),
        token_span=TokenSpan(base=span_base, len=outcome.sampled_tokens),
        finish_flags=FinishFlags(),
        product_generations=ops._output_generations(operation),
        committed_tokens=outcome.committed_tokens,
        products=(*products, *outcome.products),
    )


def non_state_outcome(
    runtime,
    operation: Operation,
    scope: PartitionState,
    *,
    products: tuple[ProductPayload, ...] = (),
    completion_tasks: tuple[_CompletionImagePayload, ...] = (),
) -> Outcome:
    session = ops._request_row(runtime, scope, operation.request_key.session_id)
    base = session.logical_position
    return Outcome(
        status=OpStatus.OK,
        selected_point=1 if operation.advances_state else 0,
        logical_lengths=ops._logical_lengths(runtime, operation, session, None),
        token_span=TokenSpan(base=base, len=0),
        finish_flags=FinishFlags(),
        product_generations=ops._output_generations(operation),
        products=products,
        completion_tasks=completion_tasks,
    )


def encode_source(
    runtime,
    operation: Operation,
    scope: PartitionState,
) -> str | tuple[torch.Tensor, DeviceProductMetadata]:
    for reference in operation.inputs:
        inline = scope.input_images.get(reference)
        if inline is not None:
            return inline
        if reference.kind is not ProductKind.ARTIFACT:
            continue
        read = ops._consume_device_product(
            runtime,
            reference,
            scope,
            consumer_op_id=operation.op_id,
            device=ops._operation_device(runtime, operation),
        )
        metadata = read.metadata
        if (
            metadata is None
            or metadata.height < 1
            or metadata.width < 1
            or metadata.value_range is None
        ):
            raise invalid_descriptor("resident image product has incomplete geometry")
        scope.device_reads.append(read)
        return read.tensor, metadata
    raise invalid_descriptor("encode operation has no source image product")


def encode_row(
    runtime,
    operation: Operation,
    mode: EncodeMode,
    prepared: PreparedImage,
    scope: PartitionState,
) -> ForwardRow:
    session = ops._request_row(runtime, scope, operation.request_key.session_id)
    return ForwardRow(
        operation=operation,
        request=session,
        weights=ops._weights(
            runtime,
        ),
        phase=(ModelPhase.ENCODE_VISION if mode is EncodeMode.VISION else ModelPhase.ENCODE_LATENT),
        encode_pixels=prepared.pixels,
        encode_grid=prepared.grid,
        encode_grid_shape=prepared.grid_shape,
    )


def vision_state_row(
    runtime,
    operation: Operation,
    features: torch.Tensor,
    height: int,
    width: int,
    conditioning_position: int,
    scope: PartitionState,
    *,
    close_image: bool,
    logits: bool,
) -> ForwardRow:
    session = ops._request_row(runtime, scope, operation.request_key.session_id)
    cache = ops._cache_coordinates(runtime, operation, scope)
    injection = ops._image_processor(
        runtime,
    ).feature_injection
    if injection is None:
        raise invalid_descriptor("vision state stage requires declared feature injection")
    embeddings = (
        features.squeeze(0) if features.ndim == 3 and int(features.shape[0]) == 1 else features
    )
    if embeddings.ndim != 2 or int(embeddings.shape[0]) < 1:
        raise invalid_descriptor("vision features must have shape [tokens, hidden]")
    leading = injection.layout is FeatureLayout.FRAMED
    trailing = leading or close_image
    query = int(leading) + int(embeddings.shape[0]) + int(trailing)
    token_ids = torch.ones(query, dtype=torch.long)
    token_embeddings = embeddings.new_zeros((query, int(embeddings.shape[1])))
    embedding_mask = torch.zeros(query, dtype=torch.bool, device=embeddings.device)
    begin = int(leading)
    token_embeddings[begin : begin + int(embeddings.shape[0])] = embeddings
    embedding_mask[begin : begin + int(embeddings.shape[0])] = True
    if leading:
        token_ids[0] = _feature_token_id(runtime, injection, start=True)
    if trailing:
        token_ids[-1] = _feature_token_id(runtime, injection, start=False)
    positions = _vision_positions(
        runtime,
        injection.positions,
        int(embeddings.shape[0]),
        conditioning_position,
        height=height,
        width=width,
        leading=leading,
        trailing=trailing,
        close_image=close_image,
    )
    return ForwardRow(
        operation=operation,
        request=session,
        weights=ops._weights(
            runtime,
        ),
        phase=ModelPhase.TEXT,
        token_ids=token_ids,
        token_embeddings=token_embeddings,
        token_embedding_mask=embedding_mask,
        positions=positions,
        selection=TokenSelection.LAST_LOGITS if logits else TokenSelection.HIDDEN,
        request_pool_idx=cache[0],
        seq_len=cache[2],
        group_id=cache[1],
        write_kv=True,
        causal=False,
        attention_indexes=positions_as_three_axis(positions, query),
    )


def _feature_token_id(runtime, injection: Any, *, start: bool) -> int:
    value = injection.start_token_id if start else injection.end_token_id
    text = injection.start_token if start else injection.end_token
    if value is not None:
        return int(value)
    if text is None or runtime.tokenizer is None:
        raise capability_mismatch("feature marker requires a worker tokenizer or token id")
    token_id = runtime.tokenizer.convert_tokens_to_ids(text)
    if token_id is None or int(token_id) < 0:
        raise invalid_descriptor("declared feature marker is absent from the tokenizer")
    return int(token_id)


def _vision_positions(
    runtime,
    layout: PositionLayout,
    feature_tokens: int,
    conditioning_position: int,
    *,
    height: int,
    width: int,
    leading: bool,
    trailing: bool,
    close_image: bool,
) -> torch.Tensor:
    query = int(leading) + feature_tokens + int(trailing)
    if layout is PositionLayout.TEMPORAL:
        return torch.full((query,), int(conditioning_position), dtype=torch.long)
    transform = ops._image_processor(
        runtime,
    ).vit
    if not isinstance(transform, PatchTransform):
        raise invalid_descriptor(
            "temporal-spatial feature injection requires a patch image transform"
        )
    raw_height, raw_width = patch_grid_shape(transform, height, width)
    factor_squared, remainder = divmod(raw_height * raw_width, feature_tokens)
    factor = math.isqrt(factor_squared)
    if remainder or factor < 1 or factor * factor != factor_squared:
        raise invalid_descriptor("vision feature count does not align with its patch grid")
    grid_height, grid_width = raw_height // factor, raw_width // factor
    if grid_height * grid_width != feature_tokens:
        raise invalid_descriptor("vision output grid is not integral")
    temporal = torch.full(
        (query,),
        int(conditioning_position + (1 if close_image else 0)),
        dtype=torch.long,
    )
    y = torch.arange(grid_height, dtype=torch.long).repeat_interleave(grid_width)
    x = torch.arange(grid_width, dtype=torch.long).repeat(grid_height)
    spatial_y = torch.zeros(query, dtype=torch.long)
    spatial_x = torch.zeros(query, dtype=torch.long)
    begin = int(leading)
    spatial_y[begin : begin + feature_tokens] = y
    spatial_x[begin : begin + feature_tokens] = x
    if trailing and close_image:
        temporal[-1] = conditioning_position + 2
    return torch.stack((temporal, spatial_y, spatial_x))


def latent_state_row(
    runtime,
    operation: Operation,
    latent: torch.Tensor,
    height: int,
    width: int,
    conditioning_position: int,
    scope: PartitionState,
) -> ForwardRow:
    session = ops._request_row(runtime, scope, operation.request_key.session_id)
    flow = ops._generation(
        runtime,
    )
    if flow.latent_layout is not LatentLayout.PATCH_TOKENS:
        raise invalid_descriptor("flow state publication requires patch-token latents")
    image_tokens = (height // int(flow.latent_downsample)) * (width // int(flow.latent_downsample))
    if int(latent.reshape(-1, latent.shape[-1]).shape[0]) != image_tokens:
        raise invalid_descriptor("state latent does not match the declared image geometry")
    query = image_tokens + int(flow.commit_marker_tokens)
    positions = get_flattened_position_ids_extrapolate(
        height,
        width,
        int(flow.latent_downsample),
        int(math.isqrt(flow.max_latent_tokens)),
    )
    temporal = torch.full((query,), conditioning_position + 1, dtype=torch.long)
    temporal[0] = conditioning_position
    temporal[-1] = conditioning_position + int(flow.rope_advance)
    indexes = torch.stack((temporal, torch.zeros_like(temporal), torch.zeros_like(temporal)))
    cache = ops._cache_coordinates(runtime, operation, scope)
    return ForwardRow(
        operation=operation,
        request=session,
        weights=ops._weights(
            runtime,
        ),
        phase=ModelPhase.DENOISE,
        positions=positions,
        timestep=latent.new_zeros(1),
        latent=latent,
        image_tokens=query,
        image_height=height,
        image_width=width,
        request_pool_idx=cache[0],
        seq_len=cache[2],
        group_id=cache[1],
        write_kv=False,
        causal=False,
        attention_indexes=indexes,
        text_local_indices=(0, query - 1),
    )


def materialize_frames(
    runtime,
    operation: Operation,
    scope: PartitionState,
) -> Outcome:
    image, metadata = transfer.fetch_product(runtime, operation, scope)
    if transfer.metadata_string(metadata, "payload_kind", "") != "image_nchw":
        raise invalid_descriptor("frame materialization source is not an image tensor")
    value_range = ImageRange(
        transfer.metadata_string(metadata, "value_range", ImageRange.SIGNED_UNIT.value)
    )
    image_task = defer_image_encoding(
        runtime,
        operation,
        image,
        value_range,
        scope,
        max_bytes=int(operation.bounds.max_completion_bytes),
    )
    return non_state_outcome(runtime, operation, scope, completion_tasks=(image_task,))


def defer_image_encoding(
    runtime,
    operation: Operation,
    image: torch.Tensor,
    value_range: ImageRange,
    scope: PartitionState,
    *,
    max_bytes: int,
) -> _CompletionImagePayload:
    if max_bytes < 1:
        raise invalid_descriptor("image materialization requires a positive completion bound")
    quantized = quantize_image_hwc(
        image,
        value_range=((-1.0, 1.0) if value_range is ImageRange.SIGNED_UNIT else (0.0, 1.0)),
    )
    if int(quantized.numel()) > max_bytes:
        raise invalid_descriptor("image staging exceeds its registered completion byte bound")
    capture = scope.completion.capture_bytes(quantized)
    identity = ops._operation_identity(operation)
    reservation = scope.cpu_tasks.get(identity)
    if reservation is None:
        raise RuntimeError("materialization has no registered CPU task slot")
    return _CompletionImagePayload(
        capture,
        reservation,
        max_bytes,
    )


def artifact_product(operation: Operation) -> ProductRef:
    for output in operation.outputs:
        if (
            output.kind is ProductKind.ARTIFACT
            and output.storage_class is StorageClass.PINNED_OUTPUT
        ):
            return output
    raise invalid_descriptor("materialize operation has no host-visible artifact output")


def bound_device_write(scope: PartitionState, reference: ProductRef) -> DeviceProductWrite:
    matches = tuple(write for write in scope.device_writes if write.reference == reference)
    if len(matches) != 1:
        raise invalid_descriptor(
            "resident product does not have exactly one atomic registration binding"
        )
    return matches[0]


def bound_encoder_write(scope: PartitionState, reference: ProductRef) -> EncoderWrite:
    matches = tuple(write for write in scope.encoder_writes if write.reference == reference)
    if len(matches) != 1:
        raise invalid_descriptor(
            "encoder feature does not have exactly one atomic registration binding"
        )
    return matches[0]


def encode_features(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("encode route did not return encoder features")
    return output


def decoded_tensor(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("decode route did not return an image tensor")
    return output


def positions_as_three_axis(positions: torch.Tensor, query: int) -> torch.Tensor:
    if positions.ndim == 1 and int(positions.numel()) == query:
        return torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
    if positions.ndim == 2 and tuple(positions.shape) == (3, query):
        return positions
    raise invalid_descriptor("state positions do not align with their physical token row")


__all__ = ["consume_forward", "pack_forward", "run_action"]
