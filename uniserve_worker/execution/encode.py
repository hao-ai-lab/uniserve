"""Image encode, latent decode, and materialization transformations."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import torch

from uniserve_worker.execution.batch import (
    EncoderTransferValue,
    FinishFlags,
    ForwardMode,
    OpStatus,
    PipelineStage,
    ScheduledRequest,
    TensorPublication,
    TensorRef,
    TensorTransfer,
)
from uniserve_worker.execution.output import (
    ImageEncoding,
)
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.media.codec import quantize_image_hwc
from uniserve_worker.models.generation import LatentLayout, Materialization
from uniserve_worker.models.inputs import FeatureInjection, FeatureLayout, PatchTransform
from uniserve_worker.models.runtime import PositionLayout
from uniserve_worker.nn.vision import get_flattened_position_ids_extrapolate
from uniserve_worker.runtime.device_products import (
    DeviceProductMetadata,
    DeviceProductWrite,
    ImageRange,
)
from uniserve_worker.runtime.encoder_cache import EncoderMetadata, EncoderWrite
from uniserve_worker.runtime.latent_pool import LatentRelease
from uniserve_worker.transfer.tickets import publish_tensor

from ..runtime.latent_pool import require_latent_pool
from . import operations as operation_geometry
from . import transfer
from .forward_batch import TokenSelection
from .image_input import PreparedImage, patch_grid_shape, prepare_image, prepare_tensor_image
from .rows import (
    ForwardRow,
    LaneState,
    OperationState,
    Outcome,
    StateOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers import PreTrainedTokenizerBase

    from ..bootstrap.worker_info import WorkerInfo
    from ..config import WorkerConfig
    from ..runtime.device_products import DeviceProducts
    from ..runtime.encoder_cache import EncoderCache
    from ..runtime.latent_pool import LatentPool
    from ..runtime.req_to_token_pool import ReqToTokenPool
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner


def pack_forward(
    state: OperationState,
    *,
    device_products: DeviceProducts,
    latent_pool: LatentPool | None,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> tuple[ForwardRow, ...]:
    """Pack an encoder or latent-decoder operation into a single-row model forward."""

    if state.phase != "initial":
        return ()
    if state.operation.kind in {PipelineStage.VISION_ENCODING, PipelineStage.LATENT_ENCODING}:
        return _pack_encode(
            state,
            device_products=device_products,
            model_runner=model_runner,
            config=config,
        )
    if state.operation.kind is PipelineStage.IMAGE_DECODING:
        return _pack_diffusion_finalize(
            state,
            latent_pool=latent_pool,
            model_runner=model_runner,
        )
    return ()


def consume_forward(
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    publication_transports: Mapping[str, Transport],
    config: WorkerConfig,
) -> None:
    """Publish encoder features or begin deferred latent reconstruction from model output."""

    if state.phase != "forward_pending" or len(outputs) != 1:
        raise RuntimeError("encode forward result is not aligned")
    if state.data["mode"] == "encode":
        _consume_encode(
            state,
            outputs[0],
            encoder_cache=encoder_cache,
            worker_info=worker_info,
            publication_transports=publication_transports,
            config=config,
        )
    else:
        state.data["image_tensor"] = decoded_tensor(outputs[0]).detach()
        state.data["image_range"] = ImageRange.UNIT
        _finish_diffusion_finalize(state, device_products=device_products)


def run_action(
    state: OperationState,
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    publication_transports: Mapping[str, Transport],
    model_runner: ModelRunner,
) -> bool:
    """Execute encoder finalization work that does not require a model forward."""

    if state.phase == "initial" and state.operation.kind is PipelineStage.TEXT_ENCODING:
        _encode_text(
            state,
            device_products=device_products,
            encoder_cache=encoder_cache,
            publication_transports=publication_transports,
            model_runner=model_runner,
        )
        return True
    if state.phase != "action":
        return False
    if state.data["mode"] == "frames":
        state.outcome = diffusion_finalize_frames(
            state.operation,
            state.lane,
            device_products=device_products,
            encoder_cache=encoder_cache,
            model_runner=model_runner,
        )
        state.phase = "done"
        return True
    _finish_diffusion_finalize(state, device_products=device_products)
    return True


def _encode_text(
    state: OperationState,
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    publication_transports: Mapping[str, Transport],
    model_runner: ModelRunner,
) -> None:
    """Encode admitted conditioning tokens and publish their declared tensors."""

    operation, scope = state.operation, state.lane
    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    admission = request.request.admission
    if admission.diffusion is None or not admission.prompt_token_ids:
        raise invalid_descriptor("text conditioning requires admitted prompt tokens")
    if not operation.outputs:
        raise invalid_descriptor("text encoder outputs must declare conditioning tensors")
    tokens = model_runner.stage_text_tokens(admission.prompt_token_ids)
    result = model_runner.run_entry(
        operation.entry,
        tokens,
    )
    if len(result.values) != len(operation.outputs):
        raise invalid_descriptor("text encoder output declarations disagree with the loaded entry")
    products = transfer.publish_tensors(
        operation,
        result.values,
        scope,
        device_products=device_products,
        encoder_cache=encoder_cache,
        publication_transports=publication_transports,
    )
    scope.observations.append(result.observation)
    state.outcome = Outcome(
        status=OpStatus.OK,
        runtime=operation_geometry.execution_runtime(request, None),
        finish_flags=FinishFlags(),
        product_generations=tuple(output.generation for output in operation.tensor_outputs()),
        products=products,
    )
    state.phase = "done"


def _pack_encode(
    state: OperationState,
    *,
    device_products: DeviceProducts,
    model_runner: ModelRunner,
    config: WorkerConfig,
) -> tuple[ForwardRow, ...]:
    """Stage image tensors and build the model batch for one encoder operation."""

    operation = state.operation
    scope = state.lane
    image_processor = model_runner.image_processor()
    mode = operation.kind
    if not isinstance(mode, PipelineStage) or mode not in {
        PipelineStage.VISION_ENCODING,
        PipelineStage.LATENT_ENCODING,
    }:
        raise invalid_descriptor("encode operation is missing an encode mode")
    feature_output = operation.encoder_output
    if feature_output is None:
        raise invalid_descriptor("encode operation requires one resident feature output")
    if int(feature_output.generation) < 1:
        raise invalid_descriptor("encode feature output requires a positive generation")
    source = encode_source(
        operation, scope, device_products=device_products, model_runner=model_runner
    )
    target_device = (
        torch.device((config.generation_device or config.device))
        if mode is PipelineStage.LATENT_ENCODING
        else torch.device(config.device)
    )
    if isinstance(source, tuple):
        source_tensor, source_metadata = source
        prepared = prepare_tensor_image(
            image_processor,
            mode,
            source_tensor,
            device=target_device,
            signed_unit=source_metadata.value_range is ImageRange.SIGNED_UNIT,
        )
    else:
        prepared = prepare_image(image_processor, mode, source, device=target_device)
    task = encode_row(operation, mode, prepared, scope)
    state.data.update(
        mode="encode",
        prepared=prepared,
        feature_output=feature_output,
        task=task,
    )
    state.rows = (task,)
    state.phase = "forward_pending"
    return state.rows


def _consume_encode(
    state: OperationState,
    output: torch.Tensor,
    *,
    encoder_cache: EncoderCache,
    worker_info: WorkerInfo,
    publication_transports: Mapping[str, Transport],
    config: WorkerConfig,
) -> None:
    """Split encoded features by request and prepare cache or product publication."""

    operation = state.operation
    scope = state.lane
    prepared = state.data["prepared"]
    feature_output = state.data["feature_output"]
    features = encode_features(output).detach()
    write = bound_encoder_write(scope, feature_output)
    resident = encoder_cache.publish(
        write,
        features,
        EncoderMetadata(height=prepared.height, width=prepared.width),
    )
    products: tuple[TensorPublication, ...] = ()
    if any(
        name != "local" for name in publication_transports
    ) and config.rank == worker_info.output_rank(operation.entry):
        locations = publish_tensor(
            publication_transports,
            resident,
            retain=partial(encoder_cache.retain_publication, write),
        )
        scope.published.extend(locations)
        scope.stage_publications[feature_output.buffer_id] = locations
        descriptor = EncoderTransferValue(
            tensor=TensorTransfer(shape=tuple(resident.shape), locations=locations),
            payload_kind="vision_feature"
            if operation.kind is PipelineStage.VISION_ENCODING
            else "latent_feature",
            height=prepared.height,
            width=prepared.width,
        )
        products = (TensorPublication(product=feature_output, value=descriptor),)
    state.outcome = non_state_outcome(operation, scope, products=products)
    state.phase = "done"


def _pack_diffusion_finalize(
    state: OperationState,
    *,
    latent_pool: LatentPool | None,
    model_runner: ModelRunner,
) -> tuple[ForwardRow, ...]:
    """Gather the final latent trajectory and build its decoder batch."""

    operation = state.operation
    scope = state.lane
    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    latent_input = operation.latent_input
    if latent_input is None:
        state.data["mode"] = "frames"
        state.phase = "action"
        return ()
    if int(latent_input.generation) < 1 or request.latent_product != latent_input:
        raise invalid_descriptor("materialization does not name the current latent generation")
    flow = model_runner.generation()
    image_params = request.request.image
    if image_params is None:
        raise invalid_descriptor("image materialization has no admitted image parameters")
    row = operation_geometry.latent_row(operation, scope)
    if int(row.params.start_step) != int(image_params.steps):
        raise invalid_descriptor("image materialization requires a completed latent trajectory")
    current = require_latent_pool(latent_pool).gather_current(
        row.request_pool_idx,
        row.staging,
        step=int(row.params.start_step),
        generation=int(latent_input.generation),
        latent_units=int(row.params.latent_units),
        height=int(row.params.height),
        width=int(row.params.width),
    )
    latent = flow.materialization_latent(current, int(row.params.height), int(row.params.width))
    state.data.update(
        mode="diffusion_finalize",
        latent_input=latent_input,
        row=row,
    )
    if flow.materialization is Materialization.DECODE_ROUTE:
        task = ForwardRow(
            operation=operation,
            request=request,
            forward_mode=PipelineStage.IMAGE_DECODING,
            latent=latent,
            image_height=int(row.params.height),
            image_width=int(row.params.width),
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


def _finish_diffusion_finalize(state: OperationState, *, device_products: DeviceProducts) -> None:
    """Decode final latents and schedule bounded image-output publication."""

    operation = state.operation
    scope = state.lane
    request = operation_geometry.request_row(state.lane, state.operation.request_key.request_id)
    row = state.data["row"]
    latent_input = state.data["latent_input"]
    image_tensor = state.data["image_tensor"]
    image_range = state.data["image_range"]
    resident_output = operation.image_output
    if resident_output is not None:
        if int(resident_output.generation) < 1:
            raise invalid_descriptor("finalized resident image requires a positive generation")
        write = bound_device_write(scope, resident_output)
        # Feedback storage uses the declared model dtype; preserve the decoder's
        # existing conversion before handing the value to the tensor store.
        storage = device_products.producer_write_views((write,))[0]
        device_products.publish_write(
            write,
            image_tensor.to(dtype=storage.dtype),
            metadata=DeviceProductMetadata(
                height=int(row.params.height),
                width=int(row.params.width),
                value_range=image_range,
            ),
        )
        scope.operation_writes.setdefault(operation_geometry.operation_identity(operation), write)
    image_task = defer_image_encoding(
        operation,
        image_tensor,
        image_range,
        scope,
        max_bytes=int(operation.bounds.max_completion_bytes),
    )
    request.latent_product = None
    request.flow_step = 0
    scope.latent_releases.append(
        LatentRelease(
            request_pool_idx=row.request_pool_idx,
            page_table=row.params.page_table,
            generation=int(latent_input.generation),
            step=int(row.params.start_step),
            latent_units=int(row.params.latent_units),
            height=int(row.params.height),
            width=int(row.params.width),
        )
    )
    state.outcome = non_state_outcome(
        operation,
        scope,
        completion_tasks=(image_task,),
    )
    state.phase = "done"


def state_outcome(
    operation: ScheduledRequest,
    outcome: StateOutcome,
    scope: LaneState,
    *,
    products: tuple[TensorPublication, ...] = (),
    request_tables: ReqToTokenPool | None,
) -> Outcome:
    """Stage execution progress and defer successor publication until stateful tensors are ready."""

    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    cache = operation_geometry.cache_coordinates(operation, scope, tables=request_tables)
    selected = scope.runtime_cache_lengths.get(cache[0], cache[2])
    if not isinstance(selected, int):
        raise RuntimeError("visual state completion has a dynamic KV length")
    cache = (cache[0], cache[1], selected, cache[3])
    return Outcome(
        status=OpStatus.OK,
        runtime=operation_geometry.execution_runtime(request, cache),
        finish_flags=FinishFlags(),
        product_generations=operation_geometry.output_generations(operation),
        committed_tokens=outcome.committed_tokens,
        sampling=outcome.sampling,
        products=products,
        logprobs=outcome.logprobs,
        prompt_logprobs=outcome.prompt_logprobs,
    )


def non_state_outcome(
    operation: ScheduledRequest,
    scope: LaneState,
    *,
    products: tuple[TensorPublication, ...] = (),
    completion_tasks: tuple[ImageEncoding, ...] = (),
) -> Outcome:
    """Record a stateless completion and its already materialized output products."""

    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    return Outcome(
        status=OpStatus.OK,
        runtime=operation_geometry.execution_runtime(request, None),
        finish_flags=FinishFlags(),
        product_generations=operation_geometry.output_generations(operation),
        products=products,
        completion_tasks=completion_tasks,
    )


def encode_source(
    operation: ScheduledRequest,
    scope: LaneState,
    *,
    device_products: DeviceProducts,
    model_runner: ModelRunner,
) -> str | tuple[torch.Tensor, DeviceProductMetadata]:
    """Resolve encoded request media and stage it according to the model’s image policy."""

    if operation.input_image is not None:
        return operation.input_image
    reference = operation.image_input
    if reference is not None:
        read = device_products.consume(
            reference,
            consumer_op_id=operation.op_id,
            device=model_runner.operation_device(operation),
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
    operation: ScheduledRequest,
    mode: PipelineStage,
    prepared: PreparedImage,
    scope: LaneState,
) -> ForwardRow:
    """Build a vision- or latent-encoder row from prepared image tensors."""

    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    return ForwardRow(
        operation=operation,
        request=request,
        forward_mode=mode,
        encode_pixels=prepared.pixels,
        encode_grid=prepared.grid,
        encode_grid_shape=prepared.grid_shape,
    )


def vision_state_row(
    operation: ScheduledRequest,
    features: torch.Tensor,
    height: int,
    width: int,
    conditioning_position: int,
    scope: LaneState,
    *,
    close_image: bool,
    logits: bool,
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
    tokenizer: PreTrainedTokenizerBase | None,
) -> ForwardRow:
    """Publish vision features and construct the request runtime that references their token span."""

    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    cache = operation_geometry.cache_coordinates(operation, scope, tables=request_tables)
    injection = model_runner.image_processor().feature_injection
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
        token_ids[0] = _feature_token_id(injection, start=True, tokenizer=tokenizer)
    if trailing:
        token_ids[-1] = _feature_token_id(injection, start=False, tokenizer=tokenizer)
    positions = _vision_positions(
        injection.positions,
        int(embeddings.shape[0]),
        conditioning_position,
        height=height,
        width=width,
        leading=leading,
        trailing=trailing,
        close_image=close_image,
        model_runner=model_runner,
    )
    return ForwardRow(
        operation=operation,
        request=request,
        forward_mode=ForwardMode.PREFILL,
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


def _feature_token_id(
    injection: FeatureInjection, *, start: bool, tokenizer: PreTrainedTokenizerBase | None
) -> int:
    """Resolve the configured opening or closing token for image-feature injection."""

    value = injection.start_token_id if start else injection.end_token_id
    text = injection.start_token if start else injection.end_token
    if value is not None:
        return int(value)
    if text is None or tokenizer is None:
        raise unsupported_setup("feature marker requires a worker tokenizer or token id")
    token_id = tokenizer.convert_tokens_to_ids(text)
    if not isinstance(token_id, int) or token_id < 0:
        raise invalid_descriptor("declared feature marker is absent from the tokenizer")
    return int(token_id)


def _vision_positions(
    layout: PositionLayout,
    feature_tokens: int,
    conditioning_position: int,
    *,
    height: int,
    width: int,
    leading: bool,
    trailing: bool,
    close_image: bool,
    model_runner: ModelRunner,
) -> torch.Tensor:
    """Build temporal-height-width positions and boundary markers for vision tokens."""

    query = int(leading) + feature_tokens + int(trailing)
    if layout is PositionLayout.TEMPORAL:
        return torch.full((query,), int(conditioning_position), dtype=torch.long)
    transform = model_runner.image_processor().vit
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
    operation: ScheduledRequest,
    latent: torch.Tensor,
    height: int,
    width: int,
    conditioning_position: int,
    scope: LaneState,
    *,
    request_tables: ReqToTokenPool | None,
    model_runner: ModelRunner,
) -> ForwardRow:
    """Publish encoded image latents and construct the request runtime for diffusion conditioning."""

    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    flow = model_runner.generation()
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
    cache = operation_geometry.cache_coordinates(operation, scope, tables=request_tables)
    return ForwardRow(
        operation=operation,
        request=request,
        forward_mode=PipelineStage.DENOISING,
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


def diffusion_finalize_frames(
    operation: ScheduledRequest,
    scope: LaneState,
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    model_runner: ModelRunner,
) -> Outcome:
    """Validate finalized diffusion output and return RGB frames with their numeric range."""

    image, metadata = transfer.fetch_product(
        operation,
        scope,
        device_products=device_products,
        encoder_cache=encoder_cache,
        model_runner=model_runner,
    )
    if transfer.metadata_string(metadata, "payload_kind", "") != "image_nchw":
        raise invalid_descriptor("frame materialization source is not an image tensor")
    value_range = ImageRange(
        transfer.metadata_string(metadata, "value_range", ImageRange.SIGNED_UNIT.value)
    )
    image_task = defer_image_encoding(
        operation,
        image,
        value_range,
        scope,
        max_bytes=int(operation.bounds.max_completion_bytes),
    )
    return non_state_outcome(operation, scope, completion_tasks=(image_task,))


def defer_image_encoding(
    operation: ScheduledRequest,
    image: torch.Tensor,
    value_range: ImageRange,
    scope: LaneState,
    *,
    max_bytes: int,
) -> ImageEncoding:
    """Reserve output storage and schedule image encoding after the device copy completes."""

    if max_bytes < 1:
        raise invalid_descriptor("image materialization requires a positive completion bound")
    quantized = quantize_image_hwc(
        image,
        value_range=((-1.0, 1.0) if value_range is ImageRange.SIGNED_UNIT else (0.0, 1.0)),
    )
    if int(quantized.numel()) > max_bytes:
        raise invalid_descriptor("image staging exceeds its registered completion byte bound")
    capture = scope.completion.capture_bytes(quantized)
    identity = operation_geometry.operation_identity(operation)
    reservation = scope.cpu_tasks.get(identity)
    if reservation is None:
        raise RuntimeError("materialization has no registered CPU task slot")
    return ImageEncoding(
        capture,
        reservation,
        max_bytes,
    )


def bound_device_write(scope: LaneState, reference: TensorRef) -> DeviceProductWrite:
    """Return the staged device-product write matching a declared output reference."""

    matches = tuple(write for write in scope.device_writes if write.reference == reference)
    if len(matches) != 1:
        raise invalid_descriptor(
            "resident product does not have exactly one atomic registration binding"
        )
    return matches[0]


def bound_encoder_write(scope: LaneState, reference: TensorRef) -> EncoderWrite:
    """Return the staged encoder-cache write matching a declared output reference."""

    matches = tuple(write for write in scope.encoder_writes if write.reference == reference)
    if len(matches) != 1:
        raise invalid_descriptor(
            "encoder feature does not have exactly one atomic registration binding"
        )
    return matches[0]


def encode_features(output: torch.Tensor) -> torch.Tensor:
    """Normalize model encoder output to a two-dimensional token-by-width tensor."""

    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("encode route did not return encoder features")
    return output


def decoded_tensor(output: torch.Tensor) -> torch.Tensor:
    """Extract the reconstructed image tensor from a supported model output wrapper."""

    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("decode route did not return an image tensor")
    return output


def positions_as_three_axis(positions: torch.Tensor, query: int) -> torch.Tensor:
    """Expand temporal positions into the three-axis layout required by multimodal decoders."""

    if positions.ndim == 1 and int(positions.numel()) == query:
        return torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
    if positions.ndim == 2 and tuple(positions.shape) == (3, query):
        return positions
    raise invalid_descriptor("state positions do not align with their physical token row")


__all__ = ["consume_forward", "pack_forward", "run_action"]
