"""Image encode, latent decode, and materialization transformations."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch

from uniserve.media import image
from uniserve_models.processing import (
    FeatureInjection,
    FeatureLayout,
    PatchTransform,
    PositionLayout,
)
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.execution.sampling import TokenSelection
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.media.codec import quantize_image_hwc, uint8_image_to_png_base64_bytes
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.operation import (
    ForwardMode,
    OpStatus,
    PipelineStage,
    ScheduledRequest,
)
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.protocol.tensor import TensorRef
from uniserve_worker.protocol.transfer import EncoderTransferValue, TensorTransfer
from uniserve_worker.runtime.cpu import CpuTask
from uniserve_worker.runtime.tensor_store import FeatureMetadata, ImageMetadata, TensorRecord
from uniserve_worker.transfer.tickets import publish_tensor

from . import operations, transfer
from .batch_state import BatchState
from .image_input import PreparedImage, patch_grid_shape, prepare_image, prepare_tensor_image
from .rows import ForwardRow

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..bootstrap.worker_info import WorkerInfo
    from ..config import WorkerConfig
    from ..runtime.block_tables import BlockTables
    from ..runtime.latent_pool import LatentPool
    from ..runtime.tensor_store import TensorStore
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner


def text(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    model_runner: ModelRunner,
) -> PendingOutput:
    """Encode admitted conditioning tokens and publish their declared tensors."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    admission = request.request.admission
    if admission.diffusion is None or not admission.prompt_token_ids:
        raise invalid_descriptor("text conditioning requires admitted prompt tokens")
    if not operation.outputs:
        raise invalid_descriptor("text encoder outputs must declare conditioning tensors")
    tokens = model_runner.stage_text_tokens(admission.prompt_token_ids)
    result = model_runner.run_encoder(
        "text",
        tokens,
    )
    if len(result.values) != len(operation.outputs):
        raise invalid_descriptor("text encoder output declarations disagree with the loaded entry")
    products = transfer.publish_tensors(
        operation,
        result.values,
        completion_group,
        tensor_store=tensor_store,
        publication_transports=publication_transports,
        state=state,
    )
    if result.stats is None:
        raise RuntimeError("module output has no execution statistics")
    state.group_forward_stats[completion_group].append(result.stats)
    request.status = OpStatus.OK
    request.projected_progress = operations.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = tuple(output.generation for output in operation.tensor_outputs())
    request.products = products
    return request


def prepare_features(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelRunner,
) -> PreparedImage:
    """Stage image tensors and build the model batch for one encoder operation."""

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
        operation,
        completion_group,
        tensor_store=tensor_store,
        model_runner=model_runner,
        state=state,
    )
    target_device = model_runner.operation_devices(operation)[1]
    if isinstance(source, tuple):
        source_tensor, source_metadata = source
        prepared = prepare_tensor_image(
            image_processor,
            mode,
            source_tensor,
            device=target_device,
            signed_unit=source_metadata.value_range == (-1.0, 1.0),
        )
    else:
        prepared = prepare_image(image_processor, mode, source, device=target_device)
    return prepared


def publish_features(
    operation: ScheduledRequest,
    completion_group: int,
    prepared: PreparedImage,
    output: torch.Tensor,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    publication_transports: Mapping[str, Transport],
    config: WorkerConfig,
) -> PendingOutput:
    """Split encoded features by request and prepare cache or product publication."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    feature_output = operation.encoder_output
    if feature_output is None:
        raise invalid_descriptor("encoder output has no feature reference")
    features = output.detach()
    write = bound_encoder_write(completion_group, feature_output, state=state)
    resident = tensor_store.publish_write(
        write,
        features,
        metadata=FeatureMetadata(height=prepared.height, width=prepared.width),
    )
    products: tuple[TensorPublication, ...] = ()
    if any(
        name != "local" for name in publication_transports
    ) and config.rank == worker_info.output_rank(operation.entry):
        locations = publish_tensor(
            publication_transports,
            resident,
            retain=partial(tensor_store.retain_publication, write),
        )
        request.exported_locators.extend(locations)
        request.tensor_exports[feature_output.buffer_id] = tuple(
            (publication_transports[location.backend], location) for location in locations
        )
        descriptor = EncoderTransferValue(
            tensor=TensorTransfer(shape=tuple(resident.shape), locations=locations),
            payload_kind="vision_feature"
            if operation.kind is PipelineStage.VISION_ENCODING
            else "latent_feature",
            height=prepared.height,
            width=prepared.width,
        )
        products = (TensorPublication(product=feature_output, value=descriptor),)
    return non_state_outcome(operation, completion_group, products=products, state=state)


def materialization_latent(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    latent_pool: LatentPool,
    model_runner: ModelRunner,
) -> torch.Tensor:
    """Gather the final latent trajectory and build its decoder batch."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    latent_input = operation.latent_input
    if latent_input is None:
        raise invalid_descriptor("latent materialization requires a latent input")
    if (
        int(latent_input.generation) < 1
        or operations.require_progress(request).latent_product != latent_input
    ):
        raise invalid_descriptor("materialization does not name the current latent generation")
    image_params = request.request.image
    if image_params is None:
        raise invalid_descriptor("image materialization has no admitted image parameters")
    row = state.pending_output(completion_group, operation.request_key.request_id)
    params = row.input_latent_params
    staging = row.latent_staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory operation has no staged latent inputs")
    if int(params.start_step) != int(image_params.steps):
        raise invalid_descriptor("image materialization requires a completed latent trajectory")
    current = latent_pool.gather_current(
        row.request.request_pool_idx,
        staging,
        step=int(params.start_step),
        generation=int(latent_input.generation),
        latent_units=int(params.latent_units),
        height=int(params.height),
        width=int(params.width),
    )
    return current


def publish_image(
    operation: ScheduledRequest,
    completion_group: int,
    image_tensor: torch.Tensor,
    image_range: tuple[float, float],
    *,
    state: BatchState,
    tensor_store: TensorStore,
) -> PendingOutput:
    """Decode final latents and schedule bounded image-output publication."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    row = state.pending_output(completion_group, operation.request_key.request_id)
    params = row.input_latent_params
    staging = row.latent_staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory operation has no staged latent inputs")
    latent_input = operation.latent_input
    if latent_input is None:
        raise invalid_descriptor("image publication lost its latent input")
    resident_output = operation.image_output
    if resident_output is not None:
        if int(resident_output.generation) < 1:
            raise invalid_descriptor("finalized resident image requires a positive generation")
        write = bound_device_write(completion_group, resident_output, state=state)
        # Feedback storage uses the declared model dtype; preserve the decoder's
        # existing conversion before handing the value to the tensor store.
        storage = tensor_store.producer_write_views((write,))[0]
        tensor_store.publish_write(
            write,
            image_tensor.to(dtype=storage.dtype),
            metadata=ImageMetadata(
                height=int(params.height),
                width=int(params.width),
                value_range=image_range,
            ),
        )
        request = state.pending_output(completion_group, operation.request_key.request_id)
        if request.producer_write is None:
            request.producer_write = write
    image_task = defer_image_encoding(
        operation,
        image_tensor,
        image_range,
        completion_group,
        max_bytes=int(operation.bounds.max_completion_bytes),
        state=state,
    )
    request.projected_progress = replace(operations.require_progress(request), latent_product=None)
    request.projected_progress = replace(operations.require_progress(request), flow_step=0)
    request.latent_params = params
    request.latent_generation = int(latent_input.generation)
    request.latent_step = int(params.start_step)
    request.latent_release = True
    return non_state_outcome(
        operation, completion_group, completion_tasks=(image_task,), state=state
    )


def state_outcome(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    products: tuple[TensorPublication, ...] = (),
    request_tables: BlockTables | None,
) -> PendingOutput:
    """Stage execution progress and defer successor publication until stateful tensors are ready."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    cache = operations.cache_coordinates(request, tables=request_tables)
    selected = request.runtime_cache_length
    if selected is None:
        selected = cache[2]
    if not isinstance(selected, int):
        raise RuntimeError("visual state completion has a dynamic KV length")
    cache = (cache[0], cache[1], selected, cache[3])
    request.status = OpStatus.OK
    request.projected_progress = operations.execution_runtime(request, cache)
    request.finish_flags = FinishFlags()
    request.product_generations = operations.output_generations(operation)
    request.committed_tokens = ()
    request.products = products
    return request


def non_state_outcome(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    products: tuple[TensorPublication, ...] = (),
    completion_tasks: tuple[CpuTask, ...] = (),
) -> PendingOutput:
    """Record a stateless completion and its already materialized output products."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    request.status = OpStatus.OK
    request.projected_progress = operations.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = operations.output_generations(operation)
    request.products = products
    request.completion_tasks = completion_tasks
    return request


def encode_source(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelRunner,
) -> str | tuple[torch.Tensor, ImageMetadata]:
    """Resolve encoded request media and stage it according to the model’s image policy."""

    if operation.input_image is not None:
        return operation.input_image
    reference = operation.image_input
    if reference is not None:
        read = tensor_store.consume(
            reference,
            consumer_op_id=operation.op_id,
            device=model_runner.operation_devices(operation)[0],
        )
        request = state.pending_output(completion_group, operation.request_key.request_id)
        request.device_reads.append(read)
        metadata = read.metadata
        if (
            not isinstance(metadata, ImageMetadata)
            or metadata.height < 1
            or metadata.width < 1
            or metadata.value_range is None
        ):
            raise invalid_descriptor("resident image product has incomplete dimensions")
        return read.tensor, metadata
    raise invalid_descriptor("encode operation has no source image product")


def encode_row(
    mode: PipelineStage,
    prepared: PreparedImage,
) -> ForwardRow:
    """Build a vision- or latent-encoder row from prepared image tensors."""

    return ForwardRow(
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
    completion_group: int,
    *,
    state: BatchState,
    close_image: bool,
    logits: bool,
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
) -> ForwardRow:
    """Publish vision features and construct the request runtime that references their token span."""

    cache = operations.cache_coordinates(
        state.pending_output(completion_group, operation.request_key.request_id),
        tables=request_tables,
    )
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
        token_ids[0] = _feature_token_id(injection, start=True)
    if trailing:
        token_ids[-1] = _feature_token_id(injection, start=False)
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
    )


def _feature_token_id(injection: FeatureInjection, *, start: bool) -> int:
    """Read a marker identity already bound by the input-asset resolver."""

    value = injection.start_token_id if start else injection.end_token_id
    if value is None:
        raise invalid_descriptor("feature injection requires a resolved marker token id")
    return value


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
    completion_group: int,
    *,
    state: BatchState,
    request_tables: BlockTables | None,
    model_runner: ModelRunner,
) -> ForwardRow:
    """Publish encoded image latents and construct the request runtime for diffusion conditioning."""

    builder = model_runner.image_builder
    if builder is None or builder.framing != 2:
        raise invalid_descriptor("latent feature publication requires framed image conditioning")
    size = image.Config(height, width)
    image_tokens = builder.denoiser.latent_shape("image", size)[0]
    if latent.reshape(-1, latent.shape[-1]).shape[0] != image_tokens:
        raise invalid_descriptor("state latent does not match the declared image dimensions")
    query = builder.sequence_length(size)
    positions = builder.positions(size, conditioning_position + 1, device=latent.device)
    positions[0, 0] = conditioning_position
    positions[0, -1] = conditioning_position + builder.rope_advance
    cache = operations.cache_coordinates(
        state.pending_output(completion_group, operation.request_key.request_id),
        tables=request_tables,
    )
    return ForwardRow(
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
        write_kv=True,
        causal=False,
    )


def diffusion_finalize_frames(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelRunner,
) -> PendingOutput:
    """Validate finalized diffusion output and return RGB frames with their numeric range."""

    image, metadata = transfer.fetch_product(
        operation,
        completion_group,
        tensor_store=tensor_store,
        model_runner=model_runner,
        state=state,
    )
    if not isinstance(metadata, ImageMetadata) or min(metadata.height, metadata.width) < 1:
        raise invalid_descriptor("frame materialization source is not an image tensor")
    value_range = metadata.value_range or (-1.0, 1.0)
    image_task = defer_image_encoding(
        operation,
        image,
        value_range,
        completion_group,
        max_bytes=int(operation.bounds.max_completion_bytes),
        state=state,
    )
    return non_state_outcome(
        operation, completion_group, completion_tasks=(image_task,), state=state
    )


def defer_image_encoding(
    operation: ScheduledRequest,
    image: torch.Tensor,
    value_range: tuple[float, float],
    completion_group: int,
    *,
    state: BatchState,
    max_bytes: int,
) -> CpuTask:
    """Reserve output storage and schedule image encoding after the device copy completes."""

    if max_bytes < 1:
        raise invalid_descriptor("image materialization requires a positive completion bound")
    quantized = quantize_image_hwc(
        image,
        value_range=value_range,
    )
    if int(quantized.numel()) > max_bytes:
        raise invalid_descriptor("image staging exceeds its registered completion byte bound")
    capture = state.group_buffers[completion_group].capture_bytes(quantized)
    pending = state.pending_output(completion_group, operation.request_key.request_id)
    if len(pending.completion_tasks) != 1:
        raise RuntimeError("materialization has no registered CPU task slot")
    reservation = pending.completion_tasks[0]

    def encode() -> bytes:
        payload = uint8_image_to_png_base64_bytes(capture)
        if not payload:
            raise RuntimeError("image encoding task produced an invalid payload")
        if len(payload) > max_bytes:
            raise RuntimeError("encoded image exceeds its registered byte bound")
        return payload

    release = state.group_buffers[completion_group].retain_cpu_reader()
    try:
        return reservation.configure(
            encode,
            input_ready=state.group_buffers[completion_group].ready,
            input_completion=state.group_buffers[completion_group].completion_future,
            release=release,
            profile_name="uniserve.image.encode",
        )
    except BaseException:
        release()
        raise


def bound_device_write(
    completion_group: int, reference: TensorRef, *, state: BatchState
) -> TensorRecord:
    """Return the staged device-product write matching a declared output reference."""

    request = state.pending_output(completion_group, reference.request_key.request_id)
    matches = tuple(
        write for write in request.writes if write.reference == reference and not write.feature
    )
    if len(matches) != 1:
        raise invalid_descriptor(
            "resident product does not have exactly one atomic registration binding"
        )
    return matches[0]


def bound_encoder_write(
    completion_group: int, reference: TensorRef, *, state: BatchState
) -> TensorRecord:
    """Return the staged encoder-cache write matching a declared output reference."""

    request = state.pending_output(completion_group, reference.request_key.request_id)
    matches = tuple(
        write for write in request.writes if write.reference == reference and write.feature
    )
    if len(matches) != 1:
        raise invalid_descriptor(
            "encoder feature does not have exactly one atomic registration binding"
        )
    return matches[0]


__all__ = [
    "text",
    "prepare_features",
    "publish_features",
    "materialization_latent",
    "publish_image",
]
