"""Image encode, latent decode, and materialization transformations.

The worker-side steps around a request's image calls: encoding conditioning
text, preparing images for the vision or latent encoder and publishing its
features, building the prefill and denoiser rows that write image features
or latents into KV, gathering a finished latent trajectory for the image
decoder, and publishing a decoded image. Image outputs leave the device as
a PNG: the image is quantized to uint8 on the device, copied into the
batch's output buffer, and encoded on the rank's ``HostLane`` once that copy
completes.

``forward``, ``token``, ``schedule`` and ``transfer`` call into this module;
the ``*_outcome`` helpers stage a call's successful ``PendingOutput``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch

from uniserve.media import image
from uniserve.processing import (
    FeatureInjection,
    FeatureLayout,
    PatchTransform,
    PositionLayout,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution import calls, transfer
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.host import HostTask
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.media.codec import (
    quantize_image_hwc,
    uint8_image_to_png_base64_bytes,
)
from uniserve_worker.model_executor.diffusion_inputs import DiffusionRow
from uniserve_worker.model_executor.image_inputs import (
    PreparedImage,
    VisionRow,
    patch_grid_shape,
    prepare_image,
    prepare_tensor_image,
)
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    ForwardMode,
    MediaCall,
)
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.protocol.tensor import TensorRef
from uniserve_worker.protocol.transfer import (
    EncoderTransferValue,
    TensorTransfer,
)
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.tensor_store import (
    FeatureMetadata,
    ImageMetadata,
    TensorRecord,
)
from uniserve_worker.transport.publication import publish_tensor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from uniserve_worker.config.execution import WorkerConfig
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.protocol.worker_info import WorkerInfo
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.latent_pool import LatentPool
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport


def text(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Encode admitted conditioning tokens and publish their declared tensors.

    Runs the text encoder on the request's admitted prompt tokens and
    publishes one tensor per declared output.
    """
    request = state.pending_output(call.request_key.request_id)
    admission = request.request.admission
    if admission.diffusion is None or not admission.prompt_token_ids:
        raise invalid_descriptor(
            "text conditioning requires admitted prompt tokens"
        )
    if not call.outputs:
        raise invalid_descriptor(
            "text encoder outputs must declare conditioning tensors"
        )

    result = model_runner.encode_text(admission.prompt_token_ids)
    if len(result.values) != len(call.outputs):
        raise invalid_descriptor(
            "text encoder output declarations disagree with the loaded entry"
        )

    products = transfer.publish_tensors(
        call,
        result.values,
        tensor_store=tensor_store,
        publication_transports=publication_transports,
        state=state,
    )
    if result.stats is None:
        raise RuntimeError("module output has no execution statistics")
    state.forward_stats.append(result.stats)

    request.status = CallStatus.OK
    request.progress = calls.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = tuple(
        output.generation for output in call.tensor_outputs()
    )
    request.products = products
    return request


def prepare_features(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
) -> PreparedImage:
    """Stage the source image of one vision or latent encoder call.

    The source is either the request's encoded image or a resident image
    product (see ``encode_source``); it is resized and normalized by the
    model's image processor onto the device of the call's execution entry.
    """
    image_processor = model_runner.image_processor()
    mode = call.kind
    if not isinstance(mode, MediaCall) or mode not in {
        MediaCall.VISION_ENCODING,
        MediaCall.LATENT_ENCODING,
    }:
        raise invalid_descriptor("encode call is missing an encode mode")
    feature_output = call.encoder_output
    if feature_output is None:
        raise invalid_descriptor(
            "encode call requires one resident feature output"
        )
    if int(feature_output.generation) < 1:
        raise invalid_descriptor(
            "encode feature output requires a positive generation"
        )

    source = encode_source(
        call,
        tensor_store=tensor_store,
        model_runner=model_runner,
        state=state,
    )
    target_device = model_runner.call_devices(call)[1]
    if isinstance(source, tuple):
        # A resident image records its numerical range: a (-1, 1) image is
        # mapped to [0, 1] before the processor's transforms, and any other
        # range is taken as [0, 1].
        source_tensor, source_metadata = source
        prepared = prepare_tensor_image(
            image_processor,
            mode,
            source_tensor,
            device=target_device,
            signed_unit=source_metadata.value_range == (-1.0, 1.0),
        )
    else:
        prepared = prepare_image(
            image_processor,
            mode,
            source,
            device=target_device,
            input_images=_input_images(
                state.pending_output(call.request_key.request_id)
            ),
        )
    return prepared


def _input_images(request: PendingOutput) -> int:
    """Return the input image count admitted with a request that has one.

    Raises:
        WorkerError: ``invalid_descriptor`` when the admission declares no
            input images.
    """
    count = int(request.request.admission.input_images)
    if count < 1:
        raise invalid_descriptor("request admission declares no input images")
    return count


def publish_features(
    call: Call,
    prepared: PreparedImage,
    output: torch.Tensor,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    worker_info: WorkerInfo,
    publication_transports: Mapping[str, Transport],
    config: WorkerConfig,
) -> PendingOutput:
    """Commit encoded features to the tensor store and export them if needed.

    The features always land in the call's encoder-cache write. They are
    also exported as a product when any non-local publication transport is
    bound and this rank is the component's output rank.
    """
    request = state.pending_output(call.request_key.request_id)
    feature_output = call.encoder_output
    if feature_output is None:
        raise invalid_descriptor("encoder output has no feature reference")
    features = output.detach()
    write = bound_encoder_write(feature_output, state=state)
    resident = tensor_store.publish_write(
        write,
        features,
        metadata=FeatureMetadata(height=prepared.height, width=prepared.width),
    )

    products: tuple[TensorPublication, ...] = ()
    if any(
        name != "local" for name in publication_transports
    ) and config.rank == worker_info.output_rank(call.component):
        locations = publish_tensor(
            publication_transports,
            resident,
            retain=partial(tensor_store.retain_publication, write),
            consumers=call.consumer_slots,
        )
        request.exported_locators.extend(locations)
        request.tensor_exports[feature_output.buffer_id] = tuple(
            (publication_transports[location.backend], location)
            for location in locations
        )
        descriptor = EncoderTransferValue(
            tensor=TensorTransfer(
                shape=tuple(resident.shape), locations=locations
            ),
            payload_kind="vision_feature"
            if call.kind is MediaCall.VISION_ENCODING
            else "latent_feature",
            height=prepared.height,
            width=prepared.width,
        )
        products = (
            TensorPublication(product=feature_output, value=descriptor),
        )
    return non_state_outcome(call, products=products, state=state)


def materialization_latent(
    call: Call,
    *,
    state: BatchState,
    latent_pool: LatentPool,
    model_runner: ModelExecutor,
) -> torch.Tensor:
    """Gather a completed latent trajectory for the image decoder.

    Returns the request's current latent, gathered from its ``LatentPool``
    pages into the call's staging. The trajectory must be at the admitted
    image's final step.
    """
    request = state.pending_output(call.request_key.request_id)
    latent_input = call.latent_input
    if latent_input is None:
        raise invalid_descriptor(
            "latent materialization requires a latent input"
        )
    if int(latent_input.generation) < 1:
        raise invalid_descriptor(
            "materialization does not name a live latent generation"
        )
    image_params = request.request.image
    if image_params is None:
        raise invalid_descriptor(
            "image materialization has no admitted image parameters"
        )

    row = state.pending_output(call.request_key.request_id)
    params = row.latent.input_params
    staging = row.latent.staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory call has no staged latent inputs")
    if int(params.start_step) != int(image_params.steps):
        raise invalid_descriptor(
            "image materialization requires a completed latent trajectory"
        )

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
    call: Call,
    image_tensor: torch.Tensor,
    image_range: tuple[float, float],
    *,
    state: BatchState,
    tensor_store: TensorStore,
) -> PendingOutput:
    """Publish a decoded image and schedule its bounded PNG encoding.

    When the call declares a resident image output, the image is also
    committed to the tensor store for later consumers. The call's latent
    trajectory is staged for release from the ``LatentPool`` at commit.
    """
    request = state.pending_output(call.request_key.request_id)
    row = state.pending_output(call.request_key.request_id)
    params = row.latent.input_params
    staging = row.latent.staging
    if params is None or staging is None:
        raise invalid_descriptor("trajectory call has no staged latent inputs")
    latent_input = call.latent_input
    if latent_input is None:
        raise invalid_descriptor("image publication lost its latent input")

    resident_output = call.image_output
    if resident_output is not None:
        if int(resident_output.generation) < 1:
            raise invalid_descriptor(
                "finalized resident image requires a positive generation"
            )
        write = bound_device_write(resident_output, state=state)
        # The reserved product storage has the product's declared dtype;
        # convert the decoder output to it before committing.
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
        request = state.pending_output(call.request_key.request_id)
        if request.producer_write is None:
            request.producer_write = write

    image_task = defer_image_encoding(
        call,
        image_tensor,
        image_range,
        max_bytes=int(call.bounds.max_completion_bytes),
        state=state,
    )
    request.progress = replace(calls.require_progress(request), flow_step=0)
    request.latent.update.params = params
    request.latent.update.generation = int(latent_input.generation)
    request.latent.update.step = int(params.start_step)
    request.latent.update.release = True
    return non_state_outcome(call, completion_tasks=(image_task,), state=state)


def state_outcome(
    call: Call,
    *,
    state: BatchState,
    products: tuple[TensorPublication, ...] = (),
    request_tables: BlockTables | None,
) -> PendingOutput:
    """Stage the successful outcome of a visual-state call, which wrote KV.

    The reported KV length is the staged ``runtime_cache_length`` when one
    exists, otherwise the request's visible KV length as checked against its
    block table by ``calls.cache_coordinates``.

    Raises:
        WorkerError: When ``calls.cache_coordinates`` rejects the request's
            KV coordinates.
        RuntimeError: When the staged length is a device tensor.
    """
    request = state.pending_output(call.request_key.request_id)
    cache = calls.cache_coordinates(request, tables=request_tables)
    selected = request.token.runtime_cache_length
    if selected is None:
        selected = cache[2]
    if not isinstance(selected, int):
        raise RuntimeError("visual state completion has a dynamic KV length")
    cache = (cache[0], cache[1], selected, cache[3])

    request.status = CallStatus.OK
    request.progress = calls.execution_runtime(request, cache)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.token.committed_tokens = ()
    request.products = products
    return request


def non_state_outcome(
    call: Call,
    *,
    state: BatchState,
    products: tuple[TensorPublication, ...] = (),
    completion_tasks: tuple[HostTask, ...] = (),
) -> PendingOutput:
    """Record a stateless completion and its already materialized products.

    ``completion_tasks`` become the call's ``host.tasks``, which must finish
    before its output is ready.
    """
    request = state.pending_output(call.request_key.request_id)
    request.status = CallStatus.OK
    request.progress = calls.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.products = products
    request.host.tasks = completion_tasks
    return request


def encode_source(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
) -> str | tuple[torch.Tensor, ImageMetadata]:
    """Resolve the source image of an encoder call.

    Returns the request's encoded image string when the call carries one;
    otherwise consumes the call's resident image product onto its compute
    device and returns it with its ``ImageMetadata``.

    Raises:
        WorkerError: When the call has no source, or the resident product's
            metadata is not ``ImageMetadata`` with positive dimensions and a
            value range.
    """
    if call.input_image is not None:
        return call.input_image

    reference = call.image_input
    if reference is not None:
        read = tensor_store.consume(
            reference,
            consumer_call_id=call.call_id,
            device=model_runner.call_devices(call)[0],
        )
        request = state.pending_output(call.request_key.request_id)
        request.device_reads.append(read)
        metadata = read.metadata
        if (
            not isinstance(metadata, ImageMetadata)
            or metadata.height < 1
            or metadata.width < 1
            or metadata.value_range is None
        ):
            raise invalid_descriptor(
                "resident image product has incomplete dimensions"
            )
        return read.tensor, metadata

    raise invalid_descriptor("encode call has no source image product")


def encode_row(
    mode: MediaCall,
    prepared: PreparedImage,
) -> VisionRow:
    """Build a vision- or latent-encoder row from prepared image tensors."""
    return VisionRow(
        forward_mode=mode,
        encode_pixels=prepared.pixels,
        encode_grid=prepared.grid,
        encode_grid_shape=prepared.grid_shape,
    )


def vision_state_row(
    call: Call,
    features: torch.Tensor,
    height: int,
    width: int,
    conditioning_position: int,
    *,
    state: BatchState,
    close_image: bool,
    logits: bool,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> TokenRow:
    """Build the prefill row that writes one image's vision features into KV.

    ``features`` are ``[tokens, hidden]`` (or with a leading singleton batch
    axis). The row is non-causal, writes KV at the request's cache
    coordinates, and selects last logits when ``logits`` is set (hidden
    states otherwise). A framed layout adds start and end marker tokens;
    ``close_image`` adds the end marker in any layout.
    """
    request = state.pending_output(call.request_key.request_id)
    cache = calls.cache_coordinates(request, tables=request_tables)
    injection = model_runner.image_processor().feature_injection
    if injection is None:
        raise invalid_descriptor(
            "vision state stage requires declared feature injection"
        )

    # Accept a leading singleton batch axis; the row layout is [tokens, hidden].
    embeddings = (
        features.squeeze(0)
        if features.ndim == 3 and int(features.shape[0]) == 1
        else features
    )
    if embeddings.ndim != 2 or int(embeddings.shape[0]) < 1:
        raise invalid_descriptor(
            "vision features must have shape [tokens, hidden]"
        )

    # The sequence is [start marker?, feature tokens..., end marker?]; marker
    # slots keep placeholder token ids while feature slots carry embeddings.
    leading = injection.layout is FeatureLayout.FRAMED
    trailing = leading or close_image
    query = int(leading) + int(embeddings.shape[0]) + int(trailing)
    token_ids = torch.ones(query, dtype=torch.long)
    token_embeddings = embeddings.new_zeros((query, int(embeddings.shape[1])))
    embedding_mask = torch.zeros(
        query, dtype=torch.bool, device=embeddings.device
    )
    begin = int(leading)
    token_embeddings[begin : begin + int(embeddings.shape[0])] = embeddings
    embedding_mask[begin : begin + int(embeddings.shape[0])] = True
    if leading:
        token_ids[0] = _feature_token_id(injection, start=True)
    if trailing:
        token_ids[-1] = _feature_token_id(injection, start=False)

    # A row that closes the image carries generated-image feedback; any
    # other vision row carries one of the request's input images, whose
    # patch grid follows the pixel budget those images share.
    positions = _vision_positions(
        injection.positions,
        int(embeddings.shape[0]),
        conditioning_position,
        height=height,
        width=width,
        leading=leading,
        trailing=trailing,
        close_image=close_image,
        input_images=None if close_image else _input_images(request),
        model_runner=model_runner,
    )
    return TokenRow(
        forward_mode=ForwardMode.PREFILL,
        token_ids=token_ids,
        token_embeddings=token_embeddings,
        token_embedding_mask=embedding_mask,
        positions=positions,
        selection=TokenSelection.LAST_LOGITS
        if logits
        else TokenSelection.HIDDEN,
        request_pool_idx=cache[0],
        seq_len=cache[2],
        group_id=cache[1],
        write_kv=True,
        causal=False,
    )


def _feature_token_id(injection: FeatureInjection, *, start: bool) -> int:
    """Read a marker token id already resolved on the feature injection."""
    value = injection.start_token_id if start else injection.end_token_id
    if value is None:
        raise invalid_descriptor(
            "feature injection requires a resolved marker token id"
        )
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
    input_images: int | None,
    model_runner: ModelExecutor,
) -> torch.Tensor:
    """Build position ids for a vision row's marker and feature slots.

    Returns ``[query]`` positions for ``PositionLayout.TEMPORAL``, all at
    ``conditioning_position``. Otherwise returns ``[3, query]`` rows of
    temporal, height and width coordinates: feature slots take their raster
    grid coordinates and marker slots zero spatial coordinates. The raster
    grid is the patch grid of a ``height`` x ``width`` canvas under the
    pixel bound for ``input_images`` (see ``patch_grid_shape``).
    """
    query = int(leading) + feature_tokens + int(trailing)
    if layout is PositionLayout.TEMPORAL:
        return torch.full(
            (query,), int(conditioning_position), dtype=torch.long
        )

    transform = model_runner.image_processor().vit
    if not isinstance(transform, PatchTransform):
        raise invalid_descriptor(
            "temporal-spatial feature injection requires a patch image "
            "transform"
        )

    # The encoder may pool patches, so the feature count can be a square
    # downscale of the raw patch grid; recover the per-axis grid factor.
    raw_height, raw_width = patch_grid_shape(
        transform, height, width, input_images
    )
    factor_squared, remainder = divmod(raw_height * raw_width, feature_tokens)
    factor = math.isqrt(factor_squared)
    if remainder or factor < 1 or factor * factor != factor_squared:
        raise invalid_descriptor(
            "vision feature count does not align with its patch grid"
        )
    grid_height, grid_width = raw_height // factor, raw_width // factor
    if grid_height * grid_width != feature_tokens:
        raise invalid_descriptor("vision output grid is not integral")

    temporal = torch.full(
        (query,),
        int(conditioning_position + (1 if close_image else 0)),
        dtype=torch.long,
    )
    # Raster-order [feature_tokens] grid coordinates for the feature slots.
    y = torch.arange(grid_height, dtype=torch.long).repeat_interleave(
        grid_width
    )
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
    call: Call,
    latent: torch.Tensor,
    height: int,
    width: int,
    conditioning_position: int,
    *,
    state: BatchState,
    request_tables: BlockTables | None,
    model_runner: ModelExecutor,
) -> DiffusionRow:
    """Build the denoiser row that writes an image latent into KV.

    The row runs the latent at timestep zero between the builder's two
    framing tokens, non-causally, at the request's cache coordinates. It
    requires an image builder with two framing tokens and a latent whose
    token count matches the declared image size.
    """
    builder = model_runner.image_builder
    if builder is None or builder.framing != 2:
        raise invalid_descriptor(
            "latent feature publication requires framed image conditioning"
        )

    size = image.Config(height, width)
    image_tokens = builder.denoiser.latent_shape("image", size)[0]
    if latent.reshape(-1, latent.shape[-1]).shape[0] != image_tokens:
        raise invalid_descriptor(
            "state latent does not match the declared image dimensions"
        )

    query = builder.sequence_length(size)
    positions = builder.positions(
        size, conditioning_position + 1, device=latent.device
    )
    # Frame markers bound the image span: the first sits at the conditioning
    # position, the last advances past the rope range of the image tokens.
    positions[0, 0] = conditioning_position
    positions[0, -1] = conditioning_position + builder.rope_advance

    cache = calls.cache_coordinates(
        state.pending_output(call.request_key.request_id),
        tables=request_tables,
    )
    return DiffusionRow(
        forward_mode=MediaCall.DENOISING,
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
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Schedule PNG encoding of a finalized resident image product.

    Used for an image-decoding call without a latent input. A product whose
    metadata records no value range is treated as ``(-1, 1)``.
    """
    image, metadata = transfer.fetch_product(
        call,
        tensor_store=tensor_store,
        model_runner=model_runner,
        state=state,
    )
    if (
        not isinstance(metadata, ImageMetadata)
        or min(metadata.height, metadata.width) < 1
    ):
        raise invalid_descriptor(
            "frame materialization source is not an image tensor"
        )
    value_range = metadata.value_range or (-1.0, 1.0)

    image_task = defer_image_encoding(
        call,
        image,
        value_range,
        max_bytes=int(call.bounds.max_completion_bytes),
        state=state,
    )
    return non_state_outcome(call, completion_tasks=(image_task,), state=state)


def defer_image_encoding(
    call: Call,
    image: torch.Tensor,
    value_range: tuple[float, float],
    *,
    state: BatchState,
    max_bytes: int,
) -> HostTask:
    """Configure the call's host task to PNG-encode an image after its copy.

    Quantizes ``image`` to HWC uint8 on its device and captures it into the
    batch's output buffer. The call's single reserved ``HostTask`` then
    encodes the host copy to base64 PNG once the buffer's device copies
    complete; it holds a CPU reader on the buffer until it finishes or, when
    cancelled, until the buffer's copy completes. The encode task itself
    fails with ``RuntimeError`` when its payload is empty or exceeds
    ``max_bytes``.

    Raises:
        WorkerError: When ``max_bytes`` is not positive or the quantized
            image alone exceeds it.
        ValueError: When ``image`` is not one RGB CHW image (optionally with
            a singleton batch axis).
        RuntimeError: When the call does not hold exactly one reserved host
            task, or ``HostTask.configure`` rejects it.
    """
    if max_bytes < 1:
        raise invalid_descriptor(
            "image materialization requires a positive completion bound"
        )

    quantized = quantize_image_hwc(
        image,
        value_range=value_range,
    )
    if int(quantized.numel()) > max_bytes:
        raise invalid_descriptor(
            "image staging exceeds its registered completion byte bound"
        )

    capture = state.output_buffer.capture_bytes(quantized)
    pending = state.pending_output(call.request_key.request_id)
    if len(pending.host.tasks) != 1:
        raise RuntimeError("materialization has no registered CPU task slot")
    reservation = pending.host.tasks[0]

    def encode() -> bytes:
        payload = uint8_image_to_png_base64_bytes(capture)
        if not payload:
            raise RuntimeError(
                "image encoding task produced an invalid payload"
            )
        if len(payload) > max_bytes:
            raise RuntimeError(
                "encoded image exceeds its registered byte bound"
            )
        return payload

    # configure transfers the reader to the task only on success.
    release = state.output_buffer.retain_cpu_reader()
    try:
        return reservation.configure(
            encode,
            input_ready=state.output_buffer.ready,
            input_completion=state.output_buffer.completion_future,
            release=release,
            profile_name="uniserve.image.encode",
        )
    except BaseException:
        release()
        raise


def bound_device_write(
    reference: TensorRef, *, state: BatchState
) -> TensorRecord:
    """Return the staged device-product write matching a declared output.

    Raises:
        WorkerError: When the request has no such write or more than one.
    """
    request = state.pending_output(reference.request_key.request_id)
    matches = tuple(
        write
        for write in request.writes
        if write.reference == reference and not write.feature
    )
    if len(matches) != 1:
        raise invalid_descriptor(
            "resident product does not have exactly one atomic registration "
            "binding"
        )
    return matches[0]


def bound_encoder_write(
    reference: TensorRef, *, state: BatchState
) -> TensorRecord:
    """Return the staged encoder-cache write matching a declared output.

    Raises:
        WorkerError: When the request has no such write or more than one.
    """
    request = state.pending_output(reference.request_key.request_id)
    matches = tuple(
        write
        for write in request.writes
        if write.reference == reference and write.feature
    )
    if len(matches) != 1:
        raise invalid_descriptor(
            "encoder feature does not have exactly one atomic registration "
            "binding"
        )
    return matches[0]


__all__ = [
    "text",
    "prepare_features",
    "publish_features",
    "materialization_latent",
    "publish_image",
]
