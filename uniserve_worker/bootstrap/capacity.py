"""Local physical arena sizing for one worker process."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from uniserve.diffusion import Branch
from uniserve.distributed.mesh import Communicator
from uniserve.math import ceil_div
from uniserve.model import (
    CausalLM,
    PatchEncoder,
    VideoDecoder,
    VideoPostprocessor,
)
from uniserve.processing import (
    FeatureLayout,
    ImageProcessor,
    PatchTransform,
)
from uniserve.runtime.device import canonical_device, device_memory_budget
from uniserve.tensors import BufferConfig

from ..config import WorkerConfig
from ..execution.input_buffers import InputBufferConfig
from ..execution.model_entry import ModelEntry
from ..execution.resources import media_state_buffers
from ..foundation.errors import unsupported_setup
from ..protocol.call import PipelineStage
from ..protocol.tensor import DeviceDim, OutputInfo
from ..runtime.cache_manager import CacheManager
from ..runtime.results import resolve_outputs
from ..runtime.tensor_store import TensorStore, device_product_capacity_bytes
from .components import pipeline_components
from .inputs import capability, image_builder, media_builder

_DEVICE_PRODUCTS_PER_CALL = 6
_DEVICE_PRODUCT_RETIREMENT_BATCHES = 1
_MAX_TRANSFER_ENTRIES = 256
_HOST_LANE_INFLIGHT = 256
_REQUEST_RELAY_ROW_BYTES = 18
_REQUEST_RELAY_RETIREMENT_LANES = 1

DEFAULT_NUM_BLOCKS_FALLBACK = 4096
DEFAULT_MAX_BATCH_OPS = 1024
DEFAULT_MAX_REQUEST_POOL_SIZE = 128
DEFAULT_BLOCK_SIZE = 64


def input_buffer_config(
    model: nn.Module,
    config: WorkerConfig,
    *,
    processor: ImageProcessor | None = None,
) -> InputBufferConfig:
    """Size staging for the admitted text span.

    Staging also covers one atomic image or CFG call.
    """
    text = capability(model, CausalLM)
    if text is None:
        raise ValueError("text input staging requires a causal language model")

    max_rows = min(
        config.max_request_pool_size,
        config.max_batch_calls,
        *(
            lane.max_batch_calls or config.max_batch_calls
            for lane in config.lanes
        ),
    )
    max_tokens = (
        min(
            config.max_batch_tokens,
            *(
                lane.max_batch_tokens or config.max_batch_tokens
                for lane in config.lanes
            ),
        )
        if config.lanes
        else config.max_batch_tokens
    )

    flow = image_builder(model)
    branches = 1 if flow is None else len(Branch)
    injection = None if processor is None else processor.feature_injection
    image_span = (
        0
        if injection is None
        else max(
            vision_tokens(model, processor),
            0 if flow is None else flow.max_tokens,
        )
        + (2 if injection.layout is FeatureLayout.FRAMED else 0)
    )
    text_tokens = max_tokens + image_span

    if flow is None:
        flow_tokens = 0
    else:
        from uniserve.media import image

        flow_tokens = branches * max(
            (
                flow.max_tokens + flow.framing,
                *(
                    flow.sequence_length(image.Config(height, width))
                    for height, width in config.flow_graph_shapes
                ),
            )
        )

    return InputBufferConfig(
        max_rows=max_rows * branches,
        # Text and diffusion use separate homogeneous calls on this lane.
        max_tokens=max(text_tokens, flow_tokens),
        max_text_tokens=text_tokens,
        max_blocks_per_row=max(
            1, ceil_div(config.max_sequence_tokens, config.block_size)
        ),
        hidden_size=text.backbone.hidden_size,
        embedding_dtype=getattr(
            torch, config.model_dtype.removeprefix("torch.")
        ),
    )


def vision_tokens(model: nn.Module, processor: ImageProcessor | None) -> int:
    """Bound feature rows from the actual image transform and encoder stride."""
    encoder = capability(model, PatchEncoder)
    if encoder is None or processor is None or processor.vit is None:
        return 0
    transform = processor.vit
    pixels = (
        transform.max_pixels
        if isinstance(transform, PatchTransform)
        else min(transform.resize.max_pixels, transform.resize.max_size**2)
    )
    return pixels // (encoder.patch_size * encoder.downsample) ** 2


def tensor_slot_capacity(
    schema: Mapping[str, BufferConfig],
    group: Communicator,
    *,
    maximum: int,
    minimum: int,
    available_bytes: int,
    auxiliary_bytes: Callable[[int], int],
) -> int:
    """Choose the largest slot count whose complete storage fits on every rank.

    ``auxiliary_bytes(slots)`` includes products and runtime arenas. Its cost
    may increase when fewer slots allow more in-flight outputs per request,
    so ranks agree on feasible counts rather than reducing local maxima.
    """
    bytes_per_slot = sum(
        field.nbytes for field in schema.values() if not field.host
    )
    if minimum < 1 or maximum < minimum:
        raise ValueError("request tensor capacity requires valid slot bounds")

    candidates = range(minimum, maximum + 1)
    requirements = [
        count * bytes_per_slot + auxiliary_bytes(count) for count in candidates
    ]
    agreed = torch.tensor(
        [required <= available_bytes for required in requirements],
        dtype=torch.int32,
        device=group.device,
    )
    group.all_reduce(agreed, op="min")

    feasible = [
        count for count, fits in zip(candidates, agreed.cpu().tolist()) if fits
    ]
    if feasible:
        return feasible[-1]

    raise RuntimeError(
        "insufficient device memory for a common request tensor slot count: "
        f"candidate range {minimum}..{maximum}, local requirements "
        f"{requirements}, {available_bytes} bytes available"
    )


def request_tensor_window(queue_depth: int, request_slots: int) -> int:
    """Return the output horizon for requests.

    One pipeline slot is reserved per request.
    """
    if request_slots < 1 or queue_depth < 3 * request_slots:
        raise ValueError(
            "request tensor pipeline requires two unresolved outputs per slot"
        )
    return queue_depth // request_slots - 1


def product_storage_bytes(
    entry_outputs: Mapping[str, tuple[OutputInfo, ...]],
) -> int:
    """Size one logical product set using the 256-byte allocation alignment."""
    return sum(
        ((output.max_bytes + 255) // 256) * 256
        for outputs in entry_outputs.values()
        for output in outputs
    )


def active_latent_capacity_tokens(
    per_image_tokens: int, concurrency_token_budget: int | None
) -> int:
    """Reserve at least one image worth of latent tokens.

    The reservation is bounded by the concurrency token budget.
    """
    per_image = max(0, int(per_image_tokens))
    if per_image == 0:
        return 0
    if concurrency_token_budget is None:
        return per_image
    return max(per_image, int(concurrency_token_budget))


def local_product_storage_bytes(
    entry_outputs: Mapping[str, tuple[OutputInfo, ...]],
    *,
    bindings: Mapping[str, ModelEntry],
    pipeline_components: Mapping[PipelineStage, str],
    max_unresolved_ops: int,
) -> int:
    """Size persistent products from placement, consumers and output horizon.

    Producers and remote consumers each need a complete logical allocation:
    disjoint regions may subsequently be imported into that allocation. A
    temporal-unit stage only retains its unresolved groups of leading-axis
    units. Non-streaming results retain their declared capacity until their
    consumers finish. Alignment follows BufferPool's allocation rules.
    """
    if max_unresolved_ops < 1:
        raise ValueError("product storage requires a positive output horizon")

    consumers: dict[str, set[str]] = {}
    # These are the concrete persistent Tensor consumers of the video path.
    # Denoising state is resident; write stages consume decoded output buffers.
    for source_stage, destination_stage in (
        (PipelineStage.TEXT_ENCODING, PipelineStage.LATENT_PREPARATION),
        (PipelineStage.DENOISING, PipelineStage.VIDEO_DECODING),
        (PipelineStage.DENOISING, PipelineStage.AUDIO_DECODING),
        (PipelineStage.VIDEO_DECODING, PipelineStage.VIDEO_ENCODING),
        (PipelineStage.AUDIO_DECODING, PipelineStage.AUDIO_ENCODING),
        (PipelineStage.VIDEO_ENCODING, PipelineStage.MUXING),
    ):
        source = pipeline_components.get(source_stage)
        destination = pipeline_components.get(destination_stage)
        if source is not None and destination is not None:
            consumers.setdefault(source, set()).add(destination)
    streamed = {
        component
        for stage in (
            PipelineStage.VIDEO_DECODING,
            PipelineStage.VIDEO_ENCODING,
        )
        if (component := pipeline_components.get(stage)) is not None
    }
    total = 0
    for entry, outputs in entry_outputs.items():
        producer = bindings.get(entry)
        if bindings:
            produces = (
                producer is not None
                and producer.process_group.global_rank in producer.output_ranks
            )
            consumes = any(
                consumer in bindings and bindings[consumer].owns
                for consumer in consumers.get(entry, ())
            )
            if not produces and not consumes:
                continue

        units_per_call = 1
        if producer is not None and entry in streamed:
            config = producer.config
            units_per_call = len(config.ranks) * config.units_per_rank

        for output in outputs:
            size = output.max_bytes
            if entry in streamed:
                dims = output.shape_bound.dims
                if not dims or not isinstance(dims[0], DeviceDim):
                    raise ValueError(
                        "streamed products require a bounded leading unit axis"
                    )
                max_units = dims[0].bound
                # Remote producers have no placement in this worker's rank
                # namespace. Cover the complete imported extent, including
                # allocation alignment if every unit arrives separately.
                if bindings and producer is None:
                    group_bytes = size // max_units
                    live_groups = max_units
                else:
                    group_bytes = (
                        size // max_units * min(max_units, units_per_call)
                    )
                    live_groups = min(
                        ceil_div(max_units, units_per_call),
                        max_unresolved_ops,
                    )
                size = live_groups * ceil_div(group_bytes, 256) * 256
            total += ceil_div(size, 256) * 256

    return total


@dataclass(frozen=True)
class RuntimeKVCapacity:
    """Resolved KV token and page capacity within the physical budget."""

    block_size: int
    bytes_per_token: int
    token_capacity: int
    num_blocks: int


def latent_trajectory_bytes(
    latent_units: int,
    latent_width: int,
    dtype_bytes: int,
) -> int:
    """Calculate storage for one latent trajectory.

    Storage is derived from unit count, width, and element size.
    """
    units = int(latent_units)
    width = int(latent_width)
    element_bytes = int(dtype_bytes)
    if units < 0 or width < 1 or element_bytes < 1:
        raise ValueError("latent trajectory dimensions are invalid")
    return units * width * element_bytes


def latent_pool_capacity_bytes(
    *,
    request_pool_size: int,
    num_pages: int,
    page_units: int,
    latent_width: int,
    dtype_bytes: int,
) -> int:
    """Calculate double-buffered latent pool storage.

    Covers latent pages, step storage, page tables, and timestep metadata.
    """
    slots = int(request_pool_size)
    pages = int(num_pages)
    units = int(page_units)
    width = int(latent_width)
    element_bytes = int(dtype_bytes)
    if min(slots, units, width, element_bytes) < 1 or pages < 2:
        raise ValueError("latent pool dimensions are invalid")

    # One page stays reserved as the sentinel; only usable pages hold
    # trajectories.
    usable_pages = pages - 1
    storage = 2 * pages * units * width * element_bytes
    step_buffer = usable_pages * units * width * element_bytes
    page_table = usable_pages * 8
    timestep_pairs = (slots + 1) * 2 * 4
    return storage + step_buffer + page_table + timestep_pairs


@dataclass(frozen=True, slots=True)
class ArenaCapacity:
    """Budgets latent storage, device products, transfers, and CPU tasks.

    All budgets are for one worker arena.
    """

    latent_pool_bytes: int
    tensor_store: int
    device_product_bytes: int
    transfer_bytes: int
    transfer_tickets: int
    host_lane_inflight: int


def call_window(queue_depth: int, max_calls: int) -> int:
    """Bound simultaneously live calls.

    The bound follows pipeline depth and per-batch capacity.
    """
    depth = int(queue_depth)
    calls = int(max_calls)
    if depth < 1 or calls < 1:
        raise ValueError("call-window sizing requires positive bounds")
    return min(depth * calls, max(2, depth))


def request_tensor_arena_capacity(
    worker_config: WorkerConfig,
    *,
    queue_depth: int,
    product_bytes_per_request: int,
    concurrent_imports: int = 0,
) -> ArenaCapacity:
    """Bound product, relay and transfer storage for fixed request tensors.

    ``concurrent_imports`` is how many remote product regions one request's
    calls can have in flight at once beyond its call window, which is what
    the artifact's assembly costs: it reads every encode round of the request,
    and each round was written by every rank that held a media unit in it.
    """
    depth = int(queue_depth)
    max_calls = int(worker_config.max_batch_calls)
    state_slots = int(worker_config.max_request_pool_size)
    slots = max(depth * max_calls, state_slots * int(concurrent_imports))
    unresolved_window = request_tensor_window(depth, state_slots)
    tensor_store = _DEVICE_PRODUCTS_PER_CALL * (
        slots + _DEVICE_PRODUCT_RETIREMENT_BATCHES * max_calls
    )
    relay_bytes = (
        (state_slots + 1)
        * (max(1, unresolved_window) + _REQUEST_RELAY_RETIREMENT_LANES)
        * _REQUEST_RELAY_ROW_BYTES
    )
    return ArenaCapacity(
        latent_pool_bytes=0,
        tensor_store=tensor_store,
        device_product_bytes=(
            device_product_capacity_bytes(tensor_store, 1, max_value_bytes=1)
            + relay_bytes
        ),
        transfer_bytes=max(1, state_slots * product_bytes_per_request),
        transfer_tickets=max(1, min(slots, _MAX_TRANSFER_ENTRIES)),
        host_lane_inflight=state_slots * (unresolved_window + 1),
    )


def artifact_import_regions(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    bindings: Mapping[str, ModelEntry],
) -> int:
    """Count the remote product regions assembling one artifact reads.

    Media units are encoded on the ranks that reconstruct them, one round at a
    time, and the muxer reads every round. Each round holds one media unit per
    participating rank and the muxer produced one of them itself.
    """
    components = pipeline_components(model)
    entry = components.get(PipelineStage.VIDEO_ENCODING)
    binding = None if entry is None else bindings.get(entry)
    decoder = capability(model, VideoDecoder)
    builder = media_builder(model, worker_config)
    if binding is None or decoder is None or builder is None:
        return 0
    ranks = len(binding.config.ranks)
    units = len(decoder.frame_slices(builder.maximum.num_frames))
    return ceil_div(units, ranks) * max(0, ranks - 1)


def model_arena_capacity(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    queue_depth: int,
    completion_payload_bytes: int,
    num_blocks: int,
    request_pool_size: int,
    num_latent_pages: int,
    latent_page_units: int,
    latent_width: int,
    max_latent_feature_bytes: int,
    max_vision_feature_bytes: int,
    bytes_per_token: int,
    bindings: Mapping[str, ModelEntry] | None = None,
    state_buffers: Mapping[str, BufferConfig] | None = None,
) -> ArenaCapacity:
    """Derive arena bounds from worker settings.

    Covers device-product, transfer, latent, and CPU arenas.
    """
    depth = int(queue_depth)
    payload_bytes = int(completion_payload_bytes)
    max_calls = int(worker_config.max_batch_calls)
    if depth < 1 or payload_bytes < 1 or max_calls < 1:
        raise ValueError("model arena sizing requires positive runtime bounds")

    slots = depth * max_calls
    if state_buffers is None:
        state_buffers = media_state_buffers(
            model, bindings or {}, worker_config
        )

    if capability(model, VideoPostprocessor) is not None or state_buffers:
        return request_tensor_arena_capacity(
            worker_config,
            queue_depth=depth,
            product_bytes_per_request=local_product_storage_bytes(
                resolve_outputs(model, worker_config),
                bindings=bindings or {},
                pipeline_components=pipeline_components(model),
                max_unresolved_ops=request_tensor_window(
                    depth, request_pool_size
                ),
            ),
            concurrent_imports=artifact_import_regions(
                model, worker_config, bindings=bindings or {}
            ),
        )

    transfer_tickets = min(slots, _MAX_TRANSFER_ENTRIES)
    block_size = int(worker_config.block_size)
    flow = image_builder(model)

    latent_pool_bytes = 0
    latent_transfer_bytes = 0
    if flow is not None:
        dtype_bytes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
        }.get(str(worker_config.model_dtype).removeprefix("torch.").lower())
        if dtype_bytes is None:
            raise ValueError(
                f"unsupported latent dtype {worker_config.model_dtype!r}"
            )
        latent_pool_bytes = latent_pool_capacity_bytes(
            request_pool_size=int(request_pool_size),
            num_pages=int(num_latent_pages),
            page_units=int(latent_page_units),
            latent_width=int(latent_width),
            dtype_bytes=dtype_bytes,
        )
        latent_transfer_bytes = latent_trajectory_bytes(
            flow.max_tokens,
            int(latent_width),
            dtype_bytes,
        )
    max_transfer_bytes = max(
        int(num_blocks) * block_size * int(bytes_per_token),
        latent_transfer_bytes,
        int(max_latent_feature_bytes),
        int(max_vision_feature_bytes),
        1,
    )

    device_product_slots = (
        slots + _DEVICE_PRODUCT_RETIREMENT_BATCHES * max_calls
    )
    tensor_store = _DEVICE_PRODUCTS_PER_CALL * device_product_slots
    device_count = len(
        {
            str(worker_config.device),
            str(worker_config.generation_device or worker_config.device),
        }
    )
    # Resident images and tensor products borrow scheduler-assigned storage
    # from BufferPool, whose complete grant is counted by the layout
    # owner. TensorStore owns scalar backing and request relays separately.
    device_product_bytes = device_product_capacity_bytes(
        tensor_store,
        device_count,
        max_value_bytes=1,
    )
    device_product_bytes += (
        (int(request_pool_size) + 1)
        * (call_window(depth, max_calls) + _REQUEST_RELAY_RETIREMENT_LANES)
        * _REQUEST_RELAY_ROW_BYTES
        * device_count
    )

    return ArenaCapacity(
        latent_pool_bytes=latent_pool_bytes,
        tensor_store=tensor_store,
        device_product_bytes=device_product_bytes,
        transfer_bytes=max_transfer_bytes * transfer_tickets,
        transfer_tickets=transfer_tickets,
        host_lane_inflight=_HOST_LANE_INFLIGHT,
    )


def derive_num_blocks(
    block_size: int,
    kv_token_capacity: int | None,
    *,
    default_blocks: int | None = None,
    floor: int = 1,
) -> int:
    """Derive the KV block count from token capacity and block size.

    When ``kv_token_capacity`` is unset or non-positive, ``default_blocks`` (or
    :data:`DEFAULT_NUM_BLOCKS_FALLBACK`) is used. The result is at least
    ``floor`` (default 1).
    """
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    min_blocks = max(1, int(floor))
    if kv_token_capacity is None or int(kv_token_capacity) <= 0:
        blocks = (
            DEFAULT_NUM_BLOCKS_FALLBACK
            if default_blocks is None
            else int(default_blocks)
        )
    else:
        blocks = int(kv_token_capacity) // block
    return max(min_blocks, blocks)


def device_total_bytes(device: str | torch.device) -> int:
    """Return CUDA capacity for a device.

    Device errors propagate so an unknown budget cannot become zero.
    """
    target = torch.device(device)
    if target.type != "cuda":
        return 0
    _free, total = torch.cuda.mem_get_info(target)
    return int(total)


def derive_runtime_kv_capacity(
    *,
    block_size: int,
    kv_token_capacity: int | None,
    bytes_per_token: int,
    device: Any = None,
    available_bytes: int | None = None,
    floor: int = 1,
    default_blocks: int | None = None,
    resident_copies: int = 1,
    co_resident_blocks: int = 0,
) -> RuntimeKVCapacity:
    """Size one KV pool from explicit tokens or a host-owned byte grant.

    Fixed-capacity callers supply their page count. Automatic CUDA sizing
    requires a granted budget; CPU capacity uses its declared default page
    policy.
    """
    block = int(block_size)
    token_bytes = int(bytes_per_token)
    if (
        block < 1
        or token_bytes < 1
        or floor < 1
        or resident_copies < 1
        or co_resident_blocks < 0
    ):
        raise ValueError("KV capacity dimensions must be positive")
    if available_bytes is not None and available_bytes < 0:
        raise ValueError("KV memory grant must not be negative")

    if kv_token_capacity is not None:
        if kv_token_capacity <= 0:
            raise ValueError("configured KV token capacity must be positive")
        blocks = derive_num_blocks(block, kv_token_capacity, floor=floor)
    elif available_bytes is not None:
        blocks = (
            available_bytes // (block * token_bytes) - co_resident_blocks
        ) // resident_copies
        if blocks < floor:
            raise ValueError(
                "device memory grant cannot hold the required KV pool"
            )
    elif device is not None and torch.device(device).type == "cuda":
        raise ValueError(
            "automatic CUDA KV sizing requires a host memory grant"
        )
    else:
        blocks = derive_num_blocks(
            block, None, default_blocks=default_blocks, floor=floor
        )

    if (
        available_bytes is not None
        and (resident_copies * blocks + co_resident_blocks)
        * block
        * token_bytes
        > available_bytes
    ):
        raise ValueError(
            "configured KV storage exceeds the device memory grant"
        )

    return RuntimeKVCapacity(
        block_size=block,
        bytes_per_token=token_bytes,
        token_capacity=blocks * block,
        num_blocks=blocks,
    )


__all__ = [
    "ArenaCapacity",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_MAX_BATCH_OPS",
    "DEFAULT_MAX_REQUEST_POOL_SIZE",
    "DEFAULT_NUM_BLOCKS_FALLBACK",
    "RuntimeKVCapacity",
    "derive_runtime_kv_capacity",
    "device_total_bytes",
    "latent_pool_capacity_bytes",
    "latent_trajectory_bytes",
    "model_arena_capacity",
    "call_window",
]


def resolve_request_capacity(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    queue_depth: int,
    capacity_group: Communicator | None,
    bindings: Mapping[str, ModelEntry] | None = None,
    state_buffers: Mapping[str, BufferConfig] | None = None,
) -> WorkerConfig:
    """Fit request tensors within the rank's fixed memory grant.

    Product arenas are fitted together with the request tensors.
    """
    if canonical_device(worker_config.device).type == "cuda":
        available, _free = device_memory_budget(
            worker_config.device, worker_config.kv_memory_fraction
        )
        worker_config = replace(worker_config, pool_memory_bytes=available)
        schema = (
            media_state_buffers(model, bindings or {}, worker_config)
            if state_buffers is None
            else state_buffers
        )

        if capability(model, VideoPostprocessor) is not None or schema:
            if capacity_group is None:
                raise unsupported_setup(
                    "request tensor sizing requires its rank group"
                )

            def auxiliary_bytes(count: int) -> int:
                capacity_config = replace(
                    worker_config,
                    max_request_pool_size=count,
                    max_batch_calls=min(count, worker_config.max_batch_calls),
                    max_batch_tokens=min(count, worker_config.max_batch_tokens),
                )
                product_bytes = local_product_storage_bytes(
                    resolve_outputs(model, worker_config),
                    bindings=bindings or {},
                    pipeline_components=pipeline_components(model),
                    max_unresolved_ops=request_tensor_window(
                        queue_depth, count
                    ),
                )
                arena = request_tensor_arena_capacity(
                    capacity_config,
                    queue_depth=queue_depth,
                    product_bytes_per_request=product_bytes,
                )
                return count * product_bytes + arena.device_product_bytes

            slots = tensor_slot_capacity(
                schema,
                capacity_group,
                maximum=min(
                    worker_config.max_request_pool_size, queue_depth // 3
                ),
                minimum=worker_config.min_request_pool_size,
                available_bytes=available,
                auxiliary_bytes=auxiliary_bytes,
            )
            worker_config = replace(
                worker_config,
                max_request_pool_size=slots,
                max_batch_calls=min(slots, worker_config.max_batch_calls),
                max_batch_tokens=min(slots, worker_config.max_batch_tokens),
            )
    return worker_config


def decode_context_blocks(
    model: nn.Module, worker_config: WorkerConfig, pool: CacheManager | None
) -> int:
    """Return the max paged-decode context blocks supported by this worker."""
    if capability(model, CausalLM) is None:
        return 0
    max_tokens = worker_config.max_sequence_tokens
    if max_tokens < 1:
        return 0
    blocks = (max_tokens + int(worker_config.block_size) - 1) // int(
        worker_config.block_size
    )
    if pool is None:
        return 0
    return min(blocks, max(0, int(pool.info.num_blocks) - 1))


def check_startup_memory(
    worker_config: WorkerConfig,
    product_capacity_bytes: int,
    tensor_store: TensorStore,
) -> None:
    """Check resident startup allocations against device grants.

    Reserved products are checked as well.
    """
    # Warmup may retain backend plans and graph pools in addition to the
    # explicit arenas. Readiness requires that these resident allocations
    # leave room for every still-lazy public product within the same grant.
    devices = tuple(
        dict.fromkeys(
            (
                worker_config.device,
                worker_config.generation_device or worker_config.device,
            )
        )
    )
    product_bytes = product_capacity_bytes // len(devices)
    for device in devices:
        if canonical_device(device).type != "cuda":
            continue
        available, free = device_memory_budget(
            device, worker_config.kv_memory_fraction
        )
        total = device_total_bytes(device)
        remaining = max(0, product_bytes - tensor_store.resident_bytes(device))
        if remaining > available or total - free > int(
            total * worker_config.kv_memory_fraction
        ):
            raise unsupported_setup(
                f"initialized runtime on {device} exceeds its static memory "
                f"grant: {total - free} resident bytes and {remaining} "
                f"reserved product bytes"
            )
