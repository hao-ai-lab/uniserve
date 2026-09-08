"""Local physical arena sizing for one worker process."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from uniserve_worker.config import WorkerConfig

from ..execution.bounded_storage import TensorSchema
from ..execution.input_buffers import InputGeometry
from ..foundation.math import ceil_div
from ..models.generation import GenerationPipeline
from ..models.inputs import FeatureLayout
from ..models.runtime import ExecutionModel
from ..nn.mesh import Communicator
from ..runtime.device_products import device_product_capacity_bytes

_DEVICE_PRODUCTS_PER_OPERATION = 6
_DEVICE_PRODUCT_RETIREMENT_BATCHES = 1
_MAX_TRANSFER_ENTRIES = 256
_CPU_TASKS = 256
_REQUEST_RELAY_ROW_BYTES = 18
_REQUEST_RELAY_RETIREMENT_LANES = 1

DEFAULT_NUM_BLOCKS_FALLBACK = 4096
DEFAULT_MAX_BATCH_OPS = 1024
DEFAULT_MAX_REQUEST_POOL_SIZE = 128
DEFAULT_BLOCK_SIZE = 64


def packed_input_geometry(model: ExecutionModel, config: WorkerConfig) -> InputGeometry:
    """Size staging for the admitted text span plus one atomic image or CFG operation."""

    max_rows = min(
        config.max_request_pool_size,
        config.max_batch_operations,
        *(lane.max_batch_operations or config.max_batch_operations for lane in config.lanes),
    )
    max_tokens = (
        min(
            config.max_batch_tokens,
            *(lane.max_batch_tokens or config.max_batch_tokens for lane in config.lanes),
        )
        if config.lanes
        else config.max_batch_tokens
    )
    flow = model.generation
    branches = 1 if flow is None else int(flow.max_cfg_branches)
    processor = model.image_processor
    injection = None if processor is None else processor.feature_injection
    image_span = (
        0
        if injection is None
        else max(
            int(model.max_vit_grid_tokens), 0 if flow is None else int(flow.max_vae_grid_tokens)
        )
        + (2 if injection.layout is FeatureLayout.FRAMED else 0)
    )
    text_tokens = max_tokens + image_span
    flow_tokens = (
        0
        if flow is None
        else branches
        * max(
            (
                int(flow.max_latent_tokens),
                *(
                    flow.physical_tokens(height, width)
                    for height, width in config.flow_graph_shapes
                ),
            )
        )
    )
    return InputGeometry(
        max_rows=max_rows * branches,
        max_tokens=text_tokens + flow_tokens,
        max_text_tokens=text_tokens,
        max_blocks_per_row=max(1, ceil_div(model.text_max_tokens, config.block_size)),
        hidden_size=model.hidden_size,
    )


def tensor_slot_capacity(
    schema: Mapping[str, TensorSchema],
    group: Communicator,
    *,
    maximum: int,
    minimum: int,
    available_bytes: int,
    product_bytes_per_request: int = 0,
) -> int:
    """Size identical request-slot counts within every participating device's grant."""

    bytes_per_slot = (
        sum(field.nbytes for field in schema.values() if field.memory != "pinned")
        + product_bytes_per_request
    )
    if bytes_per_slot < 1 or minimum < 1 or maximum < minimum:
        raise ValueError("request tensor capacity requires valid byte and slot bounds")
    available = min(maximum, available_bytes // bytes_per_slot)
    agreed = torch.tensor(available, dtype=torch.int64, device=group.device)
    group.all_reduce_min(agreed)
    count = int(agreed.item())
    if count < minimum:
        required_bytes = minimum * bytes_per_slot
        raise RuntimeError(
            "insufficient device memory for the required request tensor slots: "
            f"{required_bytes} bytes required for {minimum} slots at "
            f"{bytes_per_slot} bytes per slot, {available_bytes} bytes available"
        )
    return count


@dataclass(frozen=True)
class RuntimeKVCapacity:
    """Resolved KV token and page capacity within the assigned physical budget."""

    block_size: int
    bytes_per_token: int
    token_capacity: int
    num_blocks: int


def latent_trajectory_bytes(
    latent_units: int,
    latent_width: int,
    dtype_bytes: int,
) -> int:
    """Calculate storage for one latent trajectory from unit count, width, and element size."""

    units = int(latent_units)
    width = int(latent_width)
    element_bytes = int(dtype_bytes)
    if units < 0 or width < 1 or element_bytes < 1:
        raise ValueError("latent trajectory geometry is invalid")
    return units * width * element_bytes


def latent_pool_capacity_bytes(
    *,
    request_pool_size: int,
    num_pages: int,
    page_units: int,
    latent_width: int,
    dtype_bytes: int,
) -> int:
    """Calculate double-buffered latent pages, step storage, page tables, and timestep metadata."""

    slots = int(request_pool_size)
    pages = int(num_pages)
    units = int(page_units)
    width = int(latent_width)
    element_bytes = int(dtype_bytes)
    if min(slots, units, width, element_bytes) < 1 or pages < 2:
        raise ValueError("latent pool geometry is invalid")
    usable_pages = pages - 1
    storage = 2 * pages * units * width * element_bytes
    step_buffer = usable_pages * units * width * element_bytes
    page_table = usable_pages * 8
    timestep_pairs = (slots + 1) * 2 * 4
    return storage + step_buffer + page_table + timestep_pairs


@dataclass(frozen=True, slots=True)
class ArenaCapacity:
    """Budgets latent storage, device products, transfers, and CPU tasks for one worker arena."""

    latent_pool_bytes: int
    device_products: int
    device_product_bytes: int
    transfer_bytes: int
    transfer_tickets: int
    cpu_tasks: int


def operation_window(pipeline_depth: int, max_operations: int) -> int:
    """Bound simultaneously live operations by pipeline depth and per-batch capacity."""

    depth = int(pipeline_depth)
    operations = int(max_operations)
    if depth < 1 or operations < 1:
        raise ValueError("operation-window sizing requires positive bounds")
    return min(depth * operations, max(2, depth))


def request_tensor_arena_capacity(
    worker_config: WorkerConfig,
    *,
    pipeline_depth: int,
    product_bytes_per_request: int,
) -> ArenaCapacity:
    """Bound public product, relay and transfer storage for fixed request tensors."""

    depth = int(pipeline_depth)
    max_operations = int(worker_config.max_batch_operations)
    state_slots = int(worker_config.max_request_pool_size)
    slots = depth * max_operations
    unresolved_window = depth // state_slots - 1
    device_products = _DEVICE_PRODUCTS_PER_OPERATION * (
        slots + _DEVICE_PRODUCT_RETIREMENT_BATCHES * max_operations
    )
    relay_bytes = (
        (state_slots + 1)
        * (max(1, unresolved_window) + _REQUEST_RELAY_RETIREMENT_LANES)
        * _REQUEST_RELAY_ROW_BYTES
    )
    return ArenaCapacity(
        latent_pool_bytes=0,
        device_products=device_products,
        device_product_bytes=(
            device_product_capacity_bytes(
                device_products, 1, selected_points_per_operation=1, max_value_bytes=1
            )
            + relay_bytes
        ),
        transfer_bytes=max(1, state_slots * product_bytes_per_request),
        transfer_tickets=max(1, min(slots, _MAX_TRANSFER_ENTRIES)),
        cpu_tasks=state_slots * (unresolved_window + 1),
    )


def model_arena_capacity(
    model: ExecutionModel,
    worker_config: WorkerConfig,
    *,
    pipeline_depth: int,
    completion_payload_bytes: int,
    num_blocks: int,
    request_pool_size: int,
    num_latent_pages: int,
    latent_page_units: int,
    latent_width: int,
    max_latent_feature_bytes: int,
    max_vision_feature_bytes: int,
    bytes_per_token: int,
) -> ArenaCapacity:
    """Derive device-product, transfer, latent, and CPU arena bounds from worker_config geometry."""

    depth = int(pipeline_depth)
    payload_bytes = int(completion_payload_bytes)
    max_operations = int(worker_config.max_batch_operations)
    if depth < 1 or payload_bytes < 1 or max_operations < 1:
        raise ValueError("model arena sizing requires positive runtime bounds")

    slots = depth * max_operations
    if bool(model.resource_geometry.request_tensors):
        return request_tensor_arena_capacity(
            worker_config,
            pipeline_depth=depth,
            product_bytes_per_request=model.local_product_storage_bytes,
        )
    transfer_tickets = min(slots, _MAX_TRANSFER_ENTRIES)
    block_size = int(worker_config.block_size)
    flow = model.generation
    if flow is not None and not isinstance(flow, GenerationPipeline):
        raise ValueError("model generation behavior has an invalid type")
    latent_pool_bytes = 0
    latent_transfer_bytes = 0
    artifact_bytes = 0
    if flow is not None:
        dtype_bytes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
        }.get(str(worker_config.model_dtype).removeprefix("torch.").lower())
        if dtype_bytes is None:
            raise ValueError(f"unsupported latent dtype {worker_config.model_dtype!r}")
        latent_pool_bytes = latent_pool_capacity_bytes(
            request_pool_size=int(request_pool_size),
            num_pages=int(num_latent_pages),
            page_units=int(latent_page_units),
            latent_width=int(latent_width),
            dtype_bytes=dtype_bytes,
        )
        latent_transfer_bytes = latent_trajectory_bytes(
            int(flow.max_latent_tokens),
            int(latent_width),
            dtype_bytes,
        )
        raw_image_bytes = int(flow.max_vae_grid_tokens) * int(flow.latent_downsample) ** 2 * 3
        # Device feedback is the decoded BF16 image. Encoded PNG/base64 bytes
        # belong to the pinned CPU output owner and consume no device arena.
        artifact_bytes = raw_image_bytes * 2
    max_transfer_bytes = max(
        int(num_blocks) * block_size * int(bytes_per_token),
        latent_transfer_bytes,
        int(max_latent_feature_bytes),
        int(max_vision_feature_bytes),
        1,
    )
    max_product_bytes = max(
        1,
        artifact_bytes,
    )
    device_product_slots = slots + _DEVICE_PRODUCT_RETIREMENT_BATCHES * max_operations
    device_products = _DEVICE_PRODUCTS_PER_OPERATION * device_product_slots
    device_count = len(
        {
            str(worker_config.device),
            str(worker_config.generation_device or worker_config.device),
        }
    )
    device_product_bytes = device_product_capacity_bytes(
        device_products,
        device_count,
        selected_points_per_operation=1,
        max_value_bytes=max_product_bytes,
    )
    device_product_bytes += (
        (int(request_pool_size) + 1)
        * (operation_window(depth, max_operations) + _REQUEST_RELAY_RETIREMENT_LANES)
        * _REQUEST_RELAY_ROW_BYTES
        * device_count
    )

    return ArenaCapacity(
        latent_pool_bytes=latent_pool_bytes,
        device_products=device_products,
        device_product_bytes=device_product_bytes,
        transfer_bytes=max_transfer_bytes * transfer_tickets,
        transfer_tickets=transfer_tickets,
        cpu_tasks=_CPU_TASKS,
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
        blocks = DEFAULT_NUM_BLOCKS_FALLBACK if default_blocks is None else int(default_blocks)
    else:
        blocks = int(kv_token_capacity) // block
    return max(min_blocks, blocks)


def device_total_bytes(device: str | torch.device) -> int:
    """Return CUDA capacity, propagating device errors so an unknown budget cannot become zero."""

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
    """Size one physical KV pool from explicit tokens or a host-owned byte grant.

    Fixed-capacity callers supply their page count. Automatic CUDA sizing requires
    a granted budget; CPU geometry uses its declared default page policy.
    """

    block = int(block_size)
    token_bytes = int(bytes_per_token)
    if block < 1 or token_bytes < 1 or floor < 1 or resident_copies < 1 or co_resident_blocks < 0:
        raise ValueError("KV capacity geometry must be positive")
    if available_bytes is not None and available_bytes < 0:
        raise ValueError("KV memory grant must not be negative")
    if kv_token_capacity is not None:
        if kv_token_capacity <= 0:
            raise ValueError("configured KV token capacity must be positive")
        blocks = derive_num_blocks(block, kv_token_capacity, floor=floor)
    elif available_bytes is not None:
        blocks = (available_bytes // (block * token_bytes) - co_resident_blocks) // resident_copies
        if blocks < floor:
            raise ValueError("device memory grant cannot hold the required KV pool")
    elif device is not None and torch.device(device).type == "cuda":
        raise ValueError("automatic CUDA KV sizing requires a host memory grant")
    else:
        blocks = derive_num_blocks(block, None, default_blocks=default_blocks, floor=floor)
    if (
        available_bytes is not None
        and (resident_copies * blocks + co_resident_blocks) * block * token_bytes > available_bytes
    ):
        raise ValueError("configured KV storage exceeds the device memory grant")
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
    "operation_window",
]
