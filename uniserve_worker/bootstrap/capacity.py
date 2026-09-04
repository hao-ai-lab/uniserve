"""Local physical arena sizing for one worker process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models.generation import GenerationPipeline
from ..models.runtime import ExecutionModel, WorkerDeployment
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


@dataclass(frozen=True)
class CudaKVCapacity:
    """Records the device memory budget and resulting CUDA KV page capacity."""

    device: str
    free_bytes: int
    total_bytes: int
    memory_fraction: float
    bytes_per_token: int
    block_size: int
    token_capacity: int
    num_blocks: int


@dataclass(frozen=True)
class RuntimeKVCapacity:
    """Records resolved KV token and page capacity with optional CUDA memory evidence."""

    block_size: int
    bytes_per_token: int
    token_capacity: int
    num_blocks: int
    cuda: CudaKVCapacity | None = None


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


def model_arena_capacity(
    model: ExecutionModel,
    deployment: WorkerDeployment,
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
    """Derive device-product, transfer, latent, and CPU arena bounds from deployment geometry."""

    depth = int(pipeline_depth)
    payload_bytes = int(completion_payload_bytes)
    max_operations = int(deployment.max_batch_operations)
    if depth < 1 or payload_bytes < 1 or max_operations < 1:
        raise ValueError("model arena sizing requires positive runtime bounds")

    slots = depth * max_operations
    state_geometry = model.dedicated_state_geometry
    if state_geometry is not None:
        transfer_tickets = min(slots, _MAX_TRANSFER_ENTRIES)
        state_slots = int(state_geometry.slot_count)
        unresolved_window = depth // state_slots - 1
        device_products = _DEVICE_PRODUCTS_PER_OPERATION * (
            slots + _DEVICE_PRODUCT_RETIREMENT_BATCHES * max_operations
        )
        relay_bytes = (
            (int(request_pool_size) + 1)
            * (max(1, unresolved_window) + _REQUEST_RELAY_RETIREMENT_LANES)
            * _REQUEST_RELAY_ROW_BYTES
        )
        return ArenaCapacity(
            latent_pool_bytes=0,
            device_products=device_products,
            device_product_bytes=(
                device_product_capacity_bytes(
                    device_products,
                    1,
                    selected_points_per_operation=1,
                    max_value_bytes=1,
                )
                + relay_bytes
            ),
            transfer_bytes=max(1, transfer_tickets),
            transfer_tickets=max(1, transfer_tickets),
            cpu_tasks=state_slots * (unresolved_window + 1),
        )
    transfer_tickets = min(slots, _MAX_TRANSFER_ENTRIES)
    block_size = int(deployment.block_size)
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
        }.get(str(deployment.model_dtype).removeprefix("torch.").lower())
        if dtype_bytes is None:
            raise ValueError(f"unsupported latent dtype {deployment.model_dtype!r}")
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
        artifact_bytes = ((2 * raw_image_bytes + (1 << 20) + 2) // 3) * 4
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
            str(deployment.device),
            str(deployment.generation_device or deployment.device),
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


def device_total_bytes(device: Any) -> int:
    """Return one device's total CUDA memory, or zero for a non-CUDA device."""

    try:
        import torch
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    try:
        target = torch.device(device)
    except Exception:
        return 0
    if target.type != "cuda":
        return 0
    try:
        _free, total = torch.cuda.mem_get_info(target)
    except Exception:
        return 0
    return int(total)


def derive_cuda_kv_capacity(
    *,
    device: Any,
    block_size: int,
    bytes_per_token: int,
    memory_fraction: float,
    floor: int = 1,
    resident_copies: int = 1,
    co_resident_blocks: int = 0,
) -> CudaKVCapacity | None:
    """Derive KV token/block capacity from the device static-memory budget.

    ``None`` means CUDA sizing is unavailable and callers should use their
    non-CUDA capacity policy.

    ``memory_fraction`` is the share of total device memory that static
    residency may hold: the memory already resident when sizing runs, such as
    model weights, plus every KV pool provisioned from the derived capacity.
    A deployment that keeps ``resident_copies`` copies of the derived pool
    resident at once, plus ``co_resident_blocks`` of fixed KV storage, receives
    a block count that satisfies
    ``resident_copies * num_blocks + co_resident_blocks`` within the budget.
    """

    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None
    try:
        cuda_device = torch.device(device)
    except Exception:
        return None
    if cuda_device.type != "cuda":
        return None
    block = int(block_size)
    token_bytes = max(1, int(bytes_per_token))
    if block <= 0:
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    except Exception:
        return None
    fraction = float(memory_fraction)
    resident_bytes = max(0, int(total_bytes) - int(free_bytes))
    usable_bytes = max(0, int(float(total_bytes) * fraction) - resident_bytes)
    budget_blocks = (usable_bytes // token_bytes) // block
    copies = max(1, int(resident_copies))
    request_blocks = (budget_blocks - max(0, int(co_resident_blocks))) // copies
    num_blocks = max(max(1, int(floor)), request_blocks)
    return CudaKVCapacity(
        device=str(cuda_device),
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        memory_fraction=fraction,
        bytes_per_token=token_bytes,
        block_size=block,
        token_capacity=int(num_blocks * block),
        num_blocks=int(num_blocks),
    )


def derive_runtime_kv_capacity(
    *,
    block_size: int,
    kv_token_capacity: int | None,
    bytes_per_token: int,
    device: Any = None,
    memory_fraction: float = 1.0,
    floor: int = 1,
    default_blocks: int | None = None,
    resident_copies: int = 1,
    co_resident_blocks: int = 0,
) -> RuntimeKVCapacity:
    """Resolve the worker-facing KV capacity policy in one place.

    Explicit token capacity wins. Otherwise CUDA free-memory sizing is used
    when available, with ``resident_copies`` and ``co_resident_blocks``
    describing the KV storage that shares the memory-fraction budget with the
    request pool. CPU/unavailable-CUDA paths fall back to
    :func:`derive_num_blocks`.
    """

    block = int(block_size)
    token_bytes = max(1, int(bytes_per_token))
    if kv_token_capacity is not None and int(kv_token_capacity) > 0:
        blocks = derive_num_blocks(block, kv_token_capacity, floor=floor)
        return RuntimeKVCapacity(
            block_size=block,
            bytes_per_token=token_bytes,
            token_capacity=int(blocks * block),
            num_blocks=int(blocks),
            cuda=None,
        )

    cuda = derive_cuda_kv_capacity(
        device=device,
        block_size=block,
        bytes_per_token=token_bytes,
        memory_fraction=memory_fraction,
        floor=floor,
        resident_copies=resident_copies,
        co_resident_blocks=co_resident_blocks,
    )
    if cuda is not None:
        return RuntimeKVCapacity(
            block_size=block,
            bytes_per_token=token_bytes,
            token_capacity=int(cuda.token_capacity),
            num_blocks=int(cuda.num_blocks),
            cuda=cuda,
        )

    blocks = derive_num_blocks(block, None, default_blocks=default_blocks, floor=floor)
    return RuntimeKVCapacity(
        block_size=block,
        bytes_per_token=token_bytes,
        token_capacity=int(blocks * block),
        num_blocks=int(blocks),
        cuda=None,
    )


__all__ = [
    "ArenaCapacity",
    "CudaKVCapacity",
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
