"""Local physical arena sizing for one worker process."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.generation import GenerationPipeline
from ..models.runtime import ExecutionModel, WorkerDeployment
from ..runtime.device_products import device_product_capacity_bytes

_PRODUCTS_PER_OPERATION = 5
_MAX_TRANSFER_ENTRIES = 256
_CPU_TASKS = 256


def latent_trajectory_bytes(
    latent_units: int,
    latent_width: int,
    dtype_bytes: int,
) -> int:
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
    latent_pool_bytes: int
    device_products: int
    device_product_bytes: int
    transfer_bytes: int
    transfer_tickets: int
    cpu_tasks: int


def operation_window(pipeline_depth: int, max_operations: int) -> int:
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
    scratch_capacity_tokens: int,
    request_pool_size: int,
    num_latent_pages: int,
    latent_page_units: int,
    latent_width: int,
    max_latent_feature_bytes: int,
    max_vision_feature_bytes: int,
    bytes_per_token: int,
) -> ArenaCapacity:
    depth = int(pipeline_depth)
    payload_bytes = int(completion_payload_bytes)
    max_operations = int(deployment.max_batch_operations)
    if depth < 1 or payload_bytes < 1 or max_operations < 1:
        raise ValueError("model arena sizing requires positive runtime bounds")

    slots = depth * max_operations
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
    device_products = _PRODUCTS_PER_OPERATION * slots
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

    return ArenaCapacity(
        latent_pool_bytes=latent_pool_bytes,
        device_products=device_products,
        device_product_bytes=device_product_bytes,
        transfer_bytes=max_transfer_bytes * transfer_tickets,
        transfer_tickets=transfer_tickets,
        cpu_tasks=_CPU_TASKS,
    )


__all__ = [
    "ArenaCapacity",
    "latent_pool_capacity_bytes",
    "latent_trajectory_bytes",
    "model_arena_capacity",
    "operation_window",
]
