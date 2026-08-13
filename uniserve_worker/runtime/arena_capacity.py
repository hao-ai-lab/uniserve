"""Local physical arena sizing for one worker process."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.generation import GenerationPipeline
from ..models.runtime import ExecutionModel, WorkerDeployment
from .latent_capacity import latent_store_capacity_bytes
from .product_capacity import device_product_arena_bytes

_PRODUCTS_PER_OPERATION = 5
_MAX_TRANSFER_ENTRIES = 256
_CPU_TASKS = 256


@dataclass(frozen=True, slots=True)
class ArenaCapacity:
    latent_bytes: int
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
    latent_capacity_units: int,
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
    max_transfer_bytes = max(
        int(num_blocks) * block_size * int(bytes_per_token),
        int(max_latent_feature_bytes),
        int(max_vision_feature_bytes),
        1,
    )

    flow = model.generation
    if flow is not None and not isinstance(flow, GenerationPipeline):
        raise ValueError("model generation behavior has an invalid type")
    latent_bytes = 0
    artifact_bytes = 0
    if flow is not None:
        latent_bytes = latent_store_capacity_bytes(
            int(latent_capacity_units),
            int(flow.latent_channels),
            int(flow.latent_patch_size),
        )
        raw_image_bytes = int(flow.max_vae_grid_tokens) * int(flow.latent_downsample) ** 2 * 3
        artifact_bytes = ((2 * raw_image_bytes + (1 << 20) + 2) // 3) * 4
    max_product_bytes = max(
        1,
        artifact_bytes,
        int(max_latent_feature_bytes),
        int(max_vision_feature_bytes),
    )
    device_products = _PRODUCTS_PER_OPERATION * slots
    device_count = len(
        {
            str(deployment.device),
            str(deployment.generation_device or deployment.device),
        }
    )
    device_product_bytes = device_product_arena_bytes(
        device_products,
        device_count,
        selected_points_per_operation=1,
        max_product_bytes=max_product_bytes,
    )

    return ArenaCapacity(
        latent_bytes=latent_bytes,
        device_products=device_products,
        device_product_bytes=device_product_bytes,
        transfer_bytes=max_transfer_bytes * transfer_tickets,
        transfer_tickets=transfer_tickets,
        cpu_tasks=_CPU_TASKS,
    )


def system_arena_capacity(
    *,
    pipeline_depth: int,
    max_operations: int,
    completion_payload_bytes: int,
) -> ArenaCapacity:
    depth = int(pipeline_depth)
    operations = int(max_operations)
    payload_bytes = int(completion_payload_bytes)
    if depth < 1 or operations < 1 or payload_bytes < 1:
        raise ValueError("system arena sizing requires positive runtime bounds")
    slots = depth * operations
    return ArenaCapacity(
        latent_bytes=0,
        device_products=1,
        device_product_bytes=(1 << 20) * _PRODUCTS_PER_OPERATION * slots,
        transfer_bytes=(1 << 20) * slots,
        transfer_tickets=slots,
        cpu_tasks=_CPU_TASKS,
    )


__all__ = [
    "ArenaCapacity",
    "model_arena_capacity",
    "operation_window",
    "system_arena_capacity",
]
