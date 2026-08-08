"""Local physical arena sizing for one worker process."""

from __future__ import annotations

from dataclasses import dataclass

from ..foundation.sizing import ceil_div
from ..spec import DeploymentOverlay, ModelSpec
from .latent_capacity import latent_store_capacity_bytes
from .product_capacity import device_product_arena_bytes

_PRODUCTS_PER_OPERATION = 5
_MAX_TRANSFER_ENTRIES = 256
_CPU_TASKS = 256
_COMPLETION_FIELDS = 4


@dataclass(frozen=True, slots=True)
class ArenaCapacity:
    latent_bytes: int
    device_products: int
    device_product_bytes: int
    transfer_bytes: int
    transfer_tickets: int
    cpu_tasks: int
    pinned_staging_bytes: int


def operation_window(pipeline_depth: int, max_operations: int) -> int:
    depth = int(pipeline_depth)
    operations = int(max_operations)
    if depth < 1 or operations < 1:
        raise ValueError("operation-window sizing requires positive bounds")
    return min(depth * operations, max(2, depth))


def model_arena_capacity(
    spec: ModelSpec,
    deployment: DeploymentOverlay,
    *,
    pipeline_depth: int,
    completion_payload_bytes: int,
    num_blocks: int,
    scratch_capacity_tokens: int,
    max_latent_size: int,
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
    window = operation_window(depth, max_operations)
    transfer_tickets = min(slots, _MAX_TRANSFER_ENTRIES)
    block_size = int(deployment.block_size)
    kv_pages = int(num_blocks) + ceil_div(int(scratch_capacity_tokens), block_size)
    max_route_tokens = max(
        (int(route.shape.max_tokens_per_row) for route in spec.routes),
        default=1,
    )
    max_transfer_bytes = max(
        int(num_blocks) * block_size * int(bytes_per_token),
        int(max_latent_feature_bytes),
        int(max_vision_feature_bytes),
        1,
    )

    flow = spec.flow
    latent_bytes = 0
    artifact_bytes = 0
    if flow is not None:
        latent_bytes = latent_store_capacity_bytes(
            int(max_latent_size),
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

    per_operation_staging = payload_bytes + max_route_tokens * 32 + kv_pages * 8 + 256
    completion_words = _COMPLETION_FIELDS * max_operations + (payload_bytes + 3) // 4
    completion_arena_bytes = depth * completion_words * 8
    return ArenaCapacity(
        latent_bytes=latent_bytes,
        device_products=device_products,
        device_product_bytes=device_product_bytes,
        transfer_bytes=max_transfer_bytes * transfer_tickets,
        transfer_tickets=transfer_tickets,
        cpu_tasks=_CPU_TASKS,
        pinned_staging_bytes=max(
            completion_arena_bytes,
            per_operation_staging * window,
        ),
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
    window = operation_window(depth, operations)
    per_operation_staging = payload_bytes + 256
    completion_words = _COMPLETION_FIELDS * operations + (payload_bytes + 3) // 4
    return ArenaCapacity(
        latent_bytes=0,
        device_products=1,
        device_product_bytes=(1 << 20) * _PRODUCTS_PER_OPERATION * slots,
        transfer_bytes=(1 << 20) * slots,
        transfer_tickets=slots,
        cpu_tasks=_CPU_TASKS,
        pinned_staging_bytes=max(
            depth * completion_words * 8,
            per_operation_staging * window,
        ),
    )


__all__ = [
    "ArenaCapacity",
    "model_arena_capacity",
    "operation_window",
    "system_arena_capacity",
]
