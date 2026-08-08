"""Physical byte geometry for bounded device-product storage."""

from __future__ import annotations

from typing import Final

from ..batch import DType

DEVICE_PRODUCT_STORAGE: Final[dict[DType, tuple[str, int]]] = {
    DType.U8: ("uint8", 1),
    DType.U16: ("int32", 4),
    DType.U32: ("int64", 8),
    DType.I32: ("int32", 4),
    DType.I64: ("int64", 8),
    DType.F16: ("float16", 2),
    DType.BF16: ("bfloat16", 2),
    DType.F32: ("float32", 4),
}


def device_product_storage(dtype: DType) -> tuple[str, int]:
    return DEVICE_PRODUCT_STORAGE[DType(dtype)]


def device_product_scalar_arena_bytes(slot_capacity: int, device_count: int) -> int:
    """Backing bytes for every distinct scalar storage arena on each device."""

    slots = int(slot_capacity)
    devices = int(device_count)
    if slots < 1 or devices < 1:
        raise ValueError("device-product scalar capacity requires positive slots and devices")
    storage_bytes = {storage: width for storage, width in DEVICE_PRODUCT_STORAGE.values()}
    return slots * devices * sum(storage_bytes.values())


def device_product_arena_bytes(
    slot_capacity: int,
    device_count: int,
    *,
    selected_points_per_operation: int,
    max_product_bytes: int,
) -> int:
    """Backing bytes for every scalar arena and every bounded tensor slot."""

    slots = int(slot_capacity)
    devices = int(device_count)
    points = int(selected_points_per_operation)
    product_bytes = int(max_product_bytes)
    if slots < 1 or devices < 1 or points < 1 or product_bytes < 1:
        raise ValueError("device-product arena geometry must be positive")
    _u32_storage, u32_width = device_product_storage(DType.U32)
    _i64_storage, i64_width = device_product_storage(DType.I64)
    accepted_span_bytes = (points + 1) * u32_width
    continuation_bytes = 4 * i64_width
    tensor_slot_bytes = max(product_bytes, accepted_span_bytes, continuation_bytes)
    return device_product_scalar_arena_bytes(slots, devices) + (slots * devices * tensor_slot_bytes)


__all__ = [
    "DEVICE_PRODUCT_STORAGE",
    "device_product_arena_bytes",
    "device_product_scalar_arena_bytes",
    "device_product_storage",
]
