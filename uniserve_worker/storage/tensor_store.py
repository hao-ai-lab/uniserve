"""Numerical tensor views and metadata for native worker storage.

``TensorStore`` owns buffer visibility, read leases, imports, and retirement.
These PyTorch operations copy or view borrowed tensor storage; they do not own
request progress, allocation lifetime, or stream completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from uniserve import _slices
from uniserve_worker._uniserve_ipc import (
    Buffer as Buffer,
)
from uniserve_worker._uniserve_ipc import (
    TensorImport as TensorImport,
)
from uniserve_worker._uniserve_ipc import (
    TensorRead as TensorRead,
)
from uniserve_worker._uniserve_ipc import (
    TensorStore as TensorStore,
)
from uniserve_worker.errors import (
    WorkerError,
    WorkerErrorCode,
    invalid_descriptor,
)
from uniserve_worker.protocol.tensor import DType, StaticDim, TensorRef

_DEVICE_DTYPES: Final[dict[DType, torch.dtype]] = {
    DType.U8: torch.uint8,
    DType.I32: torch.int32,
    DType.I16: torch.int16,
    DType.I64: torch.long,
    DType.F16: torch.float16,
    DType.BF16: torch.bfloat16,
    DType.F32: torch.float32,
}

_DEVICE_TORCH_DTYPES: Final[tuple[torch.dtype, ...]] = tuple(
    dict.fromkeys(_DEVICE_DTYPES.values())
)

_DTYPE_STORAGE: Final[dict[DType, tuple[str, int]]] = {
    dtype: (
        str(torch_dtype).removeprefix("torch."),
        int(torch.empty((), dtype=torch_dtype).element_size()),
    )
    for dtype, torch_dtype in _DEVICE_DTYPES.items()
}


def device_product_storage(dtype: DType) -> tuple[str, int]:
    """Return the concrete tensor storage used for one product dtype.

    Returns:
        The torch dtype name without its ``torch.`` prefix (for example
        ``"bfloat16"``) and the element size in bytes.
    """
    return _DTYPE_STORAGE[DType(dtype)]


def device_product_capacity_bytes(
    slot_capacity: int,
    device_count: int,
    *,
    max_value_bytes: int,
) -> int:
    """Return the fixed backing bound for one ``TensorStore`` owner.

    The bound is ``slot_capacity * device_count * (scalars + max_value_bytes)``
    where ``scalars`` sums the element size of every supported product dtype.
    Request-relay arena bytes are not included;
    ``uniserve_worker.bootstrap.capacity`` adds them separately.

    Raises:
        ValueError: If any dimension is less than one.
    """
    slots = int(slot_capacity)
    devices = int(device_count)
    value_bytes = int(max_value_bytes)
    if min(slots, devices, value_bytes) < 1:
        raise ValueError("device-product dimensions must be positive")
    scalar_bytes = slots * devices * sum(dict(_DTYPE_STORAGE.values()).values())
    return scalar_bytes + slots * devices * value_bytes


def _device_dtype(dtype: DType) -> torch.dtype:
    """Resolve a product dtype descriptor to its torch dtype."""
    return _DEVICE_DTYPES[dtype]


def _device_shape(reference: TensorRef) -> tuple[int, ...]:
    """Resolve a product's bounded tensor dimensions to a concrete shape.

    Static dimensions use their extent and dynamic dimensions their upper
    bound. A zero-dimensional product is stored as shape ``(1,)``.
    """
    dims = tuple(
        dim.extent if isinstance(dim, StaticDim) else dim.bound
        for dim in reference.shape_bound.dims
    )
    return dims or (1,)


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Spatial dimensions and numerical value range of an immutable image.

    Zero height and width mean the dimensions are undeclared. ``value_range``
    is the numerical interval of the pixel values, such as ``(-1.0, 1.0)``.
    """

    height: int = 0
    width: int = 0
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Require complete, nonnegative image dimensions."""
        if self.height < 0 or self.width < 0:
            raise ValueError(
                "device-product image dimensions must be non-negative"
            )
        if (self.height == 0) != (self.width == 0):
            raise ValueError("device-product image dimensions must be complete")


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Spatial dimensions of an immutable floating-point encoder feature."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if min(self.height, self.width) < 1:
            raise ValueError("encoder feature dimensions must be positive")


def _value_view(
    tensor: torch.Tensor, shape: tuple[int, ...], extent: int
) -> torch.Tensor:
    """Borrow the published extent, preserving strides of an exact shard."""
    if tuple(tensor.shape) == shape:
        return tensor
    return tensor.reshape(-1)[:extent].reshape(shape)


def _copy_value(
    reference: TensorRef,
    target: torch.Tensor,
    region: tuple[slice, ...] | None,
    feature: bool,
    value: torch.Tensor,
    metadata: ImageMetadata | FeatureMetadata | None,
) -> torch.Tensor:
    """Copy one declared value without changing its shape or representation."""
    if feature:
        if not isinstance(metadata, FeatureMetadata):
            raise invalid_descriptor(
                "encoder feature requires spatial metadata"
            )
        value = value.to(dtype=target.dtype)
    elif isinstance(metadata, FeatureMetadata):
        raise invalid_descriptor("feature metadata requires feature admission")
    if value.dtype != target.dtype:
        raise invalid_descriptor("tensor export changes its declared dtype")
    shape = tuple(value.shape)
    matches = (
        shape == _slices.shape(region)
        if region is not None
        else reference.shape_bound.contains_shape(shape)
    )
    if not matches:
        raise invalid_descriptor("tensor export changes its declared shape")
    if value.numel() > target.numel():
        raise WorkerError(
            code=WorkerErrorCode.INVARIANT_VIOLATION,
            message="device product exceeds its registered shape bound",
            fatal=True,
        )
    view = (
        target
        if region is not None
        else _value_view(target, shape, value.numel())
    )
    source = value.detach()
    # Exact aliases need no copy; noncontiguous shard strides remain intact.
    if (
        view.data_ptr() != source.data_ptr()
        or view.dtype != source.dtype
        or view.stride() != source.stride()
    ):
        view.copy_(source, non_blocking=value.device.type == "cuda")
    return view.reshape(shape)


def _copy_scalars(
    targets: tuple[torch.Tensor, ...], values: torch.Tensor
) -> None:
    """Scatter one scalar per destination in one PyTorch copy operation."""
    flat = values.detach().reshape(-1)
    if flat.numel() != len(targets):
        raise invalid_descriptor(
            "batched device-product export requires one scalar per output"
        )
    if any(target.numel() != 1 for target in targets):
        raise invalid_descriptor(
            "batched device-product export requires scalar output bounds"
        )
    first = targets[0]
    if any(
        target.device != first.device or target.dtype != first.dtype
        for target in targets
    ):
        raise invalid_descriptor(
            "batched device-product export spans incompatible storage"
        )
    source = flat.to(dtype=first.dtype)
    torch._foreach_copy_(
        targets,
        source.reshape(-1, 1).unbind(0),
        non_blocking=source.device.type == "cuda",
    )
