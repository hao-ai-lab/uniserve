"""Output and row-layout conventions shared by numerical entry points."""

from __future__ import annotations

from enum import StrEnum
from math import prod

import torch

from uniserve.quantization import QuantizedTensor


class Rounding(StrEnum):
    """Where an elementwise expression rounds to the activation dtype.

    ``ONCE`` accumulates the complete expression in FP32 and rounds its
    result, as UniServe's fused kernels evaluate it. ``STEPWISE`` rounds after
    every tensor operation of the expression, as eager PyTorch evaluates it in
    the activation dtype: each operation computes in FP32 and stores a rounded
    result that the next operation reads. A reduction inside one operation,
    such as an RMS normalization's statistics, still accumulates in FP32 and
    rounds once. A checkpoint's numerical contract selects the recipe its
    reference implementation evaluates.
    """

    ONCE = "once"
    STEPWISE = "stepwise"


def result(value: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """Return ``value``, or copy it into matching caller output storage."""
    if out is None:
        return value
    if (
        out.shape != value.shape
        or out.dtype != value.dtype
        or out.device != value.device
    ):
        raise ValueError(
            "output must match the numerical result's shape, dtype and device"
        )
    return out.copy_(value)


def check_output(value: torch.Tensor, out: torch.Tensor) -> None:
    """Require output storage matching ``value``'s shape, dtype and device."""
    if (
        out.shape != value.shape
        or out.dtype != value.dtype
        or out.device != value.device
    ):
        raise ValueError(
            "output must match the numerical result's shape, dtype and device"
        )


def as_matrix(x: torch.Tensor) -> torch.Tensor:
    """Flatten leading axes into logical GEMM rows, preserving encoded
    layouts.
    """  # noqa: D205
    if x.ndim == 2:
        return x
    shape = (prod(x.shape[:-1]), x.shape[-1])
    if not isinstance(x, QuantizedTensor):
        return x.reshape(shape)

    fields = dict(x.buffers())
    # nvfp4 packs two values per byte, so its physical row width halves.
    fields["values"] = fields["values"].reshape(
        shape if x.quantizer.format != "nvfp4" else (shape[0], shape[1] // 2)
    )
    if x.quantizer.axis == 0:
        fields["scale"] = (
            fields["scale"].expand(*x.shape[:-1], 1).reshape(shape[0], 1)
        )
    return x.quantizer.from_tensors(
        fields, shape=shape, dtype=x.dtype, scale_layout=x.scale_layout
    )
