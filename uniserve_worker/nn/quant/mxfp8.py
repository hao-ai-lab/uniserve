"""Block-scaled MXFP8 linear execution on Blackwell GPUs.

Weights are quantized once after checkpoint loading, while activations are
quantized for each invocation. FlashInfer produces the swizzled scale-factor
layout consumed by the cuDNN MXFP8 matrix multiplication.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from .base import QuantizeMethodBase

__all__ = ["DynamicW8A8MxFp8LinearMethod"]

_MXFP8_BLOCK_SIZE = 32


def _flashinfer() -> Any:
    """Import the accelerator extension when an MXFP8 operation is requested."""

    import flashinfer

    return flashinfer


@torch.library.custom_op("uniserve_worker::mxfp8_quantize", mutates_args=())
def _mxfp8_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a rank-two tensor and return E4M3 values plus swizzled scales."""

    return _flashinfer().mxfp8_quantize(
        value,
        is_sf_swizzled_layout=True,
        backend="cuda",
    )


@_mxfp8_quantize.register_fake
def _mxfp8_quantize_fake(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe MXFP8 quantization outputs for graph tracing."""

    rows, width = value.shape

    # Scale storage pads matrix rows to 128 and groups four consecutive
    # 32-element quantization blocks for the accelerator's swizzled layout.
    scale_rows = ((rows + 127) // 128) * 128
    scale_columns = (((width // _MXFP8_BLOCK_SIZE) + 3) // 4) * 4
    return (
        value.new_empty((rows, width), dtype=torch.float8_e4m3fn),
        value.new_empty((scale_rows * scale_columns,), dtype=torch.uint8),
    )


@torch.library.custom_op("uniserve_worker::mxfp8_mm_bf16", mutates_args=())
def _mxfp8_mm_bf16(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
) -> torch.Tensor:
    """Multiply two MXFP8 matrices into a bfloat16 result."""

    return _flashinfer().mm_mxfp8(
        left,
        right,
        left_scale,
        right_scale,
        out_dtype=torch.bfloat16,
        backend="cudnn",
    )


@_mxfp8_mm_bf16.register_fake
def _mxfp8_mm_bf16_fake(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
) -> torch.Tensor:
    """Describe the bfloat16 matrix result for graph tracing."""

    del left_scale, right_scale
    return left.new_empty((left.shape[0], right.shape[1]), dtype=torch.bfloat16)


class DynamicW8A8MxFp8LinearMethod(QuantizeMethodBase):
    """MXFP8 linear method with dynamically quantized MXFP8 activations.

    Checkpoint weights enter as ordinary floating-point parameters and are
    finalized into E4M3 values with resident block scales. Calls preserve every
    leading activation dimension and apply an optional bias in bfloat16.
    """

    is_quantized = True

    def create_weights(
        self,
        module: nn.Module,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        """Create loadable floating-point parameters for an MXFP8 linear layer."""

        if int(input_size) % _MXFP8_BLOCK_SIZE:
            raise ValueError(f"MXFP8 linear input size must be divisible by {_MXFP8_BLOCK_SIZE}")

        # Parameters retain checkpoint shape until finalization quantizes the
        # weight and installs its accelerator-formatted scale buffer.
        module.register_parameter(
            "weight",
            nn.Parameter(torch.empty(int(output_size), int(input_size)), requires_grad=False),
        )
        module.register_parameter(
            "bias",
            nn.Parameter(torch.empty(int(output_size)), requires_grad=False) if bias else None,
        )
        module.register_buffer("weight_scale", None, persistent=False)

        # Standard tensor loaders populate the temporary weight and optional
        # bias before ``process_weights_after_loading`` finalizes execution state.
        from ...loader.weight_loaders import attach_weight_loader, default_weight_loader

        weight = cast(nn.Parameter, module.weight)
        attach_weight_loader(weight, default_weight_loader)
        bias_parameter = cast(nn.Parameter | None, module.bias)
        if bias_parameter is not None:
            attach_weight_loader(bias_parameter, default_weight_loader)

    @torch.no_grad()
    def process_weights_after_loading(self, module: nn.Module) -> None:
        """Finalize a loaded floating-point weight into resident MXFP8 state."""

        from ..linear import LinearBase

        linear = cast(LinearBase, module)

        # MXFP8 kernels and their swizzled scale layout are defined for
        # SM100-class CUDA execution.
        if linear.weight.device.type != "cuda":
            raise RuntimeError("MXFP8 linear execution requires a CUDA device")
        if torch.cuda.get_device_capability(linear.weight.device) < (10, 0):
            raise RuntimeError("MXFP8 linear execution requires an SM100-class CUDA device")
        if linear.weight.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise RuntimeError("MXFP8 linear weights must be loaded from a floating-point tensor")

        # Quantize once in bfloat16 so inference reuses both packed values and
        # their block scales without checkpoint-shaped floating-point storage.
        packed, block_scale = _flashinfer().mxfp8_quantize(
            linear.weight.to(torch.bfloat16),
            is_sf_swizzled_layout=True,
            backend="cuda",
        )
        linear.weight = nn.Parameter(packed.contiguous(), requires_grad=False)
        linear.weight_scale = block_scale

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply an MXFP8 linear projection while preserving leading dimensions."""

        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        if linear.weight.dtype != torch.float8_e4m3fn or linear.weight_scale is None:
            raise RuntimeError("MXFP8 linear execution requires finalized MXFP8 weights")
        if x.dtype != torch.bfloat16:
            raise RuntimeError("MXFP8 linear execution requires bfloat16 activations")

        # Matrix kernels consume rank-two operands. Collapse all token-like
        # leading dimensions and restore them after projection.
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])

        # Activations receive fresh per-block scales; the finalized weight
        # already owns scales in the matching swizzled representation.
        packed, block_scale = _mxfp8_quantize(x_2d)
        output = _mxfp8_mm_bf16(
            packed,
            linear.weight.T,
            block_scale,
            linear.weight_scale,
        )

        if linear.bias is not None:
            output = output + linear.bias.to(device=output.device, dtype=output.dtype)

        return output.reshape(*original_shape, linear.output_size)
