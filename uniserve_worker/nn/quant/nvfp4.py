"""Dynamic NVIDIA FP4 linear execution for Blackwell GPUs."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from .base import QuantizeMethodBase

__all__ = [
    "DynamicW4A4NvFp4LinearMethod",
    "NvFp4Linear",
    "replace_nvfp4_linears",
]

_NVFP4_MAX = float(torch.finfo(torch.float8_e4m3fn).max) * 6.0
_SCALE_EPS = 1.0e-12


def _flashinfer() -> Any:
    import flashinfer

    return flashinfer


def _global_scale_2(value: torch.Tensor) -> torch.Tensor:
    return value.abs().amax().float().clamp_min(_SCALE_EPS) / _NVFP4_MAX


@torch.library.custom_op("uniserve_worker::nvfp4_quantize_128x4", mutates_args=())
def _nvfp4_quantize_128x4(
    value: torch.Tensor,
    inverse_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flashinfer = _flashinfer()
    return flashinfer.nvfp4_quantize(
        value,
        inverse_global_scale,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        backend="cuda",
    )


@_nvfp4_quantize_128x4.register_fake
def _nvfp4_quantize_128x4_fake(
    value: torch.Tensor,
    inverse_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del inverse_global_scale
    rows, width = value.shape
    scale_rows = ((rows + 127) // 128) * 128
    scale_columns = (((width // 16) + 3) // 4) * 4
    return (
        value.new_empty((rows, width // 2), dtype=torch.uint8),
        value.new_empty((scale_rows, scale_columns), dtype=torch.uint8),
    )


@torch.library.custom_op("uniserve_worker::nvfp4_quantize_linear", mutates_args=())
def _nvfp4_quantize_linear(
    value: torch.Tensor,
    inverse_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flashinfer = _flashinfer()
    return flashinfer.nvfp4_quantize(
        value,
        inverse_global_scale,
        sfLayout=flashinfer.SfLayout.layout_linear,
        backend="cute-dsl",
        enable_pdl=False,
    )


@_nvfp4_quantize_linear.register_fake
def _nvfp4_quantize_linear_fake(
    value: torch.Tensor,
    inverse_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del inverse_global_scale
    rows, width = value.shape
    return (
        value.new_empty((rows, width // 2), dtype=torch.uint8),
        value.new_empty((rows, width // 16), dtype=torch.uint8),
    )


@torch.library.custom_op("uniserve_worker::nvfp4_interleave_scale", mutates_args=())
def _nvfp4_interleave_scale(linear_scale: torch.Tensor) -> torch.Tensor:
    return _flashinfer().block_scale_interleave(linear_scale)


@_nvfp4_interleave_scale.register_fake
def _nvfp4_interleave_scale_fake(linear_scale: torch.Tensor) -> torch.Tensor:
    rows, columns = linear_scale.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_columns = ((columns + 3) // 4) * 4
    return linear_scale.new_empty((padded_rows * padded_columns,))


@torch.library.custom_op("uniserve_worker::nvfp4_mm_bf16", mutates_args=())
def _nvfp4_mm_bf16(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return _flashinfer().mm_fp4(
        left,
        right,
        left_scale,
        right_scale,
        alpha,
        torch.bfloat16,
        backend="cudnn",
    )


@_nvfp4_mm_bf16.register_fake
def _nvfp4_mm_bf16_fake(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    del left_scale, right_scale, alpha
    return left.new_empty((left.shape[0], right.shape[1]), dtype=torch.bfloat16)


@torch.library.custom_op("uniserve_worker::nvfp4_mm_bf16_cute", mutates_args=())
def _nvfp4_mm_bf16_cute(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return _flashinfer().mm_fp4(
        left,
        right,
        left_scale,
        right_scale,
        alpha,
        torch.bfloat16,
        backend="cute-dsl",
        enable_pdl=False,
    )


@_nvfp4_mm_bf16_cute.register_fake
def _nvfp4_mm_bf16_cute_fake(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    del left_scale, right_scale, alpha
    return left.new_empty((left.shape[0], right.shape[1]), dtype=torch.bfloat16)


class DynamicW4A4NvFp4LinearMethod(QuantizeMethodBase):
    """Load-time FP4 weights with dynamically quantized FP4 activations."""

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
        if int(input_size) % 16 or int(output_size) % 16:
            raise ValueError("NVFP4 linear dimensions must be divisible by 16")
        module.register_parameter(
            "weight",
            nn.Parameter(torch.empty(int(output_size), int(input_size)), requires_grad=False),
        )
        module.register_parameter(
            "bias",
            nn.Parameter(torch.empty(int(output_size)), requires_grad=False) if bias else None,
        )
        module.register_buffer("weight_scale", None, persistent=False)
        module.register_buffer("weight_scale_2", None, persistent=False)
        from ...loader.weight_loaders import attach_weight_loader, default_weight_loader

        weight = cast(nn.Parameter, module.weight)
        attach_weight_loader(weight, default_weight_loader)
        bias_parameter = cast(nn.Parameter | None, module.bias)
        if bias_parameter is not None:
            attach_weight_loader(bias_parameter, default_weight_loader)

    @torch.no_grad()
    def process_weights_after_loading(self, module: nn.Module) -> None:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        if linear.weight.device.type != "cuda":
            raise RuntimeError("NVFP4 linear execution requires a CUDA device")
        if torch.cuda.get_device_capability(linear.weight.device) < (10, 0):
            raise RuntimeError("NVFP4 linear execution requires an SM100-class CUDA device")
        weight_scale_2 = _global_scale_2(linear.weight)
        flashinfer = _flashinfer()
        packed, block_scale = flashinfer.nvfp4_quantize(
            linear.weight,
            1.0 / weight_scale_2,
            sfLayout=flashinfer.SfLayout.layout_128x4,
            backend="cute-dsl",
        )
        linear.weight = nn.Parameter(packed.contiguous(), requires_grad=False)
        linear.weight_scale = block_scale
        linear.weight_scale_2 = weight_scale_2

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        weight_scale = linear.weight_scale
        weight_scale_2 = getattr(linear, "weight_scale_2", None)
        if linear.weight.dtype != torch.uint8 or weight_scale is None or weight_scale_2 is None:
            raise RuntimeError("NVFP4 linear execution requires finalized FP4 weights")
        if x.dtype != torch.bfloat16:
            raise RuntimeError("NVFP4 linear execution requires bfloat16 activations")
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        input_scale_2 = _global_scale_2(x_2d)
        packed, block_scale = _nvfp4_quantize_128x4(
            x_2d,
            1.0 / input_scale_2,
        )
        output = _nvfp4_mm_bf16(
            packed,
            linear.weight.T,
            block_scale,
            weight_scale.T,
            input_scale_2 * weight_scale_2,
        )
        if linear.bias is not None:
            output = output + linear.bias.to(device=output.device, dtype=output.dtype)
        return output.reshape(*original_shape, linear.output_size)

    def apply_sequence_parallel(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mesh: object,
        workspace: torch.Tensor,
        *,
        group: str,
    ) -> torch.Tensor:
        from ..linear import LinearBase
        from ..mesh import DeviceMesh

        linear = cast(LinearBase, module)
        device_mesh = cast(DeviceMesh, mesh)
        weight_scale = linear.weight_scale
        weight_scale_2 = getattr(linear, "weight_scale_2", None)
        if linear.weight.dtype != torch.uint8 or weight_scale is None or weight_scale_2 is None:
            raise RuntimeError("sequence-parallel NVFP4 execution requires finalized FP4 weights")
        if x.dtype != torch.bfloat16:
            raise RuntimeError("sequence-parallel NVFP4 execution requires bfloat16 activations")

        input_scale_2 = _global_scale_2(x)
        device_mesh.all_reduce_max(input_scale_2, group)
        local_packed, local_scale = _nvfp4_quantize_linear(
            x,
            1.0 / input_scale_2,
        )

        global_rows = int(x.shape[0]) * device_mesh.size(group)
        packed_elements = global_rows * int(x.shape[1]) // 2
        scale_elements = global_rows * int(x.shape[1]) // 16
        required_elements = packed_elements + scale_elements
        if int(workspace.numel()) < required_elements:
            raise RuntimeError(
                f"NVFP4 sequence-parallel workspace requires {required_elements} bytes, "
                f"got {workspace.numel()}"
            )
        gathered_packed = workspace[:packed_elements].view(
            global_rows,
            int(x.shape[1]) // 2,
        )
        gathered_linear_scale = workspace[
            packed_elements : packed_elements + scale_elements
        ].view(global_rows, int(x.shape[1]) // 16)
        device_mesh.all_gather_into_tensor(gathered_packed, local_packed, group)
        device_mesh.all_gather_into_tensor(gathered_linear_scale, local_scale, group)
        gathered_scale = _nvfp4_interleave_scale(gathered_linear_scale)
        output = _nvfp4_mm_bf16_cute(
            gathered_packed,
            linear.weight.T,
            gathered_scale,
            weight_scale.T,
            input_scale_2 * weight_scale_2,
        )
        if linear.bias is not None:
            output = output + linear.bias.to(device=output.device, dtype=output.dtype)
        return output


class NvFp4Linear(nn.Module):
    """A load-finalized NVFP4 replacement for an ordinary dense linear."""

    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        self.input_size = int(source.in_features)
        self.output_size = int(source.out_features)
        self.weight = nn.Parameter(
            source.weight.detach().to(torch.bfloat16),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(source.bias.detach().to(torch.bfloat16), requires_grad=False)
            if source.bias is not None
            else None
        )
        self.register_buffer("weight_scale", None, persistent=False)
        self.register_buffer("weight_scale_2", None, persistent=False)
        self.quant_method = DynamicW4A4NvFp4LinearMethod()
        self.quant_method.process_weights_after_loading(self)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, value.to(torch.bfloat16))


def replace_nvfp4_linears(module: nn.Module) -> int:
    """Replace every block-aligned dense child with a finalized NVFP4 linear."""

    count = 0
    for name, child in tuple(module.named_children()):
        if (
            isinstance(child, nn.Linear)
            and child.in_features % 16 == 0
            and child.out_features % 16 == 0
        ):
            setattr(module, name, NvFp4Linear(child))
            count += 1
        else:
            count += replace_nvfp4_linears(child)
    return count
