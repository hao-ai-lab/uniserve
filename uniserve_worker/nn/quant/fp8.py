"""FP8 linear quantization method with a dequantized correctness floor."""
from __future__ import annotations

from functools import partial
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...foundation.errors import compute_error
from .base import QuantizeMethodBase
from .kv_cache import fp8_quantize, fp8_scale_from
from .load_state import (
    Fp8LoadPhase,
    fp8_load_phase,
    init_fp8_phase,
    set_fp8_scale_loaded,
    set_optional_checkpoint,
    set_skip_serving_cast,
)

__all__ = [
    'DynamicW8A8Fp8LinearMethod',
    'W8A8Fp8LinearMethod',
]

# ``torch._scaled_mm`` requires the contraction (K) dim aligned to this; fixed by
# the fp8 kernel/ABI build.
_FP8_BLOCK_ALIGNMENT = 16


class DynamicW8A8Fp8LinearMethod(QuantizeMethodBase):
    """Runtime-quantized FP8 linear method with tokenwise or tensorwise scaling."""

    is_quantized = True

    def __init__(self, *, tensorwise: bool = False) -> None:
        self.tensorwise = bool(tensorwise)

    def create_weights(
        self,
        module: nn.Module,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        module.register_parameter(
            "weight",
            nn.Parameter(torch.empty(int(output_size), int(input_size)), requires_grad=False),
        )
        module.register_parameter(
            "bias",
            nn.Parameter(torch.empty(int(output_size)), requires_grad=False) if bias else None,
        )
        module.register_buffer("weight_scale", None, persistent=False)
        from ...loader.weight_loaders import attach_weight_loader, default_weight_loader

        attach_weight_loader(module.weight, default_weight_loader)
        if module.bias is not None:
            attach_weight_loader(module.bias, default_weight_loader)

    @torch.no_grad()
    def process_weights_after_loading(self, module: nn.Module) -> None:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        if linear.weight.dtype == torch.float8_e4m3fn:
            if linear.weight_scale is None:
                raise RuntimeError("dynamic FP8 checkpoint weight requires a weight scale")
            return
        dense = linear.weight.detach().to(torch.float32)
        scale = fp8_scale_from(dense, dim=None if self.tensorwise else 1)
        if self.tensorwise:
            scale = scale.reshape(1, 1)
        linear.weight = nn.Parameter(
            fp8_quantize(dense, scale).contiguous(),
            requires_grad=False,
        )
        linear.weight_scale = scale.to(device=linear.weight.device, dtype=torch.float32)

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        if linear.weight.dtype != torch.float8_e4m3fn:
            return F.linear(x, linear.weight, linear.bias)
        if linear.weight_scale is None:
            raise RuntimeError("FP8 linear weight requires a weight scale")
        return _apply_fp8_linear(
            x,
            linear.weight,
            linear.weight_scale,
            linear.bias,
            tensorwise=self.tensorwise,
        )

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

        if not self.tensorwise:
            raise RuntimeError("sequence-parallel FP8 execution requires tensorwise scaling")
        linear = cast(LinearBase, module)
        device_mesh = cast(DeviceMesh, mesh)
        if linear.weight.dtype != torch.float8_e4m3fn or linear.weight_scale is None:
            raise RuntimeError("sequence-parallel FP8 execution requires finalized FP8 weights")
        scale = fp8_scale_from(x.to(torch.float32), dim=None).reshape(1, 1)
        device_mesh.all_reduce_max(scale, group)
        quantized = fp8_quantize(x.to(torch.float32), scale)
        global_rows = int(x.shape[0]) * device_mesh.size(group)
        gathered = workspace.view(torch.float8_e4m3fn)[: global_rows * x.shape[1]].view(
            global_rows,
            x.shape[1],
        )
        device_mesh.all_gather_into_tensor(gathered, quantized, group)
        output = torch._scaled_mm(
            gathered,
            linear.weight.t(),
            scale_a=scale,
            scale_b=linear.weight_scale.reshape(1, 1),
            out_dtype=_scaled_mm_output_dtype(x.dtype),
        )
        if linear.bias is not None:
            output = output + linear.bias.to(device=output.device, dtype=output.dtype)
        return output


class W8A8Fp8LinearMethod(QuantizeMethodBase):
    """Per-channel FP8 weights and per-token dynamic FP8 activations.

    The CUDA fast path uses ``torch._scaled_mm`` when the runtime shape is
    supported.  All other cases use an explicit dequantized matmul, which keeps
    the method correct on CPU, unsupported shapes, and older CUDA stacks.
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
        module.register_parameter(
            "weight",
            nn.Parameter(torch.empty(int(output_size), int(input_size)), requires_grad=False),
        )
        module.register_parameter(
            "bias",
            nn.Parameter(torch.empty(int(output_size)), requires_grad=False) if bias else None,
        )
        module.register_parameter(
            "weight_scale",
            nn.Parameter(torch.ones(int(output_size), 1, dtype=torch.float32), requires_grad=False),
        )
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        set_optional_checkpoint(linear.weight_scale, True)
        set_skip_serving_cast(linear.weight_scale, True)
        init_fp8_phase(module)
        from ...loader.weight_loaders import (
            attach_weight_loader,
            default_weight_loader,
            fp8_scale_loader,
            fp8_weight_loader,
        )

        attach_weight_loader(linear.weight, partial(fp8_weight_loader, module=module))
        if linear.bias is not None:
            attach_weight_loader(linear.bias, default_weight_loader)
        attach_weight_loader(linear.weight_scale, partial(fp8_scale_loader, module=module))

    def process_weights_after_loading(self, module: nn.Module) -> None:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        weight = linear.weight
        if weight.dtype == torch.float8_e4m3fn:
            # An offline fp8 checkpoint weight can only be finalized once its
            # matching scale has been loaded; the load lifecycle must have
            # reached SCALE_LOADED before this finalize step runs.
            if fp8_load_phase(module) is not Fp8LoadPhase.SCALE_LOADED:
                raise compute_error(
                    "FP8 checkpoint weight requires a loaded weight_scale tensor",
                    phase="weight_finalize",
                )
            weight.data = weight.data.contiguous()
            linear.weight_scale.data = _canonical_scale(linear.weight_scale.data, weight.shape[0]).to(
                device=weight.device,
                dtype=torch.float32,
            )
            return

        dense = weight.detach().to(torch.float32)
        scale = fp8_scale_from(dense, dim=1)
        quantized = fp8_quantize(dense, scale)
        fp8_weight = nn.Parameter(quantized.contiguous(), requires_grad=False)
        from ...loader.weight_loaders import copy_parameter_loader_state

        copy_parameter_loader_state(weight, fp8_weight)
        set_skip_serving_cast(fp8_weight, True)
        linear.weight = fp8_weight
        if linear.weight_scale.is_meta:
            materialized_scale = nn.Parameter(
                torch.empty(
                    tuple(linear.weight_scale.shape),
                    device=fp8_weight.device,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            copy_parameter_loader_state(linear.weight_scale, materialized_scale)
            linear.weight_scale = materialized_scale
        linear.weight_scale.data = scale.to(device=fp8_weight.device, dtype=torch.float32)
        set_fp8_scale_loaded(module, True)

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        weight = linear.weight
        if weight.dtype != torch.float8_e4m3fn:
            return F.linear(x, weight, linear.bias)
        out = _apply_fp8_linear(x, weight, linear.weight_scale, linear.bias)
        return out


def _apply_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    tensorwise: bool = False,
) -> torch.Tensor:
    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    if _can_use_scaled_mm(x_2d, weight):
        out = _apply_scaled_mm(
            x_2d,
            weight,
            weight_scale,
            x.dtype,
            tensorwise=tensorwise,
        )
    else:
        out = _apply_dequantized(
            x_2d,
            weight,
            weight_scale,
            x.dtype,
            tensorwise=tensorwise,
        )
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out.reshape(*original_shape, weight.shape[0])


def _apply_scaled_mm(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    tensorwise: bool = False,
) -> torch.Tensor:
    x_float = x_2d.to(torch.float32)
    act_scale = fp8_scale_from(x_float, dim=None if tensorwise else 1)
    if tensorwise:
        act_scale = act_scale.reshape(1, 1)
    x_fp8 = fp8_quantize(x_float, act_scale)
    scale_b = (
        weight_scale.reshape(1, 1)
        if tensorwise
        else _canonical_scale(weight_scale, weight.shape[0]).t().contiguous()
    )
    return torch._scaled_mm(
        x_fp8,
        weight.t(),
        scale_a=act_scale,
        scale_b=scale_b,
        out_dtype=_scaled_mm_output_dtype(out_dtype),
    )


def _apply_dequantized(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    tensorwise: bool = False,
) -> torch.Tensor:
    scale = (
        weight_scale.reshape(1, 1).to(torch.float32)
        if tensorwise
        else _canonical_scale(weight_scale, weight.shape[0])
    )
    dequant_weight = weight.to(torch.float32) * scale
    out = F.linear(x_2d.to(torch.float32), dequant_weight)
    if out_dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return out.to(dtype=out_dtype)
    return out


def _can_use_scaled_mm(x_2d: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x_2d.is_cuda
        and weight.is_cuda
        and hasattr(torch, "_scaled_mm")
        and x_2d.dtype in {torch.float16, torch.bfloat16}
        and x_2d.ndim == 2
        and weight.ndim == 2
        and int(x_2d.shape[1]) == int(weight.shape[1])
        and int(x_2d.shape[1]) % _FP8_BLOCK_ALIGNMENT == 0
    )


def _scaled_mm_output_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in {torch.float16, torch.bfloat16}:
        return dtype
    return torch.bfloat16


def _canonical_scale(scale: torch.Tensor, output_size: int) -> torch.Tensor:
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    if scale.shape == (1, int(output_size)):
        scale = scale.t()
    if scale.shape != (int(output_size), 1):
        raise ValueError(f"FP8 weight_scale shape {tuple(scale.shape)} != ({int(output_size)}, 1)")
    return scale.to(torch.float32)
