"""Implements FP8 linear weight loading, activation scaling, and execution.

Checkpoint and dynamically quantized weights converge on canonical E4M3 storage
with explicit dequantization scales. Eligible CUDA operands use scaled matrix
multiplication; other operands reconstruct weights in FP32 under the same linear
projection contract.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...backends.triton import triton_available
from ...foundation.errors import compute_error
from .base import PreparedLinearInput, QuantizeMethodBase
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
    "DynamicW8A8Fp8LinearMethod",
    "W8A8Fp8LinearMethod",
    "quantize_fp8_rowwise",
]

# ``torch._scaled_mm`` requires the contraction (K) dim aligned to this; fixed by
# the fp8 kernel/ABI build.
_FP8_BLOCK_ALIGNMENT = 16

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    _FP8_MAX_TL = tl.constexpr(448.0)
    _FP8_SCALE_EPS_TL = tl.constexpr(1.0e-12)

    @triton.jit
    def _divide_rn(dividend, divisor):
        """Divide FP32 operands with explicit nearest-even PTX semantics."""

        return tl.inline_asm_elementwise(
            asm="div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[dividend, divisor],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _rowwise_fp8_quant_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        input_row_stride,
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Quantize one BF16 activation row and publish its E4M3 scale."""

        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < width
        values = tl.load(
            input_ptr + row * input_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        maximum = tl.maximum(tl.max(tl.abs(values), axis=0), _FP8_SCALE_EPS_TL)
        scale = maximum / _FP8_MAX_TL
        quantized = tl.maximum(
            tl.minimum(_divide_rn(values, scale), _FP8_MAX_TL),
            -_FP8_MAX_TL,
        )
        tl.store(output_ptr + row * width + columns, quantized, mask=mask)
        tl.store(scale_ptr + row, scale)


class DynamicW8A8Fp8LinearMethod(QuantizeMethodBase):
    """Runtime-quantized FP8 linear method with tokenwise or tensorwise scaling."""

    is_quantized = True

    def __init__(self, *, tensorwise: bool = False) -> None:
        """Select one scale per tensor or independent scales per row and channel."""

        self.tensorwise = bool(tensorwise)

    @property
    def weight_scale_domain(self):
        return "tensor" if self.tensorwise else "row"

    @property
    def input_scale_domain(self):
        return self.weight_scale_domain

    def create_weights(
        self,
        module: nn.Module,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        """Register dense checkpoint parameters and transient FP8 scale storage."""

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

        attach_weight_loader(cast(nn.Parameter, module.weight), default_weight_loader)
        if module.bias is not None:
            attach_weight_loader(cast(nn.Parameter, module.bias), default_weight_loader)

    @torch.no_grad()
    def process_weights_after_loading(self, module: nn.Module) -> None:
        """Finalize loaded weights as E4M3 values with serving-time scales."""

        from ..linear import LinearBase

        linear = cast(LinearBase, module)

        # Prequantized checkpoints must provide the scale that reconstructs weight values.
        if linear.weight.dtype == torch.float8_e4m3fn:
            if linear.weight_scale is None:
                raise RuntimeError("dynamic FP8 checkpoint weight requires a weight scale")
            return

        # Dense checkpoints are quantized once after all loader shards are installed.
        dense = linear.weight.detach().to(torch.float32)
        maximum = linear.logical_weight_absmax
        if maximum is None:
            raise RuntimeError("FP8 weights require resolved logical scale domains")
        scale = maximum.clamp_min(1.0e-12) / 448.0
        quantization_scale = scale
        if self.tensorwise and len(linear.weight_output_partitions) > 1:
            quantization_scale = torch.cat(
                [
                    scale[index].expand(rows, 1)
                    for index, rows in enumerate(linear.weight_output_partitions)
                ]
            )
        linear.weight = nn.Parameter(
            fp8_quantize(dense, quantization_scale).contiguous(),
            requires_grad=False,
        )
        linear.weight_scale = scale.to(device=linear.weight.device, dtype=torch.float32)

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply dense or dynamically activation-quantized linear projection."""

        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        if linear.weight.dtype != torch.float8_e4m3fn:
            return F.linear(x, linear.weight, linear.execution_bias)
        if linear.weight_scale is None:
            raise RuntimeError("FP8 linear weight requires a weight scale")
        if self.tensorwise and len(linear.weight_output_partitions) > 1:
            shape = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            prepared = self.prepare_input(flat, self.input_scale(flat))
            return self.apply_prepared(linear, prepared, output_dtype=x.dtype).reshape(
                *shape, linear.output_size
            )
        return _apply_fp8_linear(
            x,
            linear.weight,
            linear.weight_scale,
            linear.execution_bias,
            tensorwise=self.tensorwise,
        )

    def apply_prequantized(
        self,
        module: nn.Module,
        x: torch.Tensor,
        scale: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Multiply E4M3 activations with finalized E4M3 weights and explicit scales."""

        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        if linear.weight.dtype != torch.float8_e4m3fn or linear.weight_scale is None:
            raise RuntimeError("prequantized FP8 execution requires finalized FP8 weights")
        if x.dtype != torch.float8_e4m3fn:
            raise RuntimeError("prequantized FP8 execution requires float8_e4m3fn activations")

        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        expected_scale_shape = (1, 1) if self.tensorwise else (x_2d.shape[0], 1)
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(
                f"prequantized FP8 scale shape {tuple(scale.shape)} != {expected_scale_shape}"
            )

        # ``_scaled_mm`` expects B scales in contraction-output orientation.
        scale_b = (
            linear.weight_scale.reshape(1, 1)
            if self.tensorwise
            else _canonical_scale(linear.weight_scale, linear.weight.shape[0]).t().contiguous()
        )
        output = torch._scaled_mm(
            x_2d,
            linear.weight.t(),
            scale_a=scale,
            scale_b=scale_b,
            out_dtype=_scaled_mm_output_dtype(output_dtype),
        )
        if linear.execution_bias is not None:
            output = output + linear.execution_bias.to(device=output.device, dtype=output.dtype)
        return output.reshape(*original_shape, linear.weight.shape[0])

    def input_scale(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_scale_from(x.to(torch.float32), dim=None if self.tensorwise else 1).reshape(
            -1, 1
        )

    def prepare_input(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> PreparedLinearInput:
        if scale is None:
            raise ValueError("FP8 input preparation requires a shared activation scale")
        return PreparedLinearInput(fp8_quantize(x.to(torch.float32), scale), global_scale=scale)

    def apply_prepared(
        self,
        module: nn.Module,
        prepared: PreparedLinearInput,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        if linear.weight.dtype != torch.float8_e4m3fn or linear.weight_scale is None:
            raise RuntimeError("prepared FP8 GEMM requires finalized FP8 weights")
        if prepared.global_scale is None:
            raise ValueError("prepared FP8 input requires activation scales")
        outputs = []
        offset = 0
        for index, rows in enumerate(linear.weight_output_partitions):
            weight = linear.weight[offset : offset + rows]
            weight_scale = (
                linear.weight_scale[index].reshape(1, 1)
                if self.tensorwise
                else linear.weight_scale[offset : offset + rows].t().contiguous()
            )
            if prepared.values.is_cuda:
                output = torch._scaled_mm(
                    prepared.values,
                    weight.t(),
                    scale_a=prepared.global_scale,
                    scale_b=weight_scale,
                    out_dtype=_scaled_mm_output_dtype(output_dtype),
                )
            else:
                output = (
                    (prepared.values.float() * prepared.global_scale)
                    @ (weight.float() * weight_scale.t()).t()
                ).to(output_dtype)
            outputs.append(output)
            offset += rows
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if linear.execution_bias is not None:
            output = output + linear.execution_bias.to(device=output.device, dtype=output.dtype)
        return output


class W8A8Fp8LinearMethod(DynamicW8A8Fp8LinearMethod):
    """Per-channel FP8 weights and per-token dynamic FP8 activations.

    The CUDA fast path uses ``torch._scaled_mm`` when the runtime shape is
    supported. All other cases use an explicit dequantized matmul, preserving the
    same numerical contract on CPU and for unsupported CUDA shapes or runtimes.
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
        """Register weights, optional bias, scale state, and checkpoint loaders."""

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
        attach_weight_loader(
            cast(nn.Parameter, linear.weight_scale), partial(fp8_scale_loader, module=module)
        )

    def process_weights_after_loading(self, module: nn.Module) -> None:
        """Validate prequantized weights or quantize a completed dense checkpoint."""

        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        weight = linear.weight
        weight_scale = cast(nn.Parameter, linear.weight_scale)

        # Checkpoint FP8 data is usable only after its independently loaded scale arrives.
        if weight.dtype == torch.float8_e4m3fn:
            if fp8_load_phase(module) is not Fp8LoadPhase.SCALE_LOADED:
                raise compute_error(
                    "FP8 checkpoint weight requires a loaded weight_scale tensor",
                    phase="weight_finalize",
                )
            weight.data = weight.data.contiguous()
            weight_scale.data = _canonical_scale(weight_scale.data, weight.shape[0]).to(
                device=weight.device,
                dtype=torch.float32,
            )
            return

        # Dense data is reduced per output channel, then replaced without losing loader metadata.
        dense = weight.detach().to(torch.float32)
        if linear.logical_weight_absmax is None:
            raise RuntimeError("FP8 weights require resolved logical scale domains")
        scale = linear.logical_weight_absmax.clamp_min(1.0e-12) / 448.0
        quantized = fp8_quantize(dense, scale)
        fp8_weight = nn.Parameter(quantized.contiguous(), requires_grad=False)
        from ...loader.weight_loaders import copy_parameter_loader_state

        copy_parameter_loader_state(weight, fp8_weight)
        set_skip_serving_cast(fp8_weight, True)
        linear.weight = fp8_weight
        if weight_scale.is_meta:
            materialized_scale = nn.Parameter(
                torch.empty(
                    tuple(weight_scale.shape),
                    device=fp8_weight.device,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            copy_parameter_loader_state(weight_scale, materialized_scale)
            linear.weight_scale = materialized_scale
            weight_scale = materialized_scale
        weight_scale.data = scale.to(device=fp8_weight.device, dtype=torch.float32)
        set_fp8_scale_loaded(module, True)

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply the finalized projection through its dense or FP8 execution path."""

        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        weight = linear.weight
        if weight.dtype != torch.float8_e4m3fn:
            return F.linear(x, weight, linear.execution_bias)
        if linear.weight_scale is None:
            raise RuntimeError("FP8 linear weight requires a weight scale")
        out = _apply_fp8_linear(x, weight, linear.weight_scale, linear.execution_bias)
        return out


def _apply_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    tensorwise: bool = False,
) -> torch.Tensor:
    """Dispatch FP8 projection by runtime capability and restore leading dimensions."""

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
    """Quantize activations dynamically and execute hardware-scaled matrix multiplication."""

    if tensorwise:
        x_float = x_2d.to(torch.float32)
        act_scale = fp8_scale_from(x_float, dim=None)
        act_scale = act_scale.reshape(1, 1)
        x_fp8 = fp8_quantize(x_float, act_scale)
    else:
        x_fp8, act_scale = quantize_fp8_rowwise(x_2d)

    # Weight scales are transposed from output-channel storage into GEMM B-scale layout.
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


def quantize_fp8_rowwise(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous CUDA rows in one kernel, with an eager portable fallback."""

    width = int(x_2d.shape[1])
    if (
        triton is not None
        and x_2d.is_cuda
        and x_2d.dtype in {torch.float16, torch.bfloat16}
        and x_2d.ndim == 2
        and x_2d.stride(1) == 1
        and width <= 32_768
        and triton_available(x_2d.device)
    ):
        rows = int(x_2d.shape[0])
        output = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
        scale = torch.empty((rows, 1), dtype=torch.float32, device=x_2d.device)
        block = triton.next_power_of_2(width)
        _rowwise_fp8_quant_kernel[(rows,)](
            x_2d,
            output,
            scale,
            x_2d.stride(0),
            width=width,
            BLOCK=block,
            num_warps=8 if width <= 5_120 else 16,
        )
        return output, scale

    x_float = x_2d.to(torch.float32)
    scale = fp8_scale_from(x_float, dim=1)
    return fp8_quantize(x_float, scale), scale


def _apply_dequantized(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    tensorwise: bool = False,
) -> torch.Tensor:
    """Execute the FP8 weight contract through explicit FP32 dequantization."""

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
    """Return whether operands satisfy the CUDA scaled-matmul ABI."""

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
    """Select a scaled-matmul accumulator output supported by PyTorch."""

    if dtype in {torch.float16, torch.bfloat16}:
        return dtype
    return torch.bfloat16


def _canonical_scale(scale: torch.Tensor, output_size: int) -> torch.Tensor:
    """Normalize weight scales to one FP32 column value per output channel."""

    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    if scale.shape == (1, int(output_size)):
        scale = scale.t()
    if scale.shape != (int(output_size), 1):
        raise ValueError(f"FP8 weight_scale shape {tuple(scale.shape)} != ({int(output_size)}, 1)")
    return scale.to(torch.float32)
