"""Dynamic NVIDIA FP4 linear execution for Blackwell GPUs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn
import triton
import triton.language as tl

from uniserve.nn.quant.base import BlockScaleLayout, LinearMethod, PreparedLinearInput

if TYPE_CHECKING:
    from uniserve.nn.linear import LinearBase


__all__ = [
    "DynamicW4A4NvFp4LinearMethod",
]

_NVFP4_MAX = float(torch.finfo(torch.float8_e4m3fn).max) * 6.0
_SCALE_EPS = 1.0e-12
_FUSED_ABSMAX_MIN_ELEMENTS = 1 << 25
_FUSED_ABSMAX_BLOCK = 1 << 16


def _flashinfer() -> Any:
    """Import and return FlashInfer activation-quantization operators."""

    import flashinfer

    return flashinfer


@triton.jit
def _absmax_partial_kernel(
    source,
    partials,
    elements: tl.constexpr,
    block: tl.constexpr,
):
    """Reduce one source block into a partial absolute maximum."""

    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(source + offsets, mask=offsets < elements, other=0.0)
    tl.store(partials + tl.program_id(0), tl.max(tl.abs(values), axis=0))


@triton.jit
def _absmax_finish_kernel(
    partials,
    output,
    count: tl.constexpr,
    block: tl.constexpr,
):
    """Reduce partial maxima into one global absolute maximum."""

    offsets = tl.arange(0, block)
    values = tl.load(partials + offsets, mask=offsets < count, other=-float("inf"))
    tl.store(output, tl.max(values, axis=0))


@torch.library.custom_op("uniserve::nvfp4_absmax", mutates_args=())
def _nvfp4_absmax(value: torch.Tensor) -> torch.Tensor:
    """Compute one finite absolute maximum per NVFP4 quantization group."""

    if value.device.type != "cuda" or value.dtype != torch.bfloat16:
        raise RuntimeError("fused NVFP4 abs-max requires a CUDA bfloat16 tensor")
    if not value.is_contiguous():
        raise RuntimeError("fused NVFP4 abs-max requires contiguous storage")
    elements = int(value.numel())
    partial_count = triton.cdiv(elements, _FUSED_ABSMAX_BLOCK)
    partials = torch.empty((partial_count,), dtype=torch.float32, device=value.device)
    output = torch.empty((), dtype=value.dtype, device=value.device)
    _absmax_partial_kernel[(partial_count,)](
        value,
        partials,
        elements=elements,
        block=_FUSED_ABSMAX_BLOCK,
        num_warps=8,
    )
    finish_block = triton.next_power_of_2(partial_count)
    _absmax_finish_kernel[(1,)](
        partials,
        output,
        count=partial_count,
        block=finish_block,
        num_warps=8,
    )
    return output


@_nvfp4_absmax.register_fake
def _nvfp4_absmax_fake(value: torch.Tensor) -> torch.Tensor:
    """Infer the scalar absolute-maximum output for custom-op tracing."""

    return value.new_empty(())


def _scale_2_from_absmax(maximum: torch.Tensor) -> torch.Tensor:
    """Convert an absolute maximum into the second-level NVFP4 scale."""

    return maximum.float().clamp_min(_SCALE_EPS) / _NVFP4_MAX


def _global_scale_2(value: torch.Tensor) -> torch.Tensor:
    """Derive a finite distributed activation scale shared across tensor-parallel ranks."""

    if (
        value.device.type == "cuda"
        and value.dtype == torch.bfloat16
        and value.is_contiguous()
        and value.numel() >= _FUSED_ABSMAX_MIN_ELEMENTS
    ):
        maximum = _nvfp4_absmax(value)
    else:
        maximum = value.abs().amax()
    return _scale_2_from_absmax(maximum)


@torch.library.custom_op("uniserve::nvfp4_quantize_128x4", mutates_args=())
def _nvfp4_quantize_128x4(
    value: torch.Tensor,
    inverse_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations into packed 128x4 NVFP4 blocks and hierarchical scales."""

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
    """Infer packed NVFP4 value and scale tensor shapes for custom-op tracing."""

    del inverse_global_scale
    rows, width = value.shape
    scale_rows = ((rows + 127) // 128) * 128
    scale_columns = (((width // 16) + 3) // 4) * 4
    return (
        value.new_empty((rows, width // 2), dtype=torch.uint8),
        value.new_empty((scale_rows, scale_columns), dtype=torch.uint8),
    )


@torch.library.custom_op("uniserve::nvfp4_quantize_linear", mutates_args=())
def _nvfp4_quantize_linear(
    value: torch.Tensor,
    inverse_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a flattened activation matrix into NVFP4 values and block scales."""

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
    """Infer linear-layout NVFP4 value and scale tensors for custom-op tracing."""

    del inverse_global_scale
    rows, width = value.shape
    return (
        value.new_empty((rows, width // 2), dtype=torch.uint8),
        value.new_empty((rows, width // 16), dtype=torch.uint8),
    )


@torch.library.custom_op("uniserve::nvfp4_interleave_scale", mutates_args=())
def _nvfp4_interleave_scale(linear_scale: torch.Tensor) -> torch.Tensor:
    """Reorder linear block scales into the GEMM kernel's interleaved layout."""

    return _flashinfer().block_scale_interleave(linear_scale)


@_nvfp4_interleave_scale.register_fake
def _nvfp4_interleave_scale_fake(linear_scale: torch.Tensor) -> torch.Tensor:
    """Infer the interleaved scale tensor shape for custom-op tracing."""

    rows, columns = linear_scale.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_columns = ((columns + 3) // 4) * 4
    return linear_scale.new_empty((padded_rows * padded_columns,))


@torch.library.custom_op("uniserve::nvfp4_mm_bf16", mutates_args=())
def _nvfp4_mm_bf16(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Run scaled NVFP4 matrix multiplication and return BF16 output."""

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
    """Infer BF16 matrix-product geometry for the NVFP4 custom op."""

    del left_scale, right_scale, alpha
    return left.new_empty((left.shape[0], right.shape[1]), dtype=torch.bfloat16)


@torch.library.custom_op("uniserve::nvfp4_mm_bf16_cute", mutates_args=())
def _nvfp4_mm_bf16_cute(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Dispatch scaled NVFP4 matrix multiplication through the CuTe implementation."""

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
    """Infer BF16 matrix-product geometry for the CuTe NVFP4 custom op."""

    del left_scale, right_scale, alpha
    return left.new_empty((left.shape[0], right.shape[1]), dtype=torch.bfloat16)


class DynamicW4A4NvFp4LinearMethod(LinearMethod):
    """Load-time FP4 weights with dynamically quantized FP4 activations."""

    is_quantized = True
    preferred_block_scale_layout: ClassVar[BlockScaleLayout] = "128x4"

    @property
    def weight_scale_domain(self):
        return "tensor"

    @property
    def input_scale_domain(self):
        return "tensor"

    def create_weights(
        self,
        module: LinearBase,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        """Register BF16 staging weights whose dimensions satisfy NVFP4 packing alignment."""

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
        from uniserve.loading.weight_loaders import attach_weight_loader, default_weight_loader

        weight = module.weight
        attach_weight_loader(weight, default_weight_loader)
        bias_parameter = module.bias
        if bias_parameter is not None:
            attach_weight_loader(bias_parameter, default_weight_loader)

    @torch.no_grad()
    def process_weights_after_loading(self, module: LinearBase) -> None:
        """Quantize loaded BF16 weights into SM100 NVFP4 values and block scales."""

        linear = module
        if linear.weight.device.type != "cuda":
            raise RuntimeError("NVFP4 linear execution requires a CUDA device")
        if torch.cuda.get_device_capability(linear.weight.device) < (10, 0):
            raise RuntimeError("NVFP4 linear execution requires an SM100-class CUDA device")
        if linear.weight.dtype == torch.uint8:
            return
        if linear.logical_weight_absmax is None:
            raise RuntimeError("NVFP4 weights require resolved logical scale domains")
        weight_scale_2 = _scale_2_from_absmax(linear.logical_weight_absmax)
        flashinfer = _flashinfer()
        packed_parts, scale_parts = [], []
        offset = 0
        for index, rows in enumerate(linear.weight_output_partitions):
            if len(linear.weight_output_partitions) > 1 and rows % 128:
                raise ValueError("NVFP4 logical output partitions must align to 128 rows")
            packed, block_scale = flashinfer.nvfp4_quantize(
                linear.weight[offset : offset + rows],
                1.0 / weight_scale_2[index].reshape(()),
                sfLayout=flashinfer.SfLayout.layout_128x4,
                backend="cute-dsl",
            )
            packed_parts.append(packed)
            scale_parts.append(block_scale)
            offset += rows
        linear.weight = nn.Parameter(
            packed_parts[0] if len(packed_parts) == 1 else torch.cat(packed_parts),
            requires_grad=False,
        )
        linear.weight_scale = scale_parts[0] if len(scale_parts) == 1 else torch.cat(scale_parts)
        linear.weight_scale_2 = weight_scale_2

    def apply(self, module: LinearBase, x: torch.Tensor) -> torch.Tensor:
        """Dynamically quantize BF16 input using the swizzled NVFP4 contract."""

        value = x.to(torch.bfloat16)
        flat = value.reshape(-1, value.shape[-1])
        prepared = self.prepare_input(
            flat, self.input_scale(flat), block_scale_layout=self.preferred_block_scale_layout
        )
        output = self.apply_prepared(module, prepared, output_dtype=torch.bfloat16)
        return output.reshape(*x.shape[:-1], module.output_size)

    def input_scale(self, x: torch.Tensor, *, absmax: torch.Tensor | None = None) -> torch.Tensor:
        return (
            _global_scale_2(x.to(torch.bfloat16))
            if absmax is None
            else _scale_2_from_absmax(absmax)
        )

    def prepare_input(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None,
        *,
        block_scale_layout: BlockScaleLayout = "linear",
    ) -> PreparedLinearInput:
        if scale is None:
            raise ValueError("NVFP4 input preparation requires a shared activation scale")
        quantize = (
            _nvfp4_quantize_128x4 if block_scale_layout == "128x4" else _nvfp4_quantize_linear
        )
        values, block_scales = quantize(x.to(torch.bfloat16), 1.0 / scale)
        return PreparedLinearInput(
            values, block_scales, tensor_scale=scale, block_scale_layout=block_scale_layout
        )

    def apply_prepared(
        self,
        module: LinearBase,
        prepared: PreparedLinearInput,
        *,
        output_dtype: torch.dtype,
        include_bias: bool = True,
    ) -> torch.Tensor:
        if output_dtype != torch.bfloat16:
            raise ValueError("NVFP4 GEMM supports BF16 output")
        linear = module
        weight_scale = linear.weight_scale
        weight_scale_2 = linear.weight_scale_2
        if linear.weight.dtype != torch.uint8 or weight_scale is None or weight_scale_2 is None:
            raise RuntimeError("prepared NVFP4 GEMM requires finalized FP4 weights")
        if prepared.block_scales is None or prepared.tensor_scale is None:
            raise ValueError("prepared NVFP4 input requires block scales and a global scale")
        values = prepared.values
        if values.dtype != torch.uint8 or values.shape[-1] * 2 != linear.input_size:
            raise ValueError("NVFP4 input requires packed bytes with the linear input width")
        flat = values.reshape(-1, values.shape[-1])
        if prepared.block_scale_layout == "128x4":
            scales = prepared.block_scales
            gemm = _nvfp4_mm_bf16
        else:
            scales = _nvfp4_interleave_scale(prepared.block_scales)
            gemm = _nvfp4_mm_bf16_cute
        outputs = []
        offset = 0
        for index, rows in enumerate(linear.weight_output_partitions):
            outputs.append(
                gemm(
                    flat,
                    linear.weight[offset : offset + rows].T,
                    scales,
                    weight_scale[offset : offset + ((rows + 127) // 128) * 128].T,
                    prepared.tensor_scale * weight_scale_2[index].reshape(()),
                )
            )
            offset += rows
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if include_bias and linear.execution_bias is not None:
            output = output + linear.execution_bias.to(device=output.device, dtype=output.dtype)
        return output.reshape(*values.shape[:-1], linear.output_size)
