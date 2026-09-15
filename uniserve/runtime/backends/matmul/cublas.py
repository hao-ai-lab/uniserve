"""CUDA dense and E4M3 scaled matrix multiplication."""

import torch

from uniserve.quantization import QuantizedTensor

from . import Operator as _Operator
from .grouped import Backend as _Backend


class _CUBLASOperator(_Operator):
    def __call__(self, x, bias, *, out):
        x = self._input(x, out)
        weight = self.weight
        if not x.is_cuda:
            raise ValueError("cuBLAS requires CUDA operands")
        if x.shape[0] == 0:
            return out
        if isinstance(x, QuantizedTensor) or isinstance(weight, QuantizedTensor):
            if any(
                isinstance(value, QuantizedTensor) and value.quantizer.format != "fp8"
                for value in (x, weight)
            ):
                raise ValueError("cuBLAS encoded operands require the FP8 format")
            if not all(isinstance(value, QuantizedTensor) for value in (x, weight)):
                # One encoded operand does not authorize quantizing the other.
                # Its existing FP8 values and scales define an FP32 product.
                left = (
                    x.dequantize(dtype=torch.float32)
                    if isinstance(x, QuantizedTensor)
                    else x.float()
                )
                right = (
                    weight.dequantize(dtype=torch.float32)
                    if isinstance(weight, QuantizedTensor)
                    else weight.float()
                )
                out.copy_(torch.mm(left, right.T).to(out.dtype))
                if bias is not None:
                    out.add_(bias.to(out.dtype))
                return out
            left, right = x.buffers(), weight.buffers()
            scale_a = left["scale"].reshape(-1, 1)
            scale_b = right["scale"].reshape(-1, 1).T
            if scale_a.numel() != 1 or scale_b.numel() != 1:
                # cuBLAS requires both operands to use row/channel scale
                # vectors together. Broadcasting a scalar preserves its full
                # source domain; no operand is requantized here.
                scale_a = scale_a.expand(x.shape[0], 1)
                scale_b = scale_b.expand(1, weight.shape[0])
            scale_a, scale_b = scale_a.contiguous(), scale_b.contiguous()
            # The scaled cuBLAS entry point requires unit channel stride.
            # Dense GEMM supports more layouts, including transposed outputs.
            target = (
                out
                if out.stride(1) == 1
                else torch.empty(out.shape, dtype=out.dtype, device=out.device)
            )
            torch._scaled_mm(
                left["values"], right["values"].T, scale_a, scale_b, out_dtype=out.dtype, out=target
            )
            if target is not out:
                out.copy_(target)
            if bias is not None:
                out.add_(bias.to(out.dtype))
        elif out.dtype == x.dtype:
            if bias is None:
                torch.mm(x, weight.T, out=out)
            else:
                torch.addmm(bias, x, weight.T, out=out)
        else:
            torch.mm(x, weight.T, out_dtype=out.dtype, out=out)
            if bias is not None:
                out.add_(bias.to(out.dtype))
        return out


class Backend(_Backend):
    operator_class = _CUBLASOperator
