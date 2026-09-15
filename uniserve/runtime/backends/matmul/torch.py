"""Dense and FP8 reference matrix multiplication through PyTorch."""

import torch
import torch.nn.functional as F

from uniserve.quantization import QuantizedTensor

from . import Backend as _Backend
from . import Operator as _Operator


class _TorchOperator(_Operator):
    def __call__(self, x, bias, *, out):
        x = self._input(x, out)
        weight = self.weight
        encoded = isinstance(x, QuantizedTensor) or isinstance(weight, QuantizedTensor)
        if encoded:
            if any(
                isinstance(value, QuantizedTensor) and value.quantizer.format != "fp8"
                for value in (x, weight)
            ):
                raise ValueError("torch matmul supports dense and FP8 operands")
            x = x.dequantize(dtype=torch.float32) if isinstance(x, QuantizedTensor) else x.float()
            weight = (
                weight.dequantize(dtype=torch.float32)
                if isinstance(weight, QuantizedTensor)
                else weight.float()
            )
            # The reference projection rounds its GEMM result before adding
            # bias, matching the encoded accelerator projection contract.
            out.copy_(F.linear(x, weight).to(out.dtype))
            if bias is not None:
                out.add_(bias.to(out.dtype))
        elif out.dtype == x.dtype:
            if bias is None:
                torch.mm(x, weight.T, out=out)
            else:
                torch.addmm(bias, x, weight.T, out=out)
        else:
            out.copy_(
                F.linear(x.float(), weight.float(), None if bias is None else bias.float()).to(
                    out.dtype
                )
            )
        return out


class Backend(_Backend):
    operator_class = _TorchOperator
