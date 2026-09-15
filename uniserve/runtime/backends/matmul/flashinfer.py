"""Native MXFP8 and NVFP4 GEMM with complete encoded operands."""

import torch

from uniserve.quantization import QuantizedTensor, ScaleLayout

from . import Operator as _Operator
from .grouped import Backend as _Backend


class _FlashInferOperator(_Operator):
    def __call__(self, x, bias, *, out):
        import flashinfer

        x = self._input(x, out)
        if not isinstance(x, QuantizedTensor) or not isinstance(self.weight, QuantizedTensor):
            raise ValueError("FlashInfer block GEMM requires encoded operands")
        format = self.weight.quantizer.format
        if x.quantizer.format != format or format not in {"mxfp8", "nvfp4"}:
            raise ValueError("block GEMM operands must use the same MXFP8 or NVFP4 format")
        if out.dtype != torch.bfloat16:
            raise ValueError("block GEMM requires BF16 output")

        # FlashInfer's CuTe wrapper reconstructs the output matrix from its
        # pointer and dimensions, ignoring tensor strides. Keep that native
        # layout requirement at the backend boundary for borrowed branch views.
        target = (
            out
            if out.is_contiguous()
            else torch.empty(out.shape, dtype=out.dtype, device=out.device)
        )

        left = x.repack(scale_layout=ScaleLayout.SWIZZLED_128X4).buffers()
        right = self.weight.repack(scale_layout=ScaleLayout.SWIZZLED_128X4).buffers()
        if format == "mxfp8":
            flashinfer.mm_mxfp8(
                left["values"],
                right["values"].T,
                left["scale"],
                right["scale"],
                out=target,
                out_dtype=out.dtype,
                backend="cudnn",
            )
        else:
            # cuDNN consumes accelerator-swizzled scales; the original linear
            # scale path uses the existing CuTe GEMM after the same repacking.
            backend = "cudnn" if x.scale_layout is ScaleLayout.SWIZZLED_128X4 else "cute-dsl"
            flashinfer.mm_fp4(
                left["values"],
                right["values"].T,
                left["block_scale"],
                right["block_scale"],
                left["tensor_scale"] * right["tensor_scale"],
                out=target,
                out_dtype=out.dtype,
                backend=backend,
                enable_pdl=False,
            )

        if bias is not None:
            target.add_(bias.to(out.dtype))
        return out if target is out else out.copy_(target)


class Backend(_Backend):
    operator_class = _FlashInferOperator
