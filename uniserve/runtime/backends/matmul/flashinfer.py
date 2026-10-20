"""Native MXFP8 and NVFP4 GEMM with complete encoded operands."""

import torch

from uniserve.quantization import QuantizedTensor, ScaleLayout

from .base import Backend as _IndependentBackend
from .base import Operator as _Operator
from .grouped import Backend as _GroupedBackend


def _requires_grouped(weights):
    """Whether standalone block GEMMs reject a projection's dimensions."""
    return any(min(weight.shape) < 128 for weight in weights.values())


def _scale_matrix(scales, rows, columns):
    """View flat 128x4-swizzled block scales as their padded matrix.

    The swizzle keeps ``rows`` padded to a multiple of 128 and ``columns`` to
    a multiple of 4, so the flat storage is exactly that matrix row-major.
    """
    return scales.reshape(-(-rows // 128) * 128, -(-columns // 4) * 4)


class _FlashInferOperator(_Operator):
    def __call__(self, x, bias, *, out):
        import flashinfer

        x = self._input(x, out)
        if not isinstance(x, QuantizedTensor) or not isinstance(
            self.weight, QuantizedTensor
        ):
            raise ValueError("FlashInfer block GEMM requires encoded operands")
        format = self.weight.quantizer.format
        if x.quantizer.format != format or format not in {"mxfp8", "nvfp4"}:
            raise ValueError(
                "block GEMM operands must use the same MXFP8 or NVFP4 format"
            )
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

        # FlashInfer launches its CUTLASS NVFP4 GEMM with programmatic
        # dependent launch: the GEMM's thread blocks may start while the
        # kernel before it on the stream still runs, and its epilogue reads
        # alpha from device memory as each block starts, before waiting for
        # that kernel. Alpha is therefore computed before the operands are
        # repacked. CUTLASS serves only linear input scales, so their swizzle
        # kernels always run between the alpha write and the GEMM, which
        # cannot start before the alpha write has completed.
        alpha = (
            x.buffers()["tensor_scale"] * self.weight.buffers()["tensor_scale"]
            if format == "nvfp4"
            else None
        )
        left = x.repack(scale_layout=ScaleLayout.SWIZZLED_128X4).buffers()
        right = self.weight.repack(
            scale_layout=ScaleLayout.SWIZZLED_128X4
        ).buffers()
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
            # cuDNN runs natively swizzled activations and reads their flat
            # scales as the padded matrices they are produced as. Scales
            # repacked from the linear layout take CUTLASS, the one FP4
            # kernel FlashInfer builds for both SM100 and SM120; cuDNN's
            # products differ from it in the last bit of a few elements.
            rows, columns = left["values"].shape
            features = right["values"].shape[0]
            left_scale, right_scale = left["block_scale"], right["block_scale"]
            if x.scale_layout is ScaleLayout.SWIZZLED_128X4:
                backend = "cudnn"
                left_scale = _scale_matrix(left_scale, rows, columns // 8)
                right_scale = _scale_matrix(
                    right_scale, features, columns // 8
                ).T
            else:
                backend = "cutlass"
            flashinfer.mm_fp4(
                left["values"],
                right["values"].T,
                left_scale,
                right_scale,
                alpha,
                out=target,
                out_dtype=out.dtype,
                backend=backend,
                enable_pdl=False,
            )

        if bias is not None:
            target.add_(bias.to(out.dtype))
        return out if target is out else out.copy_(target)


class Backend(_GroupedBackend):
    """Prepare native block GEMMs with size-appropriate output ownership.

    Large independent branches write directly to their contiguous BF16
    destinations. FlashInfer rejects block GEMMs below 128 rows or columns, so
    those projections retain the grouped kernel that supports small matrices.
    """

    name = "flashinfer"
    operator_class = _FlashInferOperator

    def merged_workspace_buffers(
        self,
        weights,
        *,
        input_dtype,
        input_quantizer,
        max_rows,
        branch_width,
        output_dtype,
    ):
        if _requires_grouped(weights):
            return super().merged_workspace_buffers(
                weights,
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                branch_width=branch_width,
                output_dtype=output_dtype,
            )
        return _IndependentBackend.merged_workspace_buffers(
            self,
            weights,
            input_dtype=input_dtype,
            input_quantizer=input_quantizer,
            max_rows=max_rows,
            branch_width=branch_width,
            output_dtype=output_dtype,
        )

    def prepare_merged(
        self,
        weights,
        *,
        input_dtype,
        input_quantizer,
        max_rows,
        branch_width,
        output_dtype,
        workspace,
    ):
        if _requires_grouped(weights):
            return super().prepare_merged(
                weights,
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                branch_width=branch_width,
                output_dtype=output_dtype,
                workspace=workspace,
            )
        return _IndependentBackend.prepare_merged(
            self,
            weights,
            input_dtype=input_dtype,
            input_quantizer=input_quantizer,
            max_rows=max_rows,
            branch_width=branch_width,
            output_dtype=output_dtype,
            workspace=workspace,
        )
