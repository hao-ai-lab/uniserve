"""Immutable conversion rules and logical quantization statistics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Literal

import torch
from torch.distributed.tensor import Partial, Replicate, Shard

from .tensor import (
    QuantizedTensor,
    RowOrder,
    ScaleLayout,
    _FP8Tensor,
    _MXFP8Tensor,
    _NVFP4Tensor,
)

if TYPE_CHECKING:
    from uniserve.distributed import Distribution


@dataclass(frozen=True, slots=True)
class Quantizer:
    """Convert a complete logical statistical domain to one supported encoding.

    FP8 axis zero retains the first dimension when computing extrema. MXFP8
    uses complete K32 blocks; NVFP4 uses K16 blocks and a tensor-wide scale.
    An explicit ``amax`` must cover the same logical domain as ``distribution``.
    ``calibrated_scale`` freezes NVFP4's tensor-wide scale at a checkpoint's
    calibrated value, ModelOpt's ``input_scale = amax / (6 * 448)``; only the
    per-block K16 encoding is then computed from the input.
    ``from_tensors`` borrows existing encoding without recomputing its scales.
    """

    format: Literal["fp8", "mxfp8", "nvfp4"]
    axis: Literal[0] | None = field(default=None, kw_only=True)
    calibrated_scale: float | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.format not in {"fp8", "mxfp8", "nvfp4"}:
            raise ValueError("quantization format must be fp8, mxfp8 or nvfp4")
        if self.axis is not None and (
            type(self.axis) is not int or self.axis != 0 or self.format != "fp8"
        ):
            raise ValueError(
                "only FP8 supports the retained statistical axis zero"
            )
        if self.calibrated_scale is not None and (
            self.format != "nvfp4"
            or self.axis is not None
            or not isinstance(self.calibrated_scale, (int, float))
            or isinstance(self.calibrated_scale, bool)
            or not math.isfinite(self.calibrated_scale)
            or self.calibrated_scale <= 0
        ):
            raise ValueError(
                "calibrated scale requires one positive finite NVFP4 scalar"
            )

    @property
    def requires_complete_source(self) -> bool:
        """Return whether runtime statistics span the complete input tensor.

        Calibrated NVFP4 owns a frozen checkpoint scale, so independently
        encoded row intervals retain one common statistical domain without
        first materializing every row together.
        """
        return self.calibrated_scale is None and (
            self.format == "nvfp4"
            or (self.format == "fp8" and self.axis is None)
        )

    def _shape(self, shape, dtype):
        """Reject logical shapes and dtypes this encoding cannot represent."""
        if not isinstance(shape, tuple) or any(
            type(size) is not int or size < 0 for size in shape
        ):
            raise ValueError(
                "quantized shape must contain nonnegative integer extents"
            )
        if dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise ValueError(
                "quantized tensors require a logical floating-point dtype"
            )
        if self.axis == 0 and not shape:
            raise ValueError("axis zero requires a non-scalar tensor")
        if self.format != "fp8":
            block = 32 if self.format == "mxfp8" else 16
            if len(shape) < 2 or shape[-1] % block:
                raise ValueError(
                    f"{self.format} requires complete aligned K{block} blocks"
                )

    def _statistics_shape(self, shape):
        """Return the FP32 amax shape for a logical shape.

        Under this quantizer.
        """
        if self.format == "mxfp8":
            return (*shape[:-1], shape[-1] // 32, 1)
        if self.axis == 0:
            return (shape[0], *((1,) * (len(shape) - 1)))
        return ()

    def _distribution(
        self, shape: tuple[int, ...], distribution: Distribution | None
    ) -> None:
        """Reject placements incompatible with logical-domain statistics."""
        if distribution is None:
            return
        for placement in distribution.placements:
            if isinstance(placement, Partial):
                raise ValueError(
                    "partial tensors must be reduced before quantization"
                )
            if isinstance(placement, Shard) and not -len(
                shape
            ) <= placement.dim < len(shape):
                raise ValueError(
                    "shard dimension is outside the logical tensor"
                )

    def amax(
        self,
        x: torch.Tensor,
        *,
        distribution: Distribution | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce to the FP32 maximum magnitude.

        Over the logical statistical domain. Sharded reduction axes are
        all-reduced across the owning mesh group. ``out`` borrows caller
        storage with the statistical shape. CUDA inputs reduce with the
        magnitude kernels and raise ``ValueError`` when no kernel reads their
        layout.
        """
        self._shape(tuple(x.shape), x.dtype)
        self._distribution(tuple(x.shape), distribution)
        shape = self._statistics_shape(tuple(x.shape))
        if out is not None and (
            out.shape != shape
            or out.dtype != torch.float32
            or out.device != x.device
        ):
            raise ValueError(
                "amax output must match the statistical shape, FP32 dtype "
                "and device"
            )
        if isinstance(x, QuantizedTensor):
            x = x.dequantize(dtype=torch.float32)

        if x.is_cuda:
            maximum = self._device_amax(x, shape)
        elif self.format == "mxfp8":
            maximum = (
                x.float()
                .reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
                .abs()
                .amax(-1, keepdim=True)
            )
        elif not x.numel():
            maximum = torch.zeros(shape, dtype=torch.float32, device=x.device)
        elif self.axis == 0:
            axes = tuple(range(1, x.ndim))
            values = x.float().abs()
            maximum = values.amax(axes, keepdim=True) if axes else values
        else:
            maximum = x.float().abs().amax()

        if distribution is not None:
            # Reduction axes follow the logical value, not its execution lane
            # or projection chunk. PP layers do not share a statistical domain.
            for axis, placement in zip(
                distribution.mesh.axes, distribution.placements, strict=True
            ):
                if isinstance(placement, Partial):
                    raise ValueError(
                        "partial tensors must be reduced before quantization"
                    )
                if not isinstance(placement, Shard):
                    continue
                dim = placement.dim % x.ndim
                if self.format == "mxfp8":
                    continue
                if self.axis == 0 and dim == 0:
                    continue
                selected: tuple[str, ...] = (axis,)
                mesh = distribution.mesh
                if selected not in mesh._groups:
                    # A factored head topology can borrow its encompassing TP
                    # group. Including exact replicas leaves max unchanged.
                    replicated = {
                        name
                        for name, part in zip(
                            mesh.axes, distribution.placements, strict=True
                        )
                        if isinstance(part, Replicate)
                    }
                    candidates = [
                        axes
                        for axes in mesh._groups
                        if axis in axes
                        and set(axes).difference({axis}).issubset(replicated)
                    ]
                    if candidates:
                        selected = min(candidates, key=len)
                mesh.get_group(selected).all_reduce(maximum, op="max")
        return maximum if out is None else out.copy_(maximum)

    def quantize(
        self,
        x: torch.Tensor,
        *,
        distribution: Distribution | None = None,
        amax: torch.Tensor | None = None,
        out: QuantizedTensor | None = None,
    ) -> QuantizedTensor:
        """Encode a logical floating tensor.

        Into this quantizer's representation. An input already encoded with
        the same quantizer is returned or repacked without recomputing
        scales. ``amax`` borrows caller-computed statistics;
        ``out`` borrows caller-owned encoding storage. CUDA FP8 encoding runs
        the row kernels and raises ``ValueError`` when no kernel reads the
        input's layout.
        """
        self._shape(tuple(x.shape), x.dtype)
        self._distribution(tuple(x.shape), distribution)

        if isinstance(x, QuantizedTensor):
            if x.quantizer == self and amax is None:
                if out is None:
                    return x
                return x.repack(
                    scale_layout=out.scale_layout,
                    row_order=out.row_order,
                    out=out,
                )
            x = x.dequantize()

        if out is not None and (
            not isinstance(out, QuantizedTensor)
            or out.quantizer != self
            or out.shape != x.shape
            or out.dtype != x.dtype
            or out.device != x.device
        ):
            raise ValueError(
                "quantization output must match encoding, statistical axis, "
                "shape, dtype and device"
            )
        if amax is not None and (
            amax.shape != self._statistics_shape(tuple(x.shape))
            or amax.dtype != torch.float32
            or amax.device != x.device
        ):
            raise ValueError(
                "amax must match the logical statistical shape, FP32 dtype "
                "and device"
            )

        if self.format == "fp8" and x.is_cuda:
            return self._device_fp8(x, distribution, amax, out)

        # A calibrated activation tensor scale is a checkpoint fact. Runtime
        # callers may still supply scratch statistics used by the dynamic
        # path, but those must never replace the frozen calibration domain.
        tensor_scale = None
        if self.calibrated_scale is not None:
            # A host-to-device scalar copy is illegal while a CUDA graph is
            # being captured. Fill graph-owned device storage instead; the
            # immutable Python value remains the checkpoint fact and replay
            # never searches or changes it.
            tensor_scale = torch.empty(
                (), dtype=torch.float32, device=x.device
            ).fill_(self.calibrated_scale)
        elif amax is None and self.format != "mxfp8":
            amax = self.amax(x, distribution=distribution)

        layout = ScaleLayout.LINEAR if out is None else out.scale_layout
        if self.format == "fp8":
            # FP8 off CUDA: it never carries a calibrated scale, so amax was
            # supplied or computed above.
            assert amax is not None
            scale = amax.clamp_min(1e-12) / 448.0
            values = (
                (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            )
            fields = {"values": values, "scale": scale}
        else:
            fields = self._encode_blocks(x, amax, layout, tensor_scale)

        result = self.from_tensors(
            fields, shape=tuple(x.shape), dtype=x.dtype, scale_layout=layout
        )
        if out is None:
            return result
        if out.row_order is not result.row_order:
            # Encoding follows logical rows; ``out`` may store them permuted.
            result = result.repack(row_order=out.row_order)
        for name, value in result.buffers().items():
            out.buffers()[name].copy_(value.reshape_as(out.buffers()[name]))
        return out

    def _kernel_rows(self, x: torch.Tensor) -> tuple[torch.Tensor | None, str]:
        """Return ``x`` as the ``[rows, width]`` matrix the FP8 kernels read.

        Row scales retain axis zero of a rank-2 matrix. MXFP8 statistics
        reduce each K32 block, so the blocks become the rows. A tensor-wide
        statistic spans every element, so any rank whose leading axes
        flatten to rows without a copy qualifies. The second element names
        the unmet condition when the matrix is ``None``.
        """
        if self.format == "mxfp8":
            try:
                return x.view(-1, 32), ""
            except RuntimeError:
                return None, "K32 blocks do not flatten to rows without a copy"
        if self.axis == 0:
            if x.ndim == 2:
                return x, ""
            return None, (
                "row statistics keep one value per leading index, which only "
                "a rank-2 matrix has a kernel for"
            )
        try:
            if x.ndim == 2:
                return x, ""
            return x.view(-1, x.shape[-1]) if x.ndim else x.view(1, 1), ""
        except RuntimeError:
            return None, "leading axes do not flatten to rows without a copy"

    def _device_amax(
        self, x: torch.Tensor, shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Reduce CUDA magnitude statistics with the kernels, or raise."""
        from uniserve_kernels.triton import require_kernel

        from uniserve_kernels import quantization

        maximum = torch.empty(shape, dtype=torch.float32, device=x.device)
        if not x.numel():
            return maximum.zero_()
        matrix, reason = self._kernel_rows(x)
        require_kernel(
            "Quantizer.amax",
            reason
            if matrix is None
            else quantization.unsupported_rowwise_fp8(matrix),
            x=x,
        )
        if self.axis == 0 or self.format == "mxfp8":
            quantization.row_absmax(matrix, maximum.view(-1))
        else:
            quantization.tensor_absmax(matrix, maximum)
        return maximum

    def _reduces_rows(self, x: torch.Tensor, distribution) -> bool:
        """Return whether row statistics span columns held by other ranks.

        Columns sharded over a mesh axis of one rank are complete locally.
        """
        return distribution is not None and any(
            isinstance(placement, Shard)
            and placement.dim % x.ndim == 1
            and distribution.mesh.size((axis,)) > 1
            for axis, placement in zip(
                distribution.mesh.axes, distribution.placements, strict=True
            )
        )

    def _device_fp8(self, x, distribution, amax, out) -> QuantizedTensor:
        """Encode CUDA FP8 through the row kernels, or raise.

        Row scales computed from complete local rows fuse the statistic into
        the encoding launch. Supplied, tensor-wide or cross-rank statistics
        are reduced first (``amax``), then the encoding reads them.
        """
        from uniserve_kernels.triton import require_kernel

        from uniserve_kernels import quantization

        matrix, reason = self._kernel_rows(x)
        require_kernel(
            "Quantizer.quantize(fp8)",
            reason
            if matrix is None
            else quantization.unsupported_rowwise_fp8(matrix),
            x=x,
        )
        if amax is None and (
            self.axis is None or self._reduces_rows(x, distribution)
        ):
            amax = self.amax(x, distribution=distribution)

        target = (
            self.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
            if out is None
            else out
        )
        fields = target.buffers()
        quantization.rowwise_fp8(
            matrix,
            fields["values"].view(matrix.shape),
            fields["scale"].view(-1),
            amax=None if amax is None else amax.reshape(-1),
        )
        return target

    def _encode_blocks(self, x, maximum, layout, tensor_scale=None):
        """Encode K-blocks; NVFP4 derives its tensor scale from `maximum`
        unless a calibrated `tensor_scale` is supplied.
        """  # noqa: D205
        if not x.is_cuda or torch.cuda.get_device_capability(x.device) < (
            10,
            0,
        ):
            raise RuntimeError(
                f"{self.format} conversion requires an SM100-class CUDA device"
            )
        import flashinfer

        # Block encoders operate on a flattened [rows, K] matrix.
        rows = prod(x.shape[:-1])
        matrix = x.reshape(rows, x.shape[-1]).to(torch.bfloat16)

        if not matrix.numel():
            result = self.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
            if self.format == "nvfp4":
                result.buffers()["tensor_scale"].copy_(
                    maximum.clamp_min(1e-12) / (448.0 * 6.0)
                )
            return dict(result.repack(scale_layout=layout).buffers())

        if self.format == "mxfp8" and maximum is not None:
            # Native MXFP8 rounds a positive scale upward to E8M0. Retaining
            # exponent bits avoids log2 rounding near exact power boundaries.
            scaled = maximum / 448.0
            bits = scaled.view(torch.int32)
            exponent = (bits >> 23) & 255
            fraction = bits & 0x7FFFFF
            increment = torch.where(
                exponent == 0, fraction > 0x400000, fraction != 0
            )
            scales = (
                (exponent + increment)
                .clamp_max(254)
                .to(torch.uint8)
                .reshape(rows, x.shape[-1] // 32)
            )
            decoded = scales.view(torch.float8_e8m0fnu).float()
            # [rows, K // 32, 32] value blocks, one E8M0 scale per block
            blocks = matrix.float().reshape(rows, x.shape[-1] // 32, 32)
            values = (
                torch.where(
                    maximum.reshape(rows, -1, 1) > 0,
                    blocks / decoded.unsqueeze(-1),
                    0.0,
                )
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
            )
            encoded = self.from_tensors(
                {"values": values.reshape(x.shape), "scale": scales},
                shape=tuple(x.shape),
                dtype=x.dtype,
            )
            return dict(encoded.repack(scale_layout=layout).buffers())

        if self.format == "mxfp8":
            # The native MXFP8 encoder accepts contiguous matrices. Packing a
            # borrowed strided view preserves its complete numerical domain.
            values, scales = flashinfer.mxfp8_quantize(
                matrix.contiguous(),
                is_sf_swizzled_layout=layout is ScaleLayout.SWIZZLED_128X4,
                backend="cuda",
            )
            scales = (
                scales.reshape(-1)
                if layout is ScaleLayout.SWIZZLED_128X4
                else scales.reshape(rows, x.shape[-1] // 32)
            )
            return {"values": values.reshape(x.shape), "scale": scales}

        if tensor_scale is None:
            tensor_scale = maximum.clamp_min(1e-12) / (448.0 * 6.0)
        values, scales = flashinfer.nvfp4_quantize(
            matrix,
            1.0 / tensor_scale,
            sfLayout=flashinfer.SfLayout.layout_128x4
            if layout is ScaleLayout.SWIZZLED_128X4
            else flashinfer.SfLayout.layout_linear,
            backend="cuda"
            if layout is ScaleLayout.SWIZZLED_128X4
            else "cute-dsl",
            enable_pdl=False,
        )
        scales = (
            scales.reshape(-1)
            if layout is ScaleLayout.SWIZZLED_128X4
            else scales.reshape(rows, x.shape[-1] // 16)
        )
        return {
            "values": values.reshape(*x.shape[:-1], x.shape[-1] // 2),
            "block_scale": scales,
            "tensor_scale": tensor_scale,
        }

    def round_trip(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` with this quantizer's static scale and decode it again.

        The portable reference for kernels that consume calibrated NVFP4
        activations: each K16 block takes an E4M3 scale of its amax over
        ``6 * calibrated_scale``, and each value rounds to the nearest E2M1
        code, ties to even. Runs on any device and returns ``x``'s dtype.

        Raises:
            ValueError: This quantizer is not calibrated NVFP4.
        """
        if self.format != "nvfp4" or self.calibrated_scale is None:
            raise ValueError("round trips require calibrated NVFP4 encoding")
        self._shape(tuple(x.shape), x.dtype)
        tensor_scale = self.calibrated_scale
        blocks = x.float().reshape(*x.shape[:-1], -1, 16)
        block_scale = (
            (blocks.abs().amax(dim=-1) / (6.0 * tensor_scale))
            .clamp(max=448.0)
            .to(torch.float8_e4m3fn)
            .float()
        )
        scale = block_scale.unsqueeze(-1) * tensor_scale
        normalized = blocks / torch.where(scale == 0, 1.0, scale)
        # Midpoints between adjacent E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6.
        boundaries = torch.tensor(
            (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device=x.device
        )
        magnitude = normalized.abs().contiguous()
        code = torch.bucketize(magnitude, boundaries)
        # A value exactly on a midpoint rounds to the even (lower) code.
        boundary = boundaries[code.clamp(max=6)]
        code += ((magnitude == boundary) & (code.remainder(2) == 1)).long()
        table = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=x.device
        )
        decoded = table[code] * normalized.sign() * scale
        return decoded.reshape(x.shape).to(x.dtype)

    def empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> QuantizedTensor:
        """Allocate uninitialized encoding buffers for a logical tensor."""
        self._shape(shape, dtype)
        if self.format == "fp8":
            fields = {
                "values": torch.empty(
                    shape, dtype=torch.float8_e4m3fn, device=device
                ),
                "scale": torch.empty(
                    self._statistics_shape(shape),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        else:
            rows = prod(shape[:-1])
            block = 32 if self.format == "mxfp8" else 16
            fields = {
                "values": torch.empty(
                    shape if block == 32 else (*shape[:-1], shape[-1] // 2),
                    dtype=torch.float8_e4m3fn if block == 32 else torch.uint8,
                    device=device,
                )
            }
            fields["scale" if block == 32 else "block_scale"] = torch.empty(
                (rows, shape[-1] // block), dtype=torch.uint8, device=device
            )
            if block == 16:
                fields["tensor_scale"] = torch.empty(
                    (), dtype=torch.float32, device=device
                )

        return self.from_tensors(fields, shape=shape, dtype=dtype)

    def from_tensors(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        scale_layout: ScaleLayout = ScaleLayout.LINEAR,
        row_order: RowOrder = RowOrder.LINEAR,
    ) -> QuantizedTensor:
        """Wrap existing encoding buffers without recomputing statistics.

        Every buffer is validated against the logical shape, format dtype,
        physical scale layout and row order; borrowed buffers keep their
        storage owner. A non-linear row order describes a stacked block-
        scaled ``[E, rows, K]`` tensor whose values and block scales store
        every expert's rows permuted; it requires whole 32-row blocks.
        """
        self._shape(shape, dtype)
        if not isinstance(row_order, RowOrder):
            raise ValueError("unknown row order")
        if row_order is not RowOrder.LINEAR and (
            self.format == "fp8" or len(shape) != 3 or shape[1] % 32
        ):
            raise ValueError(
                "row orders apply only to stacked block-scaled tensors "
                "of whole 32-row blocks"
            )
        keys = (
            {"values", "block_scale", "tensor_scale"}
            if self.format == "nvfp4"
            else {"values", "scale"}
        )
        if set(tensors) != keys:
            raise ValueError(
                f"{self.format} encoding requires exactly {sorted(keys)}"
            )
        if any(
            not isinstance(value, torch.Tensor)
            or isinstance(value, QuantizedTensor)
            for value in tensors.values()
        ):
            raise TypeError("encoding buffers must be ordinary tensors")
        if len({value.device for value in tensors.values()}) != 1:
            raise ValueError("encoding buffers must share one device")
        if not isinstance(scale_layout, ScaleLayout) or (
            self.format == "fp8" and scale_layout is not ScaleLayout.LINEAR
        ):
            raise ValueError("unsupported scale layout for this encoding")

        values = tensors["values"]
        value_shape = (
            (*shape[:-1], shape[-1] // 2) if self.format == "nvfp4" else shape
        )
        value_dtype = (
            torch.uint8 if self.format == "nvfp4" else torch.float8_e4m3fn
        )
        if values.shape != value_shape or values.dtype != value_dtype:
            raise ValueError(
                "encoded values disagree with the logical shape or format dtype"
            )

        if self.format == "fp8":
            scale = tensors["scale"]
            if (
                scale.dtype != torch.float32
                or scale.shape != self._statistics_shape(shape)
            ):
                raise ValueError(
                    "FP8 scale must match the retained statistical axis and "
                    "FP32 dtype"
                )
            cls: type[QuantizedTensor] = _FP8Tensor
        else:
            block = 32 if self.format == "mxfp8" else 16
            rows, columns = prod(shape[:-1]), shape[-1] // block
            scales = tensors["scale" if block == 32 else "block_scale"]
            physical_shape = (
                (rows, columns)
                if scale_layout is ScaleLayout.LINEAR
                else (((rows + 127) // 128) * 128 * ((columns + 3) // 4) * 4,)
            )
            if (
                scales.dtype != torch.uint8
                or scales.shape != physical_shape
                or not scales.is_contiguous()
            ):
                raise ValueError(
                    "block scale shape, dtype or strides disagree with its "
                    "physical layout"
                )
            # A stacked expert tensor [E, rows, K] may carry one tensor
            # scale per expert, the leading axis; others carry one scalar.
            if block == 16 and (
                tensors["tensor_scale"].dtype != torch.float32
                or tensors["tensor_scale"].shape
                not in {(), *(((shape[0],),) if len(shape) == 3 else ())}
            ):
                raise ValueError(
                    "NVFP4 tensor scale must be an FP32 scalar or one value "
                    "per leading expert of a stacked tensor"
                )
            cls = _MXFP8Tensor if block == 32 else _NVFP4Tensor

        return cls(
            tensors,
            shape=shape,
            dtype=dtype,
            quantizer=self,
            scale_layout=scale_layout,
            row_order=row_order,
        )


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """The independently selected encodings of linear weights.

    And activations.
    """

    weight: Quantizer
    activation: Quantizer


torch.serialization.add_safe_globals([Quantizer])
