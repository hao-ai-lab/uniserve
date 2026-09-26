"""Encoded inference tensors with ordinary PyTorch parameter identity."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch.utils._pytree import tree_map

if TYPE_CHECKING:
    from .quantizer import Quantizer


class ScaleLayout(Enum):
    """Physical ordering of block scales.

    Independent of the represented values.
    """

    LINEAR = "linear"
    SWIZZLED_128X4 = "128x4"


class RowOrder(Enum):
    """Physical order of the rows of every matrix of a stacked tensor.

    Independent of the represented values, like ``ScaleLayout``. The encoded
    values and the block scales of a stacked ``[E, rows, K]`` tensor store
    each expert matrix's rows in this order; decoding restores the logical
    order. The orders are the row permutations TensorRT-LLM's grouped GEMMs
    read (``shuffleMatrixA`` and ``reorderRowsForGatedActGemm``).
    """

    LINEAR = "linear"
    # Each 32-row block stores logical row ``r`` at position
    # ``(r % 4) * 8 + r // 4``: the order a 128-row epilogue tile reads.
    SHUFFLED_128 = "shuffled-128"
    # The two row halves interleave (logical row ``i`` of the first half at
    # position ``2i`` and of the second half at ``2i + 1``), then each
    # 32-row block shuffles as in ``SHUFFLED_128``. A gated GEMM so reads
    # the linear and gate rows of one output channel side by side.
    INTERLEAVED_SHUFFLED_128 = "interleaved-shuffled-128"


def _logical_rows(order: RowOrder, rows: int) -> torch.Tensor:
    """Return the logical row stored at each physical row of one matrix.

    ``rows`` must be a multiple of 32 for a shuffled order; the result is a
    CPU ``int64`` permutation of ``range(rows)``.
    """
    positions = torch.arange(rows)
    if order is RowOrder.LINEAR:
        return positions

    # Physical position q of a 32-row block holds block row (q % 8) * 4 +
    # q // 8, the inverse of the shuffle's r -> (r % 4) * 8 + r // 4.
    within = positions % 32
    shuffled = positions - within + (within % 8) * 4 + within // 8
    if order is RowOrder.SHUFFLED_128:
        return shuffled

    # Before the shuffle, interleaved position q holds row q // 2 of the
    # first half (even q) or of the second half (odd q).
    return shuffled // 2 + (shuffled % 2) * (rows // 2)


class QuantizedTensor(torch.Tensor):
    """A logical floating tensor backed by encoded values and their scales.

    The encoding is immutable; the returned backing views remain writable by
    their storage owner. Ordinary numerical calls return dense tensors
    unless they have a representation-aware implementation. Linear calls
    consume the encoded representation directly. ``scale_layout`` and
    ``row_order`` describe only the physical arrangement of the encoded
    fields; every arrangement decodes to the same logical tensor.
    """

    @staticmethod
    def __new__(
        cls, tensors, *, shape, dtype, quantizer, scale_layout, row_order
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device=tensors["values"].device,
            requires_grad=False,
        )

    def __init__(
        self, tensors, *, shape, dtype, quantizer, scale_layout, row_order
    ):
        self._buffers = dict(tensors)
        self._quantizer = quantizer
        self._scale_layout = scale_layout
        self._row_order = row_order

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    @property
    def scale_layout(self) -> ScaleLayout:
        return self._scale_layout

    @property
    def row_order(self) -> RowOrder:
        return self._row_order

    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Borrow every encoded field.

        Without exposing a mutable field mapping.
        """
        return MappingProxyType(self._buffers)

    def dequantize(self, *, dtype=None, out=None):
        """Decode into logical shape.

        Optionally writing exact caller storage.
        """
        dtype = self.dtype if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("dequantization requires a floating-point dtype")
        if out is not None and (
            out.shape != self.shape
            or out.dtype != dtype
            or out.device != self.device
        ):
            raise ValueError(
                "dequantization output must match shape, dtype and device"
            )
        result = self._decode()
        if self.row_order is not RowOrder.LINEAR:
            # _decode follows the physical rows; each expert matrix of the
            # stacked [E, rows, K] tensor returns to its logical row order.
            logical = torch.empty_like(result)
            logical[:, _logical_rows(self.row_order, self.shape[1])] = result
            result = logical
        result = result.to(dtype)
        return result if out is None else out.copy_(result)

    def repack(
        self,
        *,
        scale_layout: ScaleLayout | None = None,
        row_order: RowOrder | None = None,
        out=None,
    ):
        """Rearrange the encoded fields without changing the encoded numbers.

        ``scale_layout`` and ``row_order`` name the target arrangement; an
        omitted one keeps this tensor's. The result decodes exactly as this
        tensor does. Returns ``self`` when nothing changes, otherwise new
        storage, or ``out`` after copying into its fields.

        Raises:
            ValueError: The target arrangement is unknown, not available for
                this encoding and shape (see ``Quantizer.from_tensors``), or
                ``out`` does not match it.
        """
        scale_layout = (
            self.scale_layout if scale_layout is None else scale_layout
        )
        row_order = self.row_order if row_order is None else row_order
        if not isinstance(scale_layout, ScaleLayout):
            raise ValueError("unknown scale layout")
        if not isinstance(row_order, RowOrder):
            raise ValueError("unknown row order")
        if (
            self.quantizer.format == "fp8"
            and scale_layout is not ScaleLayout.LINEAR
        ):
            raise ValueError("FP8 supports only linear scale storage")
        if out is not None:
            self._validate_output(
                out, scale_layout=scale_layout, row_order=row_order
            )

        if scale_layout is self.scale_layout and row_order is self.row_order:
            if out is None:
                return self
            for name, value in self._buffers.items():
                out._buffers[name].copy_(value)
            return out

        fields = dict(self._buffers)
        name = "block_scale" if self.quantizer.format == "nvfp4" else "scale"
        rows = prod(self.shape[:-1])
        columns = self.shape[-1] // (16 if name == "block_scale" else 32)
        linear = _linear_scales(fields[name], rows, columns, self.scale_layout)

        if row_order is not self.row_order:
            if self.quantizer.format == "fp8" or len(self.shape) != 3:
                raise ValueError(
                    "row orders apply only to stacked block-scaled tensors"
                )
            # Target physical row p holds logical row target[p], which this
            # tensor stores at physical row stored[target[p]]. Values and
            # block scales of one row move together.
            experts, height = self.shape[0], self.shape[1]
            stored = torch.empty(height, dtype=torch.long)
            stored[_logical_rows(self.row_order, height)] = torch.arange(height)
            source = stored[_logical_rows(row_order, height)].to(self.device)
            fields["values"] = fields["values"].index_select(1, source)
            linear = (
                linear.reshape(experts, height, columns)
                .index_select(1, source)
                .reshape(rows, columns)
            )

        fields[name] = (
            _swizzle_scales(linear)
            if scale_layout is ScaleLayout.SWIZZLED_128X4
            else linear.contiguous()
        )
        result = self.quantizer.from_tensors(
            fields,
            shape=tuple(self.shape),
            dtype=self.dtype,
            scale_layout=scale_layout,
            row_order=row_order,
        )

        if out is None:
            return result
        for name, value in result._buffers.items():
            out._buffers[name].copy_(value.reshape_as(out._buffers[name]))
        return out

    def _validate_output(self, out, *, scale_layout=None, row_order=None):
        if (
            not isinstance(out, QuantizedTensor)
            or out.quantizer != self.quantizer
            or out.shape != self.shape
            or out.dtype != self.dtype
            or out.device != self.device
            or out.scale_layout
            is not (self.scale_layout if scale_layout is None else scale_layout)
            or out.row_order
            is not (self.row_order if row_order is None else row_order)
        ):
            raise ValueError(
                "quantized output must match encoding, shape, dtype, device "
                "and layout"
            )

    def _decode(self):
        raise NotImplementedError

    def __tensor_flatten__(self):
        # Flatten attributes point at the actual backing so compile/export can
        # trace the encoded operands without allocating a logical dense value.
        names = []
        for name, value in self._buffers.items():
            attribute = f"_encoded_{name}"
            setattr(self, attribute, value)
            names.append(attribute)
        return names, (
            tuple(self.shape),
            self.dtype,
            self.quantizer,
            self.scale_layout,
            self.row_order,
        )

    @classmethod
    def __tensor_unflatten__(
        cls, tensors, metadata, outer_size=None, outer_stride=None
    ):
        shape, dtype, quantizer, layout, order = metadata
        return quantizer.from_tensors(
            {
                name.removeprefix("_encoded_"): value
                for name, value in tensors.items()
            },
            shape=shape,
            dtype=dtype,
            scale_layout=layout,
            row_order=order,
        )

    def __reduce_ex__(self, protocol):
        return _rebuild, (
            self._buffers,
            tuple(self.shape),
            self.dtype,
            self.quantizer,
            self.scale_layout,
            self.row_order,
        )

    # torch's stub for Tensor.__repr__ declares an unnamed keyword-only
    # parameter that no override can match.
    def __repr__(self):  # type: ignore[override]
        return (
            f"QuantizedTensor(format={self.quantizer.format!r}, "
            f"shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device})"
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.nn.functional.linear:
            from uniserve.nn.functional import linear

            return linear(*args, **(kwargs or {}))
        return super().__torch_function__(func, types, args, kwargs or {})

    # torch's stub assigns a plain function here, which mypy reads as an
    # instance method; torch's own subclasses override it the same way.
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}
        aten = torch.ops.aten

        # View-like calls rebuild the wrapper over transformed buffers.
        if func in (
            aten.detach.default,
            aten.alias.default,
            aten.clone.default,
            aten._to_copy.default,
        ):
            value = args[0]
            device = kwargs.get("device", value.device)
            dtype = kwargs.get("dtype", value.dtype)
            if func is aten._to_copy.default:
                fields = {
                    name: tensor.to(
                        device=device,
                        copy=True,
                        non_blocking=kwargs.get("non_blocking", False),
                    )
                    for name, tensor in value._buffers.items()
                }
            elif func is aten.clone.default:
                fields = {
                    name: tensor.clone()
                    for name, tensor in value._buffers.items()
                }
            else:
                fields = {
                    name: tensor.detach()
                    for name, tensor in value._buffers.items()
                }
            return value.quantizer.from_tensors(
                fields,
                shape=tuple(value.shape),
                dtype=dtype,
                scale_layout=value.scale_layout,
                row_order=value.row_order,
            )

        if func is aten.copy_.default:
            target, source = args[:2]
            if isinstance(target, QuantizedTensor):
                if isinstance(source, QuantizedTensor):
                    if (
                        target.quantizer != source.quantizer
                        or target.shape != source.shape
                        or target.scale_layout is not source.scale_layout
                        or target.row_order is not source.row_order
                    ):
                        raise ValueError(
                            "copy requires matching quantized shape and "
                            "encoding"
                        )
                    for name, tensor in target._buffers.items():
                        tensor.copy_(source._buffers[name])
                else:
                    target.quantizer.quantize(
                        source.to(device=target.device, dtype=target.dtype),
                        out=target,
                    )
                return target

        if func._schema.is_mutable:
            raise NotImplementedError(
                f"quantized in-place call {func} requires an "
                "encoding-aware implementation"
            )

        # Remaining calls decode their operands and produce dense results.
        def dense(value):
            return (
                value.dequantize()
                if isinstance(value, QuantizedTensor)
                else value
            )

        return func(*tree_map(dense, args), **tree_map(dense, kwargs))


class _FP8Tensor(QuantizedTensor):
    def _decode(self):
        return self._buffers["values"].float() * self._buffers["scale"]


class _MXFP8Tensor(QuantizedTensor):
    def _decode(self):
        rows, width = prod(self.shape[:-1]), self.shape[-1]
        scales = _linear_scales(
            self._buffers["scale"], rows, width // 32, self.scale_layout
        )
        scales = scales.view(torch.float8_e8m0fnu).float()
        # [rows, width // 32, 32] blocks, one E8M0 exponent per block
        values = self._buffers["values"].float().reshape(rows, width // 32, 32)
        return (values * scales.unsqueeze(-1)).reshape(self.shape)


class _NVFP4Tensor(QuantizedTensor):
    def _decode(self):
        rows, width = prod(self.shape[:-1]), self.shape[-1]
        packed = self._buffers["values"]
        # Two 4-bit codes per byte, low nibble first.
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
        # E2M1 has a sign bit and eight exactly representable magnitudes.
        magnitudes = torch.tensor(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6), device=self.device
        )
        values = magnitudes[codes & 7] * torch.where(codes < 8, 1, -1)
        scales = _linear_scales(
            self._buffers["block_scale"], rows, width // 16, self.scale_layout
        )
        # One tensor scale, or one per leading expert of a stacked tensor.
        tensor_scale = self._buffers["tensor_scale"]
        if tensor_scale.ndim:
            tensor_scale = tensor_scale.repeat_interleave(
                rows // tensor_scale.numel()
            ).unsqueeze(-1)
        scales = scales.view(torch.float8_e4m3fn).float() * tensor_scale
        return (
            values.reshape(rows, width // 16, 16) * scales.unsqueeze(-1)
        ).reshape(self.shape)


def _linear_scales(scales, rows, columns, layout):
    """Restore the [rows, columns] block-scale matrix.

    From its storage layout.
    """
    if layout is ScaleLayout.LINEAR:
        return scales.reshape(rows, columns)
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    # Each 128x4 tile stores (row % 32, row // 32, column % 4).
    return (
        scales.reshape(padded_rows // 128, padded_columns // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded_rows, padded_columns)[:rows, :columns]
    )


def _swizzle_scales(scales):
    """Pack a [rows, columns] scale matrix into 128x4-swizzled flat storage."""
    rows, columns = scales.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    padded = torch.zeros(
        (padded_rows, padded_columns), dtype=scales.dtype, device=scales.device
    )
    padded[:rows, :columns].copy_(scales)
    return (
        padded.reshape(padded_rows // 128, 4, 32, padded_columns // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )


def _rebuild(tensors, shape, dtype, quantizer, scale_layout, row_order):
    return quantizer.from_tensors(
        tensors,
        shape=shape,
        dtype=dtype,
        scale_layout=scale_layout,
        row_order=row_order,
    )


torch.serialization.add_safe_globals([_rebuild, ScaleLayout, RowOrder])
