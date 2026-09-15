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


class QuantizedTensor(torch.Tensor):
    """A logical floating tensor backed by encoded values and their scales.

    The encoding is immutable; the returned backing views remain writable by
    their storage owner. Ordinary numerical operations return dense tensors
    unless they have a representation-aware implementation. Linear operations
    consume the encoded representation directly.
    """

    @staticmethod
    def __new__(cls, tensors, *, shape, dtype, quantizer, scale_layout):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device=tensors["values"].device,
            requires_grad=False,
        )

    def __init__(self, tensors, *, shape, dtype, quantizer, scale_layout):
        self._buffers = dict(tensors)
        self._quantizer = quantizer
        self._scale_layout = scale_layout

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    @property
    def scale_layout(self) -> ScaleLayout:
        return self._scale_layout

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
        result = self._decode().to(dtype)
        return result if out is None else out.copy_(result)

    def repack(self, *, scale_layout: ScaleLayout, out=None):
        """Change scale storage order without changing the encoded numbers."""
        if not isinstance(scale_layout, ScaleLayout):
            raise ValueError("unknown scale layout")
        if (
            self.quantizer.format == "fp8"
            and scale_layout is not ScaleLayout.LINEAR
        ):
            raise ValueError("FP8 supports only linear scale storage")
        if out is not None:
            self._validate_output(out, scale_layout=scale_layout)

        if scale_layout is self.scale_layout:
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
        )

        if out is None:
            return result
        for name, value in result._buffers.items():
            out._buffers[name].copy_(value.reshape_as(out._buffers[name]))
        return out

    def _validate_output(self, out, *, scale_layout=None):
        if (
            not isinstance(out, QuantizedTensor)
            or out.quantizer != self.quantizer
            or out.shape != self.shape
            or out.dtype != self.dtype
            or out.device != self.device
            or out.scale_layout
            is not (self.scale_layout if scale_layout is None else scale_layout)
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
        )

    @classmethod
    def __tensor_unflatten__(
        cls, tensors, metadata, outer_size=None, outer_stride=None
    ):
        shape, dtype, quantizer, layout = metadata
        return quantizer.from_tensors(
            {
                name.removeprefix("_encoded_"): value
                for name, value in tensors.items()
            },
            shape=shape,
            dtype=dtype,
            scale_layout=layout,
        )

    def __reduce_ex__(self, protocol):
        return _rebuild, (
            self._buffers,
            tuple(self.shape),
            self.dtype,
            self.quantizer,
            self.scale_layout,
        )

    def __repr__(self):
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

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        aten = torch.ops.aten

        # View-like operations rebuild the wrapper over transformed buffers.
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
            )

        if func is aten.copy_.default:
            target, source = args[:2]
            if isinstance(target, QuantizedTensor):
                if isinstance(source, QuantizedTensor):
                    if (
                        target.quantizer != source.quantizer
                        or target.shape != source.shape
                        or target.scale_layout is not source.scale_layout
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
                f"quantized in-place operation {func} requires an "
                "encoding-aware implementation"
            )

        # Remaining operations decode their operands and produce dense results.
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
        scales = (
            scales.view(torch.float8_e4m3fn).float()
            * self._buffers["tensor_scale"]
        )
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


def _rebuild(tensors, shape, dtype, quantizer, scale_layout):
    return quantizer.from_tensors(
        tensors, shape=shape, dtype=dtype, scale_layout=scale_layout
    )


torch.serialization.add_safe_globals([_rebuild, ScaleLayout])
