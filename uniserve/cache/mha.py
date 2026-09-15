"""Block-indexed MHA, GQA and MQA state with scale-preserving writes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from uniserve import _slices
from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve.tensors import BufferConfig

from ._fp8 import rescale_
from .state import State as PrefixState
from .state import StateConfig, _blocks


def _spans(
    blocks: tuple[int, ...], start: int, length: int, block_size: int
) -> tuple[tuple[int, int, int], ...]:
    if (
        type(start) is not int
        or type(length) is not int
        or start < 0
        or length < 0
        or start + length > len(blocks) * block_size
    ):
        raise ValueError("token interval exceeds its block table")
    spans = []
    while length:
        logical, offset = divmod(start, block_size)
        count = min(length, block_size - offset)
        spans.append((blocks[logical], offset, count))
        start += count
        length -= count
    return tuple(spans)


@dataclass(frozen=True)
class Config(StateConfig):
    """Local K/V heads with explicit global head identities and compute dtype."""

    num_kv_heads: int
    head_dim: int
    head_indices: tuple[int, ...]
    compute_dtype: torch.dtype
    indexing: ClassVar[Literal["tokens"]] = "tokens"

    def __post_init__(self) -> None:
        if any(type(size) is not int or size < 1 for size in (self.num_kv_heads, self.head_dim)):
            raise ValueError("K/V head dimensions must be positive")
        if (
            not isinstance(self.head_indices, tuple)
            or not self.head_indices
            or len(set(self.head_indices)) != len(self.head_indices)
            or any(
                type(head) is not int or not 0 <= head < self.num_kv_heads
                for head in self.head_indices
            )
        ):
            raise ValueError("local head indices must be distinct global K/V heads")
        if self.compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("K/V computation requires FP16, BF16 or FP32")

    @property
    def local_heads(self) -> int:
        return len(self.head_indices)

    def buffers(
        self,
        *,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype | None,
        quantizer: Quantizer | None,
    ) -> Mapping[str, BufferConfig]:
        if (
            type(num_blocks) is not int
            or num_blocks < 0
            or type(block_size) is not int
            or block_size < 1
        ):
            raise ValueError(
                "K/V storage requires a nonnegative block count and positive block size"
            )
        dtype = self.compute_dtype if dtype is None else dtype
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("K/V storage dtype must be a logical floating-point dtype")
        if quantizer is not None and quantizer != Quantizer("fp8", axis=0):
            raise ValueError("MHA state supports only FP8 with one scale per block")
        shape = (num_blocks, block_size, self.local_heads, self.head_dim)
        result = {}
        for name in ("key", "value"):
            result[f"{name}.values"] = BufferConfig(
                shape, torch.float8_e4m3fn if quantizer else dtype
            )
            if quantizer:
                result[f"{name}.scale"] = BufferConfig((num_blocks, 1, 1, 1), torch.float32)
            result[f"{name}.initialized"] = BufferConfig((num_blocks,), torch.bool)
        return result

    def bind(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        block_size: int,
        dtype: torch.dtype | None,
        quantizer: Quantizer | None,
    ) -> State:
        count = tensors["key.values"].shape[0]
        expected = self.buffers(
            num_blocks=count, block_size=block_size, dtype=dtype, quantizer=quantizer
        )
        if set(tensors) != set(expected):
            raise ValueError("K/V backing fields do not match the state layout")
        for name, requirement in expected.items():
            tensor = tensors[name]
            if (
                tuple(tensor.shape) != requirement.shape
                or tensor.dtype != requirement.dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"K/V backing {name!r} disagrees with its layout")
        if len({tensor.device for tensor in tensors.values()}) != 1:
            raise ValueError("K/V backing and initialization flags must share one device")
        fields = {}
        for name in ("key", "value"):
            values = tensors[f"{name}.values"]
            fields[name] = (
                values
                if quantizer is None
                else quantizer.from_tensors(
                    {"values": values, "scale": tensors[f"{name}.scale"]},
                    shape=tuple(values.shape),
                    dtype=self.compute_dtype,
                )
            )
        return State(fields, {name: tensors[f"{name}.initialized"] for name in fields}, block_size)


@dataclass(frozen=True)
class State(PrefixState):
    """Borrow [blocks, tokens, heads, dim] keys and values.

    A block's first write chooses its FP8 scale; larger writes grow it and
    re-encode existing values while preserving uncovered token/head regions. Copying from encoded sources rounds through the source's
    logical dtype before applying the destination encoding.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            set(self.tensors) != {"key", "value"}
            or self.key.shape != self.value.shape
            or self.key.ndim != 4
            or self.key.shape[1] != self.block_size
        ):
            raise ValueError("MHA state requires matching block-indexed key and value tensors")

    @property
    def key(self) -> torch.Tensor:
        return self.tensors["key"]

    @property
    def value(self) -> torch.Tensor:
        return self.tensors["value"]

    def _read(self, name: str, block: int, interval: tuple[slice, ...]) -> torch.Tensor:
        tensor = self.tensors[name]
        index = (slice(block, block + 1), *interval)
        if not isinstance(tensor, QuantizedTensor):
            return tensor[index].squeeze(0)
        fields = tensor.buffers()
        values = fields["values"][index].float() * fields["scale"][block : block + 1]
        return values.to(tensor.dtype).squeeze(0)

    def read(
        self, block_ids: tuple[int, ...], *, start: int, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _blocks(block_ids, self.key.shape[0])
        spans = _spans(block_ids, start, length, self.block_size)
        outputs = []
        for name in ("key", "value"):
            tensor = self.tensors[name]
            values = [
                self._read(
                    name,
                    block,
                    (
                        slice(offset, offset + count),
                        slice(0, tensor.shape[2]),
                        slice(0, tensor.shape[3]),
                    ),
                )
                for block, offset, count in spans
            ]
            outputs.append(
                torch.cat(values)
                if values
                else torch.empty((0, *tensor.shape[2:]), dtype=tensor.dtype, device=tensor.device)
            )
        return outputs[0], outputs[1]

    def _write(
        self, name: str, block: int, interval: tuple[slice, ...], values: torch.Tensor
    ) -> None:
        tensor = self.tensors[name]
        index = (block, *interval)
        if isinstance(values, QuantizedTensor):
            values = values.dequantize()
        if isinstance(tensor, QuantizedTensor):
            fields = tensor.buffers()
            values = values.to(device=tensor.device, dtype=torch.float32)
            scale = fields["scale"][block]
            maximum = (
                values.abs().amax() if values.numel() else torch.zeros((), device=values.device)
            )
            proposed = maximum.clamp_min(1e-12) / 448.0
            initialized = self.initialized[name][block : block + 1]
            updated = torch.where(
                initialized.reshape(1, 1, 1), torch.maximum(scale, proposed), proposed
            )
            rescale_(
                fields["values"][block : block + 1], scale, updated, initialized, dtype=tensor.dtype
            )
            scale.copy_(updated)
            fields["values"][index].copy_(
                (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            )
        else:
            tensor[index].copy_(values)
        self.initialized[name][block] = True

    def write(
        self, block_ids: tuple[int, ...], *, start: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        if key.shape != value.shape or key.ndim != 3 or key.shape[1:] != self.key.shape[2:]:
            raise ValueError("K/V writes require matching [tokens, heads, dim] tensors")
        _blocks(block_ids, self.key.shape[0])
        spans = _spans(block_ids, start, key.shape[0], self.block_size)
        written = 0
        for block, offset, count in spans:
            interval = (
                slice(offset, offset + count),
                slice(0, key.shape[1]),
                slice(0, key.shape[2]),
            )
            for name, source in (("key", key), ("value", value)):
                self._write(name, block, interval, source[written : written + count])
            written += count

    def _validate_update(self, key, value, indices) -> None:
        """Check borrowed write views before standalone or fused computation."""

        if (
            key.shape != value.shape
            or key.ndim != 3
            or key.shape[1:] != self.key.shape[2:]
            or indices.shape != (key.shape[0],)
        ):
            raise ValueError("cache indices must align with [tokens, heads, dim] K/V")
        if (
            indices.dtype != torch.int64
            or indices.device != key.device
            or key.device != self.key.device
            or value.device != self.value.device
        ):
            raise ValueError("cache indices must be int64 on the K/V device")

    def update(self, key: torch.Tensor, value: torch.Tensor, *, indices: torch.Tensor) -> None:
        """Write linear cache slots, with -1 excluding a token from all state.

        Each FP8 block uses the maximum required by this update and its resident
        scale. Existing values round through the logical dtype before rescaling.
        """

        from uniserve.runtime.paged_kv_math import paged_kv_write

        self._validate_update(key, value, indices)
        if not indices.numel():
            return
        if not isinstance(self.key, QuantizedTensor) and not isinstance(
            self.value, QuantizedTensor
        ):
            paged_kv_write(
                self.key,
                self.value,
                indices,
                None,
                key,
                value,
                cast=key.dtype != self.key.dtype or value.dtype != self.value.dtype,
                initialized=(self.initialized["key"], self.initialized["value"]),
            )
            return
        # The dense scatter validates addresses in its owning kernel. Encoded
        # updates also index per-block scale metadata before scattering, so
        # they must validate the same domain before those accesses.
        valid = (indices >= -1) & (indices < self.key.shape[0] * self.block_size)
        if indices.is_cuda:
            torch._assert_async(valid.all(), "cache write index out of bounds")
        elif not bool(valid.all()):
            raise ValueError("cache write index out of bounds")
        count = self.key.shape[0]
        blocks = torch.where(indices >= 0, indices // self.block_size, count)
        # The final slot collects excluded tokens. Fixed-size scatter metadata
        # avoids copying addresses to the host and supports graph replay.
        touched = torch.zeros(count + 1, dtype=torch.bool, device=indices.device)
        touched.scatter_(0, blocks, True)
        touched = touched[:count]
        encoded = []
        for name, source in (("key", key), ("value", value)):
            target = self.tensors[name]
            if isinstance(source, QuantizedTensor):
                source = source.dequantize()
            if not isinstance(target, QuantizedTensor):
                encoded.append(source.to(target.dtype))
                self.initialized[name].logical_or_(touched)
                continue
            fields = target.buffers()
            maximum = source.float().abs().amax((1, 2))
            proposed = torch.zeros(count + 1, device=indices.device)
            proposed.scatter_reduce_(0, blocks, maximum, reduce="amax", include_self=True)
            proposed = proposed[:count].clamp_min(1e-12) / 448.0
            scales = fields["scale"].reshape(count)
            updated = torch.where(
                touched,
                torch.where(self.initialized[name], torch.maximum(scales, proposed), proposed),
                scales,
            )
            rescale_(fields["values"], scales, updated, self.initialized[name], dtype=target.dtype)
            scales.copy_(updated)
            # An excluded token reads a neutral scale from the extra slot and
            # never writes payload, scale or initialization state.
            selected = torch.cat((scales, torch.ones(1, device=scales.device))).index_select(
                0, blocks
            )
            encoded.append(
                (source.float() / selected[:, None, None])
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
            )
            self.initialized[name].logical_or_(touched)
        stores = tuple(
            tensor.buffers()["values"] if isinstance(tensor, QuantizedTensor) else tensor
            for tensor in (self.key, self.value)
        )
        paged_kv_write(*stores, indices, None, *encoded)

    def transfer_blocks(
        self, block_ids: tuple[int, ...], *, start: int, length: int
    ) -> Mapping[str, tuple[torch.Tensor, ...]]:
        _blocks(block_ids, self.key.shape[0])
        spans = _spans(block_ids, start, length, self.block_size)
        result = {}
        for name, tensor in self._storage().items():
            result[name] = tuple(
                tensor[block : block + 1, offset : offset + count]
                if name.endswith(".values")
                else tensor[block : block + 1]
                for block, offset, count in spans
            )
        return result

    def copy_region(
        self,
        source: torch.Tensor,
        *,
        field: Literal["key", "value"],
        block: int,
        source_slice: tuple[slice, ...],
        target_slice: tuple[slice, ...],
        workspace: Mapping[str, torch.Tensor],
    ) -> None:
        """Copy a rectangular region, using borrowed FP32 conversion scratch.

        ``workspace['values']`` must have enough FP32 elements on the target
        device. ``workspace['rounded']`` supplies source-dtype storage when an
        encoded source must round before conversion to a different dtype.
        Slices index the full source and one target block respectively.
        """

        if field not in self.tensors:
            raise ValueError("MHA field must be key or value")
        target = self.tensors[field]
        _blocks((block,), target.shape[0])
        if (
            not _slices.within(source_slice, tuple(source.shape))
            or not _slices.within(target_slice, tuple(target.shape[1:]))
            or _slices.shape(source_slice) != _slices.shape(target_slice)
        ):
            raise ValueError("copy regions must have equal shapes within their tensors")
        shape = _slices.shape(target_slice)
        if not all(shape):
            return
        if isinstance(source, QuantizedTensor):
            if source.quantizer.format != "fp8":
                raise ValueError("MHA transfer requires dense or FP8 source storage")
            fields = source.buffers()
            values = (
                workspace["values"].flatten()[: fields["values"][source_slice].numel()].view(shape)
            )
            if values.dtype != torch.float32 or values.device != target.device:
                raise ValueError(
                    "conversion workspace must provide FP32 values on the target device"
                )
            values.copy_(fields["values"][source_slice])
            scale = (
                fields["scale"]
                if source.quantizer.axis is None
                else fields["scale"][source_slice[0]]
            )
            values.mul_(scale.to(target.device))
            if source.dtype != torch.float32:
                rounded = workspace["rounded"].flatten()[: values.numel()].view(shape)
                if rounded.dtype != source.dtype or rounded.device != target.device:
                    raise ValueError("rounding workspace must match the source's logical dtype")
                rounded.copy_(values)
                values.copy_(rounded)
        else:
            values = source[source_slice]
        self._write(field, block, target_slice, values)
