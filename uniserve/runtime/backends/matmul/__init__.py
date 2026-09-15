"""Preparation and execution contracts for local matrix multiplication."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

import torch as torch_lib

from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve.tensors import BufferConfig, _join_channels


class Operator:
    """One prepared local GEMM; operands and workspace are borrowed."""

    def __init__(self, weight, *, input_dtype, input_quantizer, max_rows, output_dtype, workspace):
        self.weight = weight
        self.input_dtype = input_dtype
        self.input_quantizer = input_quantizer
        self.max_rows = max_rows
        self.output_dtype = output_dtype
        self.workspace = workspace

    def _input(self, x, out):
        if x.ndim != 2 or x.shape[-1] != self.weight.shape[-1] or x.shape[0] > self.max_rows:
            raise ValueError("linear input exceeds the prepared matrix dimensions")
        if x.dtype != self.input_dtype or x.device != self.weight.device:
            raise ValueError("linear input disagrees with the prepared dtype or device")
        if (
            out.shape != (x.shape[0], self.weight.shape[0])
            or out.dtype != self.output_dtype
            or out.device != x.device
        ):
            raise ValueError("linear output disagrees with the prepared shape, dtype or device")
        if self.input_quantizer is None:
            return x
        if isinstance(x, QuantizedTensor):
            if x.quantizer != self.input_quantizer:
                raise ValueError("input encoding disagrees with the prepared quantizer")
            return x
        target = self._input_storage(x)
        return self.input_quantizer.quantize(x, out=target)

    def _input_storage(self, x):
        """Borrow the active row prefix of this operator's encoded workspace."""

        rows = x.shape[0]
        fields = {}
        for name, value in self.workspace.items():
            if name.startswith("input."):
                field = name.removeprefix("input.")
                row_field = field in {"values", "block_scale"} or (
                    field == "scale"
                    and (self.input_quantizer.axis == 0 or self.input_quantizer.format == "mxfp8")
                )
                fields[field] = value[:rows] if row_field else value
        return self.input_quantizer.from_tensors(fields, shape=tuple(x.shape), dtype=x.dtype)

    def __call__(
        self, x: torch_lib.Tensor, bias: torch_lib.Tensor | None, *, out: torch_lib.Tensor
    ) -> torch_lib.Tensor:
        raise NotImplementedError


class MergedOperator:
    """Prepared fused projection over the logical branches of one matrix."""

    def __init__(self, operators, *, fused=None, output=None, bias=None, widths=None):
        self.operators = operators
        self.fused = fused
        self.output = output
        self.bias = bias
        self.widths = widths

    def __call__(self, x, biases, *, out):
        if set(out) != set(self.operators) or set(biases) != set(self.operators):
            raise ValueError("merged projection outputs and biases must match the named branches")
        if self.fused is None:
            # Different block encodings or NVFP4 tensor multipliers cannot be
            # merged into one scale domain without changing their mathematics.
            return {
                name: operator(x, biases[name], out=out[name])
                for name, operator in self.operators.items()
            }
        target = _join_channels(tuple(out[name] for name in self.operators), copy=False)
        direct = target is not None
        if target is None:
            target = self.output[: x.shape[0]]
        if isinstance(self.fused.weight, QuantizedTensor):
            # The scale vector is execution scratch. Refresh its branch views
            # so in-place parameter updates remain visible in every context.
            scales = self.fused.weight.buffers()["scale"]
            for operator, destination in zip(
                self.operators.values(), scales.split(self.widths), strict=True
            ):
                destination.copy_(operator.weight.buffers()["scale"].expand_as(destination))
        bias_values = tuple(biases.values())
        if all(value is None for value in bias_values):
            bias = None
        elif all(value is not None for value in bias_values):
            bias = _concatenate(bias_values)
        else:
            bias = self.bias
            for value, destination in zip(bias_values, bias.split(self.widths), strict=True):
                destination.zero_() if value is None else destination.copy_(value)
        self.fused(x, bias, out=target)
        if direct:
            return out
        for (name, operator), value in zip(
            self.operators.items(), target.split(self.widths, dim=-1), strict=True
        ):
            destination = out[name]
            if (
                destination.shape != value.shape
                or destination.dtype != value.dtype
                or destination.device != value.device
            ):
                raise ValueError("merged output disagrees with its branch representation")
            destination.copy_(value)
        return out


def _concatenate(values, *, copy=True):
    """Borrow adjacent parameter views when the loader supplied fused backing."""

    first = values[0]
    if all(
        value.is_contiguous() and value.dtype == first.dtype and value.device == first.device
        for value in values
    ):
        position = first.storage_offset()
        storage = first.untyped_storage().data_ptr()
        for value in values:
            if value.untyped_storage().data_ptr() != storage or value.storage_offset() != position:
                break
            position += value.numel()
        else:
            return first.as_strided(
                (sum(value.shape[0] for value in values), *first.shape[1:]), first.stride()
            )
    return torch_lib.cat(values, dim=0) if copy else None


def _fused_weight(weights, scale):
    """Form GEMM backing while retaining each logical branch's encoded scales."""

    values = tuple(weights.values())
    first = values[0]
    if any(value.dtype != first.dtype or value.device != first.device for value in values):
        return None
    if not any(isinstance(value, QuantizedTensor) for value in values):
        return _concatenate(values, copy=False)
    if not all(
        isinstance(value, QuantizedTensor) and value.quantizer.format == "fp8" for value in values
    ):
        return None
    # A per-tensor branch scale is broadcast over only that branch's channels.
    # This changes its physical scale view, never its original statistics.
    backing = _concatenate(tuple(value.buffers()["values"] for value in values), copy=False)
    if backing is None:
        return None
    fields = {"values": backing, "scale": scale}
    return Quantizer("fp8", axis=0).from_tensors(
        fields, shape=(sum(value.shape[0] for value in values), first.shape[1]), dtype=first.dtype
    )


class Backend:
    """A provider factory; each prepare call creates an independent operator."""

    operator_class: type[Operator]

    def workspace_buffers(
        self,
        weight: torch_lib.Tensor,
        *,
        input_dtype: torch_lib.dtype,
        input_quantizer: Quantizer | None,
        max_rows: int,
        output_dtype: torch_lib.dtype,
    ) -> Mapping[str, BufferConfig]:
        if weight.ndim != 2 or max_rows < 0:
            raise ValueError("matmul preparation requires a matrix and nonnegative row capacity")
        if input_quantizer is None:
            return {}
        # Meta conversion describes physical fields without allocating device
        # storage or conflating encoded bytes with the logical input dtype.
        encoded = input_quantizer.empty(
            (max_rows, weight.shape[1]), dtype=input_dtype, device="meta"
        )
        return {
            f"input.{name}": BufferConfig(tuple(value.shape), value.dtype)
            for name, value in encoded.buffers().items()
        }

    def prepare(
        self,
        weight: torch_lib.Tensor,
        *,
        input_dtype: torch_lib.dtype,
        input_quantizer: Quantizer | None,
        max_rows: int,
        output_dtype: torch_lib.dtype,
        workspace: Mapping[str, torch_lib.Tensor],
    ) -> Operator:
        return self.operator_class(
            weight,
            input_dtype=input_dtype,
            input_quantizer=input_quantizer,
            max_rows=max_rows,
            output_dtype=output_dtype,
            workspace=workspace,
        )

    def merged_workspace_buffers(
        self, weights, *, input_dtype, input_quantizer, max_rows, branch_width, output_dtype
    ):
        configs = {
            f"{name}.{field}": config
            for name, weight in weights.items()
            for field, config in self.workspace_buffers(
                weight,
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                output_dtype=output_dtype,
            ).items()
        }
        configs["output"] = BufferConfig(
            (max_rows, sum(weight.shape[0] for weight in weights.values())), output_dtype
        )
        configs["bias"] = BufferConfig(
            (sum(weight.shape[0] for weight in weights.values()),), input_dtype
        )
        if all(
            isinstance(value, QuantizedTensor) and value.quantizer.format == "fp8"
            for value in weights.values()
        ):
            configs["weight.scale"] = BufferConfig(
                (sum(weight.shape[0] for weight in weights.values()), 1), torch_lib.float32
            )
        return configs

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
        if branch_width is not None and (
            type(branch_width) is not int
            or branch_width < 1
            or any(weight.shape[0] % branch_width for weight in weights.values())
        ):
            raise ValueError(
                "interleaved branches require a positive group width dividing each branch"
            )
        operators = {
            name: self.prepare(
                weight,
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                output_dtype=output_dtype,
                workspace={
                    field.removeprefix(f"{name}."): value
                    for field, value in workspace.items()
                    if field.startswith(f"{name}.")
                },
            )
            for name, weight in weights.items()
        }
        fused_weight = _fused_weight(weights, workspace.get("weight.scale"))
        fused = (
            None
            if fused_weight is None
            else self.prepare(
                fused_weight,
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                output_dtype=output_dtype,
                workspace=next(iter(operators.values())).workspace,
            )
        )
        return MergedOperator(
            operators,
            fused=fused,
            output=workspace["output"],
            bias=workspace["bias"],
            widths=tuple(weight.shape[0] for weight in weights.values()),
        )


def resolve(backend: str | Backend, weight: torch_lib.Tensor) -> Backend:
    if isinstance(backend, Backend):
        return backend
    if backend == "auto":
        if isinstance(weight, QuantizedTensor) and weight.quantizer.format in {"mxfp8", "nvfp4"}:
            backend = "flashinfer"
        else:
            backend = "cublas" if weight.is_cuda else "torch"
    if backend not in {"torch", "cublas", "flashinfer"}:
        raise ValueError(f"unknown matmul backend {backend!r}")
    return import_module(f"{__name__}.{backend}").Backend()
