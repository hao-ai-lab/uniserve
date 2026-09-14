"""Immutable construction inputs shared by quantizable neural layers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import cast

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.nn.quant import LinearMethod, QuantizationConfig, UnquantizedLinearMethod
from uniserve.nn.quant.config import LinearPrecision

__all__ = ["LayerConfig"]


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """Checkpoint shard coordinates and corresponding construction-time group binding."""

    communicator: Communicator
    quantization: QuantizationConfig | None
    prefix: str = ""
    pipeline: Communicator = field(default_factory=Communicator)
    sequence: Communicator = field(default_factory=Communicator)

    dense_dtype: torch.dtype = torch.bfloat16

    @property
    def linear_precision(self) -> LinearPrecision:
        """Resolve the representation required by native projection backends."""

        if self.quantization is not None and self.quantization.method != "unquantized":
            method = self.quantization.method
            if method not in {"fp8", "mxfp8", "nvfp4"}:
                raise ValueError(f"unsupported native linear precision {method!r}")
            return cast(LinearPrecision, method)
        if self.dense_dtype == torch.float16:
            return "fp16"
        if self.dense_dtype == torch.bfloat16:
            return "bf16"
        raise ValueError("native dense projection precision must be FP16 or BF16")

    def qualify(self, name: str) -> str:
        """Resolve a child name in this component's checkpoint namespace."""

        return ".".join(part for part in (self.prefix, name) if part)

    def child(self, name: str) -> LayerConfig:
        """Keep geometry and precision policy while descending into a module."""

        return replace(self, prefix=self.qualify(name))

    def quant_method(self, prefix: str, *, packed_names: tuple[str, ...] = ()) -> LinearMethod:
        """Resolve a parameter prefix to its configured quantized linear implementation."""

        if self.quantization is None:
            return UnquantizedLinearMethod()
        full_name = self.qualify(prefix)
        parent, _, _ = full_name.rpartition(".")
        packed = tuple(".".join(part for part in (parent, name) if part) for name in packed_names)
        return self.quantization.get_quant_method(full_name, packed_prefixes=packed)
