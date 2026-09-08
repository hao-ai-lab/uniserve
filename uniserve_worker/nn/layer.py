"""Immutable construction inputs shared by quantizable neural layers."""

from __future__ import annotations

from dataclasses import dataclass

from .mesh import Communicator
from .quant import LinearMethod, QuantizationConfig, UnquantizedLinearMethod

__all__ = ["LayerConfig"]


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """Checkpoint shard coordinates and corresponding construction-time group binding."""

    communicator: Communicator
    quantization: QuantizationConfig | None
    prefix: str = ""

    def qualify(self, name: str) -> str:
        """Resolve a child name in this component's checkpoint namespace."""

        return ".".join(part for part in (self.prefix, name) if part)

    def child(self, name: str) -> LayerConfig:
        """Keep geometry and precision policy while descending into a module."""

        return LayerConfig(self.communicator, self.quantization, self.qualify(name))

    def quant_method(self, prefix: str, *, packed_names: tuple[str, ...] = ()) -> LinearMethod:
        """Resolve a parameter prefix to its configured quantized linear implementation."""

        if self.quantization is None:
            return UnquantizedLinearMethod()
        full_name = self.qualify(prefix)
        parent, _, _ = full_name.rpartition(".")
        packed = tuple(".".join(part for part in (parent, name) if part) for name in packed_names)
        return self.quantization.get_quant_method(full_name, packed_prefixes=packed)
