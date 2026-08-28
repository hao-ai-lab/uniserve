"""Immutable construction inputs shared by quantizable neural layers."""

from __future__ import annotations

from dataclasses import dataclass

from .mesh import TensorParallel
from .quant import QuantizationConfig, QuantizeMethodBase, UnquantizedLinearMethod

__all__ = ["LayerConfig"]


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """Transport-free parallel and checkpoint quantization declarations."""

    parallel: TensorParallel
    quantization: QuantizationConfig | None

    def quant_method(self, prefix: str) -> QuantizeMethodBase:
        if self.quantization is None:
            return UnquantizedLinearMethod()
        return self.quantization.get_quant_method(prefix)
