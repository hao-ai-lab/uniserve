"""Immutable construction inputs shared by quantizable neural layers."""

from __future__ import annotations

from dataclasses import dataclass

from .mesh import TensorParallelSpec
from .quant import QuantizationConfig, QuantizeMethodBase, UnquantizedLinearMethod

__all__ = ["LayerSpec"]


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """Transport-free parallel and checkpoint quantization declarations."""

    parallel: TensorParallelSpec
    quantization: QuantizationConfig | None

    def quant_method(self, prefix: str) -> QuantizeMethodBase:
        if self.quantization is None:
            return UnquantizedLinearMethod()
        return self.quantization.get_quant_method(prefix)
