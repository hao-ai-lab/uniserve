"""Immutable construction inputs shared by quantizable neural layers."""

from __future__ import annotations

from dataclasses import dataclass

from .mesh import GroupCoordinator, TensorParallel
from .quant import QuantizationConfig, QuantizeMethodBase, UnquantizedLinearMethod

__all__ = ["LayerConfig"]


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """Checkpoint shard coordinates and corresponding construction-time group binding."""

    parallel: TensorParallel
    quantization: QuantizationConfig | None
    tp_group: GroupCoordinator | None = None

    def tensor_group(self) -> GroupCoordinator:
        """Require communication to agree with the parameter shard coordinates."""

        group = self.tp_group
        if group is None:
            if self.parallel.size != 1:
                raise ValueError("distributed layers require a construction-time tp_group")
            return GroupCoordinator()
        if (group.rank_in_group, group.world_size) != (self.parallel.rank, self.parallel.size):
            raise ValueError("tp_group membership disagrees with checkpoint shard coordinates")
        return group

    def quant_method(self, prefix: str) -> QuantizeMethodBase:
        """Resolve a parameter prefix to its configured quantized linear implementation."""

        if self.quantization is None:
            return UnquantizedLinearMethod()
        return self.quantization.get_quant_method(prefix)
