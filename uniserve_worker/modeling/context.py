"""Borrowed numerical construction inputs shared by library and serving callers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..nn.diffusion.schedule import DiffusionSchedule
from ..nn.layer import LayerConfig
from ..nn.mesh import DeviceMesh
from ..nn.parallel import ParallelConfig
from ..nn.quant.config import LinearPrecision


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Describe numerical topology, precision, limits, and solver constants.

    ``parallel`` includes nonresident components' mathematical geometry. Only
    locally executable components have borrowed ``meshes`` and bound ``layers``.
    The caller owns communication and the backing of schedule tensor views;
    their lifetime must cover model calls and outstanding result readers.
    """

    parallel: Mapping[str, ParallelConfig]
    meshes: Mapping[str, DeviceMesh]
    layers: Mapping[str, LayerConfig]
    limits: Mapping[str, int | float]
    component_precisions: Mapping[str, LinearPrecision]
    schedule: DiffusionSchedule | None

    def __post_init__(self) -> None:
        for name in ("parallel", "meshes", "layers", "limits", "component_precisions"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
