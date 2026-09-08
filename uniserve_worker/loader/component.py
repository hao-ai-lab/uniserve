"""Architecture declarations consumed by the shared checkpoint loader."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..nn.diffusion.schedule import DiffusionSchedule
from ..nn.layer import LayerConfig
from ..nn.quant import QuantizationConfig
from ..nn.quant.config import LinearPrecision
from .config import LoadRequest
from .handles import WeightHandle
from .mapping import LoadReport, WeightNameMap
from .source import WeightSourceSet


@dataclass(frozen=True, slots=True)
class CheckpointComponent:
    """One source namespace and its resident tensor ownership.

    Mapping may select retained layers or assign packed shards. Postprocessing
    receives durable checkpoint handles for mathematical precomputation; file
    discovery, integrity, assignment policy, auditing and quantization belong to
    the loader. Names in reports are relative to ``module``.
    """

    module: nn.Module
    source: str = "primary"
    map_weights: Callable[[Iterable[WeightHandle]], LoadReport] | None = None
    weight_name_map: WeightNameMap = ()
    included: frozenset[str] | None = None
    optional: frozenset[str] = frozenset()
    dtype: torch.dtype | None = None
    parameter_dtypes: tuple[tuple[str, torch.dtype], ...] = ()
    module_devices: tuple[tuple[str, torch.device], ...] = ()
    persistent_buffers: bool = False
    strict: bool = True
    post_load: Callable[[Mapping[str, WeightHandle]], None] | None = None

    def __post_init__(self) -> None:
        if self.map_weights is not None and self.weight_name_map:
            raise ValueError("a checkpoint component must declare one weight-mapping policy")

    def device_for(self, path: str, default: str | torch.device) -> torch.device:
        """Resolve the most specific declared module device for a tensor or subtree."""

        matches = (
            (len(prefix), device)
            for prefix, device in self.module_devices
            if not prefix or path == prefix or path.startswith(prefix + ".")
        )
        return max(matches, key=lambda item: item[0], default=(-1, torch.device(default)))[1]


@dataclass(frozen=True, slots=True)
class ModelBuildContext:
    """Verified sources and common construction policy supplied to an architecture."""

    root: Path
    sources: tuple[WeightSourceSet, ...]
    request: LoadRequest
    quantization: QuantizationConfig | None
    component_precisions: Mapping[str, LinearPrecision]
    schedule: DiffusionSchedule | None

    def packed_decoder_layers(self, entry: str) -> LayerConfig:
        """Bind the TP layers of a packed decoder to its configured computation entry.

        Packed KV execution currently owns a complete local sequence and layer
        stack. A component needing SP or PP must supply its corresponding
        attention and layer-pipeline execution instead of this packed binding.
        """

        mesh = self.request.bindings.meshes.get(entry)
        if mesh is None:
            raise ValueError(f"packed decoder entry {entry!r} is not assigned to this rank")
        if mesh.size("sp") != 1 or mesh.size("pp") != 1:
            raise ValueError("packed decoding requires local sequence and pipeline axes")
        return LayerConfig(mesh.get_group("tp"), self.quantization)


@dataclass(frozen=True, slots=True)
class ModelConstruction:
    """Components to materialize followed by assembly of the execution model."""

    components: tuple[CheckpointComponent, ...]
    assemble: Callable[[], nn.Module]
    config: Any
    tokenizer: Any | None = None


@contextmanager
def construction_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Set floating-point defaults while constructing a component, restoring them on exit."""

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
