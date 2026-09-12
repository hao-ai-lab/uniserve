"""Architecture declarations consumed by the shared checkpoint loader."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

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
    nonresident: frozenset[str] = frozenset()
    dtype: torch.dtype | None = None
    parameter_dtypes: tuple[tuple[str, torch.dtype], ...] = ()
    module_devices: tuple[tuple[str, torch.device], ...] = ()
    buffer_pool: bool = False
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
        """Bind packed attention shards and resident layers to an execution entry."""

        mesh = self.request.bindings[entry].mesh if entry in self.request.bindings else None
        if mesh is None:
            raise ValueError(f"packed decoder entry {entry!r} is not assigned to this rank")
        if mesh.parallel_config.sequence_parallel.kind not in {"local", "ulysses"}:
            raise ValueError("paged attention sequence execution requires Ulysses head exchange")
        return LayerConfig(
            mesh.get_group("tp"),
            self.quantization,
            pipeline=mesh.get_group("pp"),
            sequence=mesh.get_group("ulysses"),
        )


@dataclass(frozen=True, slots=True)
class ModelConstruction:
    """Components to materialize followed by assembly of the execution model."""

    components: tuple[CheckpointComponent, ...]
    assemble: Callable[[], nn.Module]
    config: Any
    tokenizer: Any | None = None


_Module = TypeVar("_Module", bound=nn.Module)


def construct_owned_module(
    factory: Callable[[], _Module], *, resident: bool
) -> tuple[_Module | None, frozenset[str]]:
    """Construct owned weights or describe an off-stage checkpoint namespace.

    Off-stage construction uses metadata tensors exclusively. Its parameter
    names allow the loader to distinguish valid nonresident records from
    misspelled or unknown checkpoint weights, without retaining their modules.
    """

    if resident:
        return factory(), frozenset()
    with torch.device("meta"):
        module = factory()
    return None, frozenset(name for name, _ in module.named_parameters())


@contextmanager
def construction_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Set floating-point defaults while constructing a component, restoring them on exit."""

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
