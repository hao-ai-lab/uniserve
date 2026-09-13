"""Architecture declarations consumed by the shared checkpoint loader."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import nn

from .handles import WeightHandle
from .mapping import LoadReport, WeightNameMap


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
    post_load: Callable[[Mapping[str, WeightHandle]], None] | None = None

    def __post_init__(self) -> None:
        if self.map_weights is not None and self.weight_name_map:
            raise ValueError("a checkpoint component must declare one weight-mapping policy")


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
