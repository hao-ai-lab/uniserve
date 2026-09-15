"""Public construction and weight loading over ordinary numerical mappings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.nn.attention import AttentionParallelConfig

from . import checkpoint as checkpoint_module
from . import weights as weight_options
from .config import Config

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT", bound=nn.Module)


@dataclass(frozen=True, slots=True)
class Result(Generic[ModelT]):
    """A materialized model, its resolved sources and completeness reports."""

    model: ModelT
    sources: tuple[checkpoint_module.Source, ...]
    reports: tuple[weight_options.Report, ...]


def _load(
    model,
    checkpoint,
    mappings,
    *,
    device,
    weights,
    io,
    devices,
    excluded=(),
    selected=None,
    known_paths=None,
):
    loader = weight_options._Loader(
        model,
        checkpoint,
        mappings,
        device=device,
        weights=weights,
        io=io,
        devices=devices,
        excluded=excluded,
        selected=selected,
        known_paths=known_paths,
    )
    try:
        with torch.no_grad():
            return loader.load()
    finally:
        loader.close()


def load_weights(
    model: ModelT,
    checkpoint: tuple[checkpoint_module.Source, ...],
    *,
    mapping: Callable[[ModelT], tuple[weight_options.ModuleMapping, ...]],
    device: torch.device | str,
    weights: weight_options.Config = weight_options.Config(),
    io: Config = Config(),
    devices: Mapping[str, torch.device | str] | None = None,
) -> tuple[weight_options.Report, ...]:
    """Load an existing module via the construction-time assignment path.

    The same assignment and conversion path applies as in construction.
    Complete-source matrix assignments follow already bound mathematical
    partitions. Readers retire before return; Parameters retain their storage
    and shared identities independently of checkpoint file handles.
    """
    known_paths = frozenset(
        path for path, _ in model.named_modules(remove_duplicate=False)
    )
    return _load(
        model,
        checkpoint,
        mapping(model),
        device=device,
        weights=weights,
        io=io,
        devices=devices,
        known_paths=known_paths,
    )


def load_model(
    model_class: Callable[[ConfigT], ModelT],
    config: ConfigT,
    *,
    checkpoint: tuple[checkpoint_module.Source, ...],
    mapping: Callable[[ModelT], tuple[weight_options.ModuleMapping, ...]],
    device: torch.device | str,
    weights: weight_options.Config = weight_options.Config(),
    io: Config = Config(),
    meshes: Mapping[str, DeviceMesh] | None = None,
    attention: Mapping[str, AttentionParallelConfig] | None = None,
    devices: Mapping[str, torch.device | str] | None = None,
    modules: frozenset[str] | None = None,
) -> Result[ModelT]:
    """Construct on meta, bind partitions, then materialize selected modules.

    The bound partitions are mathematical. Module paths choose resources and
    numerical settings. Unselected modules remain on meta so architecture
    dimensions and layout queries stay available. Neither constructors nor
    the resulting model retain loading state.
    """
    with torch.device("meta"):
        model = model_class(config)
    if not isinstance(model, nn.Module):
        raise TypeError("model constructor must return torch.nn.Module")

    known_paths = frozenset(
        path for path, _ in model.named_modules(remove_duplicate=False)
    )
    selected = {
        id(child)
        for path in (("",) if modules is None else modules)
        for child in model.get_submodule(path).modules()
    }

    meshes = {} if meshes is None else meshes
    attention = {} if attention is None else attention
    if set(attention).difference(meshes):
        raise ValueError(
            "attention parallel settings require a mesh at the same module path"
        )

    # A mesh this rank does not participate in makes its whole subtree remote:
    # those modules stay on meta and are never materialized here.
    remote = {
        id(child)
        for path, mesh in meshes.items()
        if mesh.rank not in mesh.ranks
        for child in model.get_submodule(path).modules()
    }
    local = {
        id(child)
        for path, mesh in meshes.items()
        if mesh.rank in mesh.ranks
        for child in model.get_submodule(path).modules()
    }
    if remote.intersection(local):
        raise ValueError(
            "shared numerical modules cannot have conflicting mesh "
            "participation"
        )
    selected.difference_update(remote)
    for path, mesh in meshes.items():
        child = model.get_submodule(path)
        if mesh.rank in mesh.ranks:
            parallelize_(
                child,
                mesh,
                attention=attention.get(path, AttentionParallelConfig()),
            )

    declared = mapping(model)
    parameters = {
        id(parameter)
        for child in model.modules()
        if id(child) in selected
        for parameter in child.parameters(recurse=False)
    }

    # Narrow each declared mapping to the selected modules: unselected
    # parameters keep their meta placeholders and their assignments drop out.
    mapped = []
    for component in declared:
        names = {
            name
            for name, parameter in component.module.named_parameters(
                remove_duplicate=False
            )
            if id(parameter) in parameters
        }
        retained = names.intersection(component.required | component.optional)
        if not retained and id(component.module) not in selected:
            continue
        if component.post_load is not None and not component.required.issubset(
            names
        ):
            raise ValueError(
                "checkpoint-derived modules must load their complete "
                "numerical inputs"
            )

        def assign(reader, mapping=component.map_weights):
            return tuple(
                value
                for value in mapping(reader)
                if id(value.target) in parameters
            )

        mapped.append(
            replace(
                component,
                map_weights=assign,
                required=component.required & names,
                optional=component.optional & names,
            )
        )

    reports = _load(
        model,
        checkpoint,
        mapped,
        device=device,
        weights=weights,
        io=io,
        devices=devices,
        excluded=declared,
        selected=selected,
        known_paths=known_paths,
    )
    model.eval()
    return Result(model, checkpoint, reports)
