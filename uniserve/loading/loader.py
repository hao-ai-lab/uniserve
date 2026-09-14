"""Shared component construction, checkpoint assignment and finalization."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import torch
from torch import nn

from uniserve.distributed.mesh import DeviceMesh
from uniserve.distributed.parallel import ParallelConfig
from uniserve.loading.component import CheckpointComponent, construction_dtype
from uniserve.loading.config import LoadConfig, LoadFormat
from uniserve.loading.handles import TensorWeightHandle, WeightHandle, weight_handle_materialization
from uniserve.loading.io import iter_weight_handles
from uniserve.loading.mapping import LoadReport, stacked_weight_name
from uniserve.loading.source import WeightSourceSet
from uniserve.loading.validation import validate_loaded_weights
from uniserve.loading.weight_loaders import (
    WeightAssignment,
    attach_parameter_loaders,
    default_weight_loader,
    defer_parameter_weights,
    load_parameter_weight,
)
from uniserve.model.limits import ModelLimits
from uniserve.model.model import Model
from uniserve.nn.layer import LayerConfig
from uniserve.nn.quant.base import process_quantized_modules
from uniserve.runtime.branches import bind_branches, branch_device

logger = logging.getLogger(__name__)

__all__ = ["LoadedModel", "load_model"]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Materialized numerical model, checkpoint files and validated weight results."""

    model: Model
    sources: tuple[WeightSourceSet, ...]
    reports: tuple[LoadReport, ...]


def load_model(
    model_class: type[Model],
    config: Any,
    *,
    sources: tuple[WeightSourceSet, ...],
    load: LoadConfig = LoadConfig(),
    device: torch.device | str,
    dtype: torch.dtype,
    parallel: Mapping[str, ParallelConfig],
    meshes: Mapping[str, DeviceMesh],
    layers: Mapping[str, LayerConfig],
    limits: ModelLimits,
    flow_device: torch.device | str | None = None,
) -> LoadedModel:
    """Construct and load a numerical composition from resolved checkpoint files.

    Config is the model's typed architecture configuration. The caller binds
    resident meshes and numerical layers and retains their resource owners.
    ``flow_device`` places explicitly declared flow branches on a second device;
    branch delivery returns results to each invocation's input device.
    Missing weights, inconsistent shards and checksum failures reject the load.
    """

    model_class.validate_parallel(config, parallel)
    numerical_device = torch.device(device)
    capability = model_class.minimum_cuda_capability
    if capability is not None and (
        numerical_device.type != "cuda"
        or torch.cuda.get_device_capability(numerical_device) < capability
    ):
        raise ValueError(
            f"{model_class.__name__} requires CUDA compute capability {capability[0]}.{capability[1]}"
        )
    device = str(numerical_device)
    flow_device = None if flow_device is None else str(torch.device(flow_device))
    verify_checksums(sources, load.checksum_manifest)
    for layer in layers.values():
        if layer.quantization is not None:
            layer.quantization.validate_device(device, dtype)
    construction_device = "meta" if load.load_format is LoadFormat.LAYERED else device
    with construction_dtype(dtype), torch.device(construction_device):
        model = model_class(config, parallel=parallel, meshes=meshes, layers=layers, limits=limits)
    checkpoints = model.checkpoint_components()
    by_name = {source.source_name: source for source in sources}
    destinations: dict[int, torch.device] = {}
    for component in checkpoints:
        for path, _ in component.parameter_dtypes:
            component.module.get_submodule(path)
        # Aliased modules participate at every logical path. A shared
        # parameter cannot be materialized on conflicting branch devices.
        for path, owner in component.module.named_modules(remove_duplicate=False):
            destination = branch_device(
                component.module,
                path,
                device=device,
                flow_device=flow_device,
            )
            for parameter in owner.parameters(recurse=False):
                assigned = destinations.setdefault(id(parameter), destination)
                if assigned != destination:
                    raise ValueError("a shared parameter cannot belong to different devices")
            matching_dtypes = (
                (len(prefix), value)
                for prefix, value in component.parameter_dtypes
                if not prefix or path == prefix or path.startswith(prefix + ".")
            )
            parameter_dtype = max(
                matching_dtypes,
                key=lambda item: item[0],
                default=(-1, component.dtype or dtype),
            )[1]
            attach_parameter_loaders(
                owner,
                device=destination,
                dtype=parameter_dtype,
                recurse=False,
            )
            for name, buffer in owner.named_buffers(recurse=False):
                if not buffer.is_meta and buffer.device != destination:
                    owner._buffers[name] = buffer.to(destination)
    reports = tuple(
        load_component(
            component, by_name[component.source], load=load, device=device, flow_device=flow_device
        )
        for component in checkpoints
    )
    # Numerical buffers outside serialized component roots (for example
    # decoder normalization statistics) follow the complete module graph.
    for path, owner in model.named_modules():
        destination = branch_device(
            model,
            path,
            device=device,
            flow_device=flow_device,
        )
        for name, buffer in owner.named_buffers(recurse=False):
            if not buffer.is_meta and buffer.device != destination:
                owner._buffers[name] = buffer.to(destination)
    bind_branches(
        model,
        device=device,
        flow_device=flow_device,
    )
    model.eval()
    return LoadedModel(model=model, sources=sources, reports=reports)


def component_parameter_names(component: CheckpointComponent) -> set[str]:
    """Return the component's installed parameter scope in component coordinates."""

    return (
        {name for name, _ in component.module.named_parameters()}
        if component.included is None
        else set(component.included)
    )


def audit_component(component: CheckpointComponent, report: LoadReport) -> None:
    """Require all resident parameters and declared persistent buffers to be loaded."""

    included = component_parameter_names(component)
    included.update(_persistent_buffers(component.module))
    validate_loaded_weights(
        component.module,
        report,
        included=included,
        optional=component.optional,
        label=f"{component.source} checkpoint",
    )


def load_component(
    component: CheckpointComponent,
    source: WeightSourceSet,
    *,
    load: LoadConfig,
    device: str,
    flow_device: str | None = None,
) -> LoadReport:
    """Assign, audit and finalize one component using the caller's I/O policy."""

    if load.load_format is LoadFormat.DUMMY:
        if component.post_load is not None:
            raise ValueError("synthetic loading cannot supply checkpoint-dependent precomputation")
        report = _load_dummy(
            component,
            device=device,
            flow_device=flow_device,
        )
    else:
        handles: Iterable[WeightHandle] = iter_weight_handles(
            source, load, durable=component.post_load is not None
        )
        retained = None
        if component.post_load is not None:
            retained = {handle.name: handle for handle in handles}
            handles = retained.values()
        report = assign_component(
            component,
            handles,
            device=device,
            flow_device=flow_device,
            layered=load.load_format is LoadFormat.LAYERED,
        )
        if component.post_load is not None:
            assert retained is not None
            with torch.inference_mode(), weight_handle_materialization():
                component.post_load(retained)
    _materialize_scope_buffers(component, report.loaded, device, flow_device)
    _warn_skips(component.source, report)
    if load.load_format is not LoadFormat.LAYERED:
        _process_loaded_quantization(component.module, report.loaded)
    return report


def assign_component(
    component: CheckpointComponent,
    handles: Iterable[WeightHandle],
    *,
    device: str,
    flow_device: str | None = None,
    layered: bool = False,
) -> LoadReport:
    """Map and audit checkpoint values; layered mode materializes one owner at a time."""

    def assign() -> LoadReport:
        if component.map_weights is None:
            return _load_declared_weights(component, handles, device, flow_device)
        return component.map_weights(handles)

    if layered:
        with defer_parameter_weights() as assignments:
            report = assign()
        _materialize_parameter_weights(component.module, assignments)
    else:
        with torch.no_grad(), weight_handle_materialization():
            report = assign()
    audit_component(component, report)
    return report


def _persistent_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    """Select buffers that belong to the serialized state, excluding derived workspaces."""

    names = module.state_dict().keys()
    return {name: value for name, value in module.named_buffers() if name in names}


def _load_declared_weights(
    component: CheckpointComponent,
    handles: Iterable[WeightHandle],
    device: str,
    flow_device: str | None,
) -> LoadReport:
    parameters = dict(component.module.named_parameters())
    included = component_parameter_names(component)
    buffers = _persistent_buffers(component.module)
    report = LoadReport()
    for handle in handles:
        name, shard = stacked_weight_name(handle.name, component.weight_name_map)
        if name not in parameters and handle.name in parameters:
            name, shard = handle.name, None
        if name in parameters:
            if name not in included:
                continue
            load_parameter_weight(parameters[name], handle, shard)
        elif name in buffers:
            target = buffers[name]
            if target.shape != torch.Size(handle.shape):
                raise ValueError(f"checkpoint buffer {name!r} has incompatible shape")
            value = handle.full().to(
                device=branch_device(
                    component.module,
                    name.rpartition(".")[0],
                    device=device,
                    flow_device=flow_device,
                ),
                dtype=target.dtype,
            )
            path, _, field = name.rpartition(".")
            owner = component.module.get_submodule(path)
            owner.register_buffer(field, value, persistent=True)
        elif name in component.nonresident:
            report.skipped.append(handle.name)
            continue
        else:
            report.unexpected.append(name)
            continue
        report.loaded.add(name)
    return report


def _load_dummy(
    component: CheckpointComponent, *, device: str, flow_device: str | None
) -> LoadReport:
    included = component_parameter_names(component)
    loaded: set[str] = set()
    with torch.no_grad():
        for index, name in enumerate(sorted(included), start=1):
            current = dict(component.module.named_parameters()).get(name)
            if current is None:
                continue
            generator = torch.Generator(device="cpu").manual_seed(index)
            value = torch.empty(tuple(current.shape), dtype=current.dtype, device="cpu")
            if current.is_floating_point():
                value.normal_(mean=0.0, std=0.02, generator=generator)
            else:
                value.zero_()
            default_weight_loader(current, TensorWeightHandle(name, value))
            loaded.add(name)
        for index, (name, buffer) in enumerate(
            sorted(_persistent_buffers(component.module).items()), start=len(included) + 1
        ):
            generator = torch.Generator(device="cpu").manual_seed(index)
            value = torch.empty(tuple(buffer.shape), dtype=buffer.dtype, device="cpu")
            if buffer.is_floating_point():
                value.normal_(mean=0.0, std=0.02, generator=generator)
            else:
                value.zero_()
            path, _, field = name.rpartition(".")
            destination = branch_device(
                component.module, path, device=device, flow_device=flow_device
            )
            component.module.get_submodule(path).register_buffer(
                field, value.to(destination), persistent=True
            )
            loaded.add(name)
    _zero_dummy_vocab_padding(component.module, loaded)
    return LoadReport(loaded=loaded)


def _materialize_parameter_weights(
    model: nn.Module,
    assignments: list[WeightAssignment],
) -> None:
    """Group deferred assignments by owner and materialize one module subtree at a time.

    The scoped handle cache keeps each shard open only while its owner's parameters and
    quantization-derived tensors are materialized.
    """

    # Index direct parameter ownership without duplicating tied parameter identities.
    owners: dict[int, tuple[str, nn.Module]] = {}
    ordered_modules = tuple(model.named_modules())
    for module_name, module in ordered_modules:
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), (module_name, module))

    # Preserve checkpoint assignment order within each owning module.
    grouped: dict[str, list[WeightAssignment]] = defaultdict(list)
    for assignment in assignments:
        try:
            owner_name, _owner = owners[id(assignment.parameter)]
        except KeyError as error:
            raise RuntimeError("deferred checkpoint assignment has no model owner") from error
        grouped[owner_name].append(assignment)

    # Finalize quantization while the just-loaded module is the active memory unit.
    for module_name, module in ordered_modules:
        unit = grouped.get(module_name)
        if not unit:
            continue
        with torch.no_grad(), weight_handle_materialization():
            for assignment in unit:
                assignment.apply()
        process_quantized_modules((module,))


def verify_checksums(
    sources: tuple[WeightSourceSet, ...],
    manifest_location: str | None,
) -> None:
    """Verify every resolved weight file against a configured SHA-256 manifest."""

    if manifest_location is None:
        return

    # The manifest may be a local worker_config artifact or an explicit remote URI.
    if manifest_location.startswith(("http://", "https://")):
        with urlopen(manifest_location) as response:  # noqa: S310 - explicit configured URI
            payload = response.read().decode("utf-8")
    else:
        payload = Path(manifest_location).read_text(encoding="utf-8")
    value = json.loads(payload)
    if isinstance(value, dict) and isinstance(value.get("files"), dict):
        value = value["files"]
    if not isinstance(value, dict):
        raise TypeError("checksum manifest must contain a relative-path mapping")

    # Source-relative names provide stable keys across local and hub-cache roots.
    for source in sources:
        for relative, path in zip(source.relative_paths, source.weight_files):
            expected = value.get(relative)
            if not isinstance(expected, str):
                raise ValueError(f"checksum manifest has no entry for {relative!r}")
            actual = _file_sha256(path)
            if actual.lower() != expected.lower().removeprefix("sha256:"):
                raise ValueError(f"checksum mismatch for checkpoint file {relative!r}")


def _file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file using bounded read buffers."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _warn_skips(architecture: str, report: LoadReport) -> None:
    """Report checkpoint tensors the architecture intentionally left outside its load set."""

    if report.skipped:
        logger.warning(
            "%s declared %d checkpoint tensors outside its load set",
            architecture,
            len(report.skipped),
        )


def _process_loaded_quantization(model: nn.Module, loaded: set[str]) -> None:
    """Finalize quantized modules whose parameter subtree received checkpoint data."""

    selected: list[nn.Module] = []
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if any(name.startswith(prefix) for name in loaded):
            selected.append(module)
    process_quantized_modules(selected)


def _materialize_scope_buffers(
    component: CheckpointComponent,
    loaded: set[str],
    device: str,
    flow_device: str | None,
) -> None:
    """Materialize load-dependent buffers for module branches activated by parameters."""

    # Expand loaded leaves into their owning module ancestry.
    active_modules: set[str] = set()
    for name in loaded:
        parts = name.split(".")[:-1]
        active_modules.update(".".join(parts[:end]) for end in range(1, len(parts) + 1))

    # Invoke only owners within the loaded scope so excluded meta branches stay deferred.
    for module_name, module in component.module.named_modules():
        parts = module_name.split(".")
        if not any(".".join(parts[:end]) in active_modules for end in range(1, len(parts) + 1)):
            continue
        materialize = getattr(module, "materialize_load_buffers", None)
        if callable(materialize):
            materialize(
                str(
                    branch_device(
                        component.module, module_name, device=device, flow_device=flow_device
                    )
                )
            )


def _zero_dummy_vocab_padding(model: nn.Module, loaded: set[str]) -> None:
    """Zero synthetic vocabulary rows outside each rank's real token interval."""

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in loaded:
                continue
            layout = getattr(parameter, "_uniserve_vocab_layout", None)
            if not isinstance(layout, tuple) or len(layout) != 3:
                continue
            real_size, start, _end = (int(value) for value in layout)
            padding_start = max(0, real_size - start)
            if padding_start < int(parameter.shape[0]):
                parameter[padding_start:].zero_()
