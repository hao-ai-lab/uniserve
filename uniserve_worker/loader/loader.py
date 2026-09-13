"""Shared component construction, checkpoint assignment and finalization."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from urllib.request import urlopen

import torch
from torch import nn

from ..bootstrap.components import validate_components
from ..config import WorkerConfig
from ..execution.model_entry import ModelEntry
from ..foundation.errors import unsupported_setup
from ..modeling.context import BuildContext
from ..modeling.model import Model
from ..modeling.video import VideoMixin
from ..nn.diffusion.schedule import DiffusionSchedule
from ..nn.layer import LayerConfig
from ..nn.quant import QuantizationConfig
from ..nn.quant.base import process_quantized_modules
from ..runtime.branches import bind_branches, branch_device
from ..runtime.results import resolve_outputs
from ..runtime.tensors import media_calls, resolve_resources
from .audit import audit_load_report
from .component import CheckpointComponent, construction_dtype
from .config import LoadFormat, LoadRequest
from .handles import TensorWeightHandle, WeightHandle, weight_handle_materialization
from .io import iter_weight_handles
from .mapping import LoadReport, stacked_weight_name
from .source import (
    ModelSource,
    WeightSourceSet,
    resolve_weight_sources,
)
from .weight_loaders import (
    WeightAssignment,
    attach_parameter_loaders,
    default_weight_loader,
    defer_parameter_weights,
    load_parameter_weight,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..bootstrap.catalog import CatalogEntry

__all__ = ["ModelLoader", "LoadedModel", "get_model_loader", "load_model"]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Materialized model, tokenizer, sources, and resolved execution metadata."""

    model: Model
    bindings: Mapping[str, ModelEntry]
    tokenizer: Any | None
    worker_config: WorkerConfig
    sources: tuple[WeightSourceSet, ...]
    architecture_config: dict[str, Any]
    schedule: DiffusionSchedule | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))


class ModelLoader:
    """Load each declared component using one configured materialization strategy."""

    def __init__(self, load_format: LoadFormat) -> None:
        self.load_format = load_format

    def load(
        self,
        entry: CatalogEntry,
        config: dict[str, Any],
        request: LoadRequest,
        *,
        root: Path,
        repository_id: str | None,
    ) -> LoadedModel:
        """Verify the complete checkpoint before constructing or mutating its components."""

        if request.load.load_format is not self.load_format:
            raise ValueError("loader format disagrees with the load request")
        components = validate_components(
            entry.model_class,
            config,
            {name: value.config for name, value in request.bindings.items()},
        )
        entry.model_class.validate_parallel(
            config,
            {name: value.config.parallel_config for name, value in request.bindings.items()},
        )
        device = torch.device(request.execution.device)
        capability = entry.minimum_cuda_capability
        if capability is not None and (
            device.type != "cuda" or torch.cuda.get_device_capability(device) < capability
        ):
            raise unsupported_setup(
                f"{entry.architecture} requires CUDA compute capability {capability[0]}.{capability[1]}"
            )
        sources = resolve_weight_sources(
            request,
            sources=entry.sources,
            sidecars=entry.sidecars,
            root=root,
            repository_id=repository_id,
        )
        verify_checksums(sources, request.load.checksum_manifest)
        tokenizer = None
        if entry.tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                root, use_fast=False, trust_remote_code=False, local_files_only=True
            )
        dtype = _serving_dtype(request.execution.model_dtype)
        quantization = QuantizationConfig.from_model_config(
            config,
            overrides=request.quantization_config if entry.component_precisions is None else {},
        )
        if quantization is not None:
            quantization.validate_device(request.execution.device, dtype)
            config = {**config, "quantization_config": dict(quantization.raw)}
        prepared = (
            config if entry.prepare_config is None else entry.prepare_config(config, root, sources)
        )
        schedule = None if entry.create_schedule is None else entry.create_schedule(device)
        meshes = {
            name: binding.mesh
            for name, binding in request.bindings.items()
            if binding.mesh is not None
        }
        context = BuildContext(
            parallel={
                name: binding.config.parallel_config for name, binding in request.bindings.items()
            },
            meshes=meshes,
            layers={
                name: LayerConfig(
                    mesh.get_group("tp"),
                    quantization,
                    pipeline=mesh.get_group("pp"),
                    sequence=mesh.get_group("ulysses"),
                )
                for name, mesh in meshes.items()
            },
            limits={
                "text_tokens": request.max_text_rows,
                "video_seconds": request.max_video_seconds,
            },
            component_precisions=(
                {}
                if entry.component_precisions is None
                else entry.component_precisions(request.quantization_config)
            ),
            schedule=schedule,
        )
        if context.component_precisions:
            logger.info("resolved component precisions: %s", dict(context.component_precisions))
        construction_device = (
            "meta" if self.load_format is LoadFormat.LAYERED else request.execution.device
        )
        with construction_dtype(dtype), torch.device(construction_device):
            model = entry.model_class(prepared, context)
        if not isinstance(model, Model):
            raise unsupported_setup(f"{type(model).__name__} must implement Model")
        checkpoints = model.checkpoint_components()
        by_name = {source.source_name: source for source in sources}
        destinations: dict[int, torch.device] = {}
        for component in checkpoints:
            for path, _ in component.parameter_dtypes:
                component.module.get_submodule(path)
            # Aliased modules participate at every logical path. A shared
            # parameter cannot be materialized on conflicting branch devices.
            for path, owner in component.module.named_modules(remove_duplicate=False):
                device = branch_device(
                    component.module,
                    path,
                    device=request.execution.device,
                    flow_device=request.execution.generation_device,
                )
                for parameter in owner.parameters(recurse=False):
                    assigned = destinations.setdefault(id(parameter), device)
                    if assigned != device:
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
                    device=device,
                    dtype=parameter_dtype,
                    recurse=False,
                )
                for name, buffer in owner.named_buffers(recurse=False):
                    if not buffer.is_meta and buffer.device != device:
                        owner._buffers[name] = buffer.to(device)
        for component in checkpoints:
            load_component(component, by_name[component.source], request)
        # Numerical buffers outside serialized component roots (for example
        # decoder normalization statistics) follow the complete module graph.
        for path, owner in model.named_modules():
            destination = branch_device(
                model,
                path,
                device=request.execution.device,
                flow_device=request.execution.generation_device,
            )
            for name, buffer in owner.named_buffers(recurse=False):
                if not buffer.is_meta and buffer.device != destination:
                    owner._buffers[name] = buffer.to(destination)
        bind_branches(
            model,
            device=request.execution.device,
            flow_device=request.execution.generation_device,
        )
        model.eval()
        # Loading and execution retain the same placement and local mesh objects.
        bindings = MappingProxyType(dict(request.bindings))
        outputs = resolve_outputs(model)
        for name, binding in bindings.items():
            binding.output_schema = outputs.get(name, ())
            binding.calls = components[name].calls
        return LoadedModel(
            model=model,
            bindings=bindings,
            tokenizer=tokenizer,
            worker_config=request.execution,
            sources=sources,
            architecture_config=_canonical_architecture_config(prepared),
            schedule=schedule,
        )


def load_model(request: LoadRequest) -> LoadedModel:
    """Discover and load a checkpoint, including its input-token declarations."""

    return _load_model(request, ModelSource.resolve(request.model_path, request.load))


def _load_model(request: LoadRequest, source: ModelSource) -> LoadedModel:
    """Load through one resolved metadata identity for serving and direct Python calls."""

    loaded = get_model_loader(request.load.load_format).load(
        source.entry,
        source.config,
        request,
        root=source.root,
        repository_id=source.repository_id,
    )
    loaded = replace(loaded, worker_config=_loaded_worker_config(loaded.model, request))
    _resolve_input_tokens(loaded.model, loaded.tokenizer)
    logger.info(
        "loaded model architecture=%s",
        loaded.model.architecture,
    )
    return loaded


def _loaded_worker_config(model: Model, request: LoadRequest) -> WorkerConfig:
    """Close requested execution bounds over the materialized computation geometry."""

    config = request.execution
    if (
        isinstance(model, VideoMixin)
        or resolve_resources(model, media_calls(model, request.bindings)).state
    ):
        if request.pipeline_depth is None:
            raise unsupported_setup("request tensor storage requires its physical pipeline depth")
        # Two unresolved outputs and one further physical position permit
        # publication retirement before a resident request slot is reused.
        state_slots = min(config.max_batch_operations, request.pipeline_depth // 3)
        if state_slots < 2:
            raise unsupported_setup(
                "resident media execution requires two slots with two unresolved outputs each"
            )
        config = replace(
            config,
            kv_token_capacity=None,
            attention_backend=None,
            max_batch_operations=state_slots,
            max_batch_tokens=state_slots,
            max_request_pool_size=state_slots,
            min_request_pool_size=2,
            generation_device=None,
        )
    return config


def _resolve_input_tokens(model: Model, tokenizer: Any | None) -> None:
    """Resolve model-specific input token identities from tokenizer metadata."""

    processor = model.image_processor
    injection = getattr(processor, "feature_injection", None)
    if processor is None or injection is None:
        return
    updates: dict[str, int] = {}
    for token_field, id_field in (("start_token", "start_token_id"), ("end_token", "end_token_id")):
        token = getattr(injection, token_field)
        token_id = getattr(injection, id_field)
        if token_id is not None or token is None:
            continue
        if tokenizer is None:
            raise unsupported_setup(
                f"model input declaration requires tokenizer resolution for {token!r}"
            )
        resolved = tokenizer.convert_tokens_to_ids(token)
        if resolved is None or int(resolved) < 0:
            raise unsupported_setup(f"tokenizer does not define declared token {token!r}")
        updates[id_field] = int(resolved)
    if not updates:
        return
    resolved_injection = replace(
        injection,
        start_token_id=updates.get("start_token_id", injection.start_token_id),
        end_token_id=updates.get("end_token_id", injection.end_token_id),
    )
    model.image_processor = replace(processor, feature_injection=resolved_injection)


def get_model_loader(load_format: LoadFormat | str) -> ModelLoader:
    """Select checkpoint materialization without architecture-specific loader branches."""

    return ModelLoader(LoadFormat(str(load_format)))


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
    audit_load_report(
        component.module,
        report,
        included=included,
        optional=component.optional,
        label=f"{component.source} checkpoint",
    )


def load_component(
    component: CheckpointComponent, source: WeightSourceSet, request: LoadRequest
) -> LoadReport:
    """Assign, audit and finalize one component using the caller's I/O policy."""

    if request.load.load_format is LoadFormat.DUMMY:
        if component.post_load is not None:
            raise ValueError("synthetic loading cannot supply checkpoint-dependent precomputation")
        report = _load_dummy(
            component,
            device=request.execution.device,
            flow_device=request.execution.generation_device,
        )
    else:
        handles: Iterable[WeightHandle] = iter_weight_handles(
            source, request.load, durable=component.post_load is not None
        )
        retained = None
        if component.post_load is not None:
            retained = {handle.name: handle for handle in handles}
            handles = retained.values()
        report = assign_component(
            component,
            handles,
            device=request.execution.device,
            flow_device=request.execution.generation_device,
            layered=request.load.load_format is LoadFormat.LAYERED,
        )
        if component.post_load is not None:
            assert retained is not None
            with torch.inference_mode(), weight_handle_materialization():
                component.post_load(retained)
    _materialize_scope_buffers(
        component, report.loaded, request.execution.device, request.execution.generation_device
    )
    _warn_skips(component.source, report)
    if request.load.load_format is not LoadFormat.LAYERED:
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


def _canonical_architecture_config(config: Any) -> dict[str, Any]:
    """Serialize a typed or mapping configuration into an independent dictionary."""

    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(config):
        value = asdict(cast(Any, config))
    else:
        raise TypeError("load-time architecture configuration is not serializable")
    if not isinstance(value, dict):
        raise TypeError("load-time architecture configuration must serialize to an object")
    return value


def _serving_dtype(name: str) -> torch.dtype:
    """Map the execution model-dtype name to its PyTorch dtype."""

    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[str(name)]
    except KeyError as error:
        raise ValueError(
            f"unknown model dtype {name!r}; expected bfloat16, float16, or float32"
        ) from error
