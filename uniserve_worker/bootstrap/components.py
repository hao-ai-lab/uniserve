"""Bind public numerical capabilities to the worker's logical IPC entries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from importlib import import_module

from torch import nn

from uniserve.distributed import DeviceMesh
from uniserve.model import (
    AudioDecoder,
    CausalLM,
    ComponentEntry,
    Denoiser,
    Encoder,
    ImageDecoder,
    PatchEncoder,
    TextEncoder,
    VideoDecoder,
    VideoPostprocessor,
)
from uniserve.nn.vae import PatchAutoencoder

from ..execution.model_entry import Call, ModelEntry
from ..foundation.errors import unsupported_setup
from ..protocol.operation import (
    Computation,
    ForwardMode,
    PipelineStage,
    TransferMode,
)
from .config import ComponentConfig


def call_operations(calls: Iterable[Call]) -> frozenset[Computation]:
    """Resolve computation from capability type and method.

    The computation follows the declared numerical method.
    """
    operations: set[Computation] = set()
    for call in calls:
        module, method = call.module, call.entry.method
        if isinstance(module, CausalLM) and method == "forward":
            operations.update(
                (ForwardMode.PREFILL, ForwardMode.DECODE, ForwardMode.VERIFY)
            )
        elif isinstance(module, Denoiser) and method == "forward":
            operations.update(
                (PipelineStage.LATENT_PREPARATION, PipelineStage.DENOISING)
            )
        elif isinstance(module, VideoPostprocessor) and method == "forward":
            operations.update(
                (
                    PipelineStage.VIDEO_ENCODING,
                    PipelineStage.AUDIO_ENCODING,
                    PipelineStage.MUXING,
                )
            )
        elif method == "encode":
            if isinstance(module, TextEncoder):
                operations.add(PipelineStage.TEXT_ENCODING)
            elif isinstance(module, PatchEncoder):
                operations.add(PipelineStage.VISION_ENCODING)
            elif isinstance(module, PatchAutoencoder):
                operations.add(PipelineStage.LATENT_ENCODING)
            elif isinstance(module, Encoder):
                operations.add(PipelineStage.LATENT_PREPARATION)
        elif method == "decode":
            if isinstance(module, ImageDecoder):
                operations.add(PipelineStage.IMAGE_DECODING)
            elif isinstance(module, VideoDecoder):
                operations.add(PipelineStage.VIDEO_DECODING)
            elif isinstance(module, AudioDecoder):
                operations.add(PipelineStage.AUDIO_DECODING)
    return frozenset(operations)


def describe_components(
    model: nn.Module,
    *,
    entries: Mapping[str, ComponentEntry] | None = None,
) -> dict[str, tuple[Call, ...]]:
    """Bind each IPC entry's exported methods to their resident modules.

    A model declares one `ComponentEntry` per IPC entry name, naming its owning
    component and the methods serving ranks may invoke. EntryPoint methods are
    relative to that component, including nested methods such as
    ``conditioner.encode``. Numerical sharing does not imply placement
    ownership, so one component belongs to exactly one entry.
    """
    if entries is None:
        entries = import_module(type(model).__module__).entry_points(
            model.config
        )

    owners: dict[tuple[str, str], str] = {}
    calls: dict[str, list[Call]] = {name: [] for name in entries}
    for name, entry in entries.items():
        for point in entry.points:
            nested, _, method = point.method.rpartition(".")
            path = ".".join(part for part in (entry.component, nested) if part)
            key = (path, method)
            if key in owners:
                raise unsupported_setup(f"model repeats numerical method {key}")
            owners[key] = name
            try:
                module = model.get_submodule(path)
            except AttributeError as error:
                raise unsupported_setup(
                    f"entry module {path!r} does not exist"
                ) from error
            if module is None:
                # The full declaration remains available on ranks where a
                # first/last-stage submodule has no resident numerical state.
                parent_path = path
                while parent_path:
                    parent_path = parent_path.rpartition(".")[0]
                    parent = model.get_submodule(parent_path)
                    mesh = getattr(parent, "mesh", None)
                    if isinstance(mesh, DeviceMesh):
                        pipeline = mesh.get_group(
                            "pp" if "pp" in mesh.axes else ()
                        )
                        if (point.stage == "first" and pipeline.rank != 0) or (
                            point.stage == "last"
                            and pipeline.rank != pipeline.size - 1
                        ):
                            break
                        raise unsupported_setup(
                            f"participating component {path!r} was removed"
                        )
                else:
                    raise unsupported_setup(
                        f"component {path!r} has no numerical module"
                    )
                continue
            if not callable(getattr(module, method, None)):
                raise unsupported_setup(
                    f"component {path!r} has no callable {method!r}"
                )
            call = Call(path, module, replace(point, method=method))
            if not call_operations((call,)) and not (
                isinstance(module, CausalLM)
                and method in {"embed_input_ids", "compute_logits"}
            ):
                raise unsupported_setup(
                    f"worker cannot execute capability {path}.{method}"
                )
            calls[name].append(call)

    return {name: tuple(items) for name, items in calls.items()}


def supported_operations(model: nn.Module) -> frozenset[Computation]:
    """Collect every computation and transfer mode the model can serve."""
    calls = tuple(
        call for calls in describe_components(model).values() for call in calls
    )
    operations = {TransferMode.TENSOR, *call_operations(calls)}
    if any(isinstance(call.module, CausalLM) for call in calls):
        operations.update((TransferMode.KV_PUBLISH, TransferMode.KV_INSTALL))
    return frozenset(operations)


def media_components(model: nn.Module) -> dict[PipelineStage, str]:
    """Resolve the media pipeline's component routing from its capabilities."""
    components = describe_components(model)
    if not any(
        isinstance(call.module, VideoPostprocessor)
        for calls in components.values()
        for call in calls
    ):
        return {}

    stages = {
        PipelineStage.TEXT_ENCODING,
        PipelineStage.LATENT_PREPARATION,
        PipelineStage.DENOISING,
        PipelineStage.VIDEO_DECODING,
        PipelineStage.AUDIO_DECODING,
        PipelineStage.VIDEO_ENCODING,
        PipelineStage.AUDIO_ENCODING,
        PipelineStage.MUXING,
    }

    routes = {}
    for name, calls in components.items():
        for stage in call_operations(calls) & stages:
            if stage in routes:
                raise unsupported_setup(
                    f"media pipeline repeats {stage.value} computation"
                )
            routes[stage] = name

    if stages - routes.keys():
        raise unsupported_setup(
            "media pipeline lacks required numerical capabilities"
        )
    if (
        routes[PipelineStage.LATENT_PREPARATION]
        != routes[PipelineStage.DENOISING]
    ):
        raise unsupported_setup(
            "latent preparation must participate in the denoiser entry"
        )
    return routes


def validate_components(
    model: nn.Module,
    components: Mapping[str, ComponentConfig],
    *,
    entries: Mapping[str, ComponentEntry] | None = None,
) -> dict[str, tuple[Call, ...]]:
    """Validate physical placement before loading weights or creating groups."""
    declared = describe_components(model, entries=entries)
    unknown = components.keys() - declared.keys()
    if unknown:
        raise unsupported_setup(
            f"unknown computation entries {sorted(unknown)}"
        )
    for name, component in components.items():
        calls = declared[name]
        if not calls:
            raise unsupported_setup(f"entry {name!r} has no numerical methods")
        if any(isinstance(call.module, VideoDecoder) for call in calls):
            if (
                component.distribution != "temporal_units"
                or component.units_per_rank != 1
            ):
                raise unsupported_setup(
                    "video decoding requires temporal_units with one native "
                    "unit per rank"
                )
        elif component.distribution is not None:
            raise unsupported_setup(
                f"entry {name!r} requires model-parallel membership"
            )
    return declared


def bind_components(
    model: nn.Module,
    bindings: Mapping[str, ModelEntry],
    *,
    entries: Mapping[str, ComponentEntry] | None = None,
) -> None:
    """Borrow methods and communicator views for each local stage.

    Views are borrowed for every participating stage.
    """
    declared = validate_components(
        model,
        {name: binding.config for name, binding in bindings.items()},
        entries=entries,
    )
    for name, binding in bindings.items():
        binding.calls = ()
        if not binding.owns or binding.mesh is None:
            continue

        mesh = binding.mesh
        pipeline = mesh.get_group("pp")
        calls = []
        for call in declared[name]:
            stage = call.entry.stage
            if stage == "first" and pipeline.rank != 0:
                continue
            if stage == "last" and pipeline.rank != pipeline.size - 1:
                continue

            groups = {}
            for role in call.entry.groups:
                axes = (
                    tuple(
                        axis
                        for axis in mesh.axes
                        if axis.startswith("cp")
                        or (role == "sp" and axis == "ulysses")
                    )
                    if role in {"cp", "sp"}
                    else (role,)
                )
                group = mesh.get_group(axes)
                # Roles resolving to the same rank set share one group.
                if group.size > 1:
                    groups[group.ranks] = group
            calls.append(replace(call, groups=tuple(groups.values())))

        binding.calls = tuple(calls)
        binding.computations = tuple(call_operations(calls))
