"""Bind public numerical capabilities to the worker's logical IPC entries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from importlib import import_module

from torch import nn

from uniserve.distributed import Communicator, DeviceMesh
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
from ..protocol.call import (
    CallKind,
    ForwardMode,
    PipelineStage,
    TransferMode,
)
from .config import ComponentConfig

# The muxer assembles encoded media units and the encoded audio track into the
# artifact on its host lane. It owns no numerical method, so the worker declares
# it rather than the model, which declares numerical components only.
MUXER_COMPONENT = "muxer"
MUXER_CALL_KINDS = frozenset(
    {PipelineStage.AUDIO_ENCODING, PipelineStage.MUXING}
)


def call_kinds(calls: Iterable[Call]) -> frozenset[CallKind]:
    """Resolve the call kinds a capability's type and method can execute.

    The kind follows the declared numerical method.
    """
    kinds: set[CallKind] = set()
    for call in calls:
        module, method = call.module, call.entry.method
        if isinstance(module, CausalLM) and method == "forward":
            kinds.update(
                (ForwardMode.PREFILL, ForwardMode.DECODE, ForwardMode.VERIFY)
            )
        elif isinstance(module, Denoiser) and method == "forward":
            kinds.update(
                (PipelineStage.LATENT_PREPARATION, PipelineStage.DENOISING)
            )
        elif isinstance(module, VideoPostprocessor) and method == "forward":
            # The post-processor converts one media unit to RGB on the rank that
            # decoded it; encoding that unit is the host half of the same call.
            kinds.add(PipelineStage.VIDEO_ENCODING)
        elif method == "encode":
            if isinstance(module, TextEncoder):
                kinds.add(PipelineStage.TEXT_ENCODING)
            elif isinstance(module, PatchEncoder):
                kinds.add(PipelineStage.VISION_ENCODING)
            elif isinstance(module, PatchAutoencoder):
                kinds.add(PipelineStage.LATENT_ENCODING)
            elif isinstance(module, Encoder):
                kinds.add(PipelineStage.LATENT_PREPARATION)
        elif method == "decode":
            if isinstance(module, ImageDecoder):
                kinds.add(PipelineStage.IMAGE_DECODING)
            elif isinstance(module, VideoDecoder):
                kinds.add(PipelineStage.VIDEO_DECODING)
            elif isinstance(module, AudioDecoder):
                kinds.add(PipelineStage.AUDIO_DECODING)
    return frozenset(kinds)


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
            if not call_kinds((call,)) and not (
                isinstance(module, CausalLM)
                and method in {"embed_input_ids", "compute_logits"}
            ):
                raise unsupported_setup(
                    f"worker cannot execute capability {path}.{method}"
                )
            calls[name].append(call)

    described = {name: tuple(items) for name, items in calls.items()}
    if any(
        isinstance(call.module, VideoPostprocessor)
        for items in described.values()
        for call in items
    ):
        # A model that reconstructs video also needs somewhere to assemble it.
        described[MUXER_COMPONENT] = ()
    return described


def supported_calls(
    model: nn.Module, held: Iterable[str] = ()
) -> frozenset[CallKind]:
    """Collect every call kind and transfer mode this worker can serve.

    A placement names the components this worker holds, and a call the holder
    of its component cannot serve has nowhere to run. A worker given no
    placement holds every component the model declares, which is the undivided
    deployment.
    """
    components = describe_components(model)
    names = set(held)
    if names:
        components = {
            name: calls for name, calls in components.items() if name in names
        }
    calls = tuple(call for items in components.values() for call in items)
    kinds = {TransferMode.TENSOR, *call_kinds(calls)}
    if MUXER_COMPONENT in components:
        kinds.update(MUXER_CALL_KINDS)
    if any(isinstance(call.module, CausalLM) for call in calls):
        kinds.update((TransferMode.KV_PUBLISH, TransferMode.KV_INSTALL))
    return frozenset(kinds)


#: The stages a model that reconstructs video must serve between them.
MEDIA_STAGES = frozenset(
    {
        PipelineStage.TEXT_ENCODING,
        PipelineStage.LATENT_PREPARATION,
        PipelineStage.DENOISING,
        PipelineStage.VIDEO_DECODING,
        PipelineStage.AUDIO_DECODING,
        PipelineStage.VIDEO_ENCODING,
        PipelineStage.AUDIO_ENCODING,
        PipelineStage.MUXING,
    }
)


def pipeline_components(
    model: nn.Module, held: Iterable[str] = ()
) -> dict[PipelineStage, str]:
    """Resolve which component serves each pipeline stage this worker holds.

    The engine routes a stage to the entry named here, so a stage a model
    serves from a component of its own -- a patch encoder placed apart from a
    language backbone, as much as a denoiser placed apart from a muxer -- is
    reported whether or not the model reconstructs video. A stage whose
    component this placement does not hold is not reported, because this
    worker cannot serve it.
    """
    components = describe_components(model)
    names = set(held)
    if names:
        components = {
            name: calls for name, calls in components.items() if name in names
        }
    owners: dict[PipelineStage, list[str]] = {}
    for name, calls in components.items():
        owned = {
            kind
            for kind in call_kinds(calls)
            if isinstance(kind, PipelineStage)
        }
        if name == MUXER_COMPONENT:
            # The muxer's stages are host tasks with no numerical owner.
            owned = set(MUXER_CALL_KINDS)
        for stage in owned:
            owners.setdefault(stage, []).append(name)

    # A stage several components implement names no single entry, so nothing
    # can be routed to it and it is not reported.
    routes = {
        stage: holders[0]
        for stage, holders in owners.items()
        if len(holders) == 1
    }
    if MUXER_COMPONENT not in components:
        return routes

    # A model that reconstructs video serves the whole media pipeline, and
    # every stage of it from one entry.
    for stage, holders in owners.items():
        if len(holders) > 1:
            raise unsupported_setup(
                f"media pipeline repeats {stage.value} computation"
            )
    if MEDIA_STAGES - routes.keys():
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
            if name != MUXER_COMPONENT:
                raise unsupported_setup(
                    f"entry {name!r} has no numerical methods"
                )
            if component.distribution is not None:
                raise unsupported_setup(
                    "the muxer assembles one artifact and is not distributed"
                )
            continue
        if any(isinstance(call.module, VideoDecoder) for call in calls):
            if (
                component.distribution != "temporal_units"
                or component.units_per_rank != 1
            ):
                raise unsupported_setup(
                    "video decoding requires temporal_units with one native "
                    "unit per rank"
                )
        elif any(isinstance(call.module, AudioDecoder) for call in calls):
            # An audio media unit is a span of the latent timeline rather than
            # a native window, so a rank may reconstruct several of them, but
            # the division is still by media unit.
            if component.distribution not in (None, "temporal_units") or (
                component.distribution is not None
                and component.units_per_rank < 1
            ):
                raise unsupported_setup(
                    "audio decoding distributes by temporal_units with at "
                    "least one media unit per rank"
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
        binding.call_kinds = tuple(
            MUXER_CALL_KINDS if name == MUXER_COMPONENT else call_kinds(calls)
        )
        # A module that reconstructs media units borrows the ring of ranks
        # holding consecutive ones, the way a parallel module borrows its mesh.
        if binding.units is not None:
            for call in calls:
                if isinstance(call.module.__dict__.get("units"), Communicator):
                    call.module.units = binding.units
