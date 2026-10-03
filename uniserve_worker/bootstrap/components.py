"""Bind public numerical capabilities to the worker's components.

A model declares its IPC-addressable components as ``ComponentEntry`` values:
an owning module path and the ``EntryPoint`` methods serving ranks may invoke.
This module resolves those declarations against a model instance into
``Call`` values (``describe_components``), maps calls to the protocol call
kinds they serve (``call_kinds``), validates a deployment's placement against
them (``validate_components``), and attaches calls and communicator groups to
each rank's ``ComponentBinding`` (``bind_components``). It also declares the
host components, which own no numerical method.

``prepare_worker_model`` validates placement on a meta-device skeleton before
process groups exist; ``ModelExecutor`` binds the loaded model's components.
The worker's capacity report derives its advertised calls and media routes
from ``supported_calls`` and ``media_components``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from importlib import import_module

from torch import nn

from uniserve.distributed import DeviceMesh, communicators
from uniserve.model import (
    AudioDecoder,
    AudioEncoder,
    CausalLM,
    ComponentEntry,
    Denoiser,
    Encoder,
    ImageDecoder,
    PatchEncoder,
    TextEncoder,
    VideoDecoder,
    VideoEncoder,
    VideoPostprocessor,
)
from uniserve.nn.vae import PatchAutoencoder
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.errors import unsupported_setup
from uniserve_worker.model_executor.component_binding import (
    Call,
    ComponentBinding,
)
from uniserve_worker.protocol.call import (
    CallKind,
    ForwardMode,
    MediaCall,
    TransferMode,
)

# Host components run on a host worker's ranks and own no numerical method, so
# the worker declares them rather than the model, which declares numerical
# components only. The media reader decodes a request's condition media into
# the inputs of the vision and latent encoders; the video codec encodes the
# media units the video decoder reconstructs; the muxer encodes the audio
# track and assembles the artifact.
MEDIA_READER_COMPONENT = "media_reader"
VIDEO_CODEC_COMPONENT = "video_codec"
MUXER_COMPONENT = "muxer"
HOST_COMPONENTS: Mapping[str, frozenset[MediaCall]] = {
    MEDIA_READER_COMPONENT: frozenset({MediaCall.MEDIA_READING}),
    VIDEO_CODEC_COMPONENT: frozenset({MediaCall.VIDEO_ENCODING}),
    MUXER_COMPONENT: frozenset({MediaCall.AUDIO_ENCODING, MediaCall.MUXING}),
}
#: The muxer's calls, which ``bind_components`` records as its call kinds.
MUXER_CALL_KINDS = HOST_COMPONENTS[MUXER_COMPONENT]

#: Distinguishes an attribute a model never declared from one it cleared.
_MISSING = object()


def is_host_component(name: str) -> bool:
    """Report whether a component runs on host ranks rather than a device."""
    return name in HOST_COMPONENTS


def holds_host_components(names: Iterable[str]) -> bool:
    """Report whether a rank holding these components is a codec slot.

    A host rank runs one codec task at a time in its own process: one
    request's media read, one media unit's encode, the audio track's, or one
    step of a container's assembly.
    """
    return any(is_host_component(name) for name in names)


def call_kinds(calls: Iterable[Call]) -> frozenset[CallKind]:
    """Resolve the call kinds a capability's type and method can execute.

    The kind follows the module's capability type and the declared numerical
    method. A call no rule matches contributes no kind. Host components have
    no calls, so ``supported_calls`` and ``media_components`` take their
    kinds from ``HOST_COMPONENTS``.
    """
    kinds: set[CallKind] = set()
    for call in calls:
        module, method = call.module, call.entry_point.method
        if isinstance(module, CausalLM) and method == "forward":
            kinds.update(
                (ForwardMode.PREFILL, ForwardMode.DECODE, ForwardMode.VERIFY)
            )
        elif isinstance(module, Denoiser) and method == "forward":
            kinds.update((MediaCall.LATENT_PREPARATION, MediaCall.DENOISING))
        elif isinstance(module, VideoPostprocessor) and method == "forward":
            # The post-processor converts a decoded media unit to RGB on the
            # rank that decoded it, inside the same decoding call; the RGB
            # unit is that call's product, which a host rank encodes.
            kinds.add(MediaCall.VIDEO_DECODING)
        elif method == "encode":
            if isinstance(module, TextEncoder):
                kinds.add(MediaCall.TEXT_ENCODING)
            elif isinstance(module, PatchEncoder):
                kinds.add(MediaCall.VISION_ENCODING)
            elif isinstance(
                module, (PatchAutoencoder, VideoEncoder, AudioEncoder)
            ):
                kinds.add(MediaCall.LATENT_ENCODING)
            # ``TextEncoder`` and ``PatchEncoder`` subclass ``Encoder``, so
            # the generic encoder is matched only after them.
            elif isinstance(module, Encoder):
                kinds.add(MediaCall.LATENT_PREPARATION)
        elif method == "decode":
            if isinstance(module, ImageDecoder):
                kinds.add(MediaCall.IMAGE_DECODING)
            elif isinstance(module, VideoDecoder):
                kinds.add(MediaCall.VIDEO_DECODING)
            elif isinstance(module, AudioDecoder):
                kinds.add(MediaCall.AUDIO_DECODING)
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

    Args:
        model: The model instance, which may be a meta-device skeleton or a
            loaded pipeline stage.
        entries: The model's entry declarations. When ``None``, they come from
            the ``entry_points(config)`` function of the module that defines
            ``type(model)``.

    Returns:
        Every declared entry name mapped to its calls in declaration order. A
        call whose module a pipeline stage cleared on this rank is left out,
        so an entry may map to no calls. When any call's module is a
        ``VideoPostprocessor``, every host component is added with no calls.

    Raises:
        WorkerError: With ``UNSUPPORTED_SETUP`` when a (path, method) pair is
            declared twice, a path names no module, a cleared module has no
            ancestor with a ``DeviceMesh`` or is one this pipeline stage
            participates in, a method is not callable, or the worker has no
            call kind for a method other than a ``CausalLM``'s
            ``embed_input_ids`` or ``compute_logits``.
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
            module = _entry_module(model, path)
            if module is None:
                # A pipeline stage cleared this module. Walk up to the nearest
                # ancestor with a ``DeviceMesh`` and consult the entry point's
                # stage: a first- or last-stage method is excused on the other
                # stages (``break``). A cleared module on a stage that runs the
                # method is refused, and with no meshed ancestor at all the
                # ``while ... else`` refuses the declaration.
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
        # A model that reconstructs video also needs its units encoded and
        # assembled, which host components do.
        for name in HOST_COMPONENTS:
            described[name] = ()
    return described


def _entry_module(model: nn.Module, path: str) -> nn.Module | None:
    """Resolve a declared entry path, or None where its attribute is cleared.

    A pipeline stage that does not participate in a submodule clears the
    attribute, and ``get_submodule`` reports a cleared attribute the same way
    it reports a path the model never declared. This function tells the two
    apart: it returns None for a cleared attribute, which the caller answers
    by the stage rules, and raises ``WorkerError`` for a missing one, a model
    declaration this worker cannot serve.
    """
    try:
        return model.get_submodule(path)
    except AttributeError as error:
        parent_path, _, attribute = path.rpartition(".")
        try:
            parent = model.get_submodule(parent_path) if parent_path else model
        except AttributeError:
            parent = None
        if parent is not None and getattr(parent, attribute, _MISSING) is None:
            return None
        raise unsupported_setup(
            f"entry module {path!r} does not exist"
        ) from error


def supported_calls(
    model: nn.Module, held: Iterable[str] = ()
) -> frozenset[CallKind]:
    """Collect every call kind and transfer mode this worker can serve.

    A placement names the components this worker holds, and a call the holder
    of its component cannot serve has nowhere to run. A worker given no
    placement holds every component the model declares, which is the undivided
    deployment.

    ``TransferMode.TENSOR`` is always included; KV publish and install are
    included when a held component contains a ``CausalLM`` call, and a held
    host component contributes its ``HOST_COMPONENTS`` calls.
    """
    components = describe_components(model)
    names = set(held)
    if names:
        components = {
            name: calls for name, calls in components.items() if name in names
        }
    calls = tuple(call for items in components.values() for call in items)
    kinds = {TransferMode.TENSOR, *call_kinds(calls)}
    for name, host_kinds in HOST_COMPONENTS.items():
        if name in components:
            kinds.update(host_kinds)
    if any(isinstance(call.module, CausalLM) for call in calls):
        kinds.update((TransferMode.KV_PUBLISH, TransferMode.KV_INSTALL))
    return frozenset(kinds)


def media_components(
    model: nn.Module, held: Iterable[str] = ()
) -> dict[MediaCall, str]:
    """Resolve which component serves each media call this worker holds.

    The engine routes a call to the component named here, so a call a model
    serves from a component of its own -- a patch encoder placed apart from a
    language backbone, as much as a denoiser placed apart from a muxer -- is
    reported whether or not the model reconstructs video. A call whose
    component this placement does not hold is not reported, because this
    worker cannot serve it; the engine checks that the video graph is complete
    across the workers of a deployment.

    Raises:
        WorkerError: With ``UNSUPPORTED_SETUP`` when latent preparation and
            denoising are routed to different components, or when
            ``describe_components`` refuses the model.
    """
    components = describe_components(model)
    names = set(held)
    if names:
        components = {
            name: calls for name, calls in components.items() if name in names
        }
    owners: dict[MediaCall, list[str]] = {}
    for name, calls in components.items():
        owned = {
            kind for kind in call_kinds(calls) if isinstance(kind, MediaCall)
        }
        if name in HOST_COMPONENTS:
            # A host component's calls are host tasks with no numerical owner.
            owned = set(HOST_COMPONENTS[name])
        for call in owned:
            owners.setdefault(call, []).append(name)

    # A call several components implement names no single component, so it
    # is not routed: a model whose modules overlap outside the media graph
    # serves no media call through them, and a media deployment missing a
    # call is refused by the engine, which names the call, when it checks the
    # graph across workers.
    routes = {
        call: holders[0]
        for call, holders in owners.items()
        if len(holders) == 1
    }
    if (
        MediaCall.LATENT_PREPARATION in routes
        and MediaCall.DENOISING in routes
        and routes[MediaCall.LATENT_PREPARATION] != routes[MediaCall.DENOISING]
    ):
        raise unsupported_setup(
            "latent preparation must participate in the denoiser component"
        )
    return routes


def validate_components(
    model: nn.Module,
    components: Mapping[str, ComponentConfig],
    *,
    entries: Mapping[str, ComponentEntry] | None = None,
    declarations: dict[str, tuple[Call, ...]] | None = None,
) -> dict[str, tuple[Call, ...]]:
    """Validate a component placement against the model's declarations.

    ``prepare_worker_model`` calls this on a meta-device skeleton before
    loading weights or creating process groups; ``bind_components`` repeats
    it for the loaded model's bindings. Passing ``declarations`` reuses an
    earlier ``describe_components`` result instead of resolving it again.

    Returns:
        The declarations the placement was validated against.

    Raises:
        WorkerError: With ``UNSUPPORTED_SETUP`` when ``describe_components``
            refuses the model, the placement names an undeclared component, a
            non-host component has no calls, or a component's
            ``distribution`` does not fit its calls. The muxer and the
            media reader are never distributed; the video codec and a
            component with a ``VideoDecoder`` call require ``temporal_units``
            with one unit per rank; a component with an ``AudioDecoder`` or
            ``VideoEncoder`` call and no ``VideoDecoder`` call accepts no
            distribution or ``temporal_units`` with at least one unit per
            rank; every other component accepts no distribution.
    """
    declared = (
        describe_components(model, entries=entries)
        if declarations is None
        else declarations
    )
    unknown = components.keys() - declared.keys()
    if unknown:
        raise unsupported_setup(f"unknown components {sorted(unknown)}")
    for name, component in components.items():
        calls = declared[name]
        if not calls:
            if name not in HOST_COMPONENTS:
                raise unsupported_setup(
                    f"component {name!r} has no numerical methods"
                )
            if name == MUXER_COMPONENT and component.distribution is not None:
                raise unsupported_setup(
                    "the muxer assembles one artifact and is not distributed"
                )
            if (
                name == MEDIA_READER_COMPONENT
                and component.distribution is not None
            ):
                raise unsupported_setup(
                    "the media reader reads one request's conditions on one "
                    "rank and is not distributed"
                )
            if name == VIDEO_CODEC_COMPONENT and (
                component.distribution != "temporal_units"
                or component.units_per_rank != 1
            ):
                raise unsupported_setup(
                    "video encoding distributes by temporal_units with one "
                    "media unit per rank, each rank one codec slot"
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
        elif any(
            isinstance(call.module, (AudioDecoder, VideoEncoder))
            for call in calls
        ):
            # An audio media unit is a span of the latent timeline rather than
            # a native window, and a condition's encoding unit is one of its
            # encoder windows, so a rank may take several of them, but the
            # division is still by media unit.
            if component.distribution not in (None, "temporal_units") or (
                component.distribution is not None
                and component.units_per_rank < 1
            ):
                raise unsupported_setup(
                    f"{name} distributes by temporal_units with at least one "
                    "media unit per rank"
                )
        elif component.distribution is not None:
            raise unsupported_setup(
                f"entry {name!r} requires model-parallel membership"
            )
    return declared


def bind_components(
    model: nn.Module,
    bindings: Mapping[str, ComponentBinding],
    *,
    entries: Mapping[str, ComponentEntry] | None = None,
    declarations: dict[str, tuple[Call, ...]] | None = None,
) -> None:
    """Borrow methods and communicator views for each local component.

    Placement is validated first (see ``validate_components``). Every binding's
    ``calls`` is then reset; a binding this rank owns and has a mesh for
    receives the calls its pipeline stage runs and their call kinds. Each call
    carries the communicator groups it exchanges tensors in, deduplicated by
    backend handle. A temporally distributed ``VideoPostprocessor`` also has
    its ``units`` ring assigned on the module itself.
    """
    declared = validate_components(
        model,
        {name: binding.config for name, binding in bindings.items()},
        entries=entries,
        declarations=declarations,
    )
    for name, binding in bindings.items():
        binding.calls = ()
        if not binding.owns or binding.mesh is None:
            continue

        mesh = binding.mesh
        pipeline = mesh.get_group("pp")
        calls = []
        for call in declared[name]:
            stage = call.entry_point.stage
            if stage == "first" and pipeline.rank != 0:
                continue
            if stage == "last" and pipeline.rank != pipeline.size - 1:
                continue

            # Only an all-stage method keeps the module's communicators whose
            # axes include ``pp``; a first- or last-stage method runs on one
            # stage. Entry-point axes added below are not filtered.
            groups = {
                group._require(): group
                for group in communicators(call.module)
                if stage == "all" or "pp" not in group.name.split(".")
            }
            # Temporal components have a rank-local numerical mesh, but their
            # reconstruction still exchanges overlaps with adjacent ranks.
            # Bind that ring to both the model and its prepared call scope.
            if binding.units is not None and isinstance(
                call.module, VideoPostprocessor
            ):
                call.module.units = binding.units
                if binding.units.size > 1:
                    groups[binding.units._require()] = binding.units
            for axes in call.entry_point.communication_axes(mesh):
                group = mesh.get_group(axes)
                if group.size > 1:
                    groups[group._require()] = group
            calls.append(replace(call, groups=tuple(groups.values())))

        binding.calls = tuple(calls)
        binding.call_kinds = tuple(
            MUXER_CALL_KINDS if name == MUXER_COMPONENT else call_kinds(calls)
        )
