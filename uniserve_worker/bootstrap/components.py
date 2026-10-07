"""Bind public numerical capabilities to the worker's components.

A model declares its IPC-addressable components as ``ComponentEntry`` values:
an owning module path and the ``EntryPoint`` methods serving ranks may invoke.
This module resolves those declarations against a model instance into
``Call`` values (``describe_components``), maps calls to the protocol call
kinds they serve (``call_kinds``), validates a deployment's placement against
them (``validate_components``), and attaches calls and communicator groups to
each rank's ``ComponentBinding`` (``bind_components``). It also declares the
host components, which own no numerical method. Placement policy and rank-local
binding are implemented by the native component owner.

``prepare_worker_model`` validates placement on a meta-device skeleton before
process groups exist; ``ModelExecutor`` binds the loaded model's components.
The worker's capacity report derives its advertised calls and media routes
from ``supported_calls`` and ``media_components``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module

from torch import nn

from uniserve.distributed import DeviceMesh
from uniserve.model import (
    CausalLM,
    ComponentEntry,
    TokenDenoiser,
    VideoPostprocessor,
)
from uniserve_worker._uniserve_ipc import (
    bind_components as bind_components,
)
from uniserve_worker._uniserve_ipc import (
    call_kinds as call_kinds,
)
from uniserve_worker._uniserve_ipc import (
    holds_host_components as holds_host_components,
)
from uniserve_worker._uniserve_ipc import (
    host_components,
)
from uniserve_worker._uniserve_ipc import (
    is_host_component as is_host_component,
)
from uniserve_worker._uniserve_ipc import (
    media_components as media_components,
)
from uniserve_worker._uniserve_ipc import (
    supported_calls as supported_calls,
)
from uniserve_worker._uniserve_ipc import (
    validate_components as validate_components,
)
from uniserve_worker.errors import unsupported_setup
from uniserve_worker.model_executor.component_binding import (
    Call,
)
from uniserve_worker.protocol.call import MediaCall

# Host components run on a host worker's ranks and own no numerical method, so
# the worker declares them rather than the model, which declares numerical
# components only. The media reader decodes a request's condition media into
# the inputs of the vision and latent encoders; the video codec encodes the
# media units the video decoder reconstructs; the muxer encodes the audio
# track and assembles the artifact.
MEDIA_READER_COMPONENT = "media_reader"
VIDEO_CODEC_COMPONENT = "video_codec"
MUXER_COMPONENT = "muxer"
HOST_COMPONENTS: Mapping[str, frozenset[MediaCall]] = host_components()

#: The muxer's calls, which ``bind_components`` records as its call kinds.
MUXER_CALL_KINDS = HOST_COMPONENTS[MUXER_COMPONENT]

#: Distinguishes an attribute a model never declared from one it cleared.
_MISSING = object()


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
            ``embed_input_ids`` or ``compute_logits`` or a
            ``TokenDenoiser``'s ``compute_logits``.
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
                or isinstance(module, TokenDenoiser)
                and method == "compute_logits"
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
