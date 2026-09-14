"""Physical placement validation against declared numerical component calls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from uniserve.model.components import ComponentCall
from uniserve.model.decoder import DecoderMixin
from uniserve.model.encoder import EncoderMixin
from uniserve.model.text import TextMixin
from uniserve_worker.config import ComponentConfig

from ..foundation.errors import unsupported_setup
from ..protocol.batch import Computation, ForwardMode, PipelineStage, TransferMode

if TYPE_CHECKING:
    from uniserve.model.model import Model


_CALL_OPERATIONS: Mapping[str, tuple[Computation, ...]] = {
    "forward": (ForwardMode.PREFILL, ForwardMode.DECODE, ForwardMode.VERIFY),
    "encode:text": (PipelineStage.TEXT_ENCODING,),
    "encode:vision": (PipelineStage.VISION_ENCODING,),
    "encode:latent": (PipelineStage.LATENT_ENCODING,),
    "encode:conditioning": (PipelineStage.LATENT_PREPARATION,),
    "forward_diffusion": (PipelineStage.LATENT_PREPARATION, PipelineStage.DENOISING),
    "decode:image": (PipelineStage.IMAGE_DECODING,),
    "decode:video": (PipelineStage.VIDEO_DECODING,),
    "decode:audio": (PipelineStage.AUDIO_DECODING,),
    "postprocess_video": (),
}


def call_operations(methods: Iterable[str]) -> frozenset[Computation]:
    """Translate numerical entry names into worker operations during binding."""

    return frozenset(operation for method in methods for operation in _CALL_OPERATIONS[method])


def supported_operations(model: Model) -> frozenset[Computation]:
    operations: set[Computation] = {TransferMode.TENSOR}
    operations.update(call_operations(call.method for call in model.component_calls(model.config)))
    if isinstance(model, TextMixin):
        operations.update((TransferMode.KV_PUBLISH, TransferMode.KV_INSTALL))
    operations.update(media_components(model))
    return frozenset(operations)


def media_components(model: Model) -> dict[PipelineStage, str]:
    """Attach media execution to the catalog's existing logical entry names."""

    from uniserve_models.catalog import entry_paths

    calls = model.component_calls(model.config)
    if not any(call.method == "postprocess_video" for call in calls):
        return {}
    entries = {path: name for name, path in entry_paths(type(model), model.config).items()}
    required = {
        "encode:text",
        "encode:conditioning",
        "forward_diffusion",
        "decode:video",
        "decode:audio",
        "postprocess_video",
    }
    roles = {}
    for call in calls:
        if call.method not in required:
            continue
        if call.method in roles:
            raise unsupported_setup(f"video pipeline repeats {call.method} computation")
        roles[call.method] = entries[call.component]
    if roles.keys() != required:
        raise unsupported_setup(
            f"video pipeline is missing numerical calls {sorted(required - roles.keys())}"
        )
    if roles["encode:conditioning"] != roles["forward_diffusion"]:
        raise unsupported_setup("latent preparation requires the denoiser's conditioning component")
    output = roles["postprocess_video"]
    return {
        PipelineStage.TEXT_ENCODING: roles["encode:text"],
        PipelineStage.LATENT_PREPARATION: roles["forward_diffusion"],
        PipelineStage.DENOISING: roles["forward_diffusion"],
        PipelineStage.VIDEO_DECODING: roles["decode:video"],
        PipelineStage.AUDIO_DECODING: roles["decode:audio"],
        PipelineStage.VIDEO_ENCODING: output,
        PipelineStage.AUDIO_ENCODING: output,
        PipelineStage.MUXING: output,
    }


def validate_components(
    model_class: type[Model],
    config: Any,
    components: Mapping[str, ComponentConfig],
    *,
    paths: Mapping[str, str] | None = None,
) -> dict[str, tuple[ComponentCall, ...]]:
    """Validate placement against complete method/stage participation records.

    Actual module methods are resolved after construction, before loading or
    execution. An outer model's mixins do not determine a child capability.
    """

    from uniserve_models.catalog import entry_paths

    paths = entry_paths(model_class, config) if paths is None else paths
    calls = model_class.component_calls(config)
    declared = {}
    for call in calls:
        key = (call.component, call.method)
        if key in declared:
            raise unsupported_setup(f"model repeats numerical method {key}")
        if call.method not in _CALL_OPERATIONS:
            raise unsupported_setup(f"worker does not execute numerical method {call.method!r}")
        declared[key] = call
    unknown = components.keys() - paths.keys()
    if unknown:
        raise unsupported_setup(f"unknown computation entries {sorted(unknown)}")
    by_entry = {
        name: tuple(call for call in calls if call.component == path)
        for name, path in paths.items()
    }
    for name, component in components.items():
        methods = {call.method for call in by_entry[name]}
        if not methods:
            raise unsupported_setup(f"entry {name!r} has no numerical methods")
        if "decode:video" in methods:
            if component.distribution != "temporal_units" or component.units_per_rank != 1:
                raise unsupported_setup(
                    "video decoder execution requires temporal_units with one native unit per rank"
                )
        elif component.distribution is not None:
            raise unsupported_setup(f"entry {name!r} requires model-parallel membership")
    return by_entry


def bind_components(model: Model, bindings, *, paths: Mapping[str, str] | None = None) -> None:
    """Bind actual callables and groups once, resolving stage participation here."""

    from functools import partial

    from uniserve_models.catalog import entry_paths

    paths = entry_paths(type(model), model.config) if paths is None else paths
    calls = model.component_calls(model.config)
    for name, binding in bindings.items():
        if name not in paths:
            continue
        binding.component = paths[name]
        binding.methods.clear()
        if not binding.owns or binding.mesh is None:
            continue
        owner = model.get_submodule(binding.component)
        stage, stages = binding.mesh.coord("pp"), binding.mesh.size("pp")
        for call in calls:
            if call.component != binding.component:
                continue
            if call.stage == "first" and stage != 0 or call.stage == "last" and stage != stages - 1:
                continue
            method, _, kind = call.method.partition(":")
            forward = getattr(owner, method, None)
            if not callable(forward):
                raise unsupported_setup(f"component {call.component!r} has no callable {method!r}")
            if method == "encode":
                if not isinstance(owner, EncoderMixin) or kind not in owner.encoder_kinds:
                    raise unsupported_setup(f"component {call.component!r} does not encode {kind}")
                forward = partial(forward, kind)
            elif method == "decode" and kind == "image":
                if not isinstance(owner, DecoderMixin) or kind not in owner.decoder_kinds:
                    raise unsupported_setup(f"component {call.component!r} does not decode {kind}")
                forward = partial(forward, kind)
            groups = tuple(
                binding.mesh.get_group(axis) for axis in call.groups if binding.mesh.size(axis) > 1
            )
            binding.methods[call.method] = (forward, groups)
