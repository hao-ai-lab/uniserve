"""Physical placement validation against declared numerical component calls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from uniserve_worker.modeling.text import TextMixin

from ..foundation.errors import unsupported_setup
from ..modeling.components import Call, CallSpec, ComponentSpec
from ..modeling.decoder import DecoderMixin
from ..modeling.diffusion import DiffusionMixin
from ..modeling.encoder import EncoderMixin
from ..modeling.video import VideoMixin
from ..nn.parallel import ComponentConfig
from ..protocol.batch import Computation, ForwardMode, PipelineStage, TransferMode

if TYPE_CHECKING:
    from ..modeling.model import Model


_CALL_OPERATIONS: Mapping[Call, tuple[Computation, ...]] = {
    Call.TEXT: (ForwardMode.PREFILL, ForwardMode.DECODE, ForwardMode.VERIFY),
    Call.ENCODE_TEXT: (PipelineStage.TEXT_ENCODING,),
    Call.ENCODE_VISION: (PipelineStage.VISION_ENCODING,),
    Call.ENCODE_LATENT: (PipelineStage.LATENT_ENCODING,),
    Call.ENCODE_CONDITIONING: (PipelineStage.LATENT_PREPARATION,),
    Call.DIFFUSION: (PipelineStage.LATENT_PREPARATION, PipelineStage.DENOISING),
    Call.DECODE_IMAGE: (PipelineStage.IMAGE_DECODING,),
    Call.DECODE_VIDEO: (PipelineStage.VIDEO_DECODING,),
    Call.DECODE_AUDIO: (PipelineStage.AUDIO_DECODING,),
    Call.POSTPROCESS_VIDEO: (),
}


def call_operations(calls: Iterable[CallSpec]) -> frozenset[Computation]:
    """Resolve numerical capabilities into their execution operations."""

    return frozenset(operation for call in calls for operation in _CALL_OPERATIONS[call.call])


def supported_operations(model: "Model") -> frozenset[Computation]:
    """Derive numerical work and transport actions implemented by the worker.

    Numerical component calls establish computation support. Tensor delivery
    belongs to every worker; KV publication and installation additionally need
    the public text capability. Models never declare these transport actions.
    """

    operations: set[Computation] = {TransferMode.TENSOR}
    for component in model.components(model.config):
        operations.update(call_operations(component.calls))
    if isinstance(model, TextMixin):
        operations.update((TransferMode.KV_PUBLISH, TransferMode.KV_INSTALL))
    operations.update(media_components(model))
    return frozenset(operations)


def media_components(model: "Model") -> dict[PipelineStage, str]:
    """Attach codec and mux work to the declared numerical video output role.

    The current media execution pipeline reconstructs both video and audio.
    Its component edges follow numerical calls; physical ranks are resolved
    separately from the worker's component entries.
    """

    components = model.components(model.config)
    if not any(
        call.call is Call.POSTPROCESS_VIDEO for component in components for call in component.calls
    ):
        return {}
    roles: dict[Call, str] = {}
    required = {
        Call.ENCODE_TEXT,
        Call.ENCODE_CONDITIONING,
        Call.DIFFUSION,
        Call.DECODE_VIDEO,
        Call.DECODE_AUDIO,
        Call.POSTPROCESS_VIDEO,
    }
    for component in components:
        for call in component.calls:
            if call.call not in required:
                continue
            if call.call in roles:
                raise unsupported_setup(f"video pipeline repeats {call.call.value} computation")
            roles[call.call] = component.name
    if roles.keys() != required:
        missing = sorted(call.value for call in required - roles.keys())
        raise unsupported_setup(f"video pipeline is missing numerical calls {missing}")
    if roles[Call.ENCODE_CONDITIONING] != roles[Call.DIFFUSION]:
        raise unsupported_setup("latent preparation requires the denoiser's conditioning component")
    output = roles[Call.POSTPROCESS_VIDEO]
    return {
        PipelineStage.TEXT_ENCODING: roles[Call.ENCODE_TEXT],
        PipelineStage.LATENT_PREPARATION: roles[Call.DIFFUSION],
        PipelineStage.DENOISING: roles[Call.DIFFUSION],
        PipelineStage.VIDEO_DECODING: roles[Call.DECODE_VIDEO],
        PipelineStage.AUDIO_DECODING: roles[Call.DECODE_AUDIO],
        PipelineStage.VIDEO_ENCODING: output,
        PipelineStage.AUDIO_ENCODING: output,
        PipelineStage.MUXING: output,
    }


def validate_components(
    model_class: type[Model],
    config: Any,
    components: Mapping[str, ComponentConfig],
) -> dict[str, ComponentSpec]:
    """Resolve callable components and validate their physical distribution.

    Every declared call must be supplied by a numerical capability, including
    its encoder/decoder kind. This is checked before resource construction,
    even when the component is nonresident on the current caller.
    Temporal decoder execution distributes one native unit to each member.
    Other numerical entries use their configured model-parallel membership.
    Dimension divisibility and numerical algorithm limits belong to models.
    """

    declarations = {}
    for declaration in model_class.components(config):
        if declaration.name in declarations:
            raise unsupported_setup(f"model repeats component role {declaration.name!r}")
        declarations[declaration.name] = declaration
        for item in declaration.calls:
            call = item.call
            supported = False
            if call is Call.TEXT:
                supported = issubclass(model_class, TextMixin)
            elif call is Call.DIFFUSION:
                supported = issubclass(model_class, DiffusionMixin)
            elif call is Call.POSTPROCESS_VIDEO:
                supported = issubclass(model_class, VideoMixin)
            elif call.value.startswith("encode:"):
                supported = (
                    issubclass(model_class, EncoderMixin)
                    and call.value.removeprefix("encode:") in model_class.encoder_kinds
                )
            elif call.value.startswith("decode:"):
                supported = (
                    issubclass(model_class, DecoderMixin)
                    and call.value.removeprefix("decode:") in model_class.decoder_kinds
                )
            if not supported:
                raise unsupported_setup(
                    f"component {declaration.name!r} declares unsupported {call.value} computation"
                )

    unknown = components.keys() - declarations.keys()
    if unknown:
        raise unsupported_setup(f"unknown computation entries {sorted(unknown)}")
    for name, component in components.items():
        calls = {item.call for item in declarations[name].calls}
        if Call.DECODE_VIDEO in calls:
            if component.distribution != "temporal_units" or component.units_per_rank != 1:
                raise unsupported_setup(
                    "video decoder execution requires temporal_units with one native unit per rank"
                )
        elif component.distribution is not None:
            raise unsupported_setup(f"entry {name!r} requires model-parallel membership")
    return declarations
