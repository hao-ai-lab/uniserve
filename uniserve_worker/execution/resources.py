"""Worker selection of numerical components and their allocation bounds."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from uniserve.model.media import VideoSize
from uniserve.model.model import Model
from uniserve.model.text import TextSize
from uniserve.model.video import VideoMixin
from uniserve.runtime.tensors import merge_buffers
from uniserve.tensors import BufferConfig, OutputLayout

from .model_entry import ModelEntry


def media_state_buffers(
    model: Model, bindings: Mapping[str, ModelEntry]
) -> dict[str, BufferConfig]:
    """Allocate cross-step tensors only for participating stateful computations."""

    if not isinstance(model, VideoMixin):
        return {}
    size = VideoSize(
        model.output_capacity.frame_count, cast(int, getattr(model.text_encoder, "max_tokens"))
    )
    fields = []
    for binding in bindings.values():
        for method in binding.methods:
            if method in {"forward_diffusion", "postprocess_video"}:
                component = cast(Any, model.get_submodule(binding.component))
                fields.append(component.state_buffers(size))
    return merge_buffers(fields)


def media_workspace_buffers(
    model: Model, bindings: Mapping[str, ModelEntry]
) -> dict[str, BufferConfig]:
    """Bound temporary views for the locally serialized media computations."""

    if not isinstance(model, VideoMixin):
        return {}
    video = model.output_capacity
    size = VideoSize(video.frame_count, cast(int, getattr(model.text_encoder, "max_tokens")))
    fields = []
    for binding in bindings.values():
        for method in binding.methods:
            if method in {"forward_diffusion", "decode:video", "postprocess_video"}:
                component = cast(Any, model.get_submodule(binding.component))
                fields.append(component.workspace_buffers(size))
            elif method == "decode:audio":
                decoder = cast(Any, model.get_submodule(binding.component))
                samples = round(video.frame_count * video.audio_rate / video.frame_rate)
                fields.append(decoder.workspace_buffers(decoder.latent_frames(samples)))
    return merge_buffers(fields)


def output_layouts(
    model: Model,
    call: str,
    component: Any,
    *,
    frames: int | None = None,
    prompt_tokens: int | None = None,
    units: int | None = None,
) -> Mapping[str, OutputLayout]:
    """Query global tensor results at the producing numerical component.

    Media encoding publishes artifacts, while ordinary token/image operations
    use their existing scalar and feature result paths. This query covers the
    persistent tensors transferred between component computations.
    """

    if call in {"encode:conditioning", "postprocess_video"}:
        return {}
    query = getattr(component, "output_layout", None)
    if not callable(query):
        return {}
    query = cast(Callable[..., Mapping[str, OutputLayout]], query)
    if call == "encode:text":
        tokens = (
            cast(int, getattr(component, "max_tokens")) if prompt_tokens is None else prompt_tokens
        )
        return query(TextSize(tokens))
    if call == "decode:video" and not isinstance(model, VideoMixin):
        return query(units=units)
    if call not in {"forward_diffusion", "decode:video", "decode:audio"} or not isinstance(
        model, VideoMixin
    ):
        return query()
    video = model.video_info(model.output_capacity.frame_count if frames is None else frames)
    size = VideoSize(
        video.frame_count,
        cast(int, getattr(model.text_encoder, "max_tokens"))
        if prompt_tokens is None
        else prompt_tokens,
    )
    if call == "forward_diffusion":
        return query(size)
    if call == "decode:video":
        count = len(model.decode_windows(video)) if units is None else units
        return query(size, units=count)
    samples = round(video.frame_count * video.audio_rate / video.frame_rate)
    return query(samples)
