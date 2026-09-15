"""Query numerical result layouts and reserve caller-owned media input storage."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from uniserve.model import AudioDecoder, Denoiser, TextEncoder, VideoDecoder, VideoPostprocessor
from uniserve.tensors import BufferConfig, OutputLayout

from ..config import WorkerConfig
from .model_entry import Call, ModelEntry


def media_state_buffers(
    model: nn.Module, bindings: Mapping[str, ModelEntry], config: WorkerConfig
) -> dict[str, BufferConfig]:
    """Reserve resident samples and transfer staging only on participating ranks."""

    from ..bootstrap.inputs import media_builder

    builder = media_builder(model, config)
    if builder is None:
        return {}

    result = {}
    for binding in bindings.values():
        for call in binding.calls:
            if isinstance(call.module, Denoiser) and call.entry.method == "forward":
                fields = builder.capacity_buffers()
            elif isinstance(call.module, VideoPostprocessor):
                fields = call.module.state_buffers(builder.maximum.num_frames)
            else:
                continue

            for name, field in fields.items():
                if name in result and result[name] != field:
                    raise ValueError(f"media request fields disagree about {name!r}")
                result[name] = field
    return result


def output_layouts(
    model: nn.Module,
    config: WorkerConfig,
    call: Call,
    *,
    frames: int | None = None,
    prompt_tokens: int | None = None,
) -> Mapping[str, OutputLayout]:
    """Describe complete persistent tensor products for one capability call."""

    from ..bootstrap.inputs import capability, media_builder

    component = call.module
    if isinstance(component, TextEncoder) and call.entry.method == "encode":
        return component.output_layout(
            config.max_sequence_tokens if prompt_tokens is None else prompt_tokens
        )

    if not isinstance(component, (Denoiser, VideoDecoder, AudioDecoder)):
        return {}
    if isinstance(component, VideoDecoder) and frames is not None:
        return component.output_layout(frames)

    builder = media_builder(model, config)
    if builder is None:
        # Image execution returns its features and decoded raster through the
        # token/image protocol rather than persistent inter-component products.
        return {}

    size = builder.size(
        builder.maximum.num_frames if frames is None else frames,
        config.max_sequence_tokens if prompt_tokens is None else prompt_tokens,
    )
    if isinstance(component, Denoiser):
        return component.output_layout(size)
    if isinstance(component, VideoDecoder):
        return component.output_layout(size.num_frames)

    # Audio length follows from the video frame count at the declared rates.
    clock = capability(model, VideoPostprocessor)
    samples = round(size.num_frames * component.sample_rate / clock.frame_rate)
    return component.output_layout(samples)
