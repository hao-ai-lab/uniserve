"""Bind the worker's input staging to a model's declared capabilities.

The worker finds a model's capabilities (``ImageDenoiser``, ``VideoDenoiser``,
``VideoPostprocessor``, ...) by type within its ordinary module tree rather
than by concrete model identity, and builds the matching request-input
builder: ``ImageBuilder`` for image denoising and ``MediaBuilder`` for video.
"""

from torch import nn

from uniserve.model import ImageDenoiser, VideoDenoiser, VideoPostprocessor
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.model_executor.diffusion_inputs import ImageBuilder
from uniserve_worker.model_executor.media_inputs import MediaBuilder


def capability(model: nn.Module, kind: type[nn.Module]):
    """Find one capability within ordinary module composition.

    ``model.modules()`` yields a module shared under several paths once, so
    such a module counts as one capability.

    Returns:
        The single submodule (or ``model`` itself) that is an instance of
        ``kind``, or ``None`` when there is none.

    Raises:
        ValueError: More than one distinct submodule is an instance of
            ``kind``.
    """
    values = tuple(
        module for module in model.modules() if isinstance(module, kind)
    )
    if len(values) > 1:
        raise ValueError(
            f"worker requires an unambiguous {kind.__name__} capability"
        )
    return None if not values else values[0]


def image_builder(model: nn.Module):
    """Instantiate the image-denoising input builder or ``None`` without one."""
    denoiser = capability(model, ImageDenoiser)
    return None if denoiser is None else ImageBuilder(denoiser)


def media_builder(model: nn.Module, config: WorkerConfig):
    """Instantiate the video input builder within the worker's frame budget.

    The frame budget is ``max_video_seconds`` at the post-processor's output
    frame rate, rounded to the nearest frame; the text budget is
    ``max_sequence_tokens``.

    Returns:
        The builder, or ``None`` for a model without a ``VideoDenoiser``.

    Raises:
        ValueError: The model has a ``VideoDenoiser`` but no
            ``VideoPostprocessor`` to supply its frame rate, or either
            capability is ambiguous.
    """
    denoiser = capability(model, VideoDenoiser)
    if denoiser is None:
        return None

    output = capability(model, VideoPostprocessor)
    if output is None:
        raise ValueError(
            "media input construction requires its output sampling clock"
        )

    return MediaBuilder(
        denoiser,
        max_frames=int(config.max_video_seconds * output.frame_rate + 0.5),
        max_text_tokens=config.max_sequence_tokens,
    )
