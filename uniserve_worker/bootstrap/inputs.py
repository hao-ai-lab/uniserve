"""Bind the worker's input staging to a model's declared capabilities."""

from torch import nn

from uniserve.model import Denoiser, ImageDenoiser, VideoPostprocessor

from ..config import WorkerConfig
from ..execution.inputs.image import ImageBuilder
from ..execution.inputs.media import MediaBuilder


def capability(model: nn.Module, kind: type[nn.Module]):
    """Find one capability within ordinary module composition."""
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
    """Instantiate the video input builder within the worker's frame budget."""
    denoiser = capability(model, Denoiser)
    if denoiser is None or isinstance(denoiser, ImageDenoiser):
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
