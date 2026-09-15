"""Resolve caller-owned typed input factories during model discovery."""

from importlib import import_module
from typing import get_type_hints

from torch import nn

from uniserve.model import Denoiser, ImageDenoiser, VideoPostprocessor

from ..config import WorkerConfig

# Each builder implements the numerical input preparation required by the
# denoiser's annotated input type. Runtime dispatch retains the resolved object.
_BUILDERS = {
    "uniserve_models.stub.inputs.DenoiserInput": "uniserve_worker.execution.inputs.stub.StubBuilder",
    "uniserve_models.bagel.inputs.DenoiserInput": "uniserve_worker.execution.inputs.bagel.BagelBuilder",
    "uniserve_models.sensenova_u1.inputs.DenoiserInput": "uniserve_worker.execution.inputs.sensenova_u1.U1Builder",
    "uniserve_models.minimax_h3.inputs.DenoiserInput": "uniserve_worker.execution.inputs.h3.MediaBuilder",
}


def capability(model: nn.Module, kind: type[nn.Module]):
    """Find one capability within ordinary module composition."""

    values = tuple(module for module in model.modules() if isinstance(module, kind))
    if len(values) > 1:
        raise ValueError(f"worker requires an unambiguous {kind.__name__} capability")
    return None if not values else values[0]


def builder_type(denoiser: Denoiser):
    """Resolve a numerical input builder once during worker initialization."""

    input_type = get_type_hints(denoiser.forward)["inputs"]
    key = f"{input_type.__module__}.{input_type.__qualname__}"
    if key not in _BUILDERS:
        raise ValueError(f"worker has no registered input builder for {key}")
    module, _, name = _BUILDERS[key].rpartition(".")
    return getattr(import_module(module), name)


def image_builder(model: nn.Module):
    denoiser = capability(model, ImageDenoiser)
    return None if denoiser is None else builder_type(denoiser)(denoiser)


def media_builder(model: nn.Module, config: WorkerConfig):
    denoiser = capability(model, Denoiser)
    if denoiser is None or isinstance(denoiser, ImageDenoiser):
        return None
    output = capability(model, VideoPostprocessor)
    if output is None:
        raise ValueError("media input construction requires its output sampling clock")
    return builder_type(denoiser)(
        denoiser,
        max_frames=int(config.max_video_seconds * output.frame_rate + 0.5),
        max_text_tokens=config.max_sequence_tokens,
    )
