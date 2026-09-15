"""Resolve caller-owned typed input factories during model discovery."""

from importlib import import_module
from typing import get_type_hints

from torch import nn

from uniserve.model import Denoiser, ImageDenoiser, VideoPostprocessor

from ..config import WorkerConfig

# Each factory implements a concrete numerical input contract. Shared execution
# receives the constructed factory and capability, not a model identity branch.
_input_factories = {
    "uniserve_models.stub.DenoiserInput": "uniserve_worker.execution.inputs.stub.Inputs",
    "uniserve_models.bagel.DenoiserInput": "uniserve_worker.execution.inputs.bagel.Inputs",
    "uniserve_models.sensenova_u1.DenoiserInput": "uniserve_worker.execution.inputs.sensenova_u1.Inputs",
    "uniserve_models.minimax_h3.inputs.DenoiserInput": "uniserve_worker.execution.inputs.h3.Inputs",
}


def capability(model: nn.Module, kind: type[nn.Module]):
    """Find one capability within ordinary module composition."""

    values = tuple(module for module in model.modules() if isinstance(module, kind))
    if len(values) > 1:
        raise ValueError(f"worker requires an unambiguous {kind.__name__} capability")
    return None if not values else values[0]


def input_factory(denoiser: Denoiser):
    """Resolve a declared numerical input once during worker initialization."""

    input_type = get_type_hints(denoiser.forward)["inputs"]
    key = f"{input_type.__module__}.{input_type.__qualname__}"
    if key not in _input_factories:
        raise ValueError(f"worker has no registered input factory for {key}")
    module, _, name = _input_factories[key].rpartition(".")
    return getattr(import_module(module), name)


def image_inputs(model: nn.Module):
    denoiser = capability(model, ImageDenoiser)
    return None if denoiser is None else input_factory(denoiser)(denoiser)


def media_inputs(model: nn.Module, config: WorkerConfig):
    denoiser = capability(model, Denoiser)
    if denoiser is None or isinstance(denoiser, ImageDenoiser):
        return None
    output = capability(model, VideoPostprocessor)
    if output is None:
        raise ValueError("media input construction requires its output sampling clock")
    return input_factory(denoiser)(
        denoiser,
        max_frames=int(config.max_video_seconds * output.frame_rate + 0.5),
        max_text_tokens=config.max_sequence_tokens,
    )
