"""Numerical runner selection and tensor result conversion.

Native ModelRunner owns the bound context, inputs and graph lifecycle.
Concrete numerical runners evaluate ordinary model capabilities.
"""

from collections.abc import Mapping

import torch

from uniserve.tensors import OutputLayout, TensorOutput
from uniserve_worker._uniserve_ipc import ModelRunner as ModelRunner
from uniserve_worker._uniserve_ipc import joining_experts as joining_experts
from uniserve_worker.errors import ComputeError

from .output import ExecutionOutput


def _module_forward(forward, resources, inputs):
    """Evaluate captured numerical arguments with their prepared resources."""
    return forward(*inputs[0], **inputs[1], **resources)


def tensor_result(result):
    """Normalize a module call result into tensors plus optional layouts."""
    if isinstance(result, torch.Tensor):
        result = (result,)
    if isinstance(result, Mapping):
        result = tuple(value for values in result.values() for value in values)

    values: list[torch.Tensor] = []
    layouts: list[OutputLayout | None] = []
    for value in result:
        if isinstance(value, TensorOutput):
            values.append(value.tensor)
            layouts.append(value.layout)
        elif isinstance(value, torch.Tensor):
            values.append(value)
            layouts.append(None)
        else:
            raise ComputeError(
                "participating numerical call did not return a tensor"
            )
    return ExecutionOutput(tuple(values), layouts=tuple(layouts))


def runner_type(module):
    """Select a bound numerical runner from public model capabilities.

    Raises:
        TypeError: ``module`` implements none of the supported capabilities.
    """
    from uniserve.model import (
        AudioDecoder,
        AudioEncoder,
        CausalLM,
        Denoiser,
        Encoder,
        ImageDecoder,
        TokenDenoiser,
        VideoDecoder,
        VideoEncoder,
        VideoPostprocessor,
    )
    from uniserve.nn.vae import PatchAutoencoder

    # The runner modules import ``ModelRunner`` from this module, so they are
    # imported here rather than at module scope.
    from .canvas_runner import CanvasRunner
    from .decoder_runner import DecoderRunner
    from .diffusion_runner import DiffusionRunner
    from .encoder_runner import EncoderRunner
    from .text_runner import TextRunner

    if isinstance(module, CausalLM):
        return TextRunner
    if isinstance(module, Denoiser):
        return DiffusionRunner
    if isinstance(module, TokenDenoiser):
        return CanvasRunner
    if isinstance(
        module, (AudioDecoder, ImageDecoder, VideoDecoder, VideoPostprocessor)
    ):
        return DecoderRunner
    if isinstance(
        module, (Encoder, PatchAutoencoder, VideoEncoder, AudioEncoder)
    ):
        return EncoderRunner
    raise TypeError("module has no supported numerical capability")
