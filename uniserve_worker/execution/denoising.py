"""One declared diffusion evaluation, solver update, and numerical feedback."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import fields, is_dataclass
from typing import TypeVar

import torch

from uniserve.model.batch import DiffusionBatch
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.tensors import TensorViews
from uniserve.nn.diffusion.schedule import DiffusionSchedule

Size = TypeVar("Size")


def denoising_batch(
    model: DiffusionMixin[Size],
    shape: Size,
    tensors: TensorViews,
    schedule: DiffusionSchedule,
    index: int,
) -> DiffusionBatch[Size]:
    """Bind a single sequence's named latent views to a declared schedule row."""

    steps = schedule.sigmas[0].numel() - 1
    if not 0 <= index < steps:
        raise ValueError("denoising index is outside the supplied schedule")
    if len(schedule.timesteps) != len(model.modalities):
        raise ValueError("denoising schedule must follow the declared modality order")
    return DiffusionBatch(
        {name: (tensors[name],) for name in model.modalities},
        (shape,),
        timesteps={
            name: (schedule.timesteps[modality][index],)
            for modality, name in enumerate(model.modalities)
        },
        conditioning={"text": (tensors["text_condition"],)} if "text_condition" in tensors else {},
        ladder_index=index,
    )


def numerical_signature(value: object) -> Hashable:
    """Identify graph-visible geometry and fixed tensor backing without contents.

    Numerical dataclasses and mappings may be rebuilt between calls. Their
    values and borrowed tensor addresses determine reuse, not Python identity.
    Mutable tensor contents are read by replay and never copied to the host.
    """

    if isinstance(value, torch.Tensor):
        return (
            value.device,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            value.data_ptr(),
        )
    if isinstance(value, Mapping):
        return tuple((key, numerical_signature(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(numerical_signature(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple(
            (field.name, numerical_signature(getattr(value, field.name))) for field in fields(value)
        )
    if isinstance(value, Hashable):
        return value
    raise TypeError(f"unsupported numerical graph value: {type(value).__name__}")
