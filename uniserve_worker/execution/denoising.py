"""Identity of numerical views retained by worker graph execution."""

from collections.abc import Hashable, Mapping
from dataclasses import fields, is_dataclass

import torch


def numerical_signature(value: object) -> Hashable:
    """Identify static values and tensor backing independently of tensor contents.

    Immutable numerical records may be rebuilt between calls. Their values and
    borrowed tensor addresses determine reuse. Mutable device contents remain
    device inputs; this function never reads them back to the host.
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
