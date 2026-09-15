"""Output selection and borrowed storage views owned by execution."""

from collections.abc import Sequence
from enum import StrEnum

import torch


class TokenSelection(StrEnum):
    """Selects final-token logits, all-token logits, or hidden states from a model forward."""

    LAST_LOGITS = "last_logits"
    ALL_LOGITS = "all_logits"
    HIDDEN = "hidden"


def packed_tensor_views(values: Sequence[torch.Tensor]) -> torch.Tensor | None:
    """Recover one tensor from ordered contiguous views without copying."""

    if not values:
        return None
    flat = tuple(value.reshape(-1) for value in values)
    first = flat[0]
    if (
        not first.is_contiguous()
        or any(not value.is_contiguous() for value in flat)
        or any(value.dtype != first.dtype or value.device != first.device for value in flat)
    ):
        return None
    storage = first.untyped_storage().data_ptr()
    offset = int(first.storage_offset())
    expected = offset
    for value in flat:
        if value.untyped_storage().data_ptr() != storage or int(value.storage_offset()) != expected:
            return None
        expected += int(value.numel())
    return first.as_strided((expected - offset,), (1,), storage_offset=offset)


def concatenate_views(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Borrow adjacent numerical columns, or concatenate disjoint allocations."""

    tensors = tuple(value.reshape(-1) for value in values)
    packed = packed_tensor_views(tensors)
    return torch.cat(tensors, dim=0) if packed is None else packed
