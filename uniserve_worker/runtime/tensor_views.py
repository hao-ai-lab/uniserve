"""Helpers for preserving useful tensor views across runtime boundaries."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from ..foundation.errors import invalid_descriptor

__all__ = ["adjacent_one_token_view", "coalesce_one_token_rows"]


def adjacent_one_token_view(rows: Sequence[torch.Tensor]) -> torch.Tensor | None:
    """Return one strided view when one-token rows already occupy adjacent storage."""

    if not rows:
        raise invalid_descriptor("tensor rows must not be empty")
    if len(rows) == 1:
        return rows[0].reshape(-1)
    first = rows[0].reshape(-1)
    if int(first.numel()) != 1:
        return None
    elem_size = int(first.element_size())
    base_ptr = int(first.data_ptr())
    for idx, row in enumerate(rows):
        flat = row.reshape(-1)
        if (
            int(flat.numel()) != 1
            or flat.dtype != first.dtype
            or flat.device != first.device
            or int(flat.data_ptr()) != base_ptr + idx * elem_size
        ):
            return None
    try:
        return first.as_strided((len(rows),), (1,))
    except RuntimeError:
        return None


def coalesce_one_token_rows(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    view = adjacent_one_token_view(rows)
    if view is not None:
        return view
    return torch.cat([row.reshape(-1) for row in rows], dim=0)
