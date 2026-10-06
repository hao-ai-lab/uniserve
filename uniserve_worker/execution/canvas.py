"""Numerical canvas token views and completion columns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from uniserve.tensors import adjacent_view
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.storage.canvas_slots import STEP_CONTINUED


def token_values(tokens: tuple[int, ...]) -> torch.Tensor:
    """Build the shared host token vector for a call's readout canvases."""
    return torch.from_numpy(np.asarray(tokens, dtype=np.int64))


def readout_values(values: Sequence[torch.Tensor], count: int) -> torch.Tensor:
    """Pack FP32 candidate scores as integer words for completion capture."""
    logprobs = torch.cat(tuple(values)) if len(values) > 1 else values[0]
    if logprobs.dtype != torch.float32 or logprobs.numel() != count:
        raise invalid_descriptor(
            "canvas readout does not cover the call's candidates"
        )
    return logprobs.contiguous().view(torch.int32)


def step_values(
    values: Sequence[torch.Tensor], widths: Sequence[int]
) -> torch.Tensor:
    """Borrow adjacent result rows for one completion copy.

    Each int64 row contains the stop outcome followed by its canvas tokens.
    Graph outputs preserve these rows together when copying reusable storage.
    """
    width = widths[0]
    if any(
        value.dtype != torch.int64
        or value.shape != (expected,)
        or expected != width
        for value, expected in zip(values, widths, strict=True)
    ):
        raise invalid_descriptor("a canvas step does not cover its canvas")
    block = adjacent_view(tuple(values))
    if block is None:
        raise invalid_descriptor(
            "the canvas steps of a batch are not adjacent rows of one result"
        )
    return block


def step_continuations(
    block: torch.Tensor, width: int, rows: Sequence[int], dtype: torch.dtype
) -> torch.Tensor:
    """Select device continuation flags for the reserved output rows."""
    outcomes = block.view(-1, width)[:, 0]
    if len(rows) != outcomes.numel():
        outcomes = outcomes[list(rows)]
    return (outcomes == STEP_CONTINUED).to(dtype)
