"""Closed operation vocabulary shared by family and execution contracts."""

from __future__ import annotations

from enum import IntEnum

__all__ = ["OperationTag"]


class OperationTag(IntEnum):
    """Stable tags for the general model-backed operation algebra."""

    SEQUENCE_STEP = 1
    FLOW_STEP = 2
    ENCODE_STEP = 3
    MATERIALIZE_STEP = 4
