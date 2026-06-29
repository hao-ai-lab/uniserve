"""The :class:`ForwardMode` enum and op-kind -> mode resolution.

Defines the single most-imported forward-mode type in the worker. Imports only
``foundation`` and the torch-free ``contracts.op_kinds`` table.
"""
from __future__ import annotations

from enum import Enum

from ..foundation.errors import invalid_descriptor
from .op_kinds import OP_KIND_TABLE

__all__ = ["ForwardMode", "mode_for_op"]


class ForwardMode(str, Enum):
    EXTEND = "extend"
    DECODE = "decode"
    TARGET_VERIFY = "target_verify"
    ENCODE = "encode"
    DENOISE = "denoise"
    COMMIT = "commit"
    MIXED = "mixed"
    # Peeled-stage modes: a Sampler worker turns a Logits handle into a token;
    # a PostProcess worker encodes a frame.
    SAMPLE = "sample"
    ENCODE_FRAME = "encode_frame"


def mode_for_op(kind: str) -> "ForwardMode":
    spec = OP_KIND_TABLE.get(kind)
    if spec is None:
        raise invalid_descriptor(f"unsupported op kind {kind!r}")
    return ForwardMode(spec.mode)


# Every op-kind spec must name a real ``ForwardMode``; this couples the
# contract-side mode strings (kept torch-free in ``contracts.op_kinds``) to this
# enum so a typo there fails at import rather than at first dispatch.
assert all(spec.mode in ForwardMode._value2member_map_ for spec in OP_KIND_TABLE.values())
