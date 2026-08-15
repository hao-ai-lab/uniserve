"""Model-facing operator facade.

Models import this package instead of vendor kernels or backend registries.
"""
from __future__ import annotations

from .core import Dispatcher, Provider
from .facade import (
    add_rms_norm,
    attention,
    can_run_attention,
    qk_norm,
    qk_norm_packed_rope,
    qk_norm_rope,
    rms_norm,
    rope,
    silu_and_mul,
)
from .requests import (
    AddRmsNormReq,
    AttentionRegime,
    AttentionReq,
    PackedRopeReq,
    QKNormReq,
    QKNormRopeReq,
    RmsNormReq,
    SiluAndMulReq,
)

__all__ = [
    "AddRmsNormReq",
    "AttentionRegime",
    "AttentionReq",
    "Dispatcher",
    "QKNormReq",
    "QKNormRopeReq",
    "PackedRopeReq",
    "Provider",
    "RmsNormReq",
    "SiluAndMulReq",
    "add_rms_norm",
    "attention",
    "can_run_attention",
    "qk_norm",
    "qk_norm_packed_rope",
    "qk_norm_rope",
    "rope",
    "rms_norm",
    "silu_and_mul",
]
