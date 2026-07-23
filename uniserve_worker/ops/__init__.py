"""Model-facing operator facade.

Models import this package instead of vendor kernels or backend registries.
"""
from __future__ import annotations

from .core import (
    AdapterPool,
    Capabilities,
    CommDispatcher,
    CommProvider,
    Dispatcher,
    FusedOpPool,
    Handoff,
    Provider,
)
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
    tp_all_reduce,
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
    TpAllReduceReq,
)

__all__ = [
    "AddRmsNormReq",
    "AdapterPool",
    "AttentionRegime",
    "AttentionReq",
    "Capabilities",
    "CommDispatcher",
    "CommProvider",
    "Dispatcher",
    "FusedOpPool",
    "Handoff",
    "QKNormReq",
    "QKNormRopeReq",
    "PackedRopeReq",
    "Provider",
    "RmsNormReq",
    "SiluAndMulReq",
    "TpAllReduceReq",
    "add_rms_norm",
    "attention",
    "can_run_attention",
    "qk_norm",
    "qk_norm_packed_rope",
    "qk_norm_rope",
    "rope",
    "rms_norm",
    "silu_and_mul",
    "tp_all_reduce",
]
