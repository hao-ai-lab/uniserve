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
from .requests import (
    AddRmsNormReq,
    AttentionRegime,
    AttentionReq,
    QKNormReq,
    QKNormRopeReq,
    PackedRopeReq,
    RmsNormReq,
    SiluAndMulReq,
    TpAllReduceReq,
)
from .facade import (
    add_rms_norm,
    attention,
    attention_dispatcher,
    can_run_attention,
    qk_norm,
    qk_norm_packed_rope,
    qk_norm_rope,
    rope,
    rms_norm,
    silu_and_mul,
    tp_all_reduce,
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
    "attention_dispatcher",
    "can_run_attention",
    "qk_norm",
    "qk_norm_packed_rope",
    "qk_norm_rope",
    "rope",
    "rms_norm",
    "silu_and_mul",
    "tp_all_reduce",
]
