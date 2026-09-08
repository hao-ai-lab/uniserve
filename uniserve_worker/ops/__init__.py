"""Model-facing operator facade.

Models import this package instead of vendor kernels or backend registries.
"""

from __future__ import annotations

from .core import Dispatcher, Operator
from .facade import (
    add_rms_norm,
    attention,
    can_run_attention,
    qk_norm,
    qk_norm_rope,
    rms_norm,
    rope,
    silu_and_mul,
)
from .modulation import (
    gated_residual,
    gated_residual_rms_norm,
    gated_residual_rms_norm_fp8,
    modulated_rms_norm,
)
from .requests import (
    AddRmsNormReq,
    AttentionReq,
    DenseAttention,
    MultiAxisQKNormReq,
    MultiAxisQKNormRopeReq,
    PackedRopeReq,
    PagedDecodeAttention,
    QKNormReq,
    QKNormRopeReq,
    RmsNormReq,
    SiluAndMulReq,
    VarlenAttention,
    VisibleEndAttention,
)
from .rope import apply_rotary_emb, apply_rotary_pos_emb, rotate_half
from .silu import (
    silu_and_mul_fp8,
    value_first_swiglu,
    value_first_swiglu_absmax,
    value_first_swiglu_fp8,
)

__all__ = [
    "modulated_rms_norm",
    "gated_residual",
    "gated_residual_rms_norm",
    "gated_residual_rms_norm_fp8",
    "value_first_swiglu",
    "value_first_swiglu_absmax",
    "value_first_swiglu_fp8",
    "AddRmsNormReq",
    "AttentionReq",
    "DenseAttention",
    "Dispatcher",
    "MultiAxisQKNormReq",
    "MultiAxisQKNormRopeReq",
    "Operator",
    "PackedRopeReq",
    "PagedDecodeAttention",
    "QKNormReq",
    "QKNormRopeReq",
    "RmsNormReq",
    "SiluAndMulReq",
    "VarlenAttention",
    "VisibleEndAttention",
    "add_rms_norm",
    "apply_rotary_emb",
    "apply_rotary_pos_emb",
    "attention",
    "can_run_attention",
    "qk_norm",
    "qk_norm_rope",
    "rms_norm",
    "rope",
    "rotate_half",
    "silu_and_mul",
    "silu_and_mul_fp8",
]
