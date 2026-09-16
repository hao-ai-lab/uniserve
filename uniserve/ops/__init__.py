"""Model-facing operator facade.

Models import this package instead of vendor kernels or backend registries.
"""

from __future__ import annotations

from uniserve.ops.core import Dispatcher, Operator
from uniserve.ops.facade import (
    add_rms_norm,
    qk_norm,
    qk_norm_rope,
    rms_norm,
    rope,
    silu_and_mul,
)
from uniserve.ops.modulation import (
    gated_residual,
    gated_residual_rms_norm,
    gated_residual_rms_norm_fp8,
    modulated_rms_norm,
)
from uniserve.ops.requests import (
    AddRmsNormReq,
    MultiAxisQKNormReq,
    MultiAxisQKNormRopeReq,
    PackedRopeReq,
    QKNormReq,
    QKNormRopeReq,
    RmsNormReq,
    SiluAndMulReq,
)
from uniserve.ops.rope import (
    apply_rotary_emb,
    apply_rotary_pos_emb,
    rotate_half,
)
from uniserve.ops.silu import (
    silu_and_mul_fp8,
    swiglu,
    swiglu_absmax,
    value_first_swiglu,
    value_first_swiglu_absmax,
    value_first_swiglu_fp8,
)

__all__ = [
    "modulated_rms_norm",
    "gated_residual",
    "gated_residual_rms_norm",
    "gated_residual_rms_norm_fp8",
    "swiglu",
    "swiglu_absmax",
    "value_first_swiglu",
    "value_first_swiglu_absmax",
    "value_first_swiglu_fp8",
    "AddRmsNormReq",
    "Dispatcher",
    "MultiAxisQKNormReq",
    "MultiAxisQKNormRopeReq",
    "Operator",
    "PackedRopeReq",
    "QKNormReq",
    "QKNormRopeReq",
    "RmsNormReq",
    "SiluAndMulReq",
    "add_rms_norm",
    "apply_rotary_emb",
    "apply_rotary_pos_emb",
    "qk_norm",
    "qk_norm_rope",
    "rms_norm",
    "rope",
    "rotate_half",
    "silu_and_mul",
    "silu_and_mul_fp8",
]
