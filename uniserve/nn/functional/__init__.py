"""Callable numerical primitives over local tensors and borrowed operators.

This package is the single numerical entry point of the computation library.
Each function defines its mathematics, including FP32 accumulation and
rounding points, and selects an eligible UniServe kernel or evaluates the same
formula with tensor operations. ``out`` arguments receive results directly
when a kernel supports their layout.
"""

from ._activation import (
    gelu_and_mul,
    silu_and_mul,
    swiglu,
    swiglu_absmax,
    value_first_swiglu,
    value_first_swiglu_absmax,
    value_first_swiglu_fp8,
)
from ._attention import attention
from ._linear import linear, merged_linear
from ._moe import fused_moe
from ._norm import (
    add_rms_norm,
    gated_residual,
    gated_residual_rms_norm,
    gated_residual_rms_norm_fp8,
    modulated_rms_norm,
    rms_norm,
    scaled_residual_,
    scaled_residual_layer_norm,
    scaled_residual_layer_norm_absmax,
    scaled_residual_rms_norm_,
    scaled_residual_rms_norm_absmax_,
    weighted_rms_norm,
    weighted_rms_norm_absmax,
)
from ._patch import patchify, unpatchify, unpatchify_video_tokens
from ._rope import apply_rotary, qk_bias_rms_norm_rope_, qk_norm_rope

__all__ = [
    "add_rms_norm",
    "apply_rotary",
    "attention",
    "gated_residual",
    "gated_residual_rms_norm",
    "gated_residual_rms_norm_fp8",
    "fused_moe",
    "gelu_and_mul",
    "linear",
    "merged_linear",
    "modulated_rms_norm",
    "patchify",
    "qk_bias_rms_norm_rope_",
    "qk_norm_rope",
    "rms_norm",
    "scaled_residual_",
    "scaled_residual_layer_norm",
    "scaled_residual_layer_norm_absmax",
    "scaled_residual_rms_norm_",
    "scaled_residual_rms_norm_absmax_",
    "silu_and_mul",
    "swiglu",
    "swiglu_absmax",
    "unpatchify",
    "unpatchify_video_tokens",
    "value_first_swiglu",
    "value_first_swiglu_absmax",
    "value_first_swiglu_fp8",
    "weighted_rms_norm",
    "weighted_rms_norm_absmax",
]
