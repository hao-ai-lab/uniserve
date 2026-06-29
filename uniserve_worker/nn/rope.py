"""Rotary embedding utilities.

The file deliberately supports both current call patterns:
* packed decoder paths use unduplicated cos/sin with tensors shaped ``[seq, heads, dim]``.
* HF-shaped decoder paths use duplicated cos/sin with tensors shaped
  ``[batch, heads, seq, dim]`` after an unsqueeze.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from ..foundation.triton_compat import triton_device_supported, triton_fused_layers_enabled

__all__ = [
    'rotate_half',
    'apply_rotary_emb',
    'try_triton_qk_rms_norm_rope',
    'can_run_triton_sensenova_qk_rms_norm_rope_3d',
    'try_triton_sensenova_qk_rms_norm_rope_3d',
    'apply_rotary_pos_emb',
    'RotaryEmbedding',
    'HFRotaryEmbedding',
    'get_rope',
]

try:  # transformers is present in production, but keep shared layers importable in light envs.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
except Exception:  # pragma: no cover - exercised only in minimal dependency environments.
    ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}

    def dynamic_rope_update(fn):
        return fn

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


# Triton block tile for the packed RoPE kernel; fixed by the kernel build.
_TRITON_ROPE_BLOCK = 256


if triton is not None:

    @triton.jit
    def _sensenova_square_pair(ptr, base, stride: tl.constexpr, start: tl.constexpr, active, offset: tl.constexpr):
        x0 = tl.load(ptr + base + (start + offset) * stride, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(ptr + base + (start + 32 + offset) * stride, mask=active, other=0.0).to(tl.float32)
        return ((x0 * x0).to(tl.float32) + (x1 * x1).to(tl.float32)).to(tl.float32)

    @triton.jit
    def _sensenova_add_rn(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sensenova_sub_rn(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sensenova_mul_rn(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sensenova_mean_square_64(ptr, base, stride: tl.constexpr, start: tl.constexpr, active):
        p0 = _sensenova_square_pair(ptr, base, stride, start, active, 0)
        p1 = _sensenova_square_pair(ptr, base, stride, start, active, 1)
        p2 = _sensenova_square_pair(ptr, base, stride, start, active, 2)
        p3 = _sensenova_square_pair(ptr, base, stride, start, active, 3)
        p4 = _sensenova_square_pair(ptr, base, stride, start, active, 4)
        p5 = _sensenova_square_pair(ptr, base, stride, start, active, 5)
        p6 = _sensenova_square_pair(ptr, base, stride, start, active, 6)
        p7 = _sensenova_square_pair(ptr, base, stride, start, active, 7)
        p8 = _sensenova_square_pair(ptr, base, stride, start, active, 8)
        p9 = _sensenova_square_pair(ptr, base, stride, start, active, 9)
        p10 = _sensenova_square_pair(ptr, base, stride, start, active, 10)
        p11 = _sensenova_square_pair(ptr, base, stride, start, active, 11)
        p12 = _sensenova_square_pair(ptr, base, stride, start, active, 12)
        p13 = _sensenova_square_pair(ptr, base, stride, start, active, 13)
        p14 = _sensenova_square_pair(ptr, base, stride, start, active, 14)
        p15 = _sensenova_square_pair(ptr, base, stride, start, active, 15)
        p16 = _sensenova_square_pair(ptr, base, stride, start, active, 16)
        p17 = _sensenova_square_pair(ptr, base, stride, start, active, 17)
        p18 = _sensenova_square_pair(ptr, base, stride, start, active, 18)
        p19 = _sensenova_square_pair(ptr, base, stride, start, active, 19)
        p20 = _sensenova_square_pair(ptr, base, stride, start, active, 20)
        p21 = _sensenova_square_pair(ptr, base, stride, start, active, 21)
        p22 = _sensenova_square_pair(ptr, base, stride, start, active, 22)
        p23 = _sensenova_square_pair(ptr, base, stride, start, active, 23)
        p24 = _sensenova_square_pair(ptr, base, stride, start, active, 24)
        p25 = _sensenova_square_pair(ptr, base, stride, start, active, 25)
        p26 = _sensenova_square_pair(ptr, base, stride, start, active, 26)
        p27 = _sensenova_square_pair(ptr, base, stride, start, active, 27)
        p28 = _sensenova_square_pair(ptr, base, stride, start, active, 28)
        p29 = _sensenova_square_pair(ptr, base, stride, start, active, 29)
        p30 = _sensenova_square_pair(ptr, base, stride, start, active, 30)
        p31 = _sensenova_square_pair(ptr, base, stride, start, active, 31)
        s0 = _sensenova_add_rn(p0, p1)
        s1 = _sensenova_add_rn(p2, p3)
        s2 = _sensenova_add_rn(p4, p5)
        s3 = _sensenova_add_rn(p6, p7)
        s4 = _sensenova_add_rn(p8, p9)
        s5 = _sensenova_add_rn(p10, p11)
        s6 = _sensenova_add_rn(p12, p13)
        s7 = _sensenova_add_rn(p14, p15)
        s8 = _sensenova_add_rn(p16, p17)
        s9 = _sensenova_add_rn(p18, p19)
        s10 = _sensenova_add_rn(p20, p21)
        s11 = _sensenova_add_rn(p22, p23)
        s12 = _sensenova_add_rn(p24, p25)
        s13 = _sensenova_add_rn(p26, p27)
        s14 = _sensenova_add_rn(p28, p29)
        s15 = _sensenova_add_rn(p30, p31)
        t0 = _sensenova_add_rn(s0, s1)
        t1 = _sensenova_add_rn(s2, s3)
        t2 = _sensenova_add_rn(s4, s5)
        t3 = _sensenova_add_rn(s6, s7)
        t4 = _sensenova_add_rn(s8, s9)
        t5 = _sensenova_add_rn(s10, s11)
        t6 = _sensenova_add_rn(s12, s13)
        t7 = _sensenova_add_rn(s14, s15)
        u0 = _sensenova_add_rn(t0, t1)
        u1 = _sensenova_add_rn(t2, t3)
        u2 = _sensenova_add_rn(t4, t5)
        u3 = _sensenova_add_rn(t6, t7)
        v0 = _sensenova_add_rn(u0, u1)
        v1 = _sensenova_add_rn(u2, u3)
        return _sensenova_add_rn(v0, v1) / 64.0

    @triton.jit
    def _sensenova_mean_square(ptr, base, stride: tl.constexpr, start: tl.constexpr, active, n_cols: tl.constexpr, block: tl.constexpr):
        if n_cols == 64:
            return _sensenova_mean_square_64(ptr, base, stride, start, active)
        offs = tl.arange(0, block)
        mask = active & (offs < n_cols)
        x = tl.load(ptr + base + (start + offs) * stride, mask=mask, other=0.0).to(tl.float32)
        return tl.sum(x * x, axis=0) / n_cols

    @triton.jit
    def _packed_rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        total: tl.constexpr,
        heads: tl.constexpr,
        dim: tl.constexpr,
        half: tl.constexpr,
        block: tl.constexpr,
    ):
        offs = tl.program_id(0) * block + tl.arange(0, block)
        mask = offs < total
        d = offs % dim
        row = offs // dim
        token = row // heads
        d_half = d % half
        base = row * dim
        x1 = tl.load(x_ptr + base + d_half, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + base + half + d_half, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + token * half + d_half, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + token * half + d_half, mask=mask, other=0.0).to(tl.float32)
        first_half = d < half
        out = tl.where(first_half, x1 * cos - x2 * sin, x2 * cos + x1 * sin)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _qk_rms_norm_rope_kernel(
        q_ptr,
        k_ptr,
        qw_ptr,
        kw_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        q_rows: tl.constexpr,
        k_rows: tl.constexpr,
        q_heads: tl.constexpr,
        k_heads: tl.constexpr,
        q_stride_0: tl.constexpr,
        q_stride_1: tl.constexpr,
        q_stride_2: tl.constexpr,
        k_stride_0: tl.constexpr,
        k_stride_1: tl.constexpr,
        k_stride_2: tl.constexpr,
        dim: tl.constexpr,
        half: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        block: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, block)
        col_mask = offs < dim
        d_half = offs % half
        second_offs = half + d_half
        first_half = offs < half

        q_mask = (pid < q_rows) & col_mask
        q_token = pid // q_heads
        q_head = pid - q_token * q_heads
        q_base = q_token * q_stride_0 + q_head * q_stride_1
        q_vec = tl.load(q_ptr + q_base + offs * q_stride_2, mask=q_mask, other=0.0).to(tl.float32)
        q_var = tl.sum(q_vec * q_vec, axis=0) / dim
        q_inv = tl.rsqrt(q_var + q_eps)
        q1 = tl.load(q_ptr + q_base + d_half * q_stride_2, mask=q_mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + q_base + second_offs * q_stride_2, mask=q_mask, other=0.0).to(tl.float32)
        qw1 = tl.load(qw_ptr + d_half, mask=col_mask, other=0.0).to(tl.float32)
        qw2 = tl.load(qw_ptr + second_offs, mask=col_mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + q_token * half + d_half, mask=q_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + q_token * half + d_half, mask=q_mask, other=0.0).to(tl.float32)
        # Match eager RMSNorm ordering exactly: compute variance in fp32, cast
        # the normalized activation back to the tensor dtype, multiply by the
        # dtype-matched weight, then feed that rounded value into RoPE. The
        # mathematically nicer fp32-through-RoPE fusion is not numerically
        # equivalent for bf16 multimodal generation and changes images.
        q1_norm = (q1 * q_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q2_norm = (q2 * q_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q1_norm = (q1_norm * qw1).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q2_norm = (q2_norm * qw2).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_rot = tl.where(first_half, q1_norm * cos - q2_norm * sin, q2_norm * cos + q1_norm * sin)
        tl.store(q_out_ptr + pid * dim + offs, q_rot, mask=q_mask)

        k_pid = pid - q_rows
        k_mask = (k_pid >= 0) & (k_pid < k_rows) & col_mask
        k_token = k_pid // k_heads
        k_head = k_pid - k_token * k_heads
        k_base = k_token * k_stride_0 + k_head * k_stride_1
        k_vec = tl.load(k_ptr + k_base + offs * k_stride_2, mask=k_mask, other=0.0).to(tl.float32)
        k_var = tl.sum(k_vec * k_vec, axis=0) / dim
        k_inv = tl.rsqrt(k_var + k_eps)
        k1 = tl.load(k_ptr + k_base + d_half * k_stride_2, mask=k_mask, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + k_base + second_offs * k_stride_2, mask=k_mask, other=0.0).to(tl.float32)
        kw1 = tl.load(kw_ptr + d_half, mask=col_mask, other=0.0).to(tl.float32)
        kw2 = tl.load(kw_ptr + second_offs, mask=col_mask, other=0.0).to(tl.float32)
        k_cos = tl.load(cos_ptr + k_token * half + d_half, mask=k_mask, other=0.0).to(tl.float32)
        k_sin = tl.load(sin_ptr + k_token * half + d_half, mask=k_mask, other=0.0).to(tl.float32)
        k1_norm = (k1 * k_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k2_norm = (k2 * k_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k1_norm = (k1_norm * kw1).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k2_norm = (k2_norm * kw2).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_rot = tl.where(first_half, k1_norm * k_cos - k2_norm * k_sin, k2_norm * k_cos + k1_norm * k_sin)
        tl.store(k_out_ptr + k_pid * dim + offs, k_rot, mask=k_mask)

    @triton.jit
    def _sensenova_qk_rms_norm_rope_3d_kernel(
        q_ptr,
        k_ptr,
        qw_t_ptr,
        qw_hw_ptr,
        kw_t_ptr,
        kw_hw_ptr,
        cos_t_ptr,
        sin_t_ptr,
        cos_h_ptr,
        sin_h_ptr,
        cos_w_ptr,
        sin_w_ptr,
        q_out_ptr,
        k_out_ptr,
        q_rows: tl.constexpr,
        k_rows: tl.constexpr,
        q_heads: tl.constexpr,
        k_heads: tl.constexpr,
        q_stride_0: tl.constexpr,
        q_stride_1: tl.constexpr,
        q_stride_2: tl.constexpr,
        k_stride_0: tl.constexpr,
        k_stride_1: tl.constexpr,
        k_stride_2: tl.constexpr,
        dim: tl.constexpr,
        t_dim: tl.constexpr,
        t_half: tl.constexpr,
        hw_dim: tl.constexpr,
        hw_half: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        block: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, block)
        t_col = offs < t_dim
        t_first = offs < t_half
        t_pair = offs % t_half
        t_second = t_half + t_pair
        hw_col = offs < hw_dim
        hw_first = offs < hw_half
        hw_pair = offs % hw_half
        hw_second = hw_half + hw_pair

        q_active = pid < q_rows
        q_token = pid // q_heads
        q_head = pid - q_token * q_heads
        q_base = q_token * q_stride_0 + q_head * q_stride_1

        q_t_var = _sensenova_mean_square(q_ptr, q_base, q_stride_2, 0, q_active, t_dim, block)
        q_t_inv = tl.rsqrt(q_t_var + q_eps)
        q_t_1 = tl.load(q_ptr + q_base + t_pair * q_stride_2, mask=q_active & t_col, other=0.0).to(tl.float32)
        q_t_2 = tl.load(q_ptr + q_base + t_second * q_stride_2, mask=q_active & t_col, other=0.0).to(tl.float32)
        qw_t_1 = tl.load(qw_t_ptr + t_pair, mask=t_col, other=0.0).to(tl.float32)
        qw_t_2 = tl.load(qw_t_ptr + t_second, mask=t_col, other=0.0).to(tl.float32)
        q_cos_t = tl.load(cos_t_ptr + q_token * t_half + t_pair, mask=q_active & t_col, other=0.0).to(tl.float32)
        q_sin_t = tl.load(sin_t_ptr + q_token * t_half + t_pair, mask=q_active & t_col, other=0.0).to(tl.float32)
        q_t_1 = (q_t_1 * q_t_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_t_2 = (q_t_2 * q_t_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_t_1 = (q_t_1 * qw_t_1).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_t_2 = (q_t_2 * qw_t_2).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_t_left = _sensenova_mul_rn(q_t_1, q_cos_t)
        q_t_right = _sensenova_mul_rn(q_t_2, q_sin_t)
        q_t_second_left = _sensenova_mul_rn(q_t_2, q_cos_t)
        q_t_second_right = _sensenova_mul_rn(q_t_1, q_sin_t)
        q_t_rot = tl.where(t_first, _sensenova_sub_rn(q_t_left, q_t_right), _sensenova_add_rn(q_t_second_left, q_t_second_right))
        q_t_rot = q_t_rot.to(q_out_ptr.dtype.element_ty).to(tl.float32)
        tl.store(q_out_ptr + pid * dim + offs, q_t_rot, mask=q_active & t_col)

        q_hw_var = _sensenova_mean_square(q_ptr, q_base, q_stride_2, t_dim, q_active, t_dim, block)
        q_hw_inv = tl.rsqrt(q_hw_var + q_eps)
        q_h_1 = tl.load(q_ptr + q_base + (t_dim + hw_pair) * q_stride_2, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_h_2 = tl.load(q_ptr + q_base + (t_dim + hw_second) * q_stride_2, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_w_1 = tl.load(q_ptr + q_base + (t_dim + hw_dim + hw_pair) * q_stride_2, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_w_2 = tl.load(q_ptr + q_base + (t_dim + hw_dim + hw_second) * q_stride_2, mask=q_active & hw_col, other=0.0).to(tl.float32)
        qw_h_1 = tl.load(qw_hw_ptr + hw_pair, mask=hw_col, other=0.0).to(tl.float32)
        qw_h_2 = tl.load(qw_hw_ptr + hw_second, mask=hw_col, other=0.0).to(tl.float32)
        qw_w_1 = tl.load(qw_hw_ptr + hw_dim + hw_pair, mask=hw_col, other=0.0).to(tl.float32)
        qw_w_2 = tl.load(qw_hw_ptr + hw_dim + hw_second, mask=hw_col, other=0.0).to(tl.float32)
        q_cos_h = tl.load(cos_h_ptr + q_token * hw_half + hw_pair, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_sin_h = tl.load(sin_h_ptr + q_token * hw_half + hw_pair, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_cos_w = tl.load(cos_w_ptr + q_token * hw_half + hw_pair, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_sin_w = tl.load(sin_w_ptr + q_token * hw_half + hw_pair, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_h_1 = (q_h_1 * q_hw_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_h_2 = (q_h_2 * q_hw_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_w_1 = (q_w_1 * q_hw_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_w_2 = (q_w_2 * q_hw_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_h_1 = (q_h_1 * qw_h_1).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_h_2 = (q_h_2 * qw_h_2).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_w_1 = (q_w_1 * qw_w_1).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_w_2 = (q_w_2 * qw_w_2).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_h_left = _sensenova_mul_rn(q_h_1, q_cos_h)
        q_h_right = _sensenova_mul_rn(q_h_2, q_sin_h)
        q_h_second_left = _sensenova_mul_rn(q_h_2, q_cos_h)
        q_h_second_right = _sensenova_mul_rn(q_h_1, q_sin_h)
        q_w_left = _sensenova_mul_rn(q_w_1, q_cos_w)
        q_w_right = _sensenova_mul_rn(q_w_2, q_sin_w)
        q_w_second_left = _sensenova_mul_rn(q_w_2, q_cos_w)
        q_w_second_right = _sensenova_mul_rn(q_w_1, q_sin_w)
        q_h_rot = tl.where(hw_first, _sensenova_sub_rn(q_h_left, q_h_right), _sensenova_add_rn(q_h_second_left, q_h_second_right))
        q_w_rot = tl.where(hw_first, _sensenova_sub_rn(q_w_left, q_w_right), _sensenova_add_rn(q_w_second_left, q_w_second_right))
        q_h_rot = q_h_rot.to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_w_rot = q_w_rot.to(q_out_ptr.dtype.element_ty).to(tl.float32)
        tl.store(q_out_ptr + pid * dim + t_dim + offs, q_h_rot, mask=q_active & hw_col)
        tl.store(q_out_ptr + pid * dim + t_dim + hw_dim + offs, q_w_rot, mask=q_active & hw_col)

        k_pid = pid - q_rows
        k_active = (k_pid >= 0) & (k_pid < k_rows)
        k_token = k_pid // k_heads
        k_head = k_pid - k_token * k_heads
        k_base = k_token * k_stride_0 + k_head * k_stride_1

        k_t_var = _sensenova_mean_square(k_ptr, k_base, k_stride_2, 0, k_active, t_dim, block)
        k_t_inv = tl.rsqrt(k_t_var + k_eps)
        k_t_1 = tl.load(k_ptr + k_base + t_pair * k_stride_2, mask=k_active & t_col, other=0.0).to(tl.float32)
        k_t_2 = tl.load(k_ptr + k_base + t_second * k_stride_2, mask=k_active & t_col, other=0.0).to(tl.float32)
        kw_t_1 = tl.load(kw_t_ptr + t_pair, mask=t_col, other=0.0).to(tl.float32)
        kw_t_2 = tl.load(kw_t_ptr + t_second, mask=t_col, other=0.0).to(tl.float32)
        k_cos_t = tl.load(cos_t_ptr + k_token * t_half + t_pair, mask=k_active & t_col, other=0.0).to(tl.float32)
        k_sin_t = tl.load(sin_t_ptr + k_token * t_half + t_pair, mask=k_active & t_col, other=0.0).to(tl.float32)
        k_t_1 = (k_t_1 * k_t_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_t_2 = (k_t_2 * k_t_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_t_1 = (k_t_1 * kw_t_1).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_t_2 = (k_t_2 * kw_t_2).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_t_left = _sensenova_mul_rn(k_t_1, k_cos_t)
        k_t_right = _sensenova_mul_rn(k_t_2, k_sin_t)
        k_t_second_left = _sensenova_mul_rn(k_t_2, k_cos_t)
        k_t_second_right = _sensenova_mul_rn(k_t_1, k_sin_t)
        k_t_rot = tl.where(t_first, _sensenova_sub_rn(k_t_left, k_t_right), _sensenova_add_rn(k_t_second_left, k_t_second_right))
        k_t_rot = k_t_rot.to(k_out_ptr.dtype.element_ty).to(tl.float32)
        tl.store(k_out_ptr + k_pid * dim + offs, k_t_rot, mask=k_active & t_col)

        k_hw_var = _sensenova_mean_square(k_ptr, k_base, k_stride_2, t_dim, k_active, t_dim, block)
        k_hw_inv = tl.rsqrt(k_hw_var + k_eps)
        k_h_1 = tl.load(k_ptr + k_base + (t_dim + hw_pair) * k_stride_2, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_h_2 = tl.load(k_ptr + k_base + (t_dim + hw_second) * k_stride_2, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_w_1 = tl.load(k_ptr + k_base + (t_dim + hw_dim + hw_pair) * k_stride_2, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_w_2 = tl.load(k_ptr + k_base + (t_dim + hw_dim + hw_second) * k_stride_2, mask=k_active & hw_col, other=0.0).to(tl.float32)
        kw_h_1 = tl.load(kw_hw_ptr + hw_pair, mask=hw_col, other=0.0).to(tl.float32)
        kw_h_2 = tl.load(kw_hw_ptr + hw_second, mask=hw_col, other=0.0).to(tl.float32)
        kw_w_1 = tl.load(kw_hw_ptr + hw_dim + hw_pair, mask=hw_col, other=0.0).to(tl.float32)
        kw_w_2 = tl.load(kw_hw_ptr + hw_dim + hw_second, mask=hw_col, other=0.0).to(tl.float32)
        k_cos_h = tl.load(cos_h_ptr + k_token * hw_half + hw_pair, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_sin_h = tl.load(sin_h_ptr + k_token * hw_half + hw_pair, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_cos_w = tl.load(cos_w_ptr + k_token * hw_half + hw_pair, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_sin_w = tl.load(sin_w_ptr + k_token * hw_half + hw_pair, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_h_1 = (k_h_1 * k_hw_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_h_2 = (k_h_2 * k_hw_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_w_1 = (k_w_1 * k_hw_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_w_2 = (k_w_2 * k_hw_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_h_1 = (k_h_1 * kw_h_1).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_h_2 = (k_h_2 * kw_h_2).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_w_1 = (k_w_1 * kw_w_1).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_w_2 = (k_w_2 * kw_w_2).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_h_left = _sensenova_mul_rn(k_h_1, k_cos_h)
        k_h_right = _sensenova_mul_rn(k_h_2, k_sin_h)
        k_h_second_left = _sensenova_mul_rn(k_h_2, k_cos_h)
        k_h_second_right = _sensenova_mul_rn(k_h_1, k_sin_h)
        k_w_left = _sensenova_mul_rn(k_w_1, k_cos_w)
        k_w_right = _sensenova_mul_rn(k_w_2, k_sin_w)
        k_w_second_left = _sensenova_mul_rn(k_w_2, k_cos_w)
        k_w_second_right = _sensenova_mul_rn(k_w_1, k_sin_w)
        k_h_rot = tl.where(hw_first, _sensenova_sub_rn(k_h_left, k_h_right), _sensenova_add_rn(k_h_second_left, k_h_second_right))
        k_w_rot = tl.where(hw_first, _sensenova_sub_rn(k_w_left, k_w_right), _sensenova_add_rn(k_w_second_left, k_w_second_right))
        k_h_rot = k_h_rot.to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_w_rot = k_w_rot.to(k_out_ptr.dtype.element_ty).to(tl.float32)
        tl.store(k_out_ptr + k_pid * dim + t_dim + offs, k_h_rot, mask=k_active & hw_col)
        tl.store(k_out_ptr + k_pid * dim + t_dim + hw_dim + offs, k_w_rot, mask=k_active & hw_col)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Packed GPT-NeoX style RoPE.

    ``x`` is ``[seq, heads, dim]`` and ``cos``/``sin`` are ``[seq, dim/2]``.
    """

    from uniserve_worker import ops

    return ops.rope(x, cos, sin)


def try_triton_qk_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _qk_rms_norm_rope_is_eligible(q, k, q_weight, k_weight, cos, sin):
        return None
    shape = _qk_rms_norm_rope_shape(q, k)
    if shape is None:
        return None
    q_tokens, k_tokens, q_heads, k_heads, dim = shape
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    q_rows = q_tokens * q_heads
    k_rows = k_tokens * k_heads
    _qk_rms_norm_rope_kernel[(q_rows + k_rows,)](
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_out,
        q_rows,
        k_rows,
        q_heads,
        k_heads,
        int(q.stride(0)),
        int(q.stride(1)),
        int(q.stride(2)),
        int(k.stride(0)),
        int(k.stride(1)),
        int(k.stride(2)),
        dim,
        dim // 2,
        float(q_eps),
        float(k_eps),
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    return q_out, k_out


def try_triton_sensenova_qk_rms_norm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight_t: torch.Tensor,
    q_weight_hw: torch.Tensor,
    k_weight_t: torch.Tensor,
    k_weight_hw: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _sensenova_qk_rms_norm_rope_3d_is_eligible(
        q,
        k,
        q_weight_t,
        q_weight_hw,
        k_weight_t,
        k_weight_hw,
        cos_t,
        sin_t,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
    ):
        return None
    q_tokens = int(q.shape[0])
    k_tokens = int(k.shape[0])
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    dim = int(q.shape[-1])
    t_dim = dim // 2
    hw_dim = dim // 4
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    q_rows = q_tokens * q_heads
    k_rows = k_tokens * k_heads
    _sensenova_qk_rms_norm_rope_3d_kernel[(q_rows + k_rows,)](
        q,
        k,
        q_weight_t,
        q_weight_hw,
        k_weight_t,
        k_weight_hw,
        cos_t,
        sin_t,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        q_out,
        k_out,
        q_rows,
        k_rows,
        q_heads,
        k_heads,
        int(q.stride(0)),
        int(q.stride(1)),
        int(q.stride(2)),
        int(k.stride(0)),
        int(k.stride(1)),
        int(k.stride(2)),
        dim,
        t_dim,
        t_dim // 2,
        hw_dim,
        hw_dim // 2,
        float(q_eps),
        float(k_eps),
        triton.next_power_of_2(t_dim),
        num_warps=4,
    )
    return q_out, k_out


def can_run_triton_sensenova_qk_rms_norm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight_t: torch.Tensor,
    q_weight_hw: torch.Tensor,
    k_weight_t: torch.Tensor,
    k_weight_hw: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> bool:
    return _sensenova_qk_rms_norm_rope_3d_is_eligible(
        q,
        k,
        q_weight_t,
        q_weight_hw,
        k_weight_t,
        k_weight_hw,
        cos_t,
        sin_t,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
    )


def _qk_rms_norm_rope_is_eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    if triton is None or not triton_fused_layers_enabled() or torch.is_grad_enabled():
        return False
    if not _qk_rms_norm_rope_tensors_on_supported_device(q, k, q_weight, k_weight, cos, sin):
        return False
    return _qk_rms_norm_rope_shapes_match(q, k, q_weight, k_weight, cos, sin)


def _sensenova_qk_rms_norm_rope_3d_is_eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight_t: torch.Tensor,
    q_weight_hw: torch.Tensor,
    k_weight_t: torch.Tensor,
    k_weight_hw: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> bool:
    tensors = (
        q,
        k,
        q_weight_t,
        q_weight_hw,
        k_weight_t,
        k_weight_hw,
        cos_t,
        sin_t,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
    )
    if triton is None or not triton_fused_layers_enabled() or torch.is_grad_enabled():
        return False
    if not all(t.is_cuda and t.device == q.device for t in tensors):
        return False
    if not triton_device_supported(q.device):
        return False
    if q.dtype != k.dtype or q.ndim != 3 or k.ndim != 3:
        return False
    if int(q.stride(-1)) != 1 or int(k.stride(-1)) != 1:
        return False
    if not all(t.is_contiguous() for t in tensors[2:]):
        return False
    if int(q.shape[0]) <= 0 or int(k.shape[0]) <= 0 or int(q.shape[1]) <= 0 or int(k.shape[1]) <= 0:
        return False
    if int(q.shape[0]) != int(k.shape[0]) or int(q.shape[-1]) != int(k.shape[-1]):
        return False
    dim = int(q.shape[-1])
    if dim <= 0 or dim % 8 != 0 or dim > 1024:
        return False
    t_dim = dim // 2
    hw_dim = dim // 4
    hw_half = hw_dim // 2
    return (
        q_weight_t.shape == (t_dim,)
        and k_weight_t.shape == (t_dim,)
        and q_weight_hw.shape == (t_dim,)
        and k_weight_hw.shape == (t_dim,)
        and cos_t.shape == (int(q.shape[0]), t_dim // 2)
        and sin_t.shape == cos_t.shape
        and cos_h.shape == (int(q.shape[0]), hw_half)
        and sin_h.shape == cos_h.shape
        and cos_w.shape == (int(q.shape[0]), hw_half)
        and sin_w.shape == cos_w.shape
    )


def _qk_rms_norm_rope_tensors_on_supported_device(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    return (
        q.is_cuda
        and k.is_cuda
        and q_weight.is_cuda
        and k_weight.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and triton_device_supported(q.device)
        and q.device == k.device
        and q.device == cos.device
        and q.device == sin.device
    )


def _qk_rms_norm_rope_shapes_match(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    return (
        q.dtype == k.dtype
        and q.ndim == 3
        and k.ndim == 3
        and cos.ndim == 2
        and sin.shape == cos.shape
        and q.shape[0] == cos.shape[0]
        and k.shape[0] == cos.shape[0]
        and q.shape[-1] == k.shape[-1]
        and q.shape[-1] == q_weight.numel()
        and k.shape[-1] == k_weight.numel()
        and q.shape[-1] == cos.shape[-1] * 2
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and int(q.stride(-1)) == 1
        and int(k.stride(-1)) == 1
    )


def _qk_rms_norm_rope_shape(q: torch.Tensor, k: torch.Tensor) -> tuple[int, int, int, int, int] | None:
    dim = int(q.shape[-1])
    if dim <= 0 or dim % 2 != 0 or dim > 1024:
        return None
    q_tokens = int(q.shape[0])
    k_tokens = int(k.shape[0])
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    if q_tokens <= 0 or k_tokens <= 0 or q_heads <= 0 or k_heads <= 0:
        return None
    return q_tokens, k_tokens, q_heads, k_heads, dim


class _TritonPackedRope:
    def is_eligible(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
        if (
            triton is None
            or not triton_fused_layers_enabled()
            or not x.is_cuda
            or not cos.is_cuda
            or not sin.is_cuda
            or not triton_device_supported(x.device)
            or torch.is_grad_enabled()
            or not x.is_contiguous()
            or not cos.is_contiguous()
            or not sin.is_contiguous()
            or x.ndim < 2
            or cos.ndim != 2
            or sin.shape != cos.shape
            or x.shape[0] != cos.shape[0]
            or x.shape[-1] != cos.shape[-1] * 2
        ):
            return False
        dim = int(x.shape[-1])
        if dim <= 0 or dim % 2 != 0:
            return False
        tokens = int(x.shape[0])
        heads = int(x.numel() // max(1, tokens * dim))
        return tokens > 0 and heads > 0

    def run(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        dim = int(x.shape[-1])
        half = dim // 2
        tokens = int(x.shape[0])
        heads = int(x.numel() // max(1, tokens * dim))
        out = torch.empty_like(x)
        block = _TRITON_ROPE_BLOCK
        total = int(x.numel())
        _packed_rope_kernel[(triton.cdiv(total, block),)](
            x,
            cos,
            sin,
            out,
            total,
            heads,
            dim,
            half,
            block,
            num_warps=4,
        )
        return out


class _EagerPackedRope:
    def is_eligible(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
        return True

    def run(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        ro_dim = cos.shape[-1] * 2
        if ro_dim != x.shape[-1]:
            raise ValueError(f"rotary dim {ro_dim} does not match tensor dim {x.shape[-1]}")
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        out = torch.empty_like(x)
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half:] = x2 * cos + x1 * sin
        return out


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen/HF style RoPE for query and key tensors."""

    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_full_dim(q, cos, sin)
    k_embed = _apply_rotary_full_dim(k, cos, sin)
    return q_embed, k_embed


def _cos_sin_bshd(
    inv_freq: torch.Tensor,
    attention_scaling: float,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return duplicated cos/sin shaped ``[batch, seq, dim]`` for HF-shaped decoders."""

    inv_freq_expanded = inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    ).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _apply_rotary_full_dim(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if cos.shape[-1] < dim or sin.shape[-1] < dim:
        raise ValueError(
            f"rotary cos/sin dim {cos.shape[-1]}/{sin.shape[-1]} is smaller than tensor dim {dim}"
        )
    cos = cos[..., :dim]
    sin = sin[..., :dim]
    rotary_dim = dim - (dim % 2)
    if rotary_dim == 0:
        return x

    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos[..., :half] - x2 * sin[..., :half]
    out[..., half:rotary_dim] = (
        x2 * cos[..., half:rotary_dim] + x1 * sin[..., half:rotary_dim]
    )
    if rotary_dim < dim:
        out[..., rotary_dim:] = x[..., rotary_dim:]
    return out


class RotaryEmbedding(nn.Module):
    """Default rotary embedding with an optional Qwen frequency-range mode."""

    def __init__(
        self,
        dim: int,
        *,
        theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        attention_scaling: float = 1.0,
        keep_freq_range: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        # ``keep_freq_range`` selects the dual-resolution rope recipe: a closed
        # model-recipe choice, not a generic toggle. When set, inv_freq is built
        # over ``dim * 2`` and decimated by two so the kept frequencies span the same
        # range as the full-dim model while the per-axis rope uses only ``dim`` entries.
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.attention_scaling = attention_scaling
        inv_dim = dim * 2 if keep_freq_range else dim
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, inv_dim, 2, dtype=torch.float32, device=device) / inv_dim)
        )
        if keep_freq_range:
            inv_freq = inv_freq[::2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return duplicated cos/sin shaped ``[batch, seq, dim]``."""

        return _cos_sin_bshd(self.inv_freq, self.attention_scaling, x, position_ids)

    @torch.no_grad()
    def cos_sin_1d(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unduplicated packed-decoder cos/sin shaped ``[seq, dim/2]``."""

        freqs = position_ids.float()[:, None] * self.inv_freq[None, :].to(position_ids.device)
        return freqs.cos(), freqs.sin()


def _compute_default_rope_parameters(config: Any, device=None, **_kwargs) -> tuple[torch.Tensor, float]:
    """Default HF/Qwen-style RoPE frequencies with stable transformers-version behavior."""

    base = _rope_config_float(config, "rope_theta", default=10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor


def _rope_config_float(config: Any, name: str, *, default: float) -> float:
    for key in (name, f"{name}_hw"):
        try:
            value = getattr(config, key)
        except AttributeError:
            value = None
        if value is not None:
            return float(value)
    for mapping_name in ("rope_parameters", "rope_scaling"):
        mapping = getattr(config, mapping_name, None)
        if isinstance(mapping, dict) and mapping.get(name) is not None:
            return float(mapping[name])
    return float(default)


class HFRotaryEmbedding(nn.Module):
    """HF-shaped rotary embedding with optional frequency-range preservation."""

    inv_freq: torch.Tensor

    def __init__(self, config: Any, *, device=None, keep_freq_range: bool = False):
        # ``keep_freq_range`` selects the dual-resolution rope recipe: a closed
        # model-recipe choice, not a generic toggle. When set, the rope init fn
        # is wrapped (see ``_keep_freq_range``) so inv_freq is computed over a doubled
        # head_dim and decimated by two, preserving the full-dim frequency range.
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.keep_freq_range = bool(keep_freq_range)
        if self.rope_type == "default" or self.rope_type is None:
            base_rope_init_fn = _compute_default_rope_parameters
        else:
            try:
                base_rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            except KeyError as exc:
                raise ValueError(f"unknown rope type {self.rope_type!r}") from exc

        if self.keep_freq_range:
            self.rope_init_fn = self._keep_freq_range(base_rope_init_fn)
        else:
            self.rope_init_fn = base_rope_init_fn

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _keep_freq_range(self, base_rope_init_fn):
        def _rope_init_fn_keep_freq_range(cfg: Any, dev=None):
            inv_freq, attention_scaling = base_rope_init_fn(cfg, dev)
            del inv_freq

            # The base init functions only read scalar fields from the config, so a
            # shallow copy with an overridden head_dim is sufficient and avoids the
            # cost of deep-copying the entire (potentially large/nested) config.
            cfg2 = copy.copy(cfg)
            head_dim = getattr(cfg2, "head_dim", None)
            if head_dim is None:
                head_dim = cfg2.hidden_size // cfg2.num_attention_heads
            cfg2.head_dim = int(head_dim) * 2

            inv_freq_full, _ = base_rope_init_fn(cfg2, dev)
            return inv_freq_full[::2], attention_scaling

        return _rope_init_fn_keep_freq_range

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _cos_sin_bshd(self.inv_freq, self.attention_scaling, x, position_ids)

    @torch.no_grad()
    def cos_sin_1d(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float()[:, None] * self.inv_freq[None, :].to(position_ids.device)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        return cos, sin


def get_rope(
    dim: int | None = None,
    *,
    theta: float = 10000.0,
    max_position_embeddings: int = 4096,
    attention_scaling: float = 1.0,
    keep_freq_range: bool = False,
    config: Any | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Build the shared RoPE implementation for packed or HF-shaped decoders.

    ``keep_freq_range`` selects the dual-resolution rope recipe: a closed
    model-recipe choice, not a generic toggle. It is forwarded to the underlying
    ``RotaryEmbedding`` / ``HFRotaryEmbedding`` to preserve the full-dim frequency range.
    """

    if config is not None:
        return HFRotaryEmbedding(config, device=device, keep_freq_range=keep_freq_range)
    if dim is None:
        raise ValueError("dim is required when config is not provided")
    return RotaryEmbedding(
        dim,
        theta=theta,
        max_position_embeddings=max_position_embeddings,
        attention_scaling=attention_scaling,
        keep_freq_range=keep_freq_range,
        device=device,
    )
