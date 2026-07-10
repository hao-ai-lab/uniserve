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
    'apply_rotary_pos_emb',
    'RotaryEmbedding',
    'HFRotaryEmbedding',
    'get_rope',
]

try:  # transformers is present in production, but keep shared layers importable in light envs.
    from transformers.modeling_rope_utils import (
        ROPE_INIT_FUNCTIONS as _HF_ROPE_INIT_FUNCTIONS,
    )
    from transformers.modeling_rope_utils import (
        dynamic_rope_update as _hf_dynamic_rope_update,
    )
except Exception:  # pragma: no cover - exercised only in minimal dependency environments.
    ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}

    def dynamic_rope_update(fn):
        return fn
else:
    ROPE_INIT_FUNCTIONS = _HF_ROPE_INIT_FUNCTIONS
    dynamic_rope_update = _hf_dynamic_rope_update

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
    def _split_rms_norm_rope_row(
        x_ptr,
        base,
        head_w_ptr,
        tail_w_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        out_base,
        token,
        row_active,
        stride_2: tl.constexpr,
        rope_dim: tl.constexpr,
        rope_half: tl.constexpr,
        tail_dim: tl.constexpr,
        eps: tl.constexpr,
        block_a: tl.constexpr,
        block_b: tl.constexpr,
    ):
        # Head group [0, rope_dim): RMS over rope_dim + NeoX RoPE. The rounding
        # chain matches _qk_rms_norm_rope_kernel exactly (norm -> cast -> weight
        # -> cast, RoPE in fp32 over the rounded values).
        offs_a = tl.arange(0, block_a)
        mask_a = (offs_a < rope_dim) & row_active
        xa = tl.load(x_ptr + base + offs_a * stride_2, mask=mask_a, other=0.0).to(tl.float32)
        var_a = tl.sum(xa * xa, axis=0) / rope_dim
        inv_a = tl.rsqrt(var_a + eps)
        d_half = offs_a % rope_half
        second_offs = rope_half + d_half
        first_half = offs_a < rope_half
        x1 = tl.load(x_ptr + base + d_half * stride_2, mask=mask_a, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + base + second_offs * stride_2, mask=mask_a, other=0.0).to(tl.float32)
        w1 = tl.load(head_w_ptr + d_half, mask=offs_a < rope_dim, other=0.0).to(tl.float32)
        w2 = tl.load(head_w_ptr + second_offs, mask=offs_a < rope_dim, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + token * rope_half + d_half, mask=mask_a, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + token * rope_half + d_half, mask=mask_a, other=0.0).to(tl.float32)
        x1n = (x1 * inv_a).to(out_ptr.dtype.element_ty).to(tl.float32)
        x2n = (x2 * inv_a).to(out_ptr.dtype.element_ty).to(tl.float32)
        x1n = (x1n * w1).to(out_ptr.dtype.element_ty).to(tl.float32)
        x2n = (x2n * w2).to(out_ptr.dtype.element_ty).to(tl.float32)
        rot = tl.where(first_half, x1n * cos - x2n * sin, x2n * cos + x1n * sin)
        tl.store(out_ptr + out_base + offs_a, rot, mask=mask_a)

        # Tail group [rope_dim, rope_dim+tail_dim): RMS over tail_dim with its
        # own weight and no rotation (the caller declared the tail axes'
        # positions all-zero, and a zero-angle rotation is the identity). The
        # single final rounding matches _qk_rms_norm_kernel.
        offs_b = tl.arange(0, block_b)
        mask_b = (offs_b < tail_dim) & row_active
        xb = tl.load(
            x_ptr + base + (rope_dim + offs_b) * stride_2, mask=mask_b, other=0.0
        ).to(tl.float32)
        var_b = tl.sum(xb * xb, axis=0) / tail_dim
        wb = tl.load(tail_w_ptr + offs_b, mask=offs_b < tail_dim, other=0.0).to(tl.float32)
        out_b = xb * tl.rsqrt(var_b + eps) * wb
        tl.store(out_ptr + out_base + rope_dim + offs_b, out_b, mask=mask_b)

    @triton.jit
    def _qk_split_rms_norm_rope_kernel(
        q_ptr,
        k_ptr,
        q_head_w_ptr,
        q_tail_w_ptr,
        k_head_w_ptr,
        k_tail_w_ptr,
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
        rope_dim: tl.constexpr,
        rope_half: tl.constexpr,
        tail_dim: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        block_a: tl.constexpr,
        block_b: tl.constexpr,
    ):
        pid = tl.program_id(0)

        q_active = pid < q_rows
        q_token = pid // q_heads
        q_head = pid - q_token * q_heads
        q_base = q_token * q_stride_0 + q_head * q_stride_1
        _split_rms_norm_rope_row(
            q_ptr,
            q_base,
            q_head_w_ptr,
            q_tail_w_ptr,
            cos_ptr,
            sin_ptr,
            q_out_ptr,
            pid * dim,
            q_token,
            q_active,
            q_stride_2,
            rope_dim,
            rope_half,
            tail_dim,
            q_eps,
            block_a,
            block_b,
        )

        k_pid = pid - q_rows
        k_active = (k_pid >= 0) & (k_pid < k_rows)
        k_token = k_pid // k_heads
        k_head = k_pid - k_token * k_heads
        k_base = k_token * k_stride_0 + k_head * k_stride_1
        _split_rms_norm_rope_row(
            k_ptr,
            k_base,
            k_head_w_ptr,
            k_tail_w_ptr,
            cos_ptr,
            sin_ptr,
            k_out_ptr,
            k_pid * dim,
            k_token,
            k_active,
            k_stride_2,
            rope_dim,
            rope_half,
            tail_dim,
            k_eps,
            block_a,
            block_b,
        )

    @triton.jit
    def _multi_axis_rms_norm_rope_row(
        x_ptr,
        base,
        head_w_ptr,
        tail_w_ptr,
        cos0_ptr,
        sin0_ptr,
        cos1_ptr,
        sin1_ptr,
        cos2_ptr,
        sin2_ptr,
        out_ptr,
        out_base,
        token,
        stride_2: tl.constexpr,
        dim0: tl.constexpr,
        half0: tl.constexpr,
        axis_dim: tl.constexpr,
        axis_half: tl.constexpr,
        tail_dim: tl.constexpr,
        eps: tl.constexpr,
        block_a: tl.constexpr,
        block_b: tl.constexpr,
    ):
        # Head group [0, dim0): RMS over dim0 + NeoX RoPE by cos0/sin0. The
        # rounding chain matches _qk_rms_norm_rope_kernel exactly (norm ->
        # cast -> weight -> cast, RoPE in fp32 over the rounded values).
        offs_a = tl.arange(0, block_a)
        col_a = offs_a < dim0
        d_half = offs_a % half0
        second = half0 + d_half
        first_half = offs_a < half0
        xa = tl.load(x_ptr + base + offs_a * stride_2, mask=col_a, other=0.0).to(tl.float32)
        var_a = tl.sum(xa * xa, axis=0) / dim0
        inv_a = tl.rsqrt(var_a + eps)
        x1 = tl.load(x_ptr + base + d_half * stride_2, mask=col_a, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + base + second * stride_2, mask=col_a, other=0.0).to(tl.float32)
        w1 = tl.load(head_w_ptr + d_half, mask=col_a, other=0.0).to(tl.float32)
        w2 = tl.load(head_w_ptr + second, mask=col_a, other=0.0).to(tl.float32)
        cos = tl.load(cos0_ptr + token * half0 + d_half, mask=col_a, other=0.0).to(tl.float32)
        sin = tl.load(sin0_ptr + token * half0 + d_half, mask=col_a, other=0.0).to(tl.float32)
        x1n = (x1 * inv_a).to(out_ptr.dtype.element_ty).to(tl.float32)
        x2n = (x2 * inv_a).to(out_ptr.dtype.element_ty).to(tl.float32)
        x1n = (x1n * w1).to(out_ptr.dtype.element_ty).to(tl.float32)
        x2n = (x2n * w2).to(out_ptr.dtype.element_ty).to(tl.float32)
        rot = tl.where(first_half, x1n * cos - x2n * sin, x2n * cos + x1n * sin)
        tl.store(out_ptr + out_base + offs_a, rot, mask=col_a)

        # Tail group [dim0, dim0+tail_dim): one shared RMS over tail_dim
        # (matching _qk_rms_norm_kernel's (x * rsqrt) * w chain), rounded to
        # the output dtype exactly where the unfused pipeline stored the
        # normed tensor, then the two tail axes are NeoX-rotated in fp32 by
        # their own tables exactly as _packed_rope_kernel reloads and rotates
        # the rounded values.
        offs_b = tl.arange(0, block_b)
        col_b = offs_b < tail_dim
        xb = tl.load(x_ptr + base + (dim0 + offs_b) * stride_2, mask=col_b, other=0.0).to(tl.float32)
        var_b = tl.sum(xb * xb, axis=0) / tail_dim
        inv_b = tl.rsqrt(var_b + eps)
        is_second_axis = offs_b >= axis_dim
        dj = (offs_b % axis_dim) % axis_half
        src1 = tl.where(is_second_axis, axis_dim, 0) + dj
        src2 = src1 + axis_half
        y1 = tl.load(x_ptr + base + (dim0 + src1) * stride_2, mask=col_b, other=0.0).to(tl.float32)
        y2 = tl.load(x_ptr + base + (dim0 + src2) * stride_2, mask=col_b, other=0.0).to(tl.float32)
        wv1 = tl.load(tail_w_ptr + src1, mask=col_b, other=0.0).to(tl.float32)
        wv2 = tl.load(tail_w_ptr + src2, mask=col_b, other=0.0).to(tl.float32)
        y1n = (y1 * inv_b * wv1).to(out_ptr.dtype.element_ty).to(tl.float32)
        y2n = (y2 * inv_b * wv2).to(out_ptr.dtype.element_ty).to(tl.float32)
        c1 = tl.load(cos1_ptr + token * axis_half + dj, mask=col_b & (~is_second_axis), other=0.0).to(tl.float32)
        s1 = tl.load(sin1_ptr + token * axis_half + dj, mask=col_b & (~is_second_axis), other=0.0).to(tl.float32)
        c2 = tl.load(cos2_ptr + token * axis_half + dj, mask=col_b & is_second_axis, other=0.0).to(tl.float32)
        s2 = tl.load(sin2_ptr + token * axis_half + dj, mask=col_b & is_second_axis, other=0.0).to(tl.float32)
        cb = tl.where(is_second_axis, c2, c1)
        sb = tl.where(is_second_axis, s2, s1)
        first_b = (offs_b % axis_dim) < axis_half
        rot_b = tl.where(first_b, y1n * cb - y2n * sb, y2n * cb + y1n * sb)
        tl.store(out_ptr + out_base + dim0 + offs_b, rot_b, mask=col_b)

    @triton.jit
    def _qk_multi_axis_rms_norm_rope_kernel(
        q_ptr,
        k_ptr,
        q_head_w_ptr,
        q_tail_w_ptr,
        k_head_w_ptr,
        k_tail_w_ptr,
        cos0_ptr,
        sin0_ptr,
        cos1_ptr,
        sin1_ptr,
        cos2_ptr,
        sin2_ptr,
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
        dim0: tl.constexpr,
        half0: tl.constexpr,
        axis_dim: tl.constexpr,
        axis_half: tl.constexpr,
        tail_dim: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        block_a: tl.constexpr,
        block_b: tl.constexpr,
    ):
        # One program per (token, head) row; the branch predicate is uniform
        # per program and depends only on program_id, so each program executes
        # exactly one row body (capture-safe, no tensor-value branching).
        pid = tl.program_id(0)
        if pid < q_rows:
            token = pid // q_heads
            head = pid - token * q_heads
            base = token * q_stride_0 + head * q_stride_1
            _multi_axis_rms_norm_rope_row(
                q_ptr, base, q_head_w_ptr, q_tail_w_ptr,
                cos0_ptr, sin0_ptr, cos1_ptr, sin1_ptr, cos2_ptr, sin2_ptr,
                q_out_ptr, pid * dim, token,
                q_stride_2, dim0, half0, axis_dim, axis_half, tail_dim, q_eps,
                block_a, block_b,
            )
        else:
            k_pid = pid - q_rows
            token = k_pid // k_heads
            head = k_pid - token * k_heads
            base = token * k_stride_0 + head * k_stride_1
            _multi_axis_rms_norm_rope_row(
                k_ptr, base, k_head_w_ptr, k_tail_w_ptr,
                cos0_ptr, sin0_ptr, cos1_ptr, sin1_ptr, cos2_ptr, sin2_ptr,
                k_out_ptr, k_pid * dim, token,
                k_stride_2, dim0, half0, axis_dim, axis_half, tail_dim, k_eps,
                block_a, block_b,
            )



def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotation: str = "neox",
) -> torch.Tensor:
    """Packed GPT-NeoX style RoPE.

    ``x`` is ``[seq, heads, dim]`` and ``cos``/``sin`` are ``[seq, dim/2]``.
    """

    if rotation == "interleaved":
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out
    if rotation != "neox":
        raise ValueError(f"unknown RoPE rotation convention {rotation!r}")

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
    if not can_run_triton_qk_rms_norm_rope(q, k, q_weight, k_weight, cos, sin, q_eps, k_eps):
        return None
    shape = _qk_rms_norm_rope_shape(q, k)
    assert shape is not None
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


def try_triton_qk_split_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_head_weight: torch.Tensor,
    q_tail_weight: torch.Tensor,
    k_head_weight: torch.Tensor,
    k_tail_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_eps: float,
    k_eps: float,
    *,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused two-group QK RMSNorm with RoPE on the head group only.

    ``q``/``k`` are ``[tokens, heads, dim]``; dims ``[0, rope_dim)`` are
    RMS-normed with the head weight and NeoX-rotated by ``cos``/``sin``
    (``[tokens, rope_dim // 2]``); dims ``[rope_dim, dim)`` are RMS-normed with
    the tail weight and passed through unrotated. This is the one-launch form
    of the multi-axis norm+RoPE pipeline for calls whose tail-axis positions
    are all zero (a zero-angle rotation is the identity), preserving each
    group's exact rounding order.
    """

    if not can_run_triton_qk_split_rms_norm_rope(
        q,
        k,
        q_head_weight,
        q_tail_weight,
        k_head_weight,
        k_tail_weight,
        cos,
        sin,
        rope_dim=rope_dim,
    ):
        return None
    q_tokens, q_heads = int(q.shape[0]), int(q.shape[1])
    k_tokens, k_heads = int(k.shape[0]), int(k.shape[1])
    dim = int(q.shape[-1])
    rope_dim = int(rope_dim)
    tail_dim = dim - rope_dim
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    q_rows = q_tokens * q_heads
    k_rows = k_tokens * k_heads
    _qk_split_rms_norm_rope_kernel[(q_rows + k_rows,)](
        q,
        k,
        q_head_weight,
        q_tail_weight,
        k_head_weight,
        k_tail_weight,
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
        rope_dim,
        rope_dim // 2,
        tail_dim,
        float(q_eps),
        float(k_eps),
        triton.next_power_of_2(rope_dim),
        triton.next_power_of_2(tail_dim),
        num_warps=4,
    )
    return q_out, k_out


def try_triton_qk_multi_axis_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_head_weight: torch.Tensor,
    q_tail_weight: torch.Tensor,
    k_head_weight: torch.Tensor,
    k_tail_weight: torch.Tensor,
    cos: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sin: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    q_eps: float,
    k_eps: float,
    *,
    axis_dims: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused 3-axis QK RMSNorm+RoPE with a rotated two-axis tail group.

    ``q``/``k`` are ``[tokens, heads, dim]`` with ``axis_dims = (d0, d1, d1)``:
    dims ``[0, d0)`` are RMS-normed with the head weight and NeoX-rotated by
    ``cos[0]``/``sin[0]``; dims ``[d0, dim)`` share one RMS over ``2*d1`` with
    the tail weight and each ``d1``-wide axis is then NeoX-rotated by its own
    table (``cos[1]``/``cos[2]``, each ``[tokens, d1 // 2]``). This is the
    one-launch form of the general multi-axis pipeline (norm+rope kernel,
    shared-norm kernel, two packed-rope launches per tensor, plus the
    interleaving contiguous/cat copies) and preserves each group's exact
    per-element arithmetic and rounding order:
    * head group: the _qk_rms_norm_rope_kernel chain verbatim;
    * tail group: _qk_rms_norm_kernel's ``(x * rsqrt(var + eps)) * w`` with a
      single rounding to the output dtype standing in for its global store,
      then _packed_rope_kernel's fp32 rotation of the rounded values.
    Reduction order note: both groups reduce a 1D power-of-two row tile
    (<= 64 wide) with ``tl.sum`` in one shot, the same tile shapes the
    unfused kernels use. With one element per lane (any num_warps >= 2 for
    tiles <= 64) the generated tree is per-warp butterfly + one cross-warp
    combine, identical to the unfused kernels' num_warps=4 launches; this is
    verified bitwise against the unfused pipeline in
    tests/python/unit/layers/test_multi_axis_qk_norm_rope_fused.py. num_warps=1
    (two elements per lane, thread-local pre-sum) changes the tree and is not
    used. Eligibility is restricted to the verified tile family
    (next_power_of_2 of both group widths in {32, 64}).
    """

    if not can_run_triton_qk_multi_axis_rms_norm_rope(
        q,
        k,
        q_head_weight,
        q_tail_weight,
        k_head_weight,
        k_tail_weight,
        cos,
        sin,
        axis_dims=axis_dims,
    ):
        return None
    tokens, q_heads = int(q.shape[0]), int(q.shape[1])
    k_heads = int(k.shape[1])
    dim = int(q.shape[-1])
    dim0, axis_dim, _ = (int(v) for v in axis_dims)
    tail_dim = dim - dim0
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    q_rows = tokens * q_heads
    k_rows = tokens * k_heads
    _qk_multi_axis_rms_norm_rope_kernel[(q_rows + k_rows,)](
        q,
        k,
        q_head_weight,
        q_tail_weight,
        k_head_weight,
        k_tail_weight,
        cos[0],
        sin[0],
        cos[1],
        sin[1],
        cos[2],
        sin[2],
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
        dim0,
        dim0 // 2,
        axis_dim,
        axis_dim // 2,
        tail_dim,
        float(q_eps),
        float(k_eps),
        triton.next_power_of_2(dim0),
        triton.next_power_of_2(tail_dim),
        num_warps=2,
    )
    return q_out, k_out


def can_run_triton_qk_multi_axis_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_head_weight: torch.Tensor,
    q_tail_weight: torch.Tensor,
    k_head_weight: torch.Tensor,
    k_tail_weight: torch.Tensor,
    cos: tuple[torch.Tensor, ...],
    sin: tuple[torch.Tensor, ...],
    *,
    axis_dims: tuple[int, ...],
) -> bool:
    if triton is None or not triton_fused_layers_enabled() or torch.is_grad_enabled():
        return False
    if len(axis_dims) != 3 or len(cos) != 3 or len(sin) != 3:
        return False
    if not _qk_rms_norm_rope_tensors_on_supported_device(
        q, k, q_head_weight, k_head_weight, cos[0], sin[0]
    ):
        return False
    if not (q_tail_weight.is_cuda and k_tail_weight.is_cuda):
        return False
    dim0, axis_dim, axis_dim2 = (int(v) for v in axis_dims)
    if axis_dim != axis_dim2 or dim0 <= 0 or axis_dim <= 0:
        return False
    if dim0 % 2 != 0 or axis_dim % 2 != 0:
        return False
    tail_dim = 2 * axis_dim
    # Only the bitwise-verified reduction tile family (see the launch note).
    if triton.next_power_of_2(dim0) not in (32, 64):
        return False
    if triton.next_power_of_2(tail_dim) not in (32, 64):
        return False
    if q.ndim != 3 or k.ndim != 3 or q.dtype != k.dtype:
        return False
    tokens = int(q.shape[0])
    dim = int(q.shape[-1])
    if dim != dim0 + tail_dim or int(k.shape[-1]) != dim:
        return False
    if tokens <= 0 or int(k.shape[0]) != tokens:
        return False
    if int(q.shape[1]) <= 0 or int(k.shape[1]) <= 0:
        return False
    if int(q.stride(-1)) != 1 or int(k.stride(-1)) != 1:
        return False
    for weight, width in (
        (q_head_weight, dim0),
        (k_head_weight, dim0),
        (q_tail_weight, tail_dim),
        (k_tail_weight, tail_dim),
    ):
        if int(weight.numel()) != width or not weight.is_contiguous():
            return False
    for axis, width in ((0, dim0), (1, axis_dim), (2, axis_dim)):
        cos_a, sin_a = cos[axis], sin[axis]
        if not (cos_a.is_cuda and sin_a.is_cuda and cos_a.device == q.device and sin_a.device == q.device):
            return False
        if cos_a.ndim != 2 or sin_a.shape != cos_a.shape:
            return False
        if int(cos_a.shape[0]) != tokens or int(cos_a.shape[-1]) != width // 2:
            return False
        if not (cos_a.is_contiguous() and sin_a.is_contiguous()):
            return False
    return True


def can_run_triton_qk_split_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_head_weight: torch.Tensor,
    q_tail_weight: torch.Tensor,
    k_head_weight: torch.Tensor,
    k_tail_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rope_dim: int,
) -> bool:
    if triton is None or not triton_fused_layers_enabled() or torch.is_grad_enabled():
        return False
    if not _qk_rms_norm_rope_tensors_on_supported_device(
        q, k, q_head_weight, k_head_weight, cos, sin
    ):
        return False
    if not (q_tail_weight.is_cuda and k_tail_weight.is_cuda):
        return False
    rope_dim = int(rope_dim)
    dim = int(q.shape[-1]) if q.ndim == 3 else 0
    tail_dim = dim - rope_dim
    return (
        q.ndim == 3
        and k.ndim == 3
        and q.dtype == k.dtype
        and 0 < rope_dim < dim <= 1024
        and rope_dim % 2 == 0
        and tail_dim > 0
        and cos.ndim == 2
        and sin.shape == cos.shape
        and cos.shape[0] == q.shape[0]
        and cos.shape[0] == k.shape[0]
        and int(cos.shape[-1]) == rope_dim // 2
        and q.shape[-1] == k.shape[-1]
        and int(q_head_weight.numel()) == rope_dim
        and int(k_head_weight.numel()) == rope_dim
        and int(q_tail_weight.numel()) == tail_dim
        and int(k_tail_weight.numel()) == tail_dim
        and q_head_weight.is_contiguous()
        and q_tail_weight.is_contiguous()
        and k_head_weight.is_contiguous()
        and k_tail_weight.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and int(q.stride(-1)) == 1
        and int(k.stride(-1)) == 1
        and int(q.shape[0]) > 0
        and int(k.shape[0]) > 0
        and int(q.shape[1]) > 0
        and int(k.shape[1]) > 0
    )


def can_run_triton_qk_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> bool:
    del q_eps, k_eps
    return (
        _qk_rms_norm_rope_is_eligible(q, k, q_weight, k_weight, cos, sin)
        and _qk_rms_norm_rope_shape(q, k) is not None
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

    inv_freq: torch.Tensor
    original_inv_freq: torch.Tensor

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
) -> RotaryEmbedding | HFRotaryEmbedding:
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
