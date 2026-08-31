"""Triton and eager launchers for packed RoPE and fused QK norm+RoPE."""
from __future__ import annotations

import torch

from ..backends.triton import triton_available

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
    def _qk_rms_norm_partial_rope_inplace_kernel(
        query,
        key,
        query_weight,
        key_weight,
        cosine,
        sine,
        query_stride_row: tl.constexpr,
        query_stride_head: tl.constexpr,
        key_stride_row: tl.constexpr,
        key_stride_head: tl.constexpr,
        rotary_stride_row: tl.constexpr,
        rows: tl.constexpr,
        heads: tl.constexpr,
        eps: tl.constexpr,
        block_rows: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        head = tl.program_id(1)
        columns = tl.arange(0, head_dim)
        valid = row_offsets[:, None] < rows
        query_offsets = (
            row_offsets[:, None] * query_stride_row
            + head * query_stride_head
            + columns[None, :]
        )
        key_offsets = (
            row_offsets[:, None] * key_stride_row
            + head * key_stride_head
            + columns[None, :]
        )
        query_values = tl.load(query + query_offsets, mask=valid, other=0.0).to(tl.float32)
        key_values = tl.load(key + key_offsets, mask=valid, other=0.0).to(tl.float32)
        query_weights = tl.load(query_weight + columns)[None, :].to(tl.float32)
        key_weights = tl.load(key_weight + columns)[None, :].to(tl.float32)
        query_rstd = tl.rsqrt(tl.sum(query_values * query_values, axis=1) / head_dim + eps)
        key_rstd = tl.rsqrt(tl.sum(key_values * key_values, axis=1) / head_dim + eps)
        normalized_query = (query_values * query_rstd[:, None] * query_weights).to(tl.bfloat16).to(tl.float32)
        normalized_key = (key_values * key_rstd[:, None] * key_weights).to(tl.bfloat16).to(tl.float32)

        half_rotary: tl.constexpr = rotary_dim // 2
        partner_columns = tl.where(
            columns < half_rotary,
            columns + half_rotary,
            columns - half_rotary,
        )
        partner_columns = tl.where(columns < rotary_dim, partner_columns, columns)
        partner_query = tl.load(
            query
            + row_offsets[:, None] * query_stride_row
            + head * query_stride_head
            + partner_columns[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        partner_key = tl.load(
            key
            + row_offsets[:, None] * key_stride_row
            + head * key_stride_head
            + partner_columns[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        partner_query_weight = tl.load(query_weight + partner_columns)[None, :].to(tl.float32)
        partner_key_weight = tl.load(key_weight + partner_columns)[None, :].to(tl.float32)
        partner_query = (partner_query * query_rstd[:, None] * partner_query_weight).to(tl.bfloat16).to(tl.float32)
        partner_key = (partner_key * key_rstd[:, None] * partner_key_weight).to(tl.bfloat16).to(tl.float32)

        rotary_mask = columns[None, :] < rotary_dim
        cosine_values = tl.load(
            cosine + row_offsets[:, None] * rotary_stride_row + columns[None, :],
            mask=valid & rotary_mask,
            other=1.0,
        ).to(tl.float32)
        sine_values = tl.load(
            sine + row_offsets[:, None] * rotary_stride_row + columns[None, :],
            mask=valid & rotary_mask,
            other=0.0,
        ).to(tl.float32)
        sign = tl.where(columns[None, :] < half_rotary, -1.0, 1.0)
        query_output = tl.where(
            rotary_mask,
            normalized_query * cosine_values + sign * partner_query * sine_values,
            normalized_query,
        )
        key_output = tl.where(
            rotary_mask,
            normalized_key * cosine_values + sign * partner_key * sine_values,
            normalized_key,
        )
        tl.store(query + query_offsets, query_output, mask=valid)
        tl.store(key + key_offsets, key_output, mask=valid)

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


def can_run_triton_qk_rms_norm_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> bool:
    return not (
        triton is None
        or torch.is_grad_enabled()
        or not triton_available(query.device)
        or not query.is_cuda
        or query.ndim != 3
        or query.shape != key.shape
        or query.device != key.device
        or query.dtype != key.dtype
        or int(query.shape[0]) <= 0
        or int(query.shape[1]) <= 0
        or int(query.shape[2]) <= 0
        or int(query.shape[2]) > 1024
        or not query_weight.is_cuda
        or not key_weight.is_cuda
        or query_weight.device != query.device
        or key_weight.device != query.device
        or query_weight.numel() != query.shape[2]
        or key_weight.numel() != query.shape[2]
        or not query_weight.is_contiguous()
        or not key_weight.is_contiguous()
        or cosine.ndim != 2
        or sine.shape != cosine.shape
        or cosine.device != query.device
        or sine.device != query.device
        or not cosine.is_contiguous()
        or not sine.is_contiguous()
        or int(cosine.shape[0]) != int(query.shape[0])
        or int(cosine.shape[1]) <= 0
        or int(cosine.shape[1]) % 2 != 0
        or int(cosine.shape[1]) > int(query.shape[2])
    )


@torch.library.custom_op(
    "uniserve_worker::qk_rms_norm_partial_rope_inplace",
    mutates_args=("query", "key"),
)
def triton_qk_rms_norm_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    eps: float,
) -> None:
    if not can_run_triton_qk_rms_norm_rope_inplace(
        query, key, query_weight, key_weight, cosine, sine
    ):
        raise RuntimeError("in-place partial QK RMSNorm and RoPE requires eligible Triton tensors")
    rows, heads, head_dim = (int(size) for size in query.shape)
    rotary_dim = int(cosine.shape[-1])
    block_rows = 8
    _qk_rms_norm_partial_rope_inplace_kernel[(triton.cdiv(rows, block_rows), heads)](
        query,
        key,
        query_weight,
        key_weight,
        cosine,
        sine,
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(cosine.stride(0)),
        rows,
        heads,
        float(eps),
        block_rows,
        head_dim,
        rotary_dim,
        num_warps=8,
        num_stages=1,
    )


@triton_qk_rms_norm_rope_inplace.register_fake
def _triton_qk_rms_norm_rope_inplace_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    eps: float,
) -> None:
    del query, key, query_weight, key_weight, cosine, sine, eps


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
    combine, identical to the unfused kernels' num_warps=4 launches. num_warps=1
    (two elements per lane, thread-local pre-sum) changes the tree and is not
    used. Eligibility is restricted to the tile family
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
    if triton is None or torch.is_grad_enabled():
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
    if triton is None or torch.is_grad_enabled():
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
    if triton is None or torch.is_grad_enabled():
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
        and triton_available(q.device)
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
            or not x.is_cuda
            or not cos.is_cuda
            or not sin.is_cuda
            or not triton_available(x.device)
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
