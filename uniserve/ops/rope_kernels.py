"""Packed RoPE and fused QK normalization-plus-RoPE kernels.

The launchers enforce tensor shapes before entering Triton, accumulate
normalization and rotation in FP32, and combine query and key head rows into shared
launch domains. Specialized kernels cover partial in-place rotation and the
two multi-axis head/tail layouts selected by :mod:`uniserve.ops.qk_plan`.
"""

from __future__ import annotations

import torch

from uniserve.runtime.triton import triton_available

try:  # pragma: no cover - depends on the installed accelerator stack.
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except Exception:  # pragma: no cover
    triton = None
    tl = None

# Each packed-RoPE program handles this many flattened feature elements.
_TRITON_ROPE_BLOCK = 256


if triton is not None:

    @triton.jit(do_not_specialize=["total"])
    def _rotary_factors_kernel(
        positions,
        frequencies,
        cosine,
        sine,
        total: tl.int64,
        width: tl.constexpr,
        shape: tl.constexpr,
        strides: tl.constexpr,
        frequency_stride: tl.constexpr,
        scale: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        rows = offsets // width
        position_offsets = tl.full((block,), 0, tl.int64)
        # The leading extent only bounds the launch. Remaining dimensions
        # describe the strided row layout; a new token count needs no kernel.
        for axis in tl.static_range(len(shape) - 1, -1, -1):
            position_offsets += (rows % shape[axis]) * strides[axis + 1]
            rows //= shape[axis]
        position_offsets += rows * strides[0] if len(strides) else 0
        position = tl.load(positions + position_offsets, offsets < total, other=0).to(tl.float32)
        frequency = tl.load(frequencies + (offsets % width) * frequency_stride).to(tl.float32)
        # Match the public FP32 phase domain, including range reduction for
        # large or negative positions. Approximate sin/cos instructions do not
        # provide that domain; libdevice uses the CUDA numerical functions.
        phase = position * frequency
        factor = tl.full((), scale, tl.float32)
        tl.store(cosine + offsets, libdevice.cos(phase) * factor, offsets < total)
        tl.store(sine + offsets, libdevice.sin(phase) * factor, offsets < total)

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
        """Rotate flattened packed token/head rows in GPT-NeoX half layout."""

        offs = tl.program_id(0) * block + tl.arange(0, block)
        mask = offs < total
        d = offs % dim
        row = offs // dim
        token = row // heads
        d_half = d % half
        base = row * dim

        # Every output feature reads its paired values and the token-specific
        # factor indexed by position within a half.
        x1 = tl.load(x_ptr + base + d_half, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + base + half + d_half, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + token * half + d_half, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + token * half + d_half, mask=mask, other=0.0).to(tl.float32)
        first_half = d < half
        out = tl.where(first_half, x1 * cos - x2 * sin, x2 * cos + x1 * sin)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _rms_norm_rope_rows(
        source,
        weight,
        cosine,
        sine,
        output,
        first_row,
        row_count,
        heads: tl.constexpr,
        stride_token: tl.constexpr,
        stride_head: tl.constexpr,
        stride_feature: tl.constexpr,
        dim: tl.constexpr,
        half: tl.constexpr,
        epsilon: tl.constexpr,
        block: tl.constexpr,
        rows_per_program: tl.constexpr,
    ):
        """Normalize and rotate independent head rows without crossing row reductions."""

        rows = first_row + tl.arange(0, rows_per_program)
        columns = tl.arange(0, block)
        tokens = rows // heads
        head = rows % heads
        bases = tokens * stride_token + head * stride_head
        mask = (rows[:, None] < row_count) & (columns[None, :] < dim)
        values = tl.load(
            source + bases[:, None] + columns[None, :] * stride_feature,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        variance = tl.sum(values * values, axis=1) / dim
        inverse = tl.rsqrt(variance + epsilon)
        offsets = columns % half
        left = tl.load(
            source + bases[:, None] + offsets[None, :] * stride_feature,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        right = tl.load(
            source + bases[:, None] + (half + offsets[None, :]) * stride_feature,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        left_weight = tl.load(weight + offsets, mask=columns < dim, other=0.0).to(tl.float32)
        right_weight = tl.load(weight + half + offsets, mask=columns < dim, other=0.0).to(
            tl.float32
        )
        factors = tokens[:, None] * half + offsets[None, :]
        cos = tl.load(cosine + factors, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sine + factors, mask=mask, other=0.0).to(tl.float32)
        # Keep normalization, learned scale and rotation in FP32 until store.
        left = left * inverse[:, None] * left_weight[None, :]
        right = right * inverse[:, None] * right_weight[None, :]
        rotated = tl.where(
            columns[None, :] < half, left * cos - right * sin, right * cos + left * sin
        )
        tl.store(output + rows[:, None] * dim + columns[None, :], rotated, mask=mask)

    @triton.jit(do_not_specialize=["q_rows", "k_rows"])
    def _qk_rms_norm_rope_kernel(
        q_ptr,
        k_ptr,
        qw_ptr,
        kw_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        q_rows: tl.int32,
        k_rows: tl.int32,
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
        rows_per_program: tl.constexpr,
    ):
        """Process Q/K tiles with live row bounds and one normalization per row."""

        pid = tl.program_id(0)
        query_programs = tl.cdiv(q_rows, rows_per_program)
        # The branch is uniform within each CTA. Separate domains also preserve
        # independent Q/K weights, strides, epsilons and incomplete final tiles.
        if pid < query_programs:
            _rms_norm_rope_rows(
                q_ptr,
                qw_ptr,
                cos_ptr,
                sin_ptr,
                q_out_ptr,
                pid * rows_per_program,
                q_rows,
                q_heads,
                q_stride_0,
                q_stride_1,
                q_stride_2,
                dim,
                half,
                q_eps,
                block,
                rows_per_program,
            )
        else:
            _rms_norm_rope_rows(
                k_ptr,
                kw_ptr,
                cos_ptr,
                sin_ptr,
                k_out_ptr,
                (pid - query_programs) * rows_per_program,
                k_rows,
                k_heads,
                k_stride_0,
                k_stride_1,
                k_stride_2,
                dim,
                half,
                k_eps,
                block,
                rows_per_program,
            )

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
        compact: tl.constexpr,
    ):
        """Normalize Q/K in place and rotate an even prefix of each head."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        head = tl.program_id(1)
        columns = tl.arange(0, head_dim)
        valid = row_offsets[:, None] < rows
        query_offsets = (
            row_offsets[:, None] * query_stride_row + head * query_stride_head + columns[None, :]
        )
        key_offsets = (
            row_offsets[:, None] * key_stride_row + head * key_stride_head + columns[None, :]
        )
        query_values = tl.load(query + query_offsets, mask=valid, other=0.0).to(tl.float32)
        key_values = tl.load(key + key_offsets, mask=valid, other=0.0).to(tl.float32)
        query_weights = tl.load(query_weight + columns)[None, :].to(tl.float32)
        key_weights = tl.load(key_weight + columns)[None, :].to(tl.float32)

        # Normalization spans the complete head even though only the rotary
        # prefix consumes sine and cosine factors.
        query_rstd = tl.rsqrt(tl.sum(query_values * query_values, axis=1) / head_dim + eps)
        key_rstd = tl.rsqrt(tl.sum(key_values * key_values, axis=1) / head_dim + eps)
        normalized_query = query_values * query_rstd[:, None] * query_weights
        normalized_key = key_values * key_rstd[:, None] * key_weights

        # Map each feature in the rotary prefix to its partner in the opposite
        # half. Tail features map to themselves and bypass rotation below.
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
        partner_query = partner_query * query_rstd[:, None] * partner_query_weight
        partner_key = partner_key * key_rstd[:, None] * partner_key_weight

        rotary_mask = columns[None, :] < rotary_dim
        factor_columns = columns % half_rotary if compact else columns
        cosine_values = tl.load(
            cosine + row_offsets[:, None] * rotary_stride_row + factor_columns[None, :],
            mask=valid & rotary_mask,
            other=1.0,
        ).to(tl.float32)
        sine_values = tl.load(
            sine + row_offsets[:, None] * rotary_stride_row + factor_columns[None, :],
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
        """Normalize a two-group row and rotate only its leading group."""

        # The head group uses an independent RMS reduction before NeoX rotation.
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
        x1n = x1 * inv_a * w1
        x2n = x2 * inv_a * w2
        rot = tl.where(first_half, x1n * cos - x2n * sin, x2n * cos + x1n * sin)
        tl.store(out_ptr + out_base + offs_a, rot, mask=mask_a)

        # The tail group has its own RMS reduction and weight. Its declared
        # positions are zero, so normalized values pass through unrotated.
        offs_b = tl.arange(0, block_b)
        mask_b = (offs_b < tail_dim) & row_active
        xb = tl.load(x_ptr + base + (rope_dim + offs_b) * stride_2, mask=mask_b, other=0.0).to(
            tl.float32
        )
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
        """Apply the split head/tail row operation across Q and K ranges."""

        pid = tl.program_id(0)

        # Query rows occupy the leading program-id range.
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

        # Key rows reuse the same row helper with their own dimensions and weights.
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
        """Normalize and rotate a head axis plus two shared-normalization axes."""

        # Axis zero has an independent RMS reduction and factor table.
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
        x1n = x1 * inv_a * w1
        x2n = x2 * inv_a * w2
        rot = tl.where(first_half, x1n * cos - x2n * sin, x2n * cos + x1n * sin)
        tl.store(out_ptr + out_base + offs_a, rot, mask=col_a)

        # The two equal-width tail axes share one RMS reduction and weight, then
        # select separate rotary tables while retaining their local half pairing.
        offs_b = tl.arange(0, block_b)
        col_b = offs_b < tail_dim
        xb = tl.load(
            x_ptr + base + (dim0 + offs_b) * stride_2,
            mask=col_b,
            other=0.0,
        ).to(tl.float32)
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
        y1n = y1 * inv_b * wv1
        y2n = y2 * inv_b * wv2
        c1 = tl.load(
            cos1_ptr + token * axis_half + dj,
            mask=col_b & (~is_second_axis),
            other=0.0,
        ).to(tl.float32)
        s1 = tl.load(
            sin1_ptr + token * axis_half + dj,
            mask=col_b & (~is_second_axis),
            other=0.0,
        ).to(tl.float32)
        c2 = tl.load(
            cos2_ptr + token * axis_half + dj,
            mask=col_b & is_second_axis,
            other=0.0,
        ).to(tl.float32)
        s2 = tl.load(
            sin2_ptr + token * axis_half + dj,
            mask=col_b & is_second_axis,
            other=0.0,
        ).to(tl.float32)
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
        """Dispatch each flattened Q/K head row to the three-axis row helper."""

        # Program ids select either a complete query or key row; the branch is
        # uniform within a program and independent of tensor values.
        pid = tl.program_id(0)
        q_rows = tl.num_programs(0) // (q_heads + k_heads) * q_heads
        if pid < q_rows:
            token = pid // q_heads
            head = pid - token * q_heads
            base = token * q_stride_0 + head * q_stride_1
            _multi_axis_rms_norm_rope_row(
                q_ptr,
                base,
                q_head_w_ptr,
                q_tail_w_ptr,
                cos0_ptr,
                sin0_ptr,
                cos1_ptr,
                sin1_ptr,
                cos2_ptr,
                sin2_ptr,
                q_out_ptr,
                pid * dim,
                token,
                q_stride_2,
                dim0,
                half0,
                axis_dim,
                axis_half,
                tail_dim,
                q_eps,
                block_a,
                block_b,
            )
        else:
            k_pid = pid - q_rows
            token = k_pid // k_heads
            head = k_pid - token * k_heads
            base = token * k_stride_0 + head * k_stride_1
            _multi_axis_rms_norm_rope_row(
                k_ptr,
                base,
                k_head_w_ptr,
                k_tail_w_ptr,
                cos0_ptr,
                sin0_ptr,
                cos1_ptr,
                sin1_ptr,
                cos2_ptr,
                sin2_ptr,
                k_out_ptr,
                k_pid * dim,
                token,
                k_stride_2,
                dim0,
                half0,
                axis_dim,
                axis_half,
                tail_dim,
                k_eps,
                block_a,
                block_b,
            )


def try_triton_rotary_factors(positions, frequencies, scale, *, dtype):
    """Generate compact factors on supported CUDA tensors, or decline eligibility."""

    floating = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if (
        triton is None
        or not triton_available(positions.device)
        or frequencies.device != positions.device
        or positions.layout != torch.strided
        or frequencies.layout != torch.strided
        or positions.dtype not in floating | {torch.int32, torch.int64}
        or frequencies.dtype not in floating
        or dtype not in floating
    ):
        return None
    shape = (*positions.shape, frequencies.numel())
    cosine = torch.empty(shape, device=positions.device, dtype=dtype)
    sine = torch.empty_like(cosine)
    total = cosine.numel()
    if total:
        # Triton launches on the thread's current device, whereas this public
        # numerical operation follows its input tensors, as PyTorch does.
        with torch.cuda.device(positions.device):
            _rotary_factors_kernel[(triton.cdiv(total, 256),)](
                positions,
                frequencies,
                cosine,
                sine,
                total,
                frequencies.numel(),
                tuple(positions.shape[1:]),
                tuple(positions.stride()),
                frequencies.stride(0),
                scale,
                256,
            )
    return cosine, sine


def can_run_triton_qk_rms_norm_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    compact: bool = False,
) -> bool:
    """Check packed row/head dimensions for in-place partial QK rotation."""

    # In-place execution requires co-located rank-three Q/K tensors, full-width
    # normalization weights, and contiguous duplicated factors for an even
    # rotary prefix.
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
        or (not compact and int(cosine.shape[1]) % 2 != 0)
        or int(cosine.shape[1]) * (2 if compact else 1) > int(query.shape[2])
    )


@torch.library.custom_op(
    "uniserve::qk_rms_norm_partial_rope_inplace",
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
    compact: bool = False,
) -> None:
    """Normalize Q/K and rotate their leading feature prefix in place.

    Query and key use ``[rows, heads, head_dim]`` layout. ``cosine`` and
    ``sine`` contain duplicated full-width factors for the even rotary prefix.
    """

    if not can_run_triton_qk_rms_norm_rope_inplace(
        query, key, query_weight, key_weight, cosine, sine, compact=compact
    ):
        raise RuntimeError("in-place partial QK RMSNorm and RoPE requires eligible Triton tensors")
    rows, heads, head_dim = (int(size) for size in query.shape)
    rotary_dim = int(cosine.shape[-1]) * (2 if compact else 1)

    # Each program covers eight rows for one head, normalizing the full head
    # before applying factors only to the declared rotary prefix.
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
        compact,
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
    compact: bool = False,
) -> None:
    """Declare fake-tensor mutation for the custom operator."""

    del query, key, query_weight, key_weight, cosine, sine, eps, compact


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
    """Try one-launch QK RMSNorm plus full-width NeoX rotation.

    Returns contiguous output tensors, or ``None`` when device, layout, or
    feature shape falls outside the Triton kernel requirements.
    """

    if not can_run_triton_qk_rms_norm_rope(q, k, q_weight, k_weight, cos, sin, q_eps, k_eps):
        return None

    # Eligibility guarantees an even, bounded feature width and nonempty Q/K
    # token/head ranges; those ranges share one flattened launch.
    shape = _qk_rms_norm_rope_shape(q, k)
    assert shape is not None
    q_tokens, k_tokens, q_heads, k_heads, dim = shape
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    q_rows = q_tokens * q_heads
    k_rows = k_tokens * k_heads
    block = triton.next_power_of_2(dim)
    # Small heads share a CTA; bounding the feature tile limits register growth
    # for wider heads. At width 128, each of four warps reduces one head.
    rows_per_program = min(4, max(1, 512 // block))
    grid = (triton.cdiv(q_rows, rows_per_program) + triton.cdiv(k_rows, rows_per_program),)
    _qk_rms_norm_rope_kernel[grid](
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
        block,
        rows_per_program,
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
    """Try two-group QK RMSNorm with rotation on the leading group.

    ``q`` and ``k`` use ``[tokens, heads, dim]`` layout. Features before
    ``rope_dim`` are normalized with the head weight and NeoX-rotated by
    half-width factor tables. Remaining features use the tail weight and pass
    through unrotated because their positions are zero. Returns ``None`` when
    the tensors do not satisfy this layout.
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

    # Query and key may have different head counts but share feature grouping
    # and token-indexed rotary tables.
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
    """Try three-axis QK normalization and rotation in one Triton launch.

    ``q`` and ``k`` use ``[tokens, heads, dim]`` layout with
    ``axis_dims = (head, tail, tail)``. The head axis has its own RMS weight and
    rotary table. Both tail axes share one RMS weight while retaining separate
    rotary tables. Masked reduction tiles cover arbitrary even axis widths
    within the fused head-size bound. Returns ``None`` when the tensors fall
    outside these requirements.
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

    # Axis metadata determines the two normalization domains. Query and key
    # share tokens and feature widths but may have different head counts.
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
    """Return whether tensors satisfy the specialized three-axis kernel requirements."""

    # The kernel is specialized for a leading axis plus two equal-width axes
    # that share one tail normalization weight.
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

    # Match the bounded head dimensions of the other fused QK kernels.
    if dim0 + tail_dim > 1024:
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

    # Each normalization weight covers its complete group contiguously.
    for weight, width in (
        (q_head_weight, dim0),
        (k_head_weight, dim0),
        (q_tail_weight, tail_dim),
        (k_tail_weight, tail_dim),
    ):
        if int(weight.numel()) != width or not weight.is_contiguous():
            return False

    # Every rotary axis owns one contiguous half-width factor table per token.
    for axis, width in ((0, dim0), (1, axis_dim), (2, axis_dim)):
        cos_a, sin_a = cos[axis], sin[axis]
        if not (
            cos_a.is_cuda
            and sin_a.is_cuda
            and cos_a.device == q.device
            and sin_a.device == q.device
        ):
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
    """Return whether tensors fit the rotated-head, identity-tail kernel."""

    # Shared residency checks cover the rotated head operands; tail weights add
    # the second independent normalization group.
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

    # Both Q/K ranges share tokens, feature partitions, and one half-width
    # rotary table while retaining independent head counts and weights.
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
    """Return whether tensors fit full-width fused QK RMSNorm plus RoPE."""

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
    """Check device and shape requirements shared by the full-width kernel."""

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
    """Check CUDA residency and co-location for fused QK operands."""

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
    """Check contiguous packed-QK and half-width rotary table dimensions."""

    # The kernel flattens token/head rows directly, so only the feature axis may
    # be strided through its explicit stride and must remain unit-contiguous.
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


def _qk_rms_norm_rope_shape(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[int, int, int, int, int] | None:
    """Return validated token, head, and feature dimensions for Q and K."""

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
    """Triton eligibility and launch implementation for packed RoPE."""

    def is_eligible(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
        """Return whether packed values and factor tables fit the Triton kernel."""

        # Factors provide one half-width row per token and all operands must be
        # contiguous CUDA tensors with gradient tracking disabled.
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
        """Launch packed RoPE over every flattened feature element."""

        dim = int(x.shape[-1])
        half = dim // 2
        tokens = int(x.shape[0])
        heads = int(x.numel() // max(1, tokens * dim))
        out = torch.empty_like(x)
        block = _TRITON_ROPE_BLOCK
        total = int(x.numel())

        # Flattened indexing lets one grid cover tokens, arbitrary leading head
        # dimensions, and both rotary halves without reshaping the input.
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
    """Portable tensor implementation of packed RoPE."""

    def is_eligible(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
        """Return ``True`` because tensor operations handle the general case."""

        return True

    def run(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate contiguous feature halves using token-indexed factor tables."""

        ro_dim = cos.shape[-1] * 2
        if ro_dim != x.shape[-1]:
            raise ValueError(f"rotary dim {ro_dim} does not match tensor dim {x.shape[-1]}")
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]

        # Factor tables gain a singleton head axis and broadcast across all
        # heads associated with each token.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        out = torch.empty_like(x)
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half:] = x2 * cos + x1 * sin
        return out


if triton is not None:

    @triton.jit
    def _qk_rms_norm_partial_rope_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        query_bias_ptr,
        key_bias_ptr,
        value_bias_ptr,
        cosine_ptr,
        sine_ptr,
        rows: tl.constexpr,
        heads: tl.constexpr,
        row_stride: tl.constexpr,
        head_stride: tl.constexpr,
        rotary_row_stride: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_VALUE_BIAS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        EPS: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        """Normalize Q/K heads, rotate their leading coordinates, and apply biases."""

        row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        head = tl.program_id(1)
        columns = tl.arange(0, HEAD_BLOCK)
        valid = (row[:, None] < rows) & (columns[None, :] < HEAD_DIM)
        offsets = row[:, None] * row_stride + head * head_stride + columns[None, :]
        query = tl.load(query_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        key = tl.load(key_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        if HAS_BIAS:
            bias_offsets = head * HEAD_DIM + columns[None, :]
            query += tl.load(
                query_bias_ptr + bias_offsets, mask=columns[None, :] < HEAD_DIM, other=0.0
            ).to(tl.float32)
            key += tl.load(
                key_bias_ptr + bias_offsets, mask=columns[None, :] < HEAD_DIM, other=0.0
            ).to(tl.float32)
        if HAS_VALUE_BIAS:
            value = tl.load(value_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
            value += tl.load(
                value_bias_ptr + head * HEAD_DIM + columns[None, :],
                mask=columns[None, :] < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
            tl.store(value_ptr + offsets, value.to(value_ptr.dtype.element_ty), mask=valid)
        query_rstd = tl.rsqrt(tl.sum(query * query, axis=1) / HEAD_DIM + EPS)
        key_rstd = tl.rsqrt(tl.sum(key * key, axis=1) / HEAD_DIM + EPS)
        query = query * query_rstd[:, None]
        key = key * key_rstd[:, None]

        # Partner coordinates come from the opposite half of the rotary subspace.
        half_rotary: tl.constexpr = ROTARY_DIM // 2
        partner_columns = tl.where(
            columns < half_rotary,
            columns + half_rotary,
            columns - half_rotary,
        )
        partner_columns = tl.where(columns < ROTARY_DIM, partner_columns, columns)
        partner_offsets = row[:, None] * row_stride + head * head_stride + partner_columns[None, :]
        query_partner = tl.load(query_ptr + partner_offsets, mask=valid, other=0.0).to(tl.float32)
        key_partner = tl.load(key_ptr + partner_offsets, mask=valid, other=0.0).to(tl.float32)
        if HAS_BIAS:
            partner_bias_offsets = head * HEAD_DIM + partner_columns[None, :]
            query_partner += tl.load(
                query_bias_ptr + partner_bias_offsets, mask=columns[None, :] < HEAD_DIM, other=0.0
            ).to(tl.float32)
            key_partner += tl.load(
                key_bias_ptr + partner_bias_offsets, mask=columns[None, :] < HEAD_DIM, other=0.0
            ).to(tl.float32)
        query_partner = query_partner * query_rstd[:, None]
        key_partner = key_partner * key_rstd[:, None]

        rotary_mask = valid & (columns[None, :] < ROTARY_DIM)
        rotary_offsets = row[:, None] * rotary_row_stride + columns[None, :]
        cosine = tl.load(cosine_ptr + rotary_offsets, mask=rotary_mask, other=1.0).to(tl.float32)
        sine = tl.load(sine_ptr + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
        sign = tl.where(columns[None, :] < half_rotary, -1.0, 1.0)
        query_rotated = query * cosine + sign * query_partner * sine
        key_rotated = key * cosine + sign * key_partner * sine
        tl.store(query_ptr + offsets, tl.where(rotary_mask, query_rotated, query), mask=valid)
        tl.store(key_ptr + offsets, tl.where(rotary_mask, key_rotated, key), mask=valid)
