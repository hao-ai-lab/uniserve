"""Packed RoPE and fused QK normalization-plus-RoPE kernels.

These Triton kernels back the rotary entry points of
``uniserve.nn.functional`` (``apply_rotary``, ``qk_norm_rope`` and
``qk_bias_rms_norm_rope_``) and the factor tables of
``uniserve.nn.rope.RotaryEmbedding``. Those callers validate the public
contract and fall back to tensor operations when no kernel matches the call
or its ``can_run_*`` check rejects the operands. ``qk_norm_rope`` selects a
kernel from the explicit normalization domains and rotary axes of a call.

Every launcher has a separate ``can_run_*`` eligibility check and does not
revalidate its operands, so callers must run the check first. Launchers
accumulate normalization and rotation in FP32 and round once into
caller-supplied outputs.

Rotary factors are compact: one ``[tokens, rotated / 2]`` cosine and sine
table per rotary axis, shared by every head of a token. Rotation is split-half
(GPT-NeoX layout): feature ``i`` of an axis pairs with feature
``i + rotated / 2``.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

if triton is not None:  # pragma: no cover - depends on the accelerator stack.
    from triton.language.extra.cuda import libdevice

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
        """Store scaled cosine and sine factors for strided positions.

        ``offsets`` flatten the contiguous ``[*positions.shape, width]``
        outputs. ``shape`` holds the trailing position extents and ``strides``
        every position stride, both as compile-time tuples.
        """
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        rows = offsets // width
        position_offsets = tl.full((block,), 0, tl.int64)

        # The leading extent only bounds the launch through ``total``. The
        # trailing extents and strides are compile-time constants, so a new
        # leading extent with unchanged strides reuses the compiled kernel.
        for axis in tl.static_range(len(shape) - 1, -1, -1):
            position_offsets += (rows % shape[axis]) * strides[axis + 1]
            rows //= shape[axis]
        position_offsets += rows * strides[0] if len(strides) else 0

        position = tl.load(
            positions + position_offsets, offsets < total, other=0
        ).to(tl.float32)
        frequency = tl.load(
            frequencies + (offsets % width) * frequency_stride
        ).to(tl.float32)

        # Match the public FP32 phase domain, including range reduction for
        # large or negative positions. Approximate sin/cos instructions do not
        # provide that domain; libdevice uses the CUDA numerical functions.
        phase = position * frequency
        factor = tl.full((), scale, tl.float32)
        tl.store(
            cosine + offsets, libdevice.cos(phase) * factor, offsets < total
        )
        tl.store(sine + offsets, libdevice.sin(phase) * factor, offsets < total)

    @triton.jit(do_not_specialize=["total"])
    def _packed_rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        total,
        heads: tl.constexpr,
        dim: tl.constexpr,
        half: tl.constexpr,
        block: tl.constexpr,
    ):
        """Rotate flattened packed token/head rows in GPT-NeoX half layout.

        Input and output are contiguous ``[tokens, heads, dim]``; factors are
        contiguous ``[tokens, half]``. ``total`` is excluded from Triton's
        integer specialization, so a new token count reuses the kernel.
        """
        # offs flattens the [tokens, heads, dim] layout of the input tensor.
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
        x2 = tl.load(x_ptr + base + half + d_half, mask=mask, other=0.0).to(
            tl.float32
        )
        cos = tl.load(cos_ptr + token * half + d_half, mask=mask, other=0.0).to(
            tl.float32
        )
        sin = tl.load(sin_ptr + token * half + d_half, mask=mask, other=0.0).to(
            tl.float32
        )

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
        output_stride_token: tl.constexpr,
        output_stride_head: tl.constexpr,
        dim: tl.constexpr,
        half: tl.constexpr,
        epsilon: tl.constexpr,
        block: tl.constexpr,
        rows_per_program: tl.constexpr,
    ):
        """Normalize and rotate independent head rows.

        Every row is loaded completely before its store, and programs own
        disjoint rows, so the output may alias the source with equal strides.
        """
        rows = first_row + tl.arange(0, rows_per_program)
        columns = tl.arange(0, block)
        tokens = rows // heads
        head = rows % heads
        bases = tokens * stride_token + head * stride_head
        mask = (rows[:, None] < row_count) & (columns[None, :] < dim)

        # values: [rows_per_program, block] tile of head features.
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
            source
            + bases[:, None]
            + (half + offsets[None, :]) * stride_feature,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        left_weight = tl.load(
            weight + offsets, mask=columns < dim, other=0.0
        ).to(tl.float32)
        right_weight = tl.load(
            weight + half + offsets, mask=columns < dim, other=0.0
        ).to(tl.float32)

        factors = tokens[:, None] * half + offsets[None, :]
        cos = tl.load(cosine + factors, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sine + factors, mask=mask, other=0.0).to(tl.float32)

        # Keep normalization, learned scale and rotation in FP32 until store.
        left = left * inverse[:, None] * left_weight[None, :]
        right = right * inverse[:, None] * right_weight[None, :]
        rotated = tl.where(
            columns[None, :] < half,
            left * cos - right * sin,
            right * cos + left * sin,
        )
        destination = tokens * output_stride_token + head * output_stride_head
        tl.store(
            output + destination[:, None] + columns[None, :], rotated, mask=mask
        )

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
        q_out_stride_0: tl.constexpr,
        q_out_stride_1: tl.constexpr,
        k_out_stride_0: tl.constexpr,
        k_out_stride_1: tl.constexpr,
        dim: tl.constexpr,
        half: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        block: tl.constexpr,
        rows_per_program: tl.constexpr,
    ):
        """Normalize and rotate Q rows, then K rows, over one program grid.

        Programs below ``cdiv(q_rows, rows_per_program)`` process query rows
        and the remaining programs process key rows, ``rows_per_program``
        flattened token/head rows each. ``q_rows`` and ``k_rows`` are runtime
        arguments excluded from integer specialization; head counts, strides
        and widths are compile-time constants.
        """
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
                q_out_stride_0,
                q_out_stride_1,
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
                k_out_stride_0,
                k_out_stride_1,
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
        stepwise: tl.constexpr,
    ):
        """Normalize Q/K in place and rotate an even prefix of each head.

        Compact factors hold one ``[rows, rotary_dim / 2]`` phase per rotated
        feature pair. Each program owns a disjoint ``[block_rows, head_dim]``
        tile of one head and loads every value it reads, including partner
        features, before its stores, so updating in place is safe. ``rows`` is
        a compile-time constant, so each distinct row count compiles its own
        variant. ``stepwise`` rounds the weighted normalization, the factors
        and each rotation product to the input dtype before the sum, as eager
        PyTorch does; the launch then disables floating-point contraction.
        """
        row_offsets = (
            tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        ).to(tl.int64)
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

        # query/key tiles: [block_rows, head_dim] for one head.
        query_values = tl.load(query + query_offsets, mask=valid, other=0.0).to(
            tl.float32
        )
        key_values = tl.load(key + key_offsets, mask=valid, other=0.0).to(
            tl.float32
        )
        # ``head_dim`` is the power-of-two tile width, so the weight loads
        # need no feature mask.
        query_weights = tl.load(query_weight + columns)[None, :].to(tl.float32)
        key_weights = tl.load(key_weight + columns)[None, :].to(tl.float32)

        # Normalization spans the complete head even though only the rotary
        # prefix consumes sine and cosine factors.
        query_rstd = tl.rsqrt(
            tl.sum(query_values * query_values, axis=1) / head_dim + eps
        )
        key_rstd = tl.rsqrt(
            tl.sum(key_values * key_values, axis=1) / head_dim + eps
        )
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
        partner_columns = tl.where(
            columns < rotary_dim, partner_columns, columns
        )

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
        partner_query_weight = tl.load(query_weight + partner_columns)[
            None, :
        ].to(tl.float32)
        partner_key_weight = tl.load(key_weight + partner_columns)[None, :].to(
            tl.float32
        )
        partner_query = (
            partner_query * query_rstd[:, None] * partner_query_weight
        )
        partner_key = partner_key * key_rstd[:, None] * partner_key_weight

        rotary_mask = columns[None, :] < rotary_dim
        factor_columns = columns % half_rotary
        cosine_values = tl.load(
            cosine
            + row_offsets[:, None] * rotary_stride_row
            + factor_columns[None, :],
            mask=valid & rotary_mask,
            other=1.0,
        ).to(tl.float32)
        sine_values = tl.load(
            sine
            + row_offsets[:, None] * rotary_stride_row
            + factor_columns[None, :],
            mask=valid & rotary_mask,
            other=0.0,
        ).to(tl.float32)

        sign = tl.where(columns[None, :] < half_rotary, -1.0, 1.0)
        if stepwise:
            # Eager order: the normalized head and the factors round to the
            # input dtype, then each product rounds before the sum. Rounding
            # is symmetric, so the sign may apply after the partner product.
            dtype = query.dtype.element_ty
            normalized_query = normalized_query.to(dtype).to(tl.float32)
            normalized_key = normalized_key.to(dtype).to(tl.float32)
            partner_query = partner_query.to(dtype).to(tl.float32)
            partner_key = partner_key.to(dtype).to(tl.float32)
            cosine_values = cosine_values.to(dtype).to(tl.float32)
            sine_values = sine_values.to(dtype).to(tl.float32)
            query_output = tl.where(
                rotary_mask,
                (normalized_query * cosine_values).to(dtype).to(tl.float32)
                + sign * (partner_query * sine_values).to(dtype).to(tl.float32),
                normalized_query,
            )
            key_output = tl.where(
                rotary_mask,
                (normalized_key * cosine_values).to(dtype).to(tl.float32)
                + sign * (partner_key * sine_values).to(dtype).to(tl.float32),
                normalized_key,
            )
        else:
            query_output = tl.where(
                rotary_mask,
                normalized_query * cosine_values
                + sign * partner_query * sine_values,
                normalized_query,
            )
            key_output = tl.where(
                rotary_mask,
                normalized_key * cosine_values
                + sign * partner_key * sine_values,
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
        xa = tl.load(
            x_ptr + base + offs_a * stride_2, mask=mask_a, other=0.0
        ).to(tl.float32)
        var_a = tl.sum(xa * xa, axis=0) / rope_dim
        inv_a = tl.rsqrt(var_a + eps)

        d_half = offs_a % rope_half
        second_offs = rope_half + d_half
        first_half = offs_a < rope_half
        x1 = tl.load(
            x_ptr + base + d_half * stride_2, mask=mask_a, other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr + base + second_offs * stride_2, mask=mask_a, other=0.0
        ).to(tl.float32)
        w1 = tl.load(head_w_ptr + d_half, mask=offs_a < rope_dim, other=0.0).to(
            tl.float32
        )
        w2 = tl.load(
            head_w_ptr + second_offs, mask=offs_a < rope_dim, other=0.0
        ).to(tl.float32)
        cos = tl.load(
            cos_ptr + token * rope_half + d_half, mask=mask_a, other=0.0
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + token * rope_half + d_half, mask=mask_a, other=0.0
        ).to(tl.float32)

        x1n = x1 * inv_a * w1
        x2n = x2 * inv_a * w2
        rot = tl.where(first_half, x1n * cos - x2n * sin, x2n * cos + x1n * sin)
        tl.store(out_ptr + out_base + offs_a, rot, mask=mask_a)

        # The tail group has its own RMS reduction and weight. ``qk_norm_rope``
        # selects this kernel only when the tail's rotary factors have zero
        # width, so it reads no tail factors and stores the tail unrotated.
        offs_b = tl.arange(0, block_b)
        mask_b = (offs_b < tail_dim) & row_active
        xb = tl.load(
            x_ptr + base + (rope_dim + offs_b) * stride_2,
            mask=mask_b,
            other=0.0,
        ).to(tl.float32)
        var_b = tl.sum(xb * xb, axis=0) / tail_dim
        wb = tl.load(tail_w_ptr + offs_b, mask=offs_b < tail_dim, other=0.0).to(
            tl.float32
        )

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
        q_out_stride_0: tl.constexpr,
        q_out_stride_1: tl.constexpr,
        k_out_stride_0: tl.constexpr,
        k_out_stride_1: tl.constexpr,
        rope_dim: tl.constexpr,
        rope_half: tl.constexpr,
        tail_dim: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        block_a: tl.constexpr,
        block_b: tl.constexpr,
    ):
        """Apply the split head/tail row call across Q and K ranges.

        The head group is stored before the tail group is read, and the two
        groups occupy disjoint features, so outputs may alias equal-stride
        inputs. ``q_rows`` and ``k_rows`` are compile-time constants, so each
        distinct token count compiles its own variant.
        """
        pid = tl.program_id(0)

        # Query rows occupy the leading program-id range. Every program runs
        # both row calls; ``q_active`` and ``k_active`` mask the source and
        # factor loads and the stores of the call whose range does not
        # contain ``pid``. Its weight loads stay in bounds regardless.
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
            q_token * q_out_stride_0 + q_head * q_out_stride_1,
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

        # Key rows reuse the same row helper with their own dimensions and
        # weights.
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
            k_token * k_out_stride_0 + k_head * k_out_stride_1,
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
        """Normalize and rotate a head axis plus two shared-domain axes.

        The head axis (``dim0`` features) has its own RMS reduction, weight
        and factor table. The two tail axes (``axis_dim`` features each, for
        ``tail_dim`` in total) share one RMS reduction and weight but rotate
        with separate factor tables, each pairing features within its axis.
        """
        # Axis zero has an independent RMS reduction and factor table.
        offs_a = tl.arange(0, block_a)
        col_a = offs_a < dim0
        d_half = offs_a % half0
        second = half0 + d_half
        first_half = offs_a < half0

        xa = tl.load(
            x_ptr + base + offs_a * stride_2, mask=col_a, other=0.0
        ).to(tl.float32)
        var_a = tl.sum(xa * xa, axis=0) / dim0
        inv_a = tl.rsqrt(var_a + eps)

        x1 = tl.load(
            x_ptr + base + d_half * stride_2, mask=col_a, other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr + base + second * stride_2, mask=col_a, other=0.0
        ).to(tl.float32)
        w1 = tl.load(head_w_ptr + d_half, mask=col_a, other=0.0).to(tl.float32)
        w2 = tl.load(head_w_ptr + second, mask=col_a, other=0.0).to(tl.float32)
        cos = tl.load(
            cos0_ptr + token * half0 + d_half, mask=col_a, other=0.0
        ).to(tl.float32)
        sin = tl.load(
            sin0_ptr + token * half0 + d_half, mask=col_a, other=0.0
        ).to(tl.float32)

        x1n = x1 * inv_a * w1
        x2n = x2 * inv_a * w2
        rot = tl.where(first_half, x1n * cos - x2n * sin, x2n * cos + x1n * sin)
        tl.store(out_ptr + out_base + offs_a, rot, mask=col_a)

        # The two equal-width tail axes share one RMS reduction and weight, then
        # select separate rotary tables while retaining their local half
        # pairing.
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

        y1 = tl.load(
            x_ptr + base + (dim0 + src1) * stride_2, mask=col_b, other=0.0
        ).to(tl.float32)
        y2 = tl.load(
            x_ptr + base + (dim0 + src2) * stride_2, mask=col_b, other=0.0
        ).to(tl.float32)
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
        q_out_stride_0: tl.constexpr,
        q_out_stride_1: tl.constexpr,
        k_out_stride_0: tl.constexpr,
        k_out_stride_1: tl.constexpr,
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
        """Dispatch each flattened Q/K head row to the three-axis row helper.

        The head axis is stored before the shared tail is read, and the two
        domains occupy disjoint features, so outputs may alias equal-stride
        inputs.
        """
        # Program ids select either a complete query or key row; the branch is
        # uniform within a program and independent of tensor values. The grid
        # holds ``tokens * (q_heads + k_heads)`` programs, so the query row
        # count follows from its size instead of a row-count argument.
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
                token * q_out_stride_0 + head * q_out_stride_1,
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
                token * k_out_stride_0 + head * k_out_stride_1,
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


def can_run_rotary_factors(
    positions: torch.Tensor, frequencies: torch.Tensor, dtype: torch.dtype
) -> bool:
    """Return whether strided positions and frequencies fit the kernel.

    The check covers devices, layouts and dtypes. Callers of
    :func:`rotary_factors` must also supply one-dimensional ``frequencies``
    and contiguous outputs, which this check does not inspect.
    """
    floating = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    return (
        launchable(positions.device)
        and frequencies.device == positions.device
        and positions.layout == torch.strided
        and frequencies.layout == torch.strided
        and positions.dtype in floating | {torch.int32, torch.int64}
        and frequencies.dtype in floating
        and dtype in floating
    )


def rotary_factors(
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    scale: float,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> None:
    """Store ``cos/sin(position * frequency) * scale`` compact factors.

    ``cosine`` and ``sine`` are contiguous ``[*positions.shape, frequencies]``
    outputs. Phases use FP32 with full trigonometric range reduction.
    """
    total = cosine.numel()
    if not total:
        return
    # Triton launches on the thread's current device, whereas this numerical
    # call follows its input tensors, as PyTorch does.
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


def _resident(*tensors: torch.Tensor) -> bool:
    """Check the conditions shared by the ``can_run_triton_*`` checks.

    Fused launches record no autograd graph, so these checks refuse them
    while grad mode is enabled and callers use their tensor-operation
    fallback. Every operand must reside on the first operand's CUDA device.
    """
    device = tensors[0].device
    return (
        triton is not None
        and not torch.is_grad_enabled()
        and launchable(device)
        and all(value.is_cuda and value.device == device for value in tensors)
    )


def _shares_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two tensors view one storage, overlapping or not."""
    return (
        first.untyped_storage().data_ptr()
        == second.untyped_storage().data_ptr()
    )


def can_run_triton_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Return whether a full-width split rotation fits the packed kernel.

    ``x`` and ``out`` are contiguous ``[..., heads, dim]`` tensors whose
    leading axes flatten to the rows of contiguous ``[..., dim / 2]`` factors.
    Only the factor width is checked here; ``apply_rotary`` in
    ``uniserve.nn.functional`` checks that the factor token axes match ``x``.
    Programs read partner features that other programs may already have
    written, so ``out`` must not share storage with ``x``.
    """
    return (
        _resident(x, cos, sin, out)
        and x.ndim >= 2
        and x.numel() > 0
        and x.shape[-1] == 2 * cos.shape[-1]
        and x.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and out.is_contiguous()
        and not _shares_storage(x, out)
    )


def triton_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Rotate every head's contiguous halves into ``out``.

    Arithmetic uses FP32 and rounds once to ``out``'s dtype. Callers first
    check :func:`can_run_triton_rope`.
    """
    dim = int(x.shape[-1])
    total = int(x.numel())

    # Flattened indexing lets one grid cover tokens, heads, and both halves.
    _packed_rope_kernel[(triton.cdiv(total, _TRITON_ROPE_BLOCK),)](
        x,
        cos,
        sin,
        out,
        total,
        int(x.shape[-2]),
        dim,
        dim // 2,
        _TRITON_ROPE_BLOCK,
        num_warps=4,
    )


def can_run_triton_qk_rms_norm_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> bool:
    """Check ``[rows, heads, head_dim]`` Q/K for in-place partial rotation.

    A program tiles the complete head with ``tl.arange(0, head_dim)`` and no
    feature mask, so ``head_dim`` must be a power of two; it is also bounded
    by 1024 features. Compact factors hold one row per token and cover an even
    prefix of the head.
    """
    head_dim = int(query.shape[-1]) if query.ndim == 3 else 0
    return (
        _resident(query, key, query_weight, key_weight, cosine, sine)
        and query.ndim == 3
        and query.shape == key.shape
        and query.dtype == key.dtype
        and query.numel() > 0
        and 0 < head_dim <= 1024
        and head_dim & (head_dim - 1) == 0
        and int(query.stride(-1)) == 1
        and int(key.stride(-1)) == 1
        and query_weight.shape == (head_dim,)
        and key_weight.shape == (head_dim,)
        and query_weight.is_contiguous()
        and key_weight.is_contiguous()
        and cosine.ndim == 2
        and sine.shape == cosine.shape
        and cosine.is_contiguous()
        and sine.is_contiguous()
        and int(cosine.shape[0]) == int(query.shape[0])
        and 0 < int(cosine.shape[1]) * 2 <= head_dim
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
    stepwise: bool = False,
) -> None:
    """Normalize complete Q/K heads and rotate their leading prefix in place.

    Query and key use ``[rows, heads, head_dim]`` layout; compact factors have
    shape ``[rows, rotary_dim / 2]``. ``stepwise`` rounds each operation of
    the eager expression to the input dtype instead of rounding once.
    Callers first check :func:`can_run_triton_qk_rms_norm_rope_inplace`.
    Registration as a ``torch.library`` custom operator with ``mutates_args``
    declares the in-place update of ``query`` and ``key`` to PyTorch.
    """
    rows, heads, head_dim = (int(size) for size in query.shape)

    # Each program covers eight rows for one head, normalizing the full head
    # before applying factors only to the declared rotary prefix.
    block_rows = 8
    _qk_rms_norm_partial_rope_inplace_kernel[
        (triton.cdiv(rows, block_rows), heads)
    ](
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
        int(cosine.shape[-1]) * 2,
        stepwise,
        num_warps=8,
        num_stages=1,
        # A contracted multiply-add would skip a stepwise rounding.
        enable_fp_fusion=not stepwise,
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
    stepwise: bool = False,
) -> None:
    """Fake-tensor implementation: the operator returns nothing.

    The mutation of ``query`` and ``key`` is declared by ``mutates_args``.
    """
    del query, key, query_weight, key_weight, cosine, sine, eps, stepwise


def _qk_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> bool:
    """Check nonempty ``[tokens, heads, dim]`` Q/K and outputs.

    Rows may be strided, but each feature vector is unit-strided. Q and K share
    tokens and width and may differ in head count.
    """
    return (
        q.ndim == 3
        and k.ndim == 3
        and q.dtype == k.dtype
        and q.shape[0] == k.shape[0]
        and q.shape[-1] == k.shape[-1]
        and all(int(size) > 0 for size in (*q.shape, *k.shape))
        and all(int(value.stride(-1)) == 1 for value in (q, k, q_out, k_out))
    )


def _factors(cos: torch.Tensor, sin: torch.Tensor, tokens: int, width: int):
    """Check contiguous compact factors for one fully rotated axis.

    ``tokens`` is the leading Q/K extent, so factor rows index tokens and are
    shared across heads.
    """
    return (
        cos.ndim == 2
        and sin.shape == cos.shape
        and int(cos.shape[0]) == tokens
        and int(cos.shape[1]) * 2 == width
        and cos.is_contiguous()
        and sin.is_contiguous()
    )


def _weights(*pairs: tuple[torch.Tensor, int]) -> bool:
    """Check that each normalization weight covers its domain contiguously."""
    return all(
        weight.shape == (width,) and weight.is_contiguous()
        for weight, width in pairs
    )


def can_run_triton_qk_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> bool:
    """Return whether one domain and one fully rotated axis fit the kernel."""
    dim = int(q.shape[-1])
    return (
        _resident(q, k, q_weight, k_weight, cos, sin, q_out, k_out)
        and _qk_rows(q, k, q_out, k_out)
        and 0 < dim <= 1024
        and dim % 2 == 0
        and _weights((q_weight, dim), (k_weight, dim))
        and _factors(cos, sin, int(q.shape[0]), dim)
    )


def triton_qk_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> None:
    """Normalize complete Q/K heads and rotate their split halves.

    Outputs may be strided views and may alias equal-stride inputs.
    """
    tokens, q_heads, dim = (int(size) for size in q.shape)
    k_heads = int(k.shape[1])
    q_rows, k_rows = tokens * q_heads, tokens * k_heads

    block = triton.next_power_of_2(dim)
    # Small heads share a CTA; wider heads get fewer rows per program, keeping
    # the tile within 512 elements where one row allows it, to limit register
    # growth. A program covers four heads up to width 128, two up to width
    # 256 and one above that.
    rows_per_program = min(4, max(1, 512 // block))
    grid = (
        triton.cdiv(q_rows, rows_per_program)
        + triton.cdiv(k_rows, rows_per_program),
    )
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
        int(q_out.stride(0)),
        int(q_out.stride(1)),
        int(k_out.stride(0)),
        int(k_out.stride(1)),
        dim,
        dim // 2,
        float(eps),
        float(eps),
        block,
        rows_per_program,
        num_warps=4,
    )


def can_run_triton_qk_split_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weights: tuple[torch.Tensor, torch.Tensor],
    k_weights: tuple[torch.Tensor, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> bool:
    """Return whether a rotated head domain and unrotated tail domain fit.

    The first weight covers the fully rotated leading axis; the second covers
    the remaining features, which are normalized but not rotated.
    """
    dim = int(q.shape[-1])
    rope_dim = int(cos.shape[-1]) * 2
    tail_dim = dim - rope_dim
    return (
        _resident(q, k, *q_weights, *k_weights, cos, sin, q_out, k_out)
        and _qk_rows(q, k, q_out, k_out)
        and 0 < rope_dim < dim <= 1024
        and _weights(
            (q_weights[0], rope_dim),
            (k_weights[0], rope_dim),
            (q_weights[1], tail_dim),
            (k_weights[1], tail_dim),
        )
        and _factors(cos, sin, int(q.shape[0]), rope_dim)
    )


def triton_qk_split_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weights: tuple[torch.Tensor, torch.Tensor],
    k_weights: tuple[torch.Tensor, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> None:
    """Normalize two domains and rotate only the leading one."""
    tokens, q_heads, dim = (int(size) for size in q.shape)
    k_heads = int(k.shape[1])
    rope_dim = int(cos.shape[-1]) * 2
    tail_dim = dim - rope_dim
    q_rows, k_rows = tokens * q_heads, tokens * k_heads

    # One program per flattened token/head row across the Q and K ranges.
    _qk_split_rms_norm_rope_kernel[(q_rows + k_rows,)](
        q,
        k,
        q_weights[0],
        q_weights[1],
        k_weights[0],
        k_weights[1],
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
        int(q_out.stride(0)),
        int(q_out.stride(1)),
        int(k_out.stride(0)),
        int(k_out.stride(1)),
        rope_dim,
        rope_dim // 2,
        tail_dim,
        float(eps),
        float(eps),
        triton.next_power_of_2(rope_dim),
        triton.next_power_of_2(tail_dim),
        num_warps=4,
    )


def can_run_triton_qk_multi_axis_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weights: tuple[torch.Tensor, torch.Tensor],
    k_weights: tuple[torch.Tensor, torch.Tensor],
    cos: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sin: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> bool:
    """Return whether a head axis and two shared-domain axes fit the kernel.

    The first domain is one fully rotated axis. The second domain holds two
    equal fully rotated axes that share one RMS denominator and weight while
    keeping separate factors.
    """
    dim0 = int(cos[0].shape[-1]) * 2
    axis_dim = int(cos[1].shape[-1]) * 2
    tokens = int(q.shape[0])
    return (
        _resident(q, k, *q_weights, *k_weights, *cos, *sin, q_out, k_out)
        and _qk_rows(q, k, q_out, k_out)
        and dim0 > 0
        and axis_dim > 0
        and int(cos[2].shape[-1]) * 2 == axis_dim
        and int(q.shape[-1]) == dim0 + 2 * axis_dim <= 1024
        and _weights(
            (q_weights[0], dim0),
            (k_weights[0], dim0),
            (q_weights[1], 2 * axis_dim),
            (k_weights[1], 2 * axis_dim),
        )
        and all(
            _factors(cosine, sine, tokens, width)
            for cosine, sine, width in zip(
                cos, sin, (dim0, axis_dim, axis_dim), strict=True
            )
        )
    )


def triton_qk_multi_axis_rms_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weights: tuple[torch.Tensor, torch.Tensor],
    k_weights: tuple[torch.Tensor, torch.Tensor],
    cos: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sin: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> None:
    """Normalize the head and shared-tail domains and rotate all three axes."""
    tokens, q_heads, dim = (int(size) for size in q.shape)
    k_heads = int(k.shape[1])
    dim0 = int(cos[0].shape[-1]) * 2
    axis_dim = int(cos[1].shape[-1]) * 2
    tail_dim = dim - dim0

    # One program per flattened token/head row across the Q and K ranges.
    _qk_multi_axis_rms_norm_rope_kernel[(tokens * (q_heads + k_heads),)](
        q,
        k,
        q_weights[0],
        q_weights[1],
        k_weights[0],
        k_weights[1],
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
        int(q_out.stride(0)),
        int(q_out.stride(1)),
        int(k_out.stride(0)),
        int(k_out.stride(1)),
        dim0,
        dim0 // 2,
        axis_dim,
        axis_dim // 2,
        tail_dim,
        float(eps),
        float(eps),
        triton.next_power_of_2(dim0),
        triton.next_power_of_2(tail_dim),
        num_warps=2,
    )


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
        HAS_BIAS: tl.constexpr,  # noqa: N803
        HAS_VALUE_BIAS: tl.constexpr,  # noqa: N803
        HEAD_DIM: tl.constexpr,  # noqa: N803
        ROTARY_DIM: tl.constexpr,  # noqa: N803
        EPS: tl.constexpr,  # noqa: N803
        HEAD_BLOCK: tl.constexpr,  # noqa: N803
        ROW_BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Bias and normalize Q/K heads, then rotate leading coordinates.

        Each program owns a disjoint ``[ROW_BLOCK, head]`` tile and loads all
        Q/K values it reads, including partners, before storing in place.
        ``HAS_BIAS`` covers both Q and K biases. With ``HAS_VALUE_BIAS`` the
        value tile, which shares the Q/K strides, receives its bias here.
        ``rows`` is a compile-time constant, so each distinct row count
        compiles its own variant.
        """
        row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        head = tl.program_id(1)
        columns = tl.arange(0, HEAD_BLOCK)
        valid = (row[:, None] < rows) & (columns[None, :] < HEAD_DIM)
        offsets = (
            row[:, None] * row_stride + head * head_stride + columns[None, :]
        )

        # query/key tiles: [ROW_BLOCK, HEAD_BLOCK] for one head.
        query = tl.load(query_ptr + offsets, mask=valid, other=0.0).to(
            tl.float32
        )
        key = tl.load(key_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        if HAS_BIAS:
            bias_offsets = head * HEAD_DIM + columns[None, :]
            query += tl.load(
                query_bias_ptr + bias_offsets,
                mask=columns[None, :] < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
            key += tl.load(
                key_bias_ptr + bias_offsets,
                mask=columns[None, :] < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
        if HAS_VALUE_BIAS:
            value = tl.load(value_ptr + offsets, mask=valid, other=0.0).to(
                tl.float32
            )
            value += tl.load(
                value_bias_ptr + head * HEAD_DIM + columns[None, :],
                mask=columns[None, :] < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                value_ptr + offsets,
                value.to(value_ptr.dtype.element_ty),
                mask=valid,
            )

        query_rstd = tl.rsqrt(tl.sum(query * query, axis=1) / HEAD_DIM + EPS)
        key_rstd = tl.rsqrt(tl.sum(key * key, axis=1) / HEAD_DIM + EPS)
        query = query * query_rstd[:, None]
        key = key * key_rstd[:, None]

        # Partner coordinates come from the opposite half of the rotary
        # subspace.
        half_rotary: tl.constexpr = ROTARY_DIM // 2
        partner_columns = tl.where(
            columns < half_rotary,
            columns + half_rotary,
            columns - half_rotary,
        )
        partner_columns = tl.where(
            columns < ROTARY_DIM, partner_columns, columns
        )
        partner_offsets = (
            row[:, None] * row_stride
            + head * head_stride
            + partner_columns[None, :]
        )

        query_partner = tl.load(
            query_ptr + partner_offsets, mask=valid, other=0.0
        ).to(tl.float32)
        key_partner = tl.load(
            key_ptr + partner_offsets, mask=valid, other=0.0
        ).to(tl.float32)
        if HAS_BIAS:
            partner_bias_offsets = head * HEAD_DIM + partner_columns[None, :]
            query_partner += tl.load(
                query_bias_ptr + partner_bias_offsets,
                mask=columns[None, :] < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
            key_partner += tl.load(
                key_bias_ptr + partner_bias_offsets,
                mask=columns[None, :] < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
        query_partner = query_partner * query_rstd[:, None]
        key_partner = key_partner * key_rstd[:, None]

        # Compact factors hold one phase per rotated pair: [rows, ROTARY/2].
        rotary_mask = valid & (columns[None, :] < ROTARY_DIM)
        rotary_offsets = (
            row[:, None] * rotary_row_stride + (columns % half_rotary)[None, :]
        )
        cosine = tl.load(
            cosine_ptr + rotary_offsets, mask=rotary_mask, other=1.0
        ).to(tl.float32)
        sine = tl.load(
            sine_ptr + rotary_offsets, mask=rotary_mask, other=0.0
        ).to(tl.float32)

        sign = tl.where(columns[None, :] < half_rotary, -1.0, 1.0)
        query_rotated = query * cosine + sign * query_partner * sine
        key_rotated = key * cosine + sign * key_partner * sine
        tl.store(
            query_ptr + offsets,
            tl.where(rotary_mask, query_rotated, query),
            mask=valid,
        )
        tl.store(
            key_ptr + offsets,
            tl.where(rotary_mask, key_rotated, key),
            mask=valid,
        )


def can_run_qk_bias_rms_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor | None,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *biases: torch.Tensor | None,
) -> bool:
    """Return whether strided Q/K/V heads and compact factors fit the kernel.

    Leading token axes must flatten affinely to rows whose heads own disjoint
    unit-strided coordinates, as merged projections lend them. Q, K and V
    share one layout; factors and biases are contiguous. This check covers
    strides, contiguity, the head width bound and launchability only;
    ``qk_bias_rms_norm_rope_`` in ``uniserve.nn.functional`` validates shapes,
    dtypes, device agreement and factor token axes before calling it.
    """
    head_dim, heads = int(query.shape[-1]), int(query.shape[-2])
    row_stride, head_stride = int(query.stride(-3)), int(query.stride(-2))
    # Each leading token axis of extent above one must have a stride of
    # ``row_stride`` times the token extents after it, so the token axes
    # flatten to rows addressed as ``row * row_stride``.
    row_span = row_stride
    for dimension in range(query.ndim - 4, -1, -1):
        row_span *= query.shape[dimension + 1]
        if query.shape[dimension] != 1 and query.stride(dimension) != row_span:
            return False

    projections = (query, key) if value is None else (query, key, value)
    return (
        launchable(query.device)
        and query.is_cuda
        and head_dim <= 256
        and query.stride(-1) == 1
        # Disjoint (row, head) vectors keep the in-place per-tile update from
        # reading coordinates that another program writes.
        and head_stride >= head_dim
        and row_stride >= (heads - 1) * head_stride + head_dim
        and all(tensor.stride() == query.stride() for tensor in projections)
        and all(
            tensor is None or tensor.is_contiguous()
            for tensor in (cosine, sine, *biases)
        )
    )


def qk_bias_rms_norm_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    query_bias: torch.Tensor | None,
    key_bias: torch.Tensor | None,
    value: torch.Tensor | None,
    value_bias: torch.Tensor | None,
    eps: float,
) -> None:
    """Bias, normalize and partially rotate Q/K heads in place.

    Heads are RMS-normalized without a learned weight; the leading
    ``2 * cosine.shape[-1]`` coordinates rotate split-half. The value bias is
    applied in the same launch. ``query_bias`` and ``key_bias`` must be both
    present or both ``None`` because the kernel applies both when
    ``query_bias`` is present; ``value`` and ``value_bias`` likewise pair,
    keyed on ``value_bias``. ``qk_bias_rms_norm_rope_`` in
    ``uniserve.nn.functional`` enforces both pairings.
    """
    head_dim, heads = int(query.shape[-1]), int(query.shape[-2])
    rows = query.numel() // (heads * head_dim)
    rotary_dim = int(cosine.shape[-1]) * 2

    # Each program covers a tile of eight token rows for one head.
    row_block = 8
    _qk_rms_norm_partial_rope_kernel[(triton.cdiv(rows, row_block), heads)](
        query,
        key,
        value,
        query_bias,
        key_bias,
        value_bias,
        cosine,
        sine,
        rows,
        heads,
        int(query.stride(-3)),
        int(query.stride(-2)),
        int(cosine.shape[-1]),
        HAS_BIAS=query_bias is not None,
        HAS_VALUE_BIAS=value_bias is not None,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        EPS=eps,
        HEAD_BLOCK=triton.next_power_of_2(head_dim),
        ROW_BLOCK=row_block,
        num_warps=4,
    )
