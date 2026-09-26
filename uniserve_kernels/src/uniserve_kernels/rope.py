"""Rotary factor tables, rotary rotation and fused Q/K normalization + RoPE.

These Triton kernels back the rotary entry points of
``uniserve.nn.functional`` (``apply_rotary``, ``qk_norm_rope`` and
``qk_bias_rms_norm_rope_``) and the factor tables of
``uniserve.nn.rope.RotaryEmbedding``. Those callers validate the public
contract, run the matching ``unsupported_*`` check and raise on CUDA with its
reason; tensor operations evaluate the same formulas only off CUDA.

Launchers do not revalidate their operands, so callers must run the check
first. Launchers accumulate normalization and rotation in FP32 and round once
into caller-supplied outputs.

Rotary factors are compact: one ``[tokens, rotated / 2]`` cosine and sine
table per rotary axis, shared by every head of a token. Split-half rotation
(GPT-NeoX layout) pairs feature ``i`` of an axis with ``i + rotated / 2``;
interleaved rotation pairs adjacent features. One row kernel serves every
layout of RMS domains and rotary axes: each program owns complete
token/head rows, so outputs may alias their sources.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import tl, triton, unsupported_operands

if triton is not None:  # pragma: no cover - depends on the accelerator stack.
    from triton.language.extra.cuda import libdevice


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

    @triton.jit
    def _norm_rope_rows(
        source,
        weights,
        cosines,
        sines,
        output,
        first_row,
        row_count,
        HEADS: tl.constexpr,  # noqa: N803
        SOURCE_STRIDES: tl.constexpr,  # noqa: N803
        OUTPUT_STRIDES: tl.constexpr,  # noqa: N803
        DIM: tl.constexpr,  # noqa: N803
        DOMAIN_STARTS: tl.constexpr,  # noqa: N803
        DOMAIN_ENDS: tl.constexpr,  # noqa: N803
        AXIS_STARTS: tl.constexpr,  # noqa: N803
        AXIS_ROTATED: tl.constexpr,  # noqa: N803
        FACTOR_STRIDES: tl.constexpr,  # noqa: N803
        INTERLEAVED: tl.constexpr,  # noqa: N803
        EPS: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
        ROWS: tl.constexpr,  # noqa: N803
    ):
        """Normalize and rotate ``ROWS`` token/head rows of one tensor.

        Row ``r`` is head ``r % HEADS`` of token ``r // HEADS``, addressed
        through ``SOURCE_STRIDES`` / ``OUTPUT_STRIDES`` = ``(token, head)``
        element strides with unit feature stride. ``DOMAIN_STARTS`` and
        ``DOMAIN_ENDS`` partition ``[0, DIM)`` into RMS domains, each scaled by
        its ``weights[domain]`` vector; empty tuples skip normalization.
        Rotary axis ``a`` starts at ``AXIS_STARTS[a]`` and rotates its leading
        ``AXIS_ROTATED[a]`` features with the ``[tokens, rotated / 2]`` tables
        ``cosines[a]`` and ``sines[a]`` (token stride ``FACTOR_STRIDES[a]``):
        split-half pairs feature ``i`` with ``i + rotated / 2``,
        ``INTERLEAVED`` pairs adjacent features. Every other feature is only
        normalized. Each program loads all features and partners of its own
        rows before storing, so the output may alias the source.
        """
        rows = first_row + tl.arange(0, ROWS)
        columns = tl.arange(0, BLOCK)
        tokens = (rows // HEADS).to(tl.int64)
        heads = rows % HEADS
        mask = (rows[:, None] < row_count) & (columns[None, :] < DIM)
        source_rows = tokens * SOURCE_STRIDES[0] + heads * SOURCE_STRIDES[1]

        # Per rotated feature: its pair index within the factor table, its
        # partner feature, and whether it leads the pair (sine sign -1).
        # Unrotated features keep themselves as partner with no factors.
        partner = columns
        rotated = columns < 0
        leading = columns < 0
        cosine = tl.full((ROWS, BLOCK), 1.0, tl.float32)
        sine = tl.zeros((ROWS, BLOCK), tl.float32)
        for axis in tl.static_range(len(AXIS_STARTS)):
            if AXIS_ROTATED[axis] > 0:
                local = columns - AXIS_STARTS[axis]
                in_axis = (local >= 0) & (local < AXIS_ROTATED[axis])
                if INTERLEAVED:
                    pair = local // 2
                    partner_local = local ^ 1
                    first = local % 2 == 0
                else:
                    pair = local % (AXIS_ROTATED[axis] // 2)
                    first = local < AXIS_ROTATED[axis] // 2
                    partner_local = tl.where(
                        first,
                        local + AXIS_ROTATED[axis] // 2,
                        local - AXIS_ROTATED[axis] // 2,
                    )
                partner = tl.where(
                    in_axis, AXIS_STARTS[axis] + partner_local, partner
                )
                rotated = rotated | in_axis
                leading = tl.where(in_axis, first, leading)

                factors = tokens[:, None] * FACTOR_STRIDES[axis] + pair[None, :]
                factor_mask = mask & in_axis[None, :]
                cosine = tl.where(
                    in_axis[None, :],
                    tl.load(
                        cosines[axis] + factors, mask=factor_mask, other=1.0
                    ).to(tl.float32),
                    cosine,
                )
                sine = tl.where(
                    in_axis[None, :],
                    tl.load(
                        sines[axis] + factors, mask=factor_mask, other=0.0
                    ).to(tl.float32),
                    sine,
                )

        # values, partners: [ROWS, BLOCK] FP32 tiles of the source rows.
        values = tl.load(
            source + source_rows[:, None] + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        partners = tl.load(
            source + source_rows[:, None] + partner[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        if len(DOMAIN_ENDS) > 0:
            # A rotary axis lies inside one domain, so a partner shares its
            # feature's RMS denominator and reads its own weight entry.
            scale = tl.zeros((ROWS, BLOCK), tl.float32)
            partner_scale = tl.zeros((ROWS, BLOCK), tl.float32)
            for domain in tl.static_range(len(DOMAIN_ENDS)):
                start = DOMAIN_STARTS[domain]
                in_domain = (columns >= start) & (columns < DOMAIN_ENDS[domain])
                squares = tl.where(in_domain[None, :], values * values, 0.0)
                inverse = tl.rsqrt(
                    tl.sum(squares, axis=1) / (DOMAIN_ENDS[domain] - start)
                    + EPS
                )
                weight = tl.load(
                    weights[domain] + (columns - start),
                    mask=in_domain,
                    other=0.0,
                ).to(tl.float32)
                partner_weight = tl.load(
                    weights[domain] + (partner - start),
                    mask=in_domain,
                    other=0.0,
                ).to(tl.float32)
                scale = tl.where(
                    in_domain[None, :],
                    inverse[:, None] * weight[None, :],
                    scale,
                )
                partner_scale = tl.where(
                    in_domain[None, :],
                    inverse[:, None] * partner_weight[None, :],
                    partner_scale,
                )
            values = values * scale
            partners = partners * partner_scale

        # Normalization and rotation stay in FP32; the store rounds once.
        turned = tl.where(
            leading[None, :],
            values * cosine - partners * sine,
            values * cosine + partners * sine,
        )
        output_rows = tokens * OUTPUT_STRIDES[0] + heads * OUTPUT_STRIDES[1]
        tl.store(
            output + output_rows[:, None] + columns[None, :],
            tl.where(rotated[None, :], turned, values),
            mask=mask,
        )

    @triton.jit(do_not_specialize=["row_count"])
    def _rotary_kernel(
        source,
        cosines,
        sines,
        output,
        row_count,
        HEADS: tl.constexpr,  # noqa: N803
        SOURCE_STRIDES: tl.constexpr,  # noqa: N803
        OUTPUT_STRIDES: tl.constexpr,  # noqa: N803
        DIM: tl.constexpr,  # noqa: N803
        ROTATED: tl.constexpr,  # noqa: N803
        FACTOR_STRIDE: tl.constexpr,  # noqa: N803
        INTERLEAVED: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
        ROWS: tl.constexpr,  # noqa: N803
    ):
        """Rotate the leading ``ROTATED`` features of token/head rows.

        One rotary axis spans the head; ``row_count`` is excluded from
        integer specialization, so a new token count reuses the kernel. The
        cosine table stands in for the unused normalization weights.
        """
        _norm_rope_rows(
            source,
            cosines,
            cosines,
            sines,
            output,
            tl.program_id(0) * ROWS,
            row_count,
            HEADS,
            SOURCE_STRIDES,
            OUTPUT_STRIDES,
            DIM,
            (),
            (),
            (0,),
            (ROTATED,),
            (FACTOR_STRIDE,),
            INTERLEAVED,
            0.0,
            BLOCK,
            ROWS,
        )

    @triton.jit(do_not_specialize=["q_rows", "k_rows"])
    def _qk_norm_rope_kernel(
        q,
        k,
        q_weights,
        k_weights,
        cosines,
        sines,
        q_out,
        k_out,
        q_rows,
        k_rows,
        Q_HEADS: tl.constexpr,  # noqa: N803
        K_HEADS: tl.constexpr,  # noqa: N803
        Q_STRIDES: tl.constexpr,  # noqa: N803
        K_STRIDES: tl.constexpr,  # noqa: N803
        Q_OUT_STRIDES: tl.constexpr,  # noqa: N803
        K_OUT_STRIDES: tl.constexpr,  # noqa: N803
        DIM: tl.constexpr,  # noqa: N803
        DOMAIN_STARTS: tl.constexpr,  # noqa: N803
        DOMAIN_ENDS: tl.constexpr,  # noqa: N803
        AXIS_STARTS: tl.constexpr,  # noqa: N803
        AXIS_ROTATED: tl.constexpr,  # noqa: N803
        FACTOR_STRIDES: tl.constexpr,  # noqa: N803
        EPS: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
        ROWS: tl.constexpr,  # noqa: N803
    ):
        """Normalize and rotate Q rows, then K rows, over one program grid.

        Programs below ``cdiv(q_rows, ROWS)`` process query rows and the
        rest process key rows; Q and K share domains, axes and factor tables
        but keep their own weights, head counts and strides. The branch is
        uniform within each program.
        """
        program = tl.program_id(0)
        query_programs = tl.cdiv(q_rows, ROWS)
        if program < query_programs:
            _norm_rope_rows(
                q,
                q_weights,
                cosines,
                sines,
                q_out,
                program * ROWS,
                q_rows,
                Q_HEADS,
                Q_STRIDES,
                Q_OUT_STRIDES,
                DIM,
                DOMAIN_STARTS,
                DOMAIN_ENDS,
                AXIS_STARTS,
                AXIS_ROTATED,
                FACTOR_STRIDES,
                False,
                EPS,
                BLOCK,
                ROWS,
            )
        else:
            _norm_rope_rows(
                k,
                k_weights,
                cosines,
                sines,
                k_out,
                (program - query_programs) * ROWS,
                k_rows,
                K_HEADS,
                K_STRIDES,
                K_OUT_STRIDES,
                DIM,
                DOMAIN_STARTS,
                DOMAIN_ENDS,
                AXIS_STARTS,
                AXIS_ROTATED,
                FACTOR_STRIDES,
                False,
                EPS,
                BLOCK,
                ROWS,
            )


def unsupported_rotary_factors(
    positions: torch.Tensor, frequencies: torch.Tensor, dtype: torch.dtype
) -> str | None:
    """Return why the factor kernel cannot take these operands, or ``None``.

    Positions are strided integer or floating tensors of any rank,
    frequencies a strided floating vector, and ``dtype`` the floating output
    dtype. Callers of :func:`rotary_factors` supply contiguous outputs,
    which this check does not inspect.
    """
    floating = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    reason = unsupported_operands(positions, frequencies)
    if reason is not None:
        return reason
    if positions.layout != torch.strided or frequencies.layout != torch.strided:
        return "positions or frequencies are not strided tensors"
    if positions.dtype not in floating | {torch.int32, torch.int64}:
        return f"position dtype {positions.dtype} is not int32, int64 or float"
    if frequencies.ndim != 1 or frequencies.dtype not in floating:
        return "frequencies are not one floating vector"
    if dtype not in floating:
        return f"factor dtype {dtype} is not floating"
    return None


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


#: Widest head the row kernels tile in one program.
MAX_HEAD_DIM = 1024
_FLOATING = (torch.float16, torch.bfloat16, torch.float32)


def _merged_axes(shape, strides) -> list[list[int]]:
    """Merge adjacent axes that step over each other; drop extent-one axes."""
    axes: list[list[int]] = []
    for extent, stride in zip(shape, strides, strict=True):
        if extent == 1:
            continue
        if axes and axes[-1][1] == extent * stride:
            axes[-1] = [axes[-1][0] * extent, stride]
        else:
            axes.append([int(extent), int(stride)])
    return axes


def token_rows(value: torch.Tensor, trailing: int) -> tuple[int, ...] | None:
    """Return ``value``'s token stride and its trailing strides, or ``None``.

    The leading ``value.ndim - trailing`` axes are token axes; they must
    merge into one strided token axis (stride 0 when there is at most one
    token), and the last axis must be unit-strided. The result is ``(token
    stride, *strides of the trailing axes)``.
    """
    if value.ndim < trailing or (value.shape[-1] > 1 and value.stride(-1) != 1):
        return None
    leading = value.ndim - trailing
    axes = _merged_axes(value.shape[:leading], value.stride()[:leading])
    if len(axes) > 1:
        return None
    return (axes[0][1] if axes else 0, *value.stride()[leading:])


def _shares_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two tensors view one storage, overlapping or not."""
    return (
        first.untyped_storage().data_ptr()
        == second.untyped_storage().data_ptr()
    )


def _aliases(source: torch.Tensor, output: torch.Tensor) -> str | None:
    """Reject an output that shares storage with its source unless it is
    the source itself.

    Programs own disjoint rows and load a row's features and partners before
    storing it, so writing in place is safe; any other overlap could let one
    program overwrite rows another has yet to read.
    """  # noqa: D205
    if _shares_storage(source, output) and (
        source.data_ptr() != output.data_ptr()
        or source.stride() != output.stride()
    ):
        return "an output overlaps its source without aliasing it exactly"
    return None


def _rows_per_program(block: int) -> int:
    """Rows per program: about 2048 features per tile, at most 16 rows."""
    return min(16, max(1, 2048 // block))


def _starts(widths) -> tuple[int, ...]:
    """Return the first feature of each consecutive width."""
    starts, start = [], 0
    for width in widths:
        starts.append(start)
        start += width
    return tuple(starts)


def _factor_tables(tokens: int, cosines, sines, rotated) -> str | None:
    """Check each axis's ``[..., rotated / 2]`` cosine and sine tables."""
    for cosine, sine, width in zip(cosines, sines, rotated, strict=True):
        if cosine.shape != sine.shape or cosine.stride() != sine.stride():
            return "cosine and sine tables differ in shape or strides"
        if cosine.dtype not in _FLOATING or sine.dtype not in _FLOATING:
            return "factor tables are not floating"
        if width and token_rows(cosine, 1) is None:
            return (
                "factor token axes do not merge into one strided axis with "
                "unit-strided pairs"
            )
        if width and cosine.numel() // max(1, width // 2) != tokens:
            return "factor tables do not hold one row per token"
    return None


def unsupported_rotary(
    x: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    out: torch.Tensor,
) -> str | None:
    """Return why the rotary kernel cannot rotate ``x``, or ``None``.

    ``x`` and ``out`` are floating ``[..., heads, head_dim]`` tensors whose
    token axes merge into one strided axis with unit feature stride;
    ``out`` is ``x`` itself or does not overlap it. The factors are
    ``[..., rotated / 2]`` tables over the same tokens. ``head_dim`` is at
    most :data:`MAX_HEAD_DIM`.
    """
    reason = unsupported_operands(x, cosine, sine, out)
    if reason is not None:
        return reason
    if x.dtype not in _FLOATING or out.dtype not in _FLOATING:
        return "rotated values are not float16, bfloat16 or float32"
    if not 0 < x.shape[-1] <= MAX_HEAD_DIM:
        return f"head width {x.shape[-1]} is outside 1..{MAX_HEAD_DIM}"
    if token_rows(x, 2) is None or token_rows(out, 2) is None:
        return (
            "token axes do not merge into one strided axis with unit-strided "
            "features"
        )
    tokens = x.numel() // (x.shape[-2] * x.shape[-1]) if x.numel() else 0
    return _factor_tables(
        tokens, (cosine,), (sine,), (cosine.shape[-1] * 2,)
    ) or _aliases(x, out)


def rotary(
    x: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    out: torch.Tensor,
    *,
    interleaved: bool,
) -> None:
    """Rotate the leading ``2 * cosine.shape[-1]`` features of every head.

    ``interleaved`` pairs adjacent features; otherwise feature ``i`` pairs
    with ``i + rotated / 2``. Remaining features are copied unchanged.
    Arithmetic uses FP32 and rounds once to ``out``'s dtype. Callers first
    check :func:`unsupported_rotary`; ``out`` may be ``x``.
    """
    heads, dim = int(x.shape[-2]), int(x.shape[-1])
    rows = x.numel() // dim
    if rows == 0:
        return
    rotated = int(cosine.shape[-1]) * 2
    block = triton.next_power_of_2(dim)
    tile_rows = _rows_per_program(block)
    # An empty factor table rotates nothing; ``x`` stands in for its pointer
    # and is never read through it.
    tables = (cosine, sine) if rotated else (x, x)
    _rotary_kernel[(triton.cdiv(rows, tile_rows),)](
        x,
        (tables[0],),
        (tables[1],),
        out,
        rows,
        heads,
        token_rows(x, 2)[:2],
        token_rows(out, 2)[:2],
        dim,
        rotated,
        token_rows(cosine, 1)[0] if rotated else 0,
        interleaved,
        block,
        tile_rows,
        num_warps=4,
    )


def unsupported_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weights: tuple[torch.Tensor, ...],
    k_weights: tuple[torch.Tensor, ...],
    cosines: tuple[torch.Tensor, ...],
    sines: tuple[torch.Tensor, ...],
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> str | None:
    """Return why the fused kernel cannot take a Q/K call, or ``None``.

    ``uniserve.nn.functional.qk_norm_rope`` validates the domain and axis
    partition first. Here, Q, K and their outputs are floating
    ``[..., heads, head_dim]`` tensors whose token axes merge into one
    strided axis with unit feature stride; each output is its source itself
    or does not overlap it. Weights are contiguous vectors, factor tables
    ``[..., rotated / 2]`` over the same tokens, and ``head_dim`` is at most
    :data:`MAX_HEAD_DIM`.
    """
    reason = unsupported_operands(
        q, k, *q_weights, *k_weights, *cosines, *sines, q_out, k_out
    )
    if reason is not None:
        return reason
    if any(
        value.dtype not in _FLOATING
        for value in (q, k, q_out, k_out, *q_weights, *k_weights)
    ):
        return "values or weights are not float16, bfloat16 or float32"
    if not 0 < q.shape[-1] <= MAX_HEAD_DIM:
        return f"head width {q.shape[-1]} is outside 1..{MAX_HEAD_DIM}"
    if any(token_rows(value, 2) is None for value in (q, k, q_out, k_out)):
        return (
            "token axes do not merge into one strided axis with unit-strided "
            "features"
        )
    if any(not weight.is_contiguous() for weight in (*q_weights, *k_weights)):
        return "a normalization weight is not contiguous"
    tokens = q.numel() // (q.shape[-2] * q.shape[-1]) if q.numel() else 0
    return (
        _factor_tables(
            tokens,
            cosines,
            sines,
            tuple(int(cosine.shape[-1]) * 2 for cosine in cosines),
        )
        or _aliases(q, q_out)
        or _aliases(k, k_out)
    )


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weights: tuple[torch.Tensor, ...],
    k_weights: tuple[torch.Tensor, ...],
    cosines: tuple[torch.Tensor, ...],
    sines: tuple[torch.Tensor, ...],
    eps: float,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    *,
    axis_dims: tuple[int, ...],
) -> None:
    """RMS-normalize Q/K domains and rotate every axis in one launch.

    Weight ``i`` scales the ``i``-th consecutive normalization domain, whose
    width is the weight length. Axis ``a`` spans ``axis_dims[a]`` features
    and rotates its leading ``2 * cosines[a].shape[-1]`` split-half. Callers
    first check :func:`unsupported_qk_norm_rope`; outputs may be the sources.
    """
    q_heads, dim = int(q.shape[-2]), int(q.shape[-1])
    k_heads = int(k.shape[-2])
    tokens = q.numel() // (q_heads * dim)
    if tokens == 0:
        return
    widths = tuple(int(weight.shape[0]) for weight in q_weights)
    starts = _starts(widths)
    rotated = tuple(int(cosine.shape[-1]) * 2 for cosine in cosines)
    block = triton.next_power_of_2(dim)
    tile_rows = _rows_per_program(block)
    q_rows, k_rows = tokens * q_heads, tokens * k_heads
    grid = (triton.cdiv(q_rows, tile_rows) + triton.cdiv(k_rows, tile_rows),)
    # Axes without rotation read no factors; ``q`` stands in for their empty
    # tables' pointers.
    cosines = tuple(cosine if cosine.shape[-1] else q for cosine in cosines)
    sines = tuple(sine if sine.shape[-1] else q for sine in sines)
    _qk_norm_rope_kernel[grid](
        q,
        k,
        tuple(q_weights),
        tuple(k_weights),
        cosines,
        sines,
        q_out,
        k_out,
        q_rows,
        k_rows,
        q_heads,
        k_heads,
        token_rows(q, 2)[:2],
        token_rows(k, 2)[:2],
        token_rows(q_out, 2)[:2],
        token_rows(k_out, 2)[:2],
        dim,
        starts,
        tuple(start + width for start, width in zip(starts, widths)),
        _starts(axis_dims),
        rotated,
        tuple(
            token_rows(cosine, 1)[0] if width else 0
            for cosine, width in zip(cosines, rotated, strict=True)
        ),
        float(eps),
        block,
        tile_rows,
        num_warps=4,
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


def unsupported_qk_bias_rms_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor | None,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *biases: torch.Tensor | None,
) -> str | None:
    """Return why the in-place bias kernel cannot take Q/K/V, or ``None``.

    Leading token axes must flatten affinely to rows whose heads own disjoint
    unit-strided coordinates, as merged projections lend them. Q, K and V
    share one layout; factors and biases are contiguous; ``head_dim`` is at
    most 256. ``qk_bias_rms_norm_rope_`` in ``uniserve.nn.functional``
    validates shapes, dtypes, device agreement and factor token axes before
    calling it.
    """
    reason = unsupported_operands(query, key, value, cosine, sine, *biases)
    if reason is not None:
        return reason
    head_dim, heads = int(query.shape[-1]), int(query.shape[-2])
    row_stride, head_stride = int(query.stride(-3)), int(query.stride(-2))
    # Each leading token axis of extent above one must have a stride of
    # ``row_stride`` times the token extents after it, so the token axes
    # flatten to rows addressed as ``row * row_stride``.
    row_span = row_stride
    for dimension in range(query.ndim - 4, -1, -1):
        row_span *= query.shape[dimension + 1]
        if query.shape[dimension] != 1 and query.stride(dimension) != row_span:
            return "token axes do not flatten to rows of one stride"

    projections = (query, key) if value is None else (query, key, value)
    if head_dim > 256:
        return f"head width {head_dim} exceeds the kernel's 256"
    if query.stride(-1) != 1:
        return "head features are not unit-strided"
    # Disjoint (row, head) vectors keep the in-place per-tile update from
    # reading coordinates that another program writes.
    if (
        head_stride < head_dim
        or row_stride < (heads - 1) * head_stride + head_dim
    ):
        return "heads of one row overlap"
    if any(tensor.stride() != query.stride() for tensor in projections):
        return "Q, K and V do not share one layout"
    if any(
        tensor is not None and not tensor.is_contiguous()
        for tensor in (cosine, sine, *biases)
    ):
        return "factor tables or biases are not contiguous"
    return None


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
