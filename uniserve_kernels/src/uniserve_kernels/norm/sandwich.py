"""Sandwich RMS normalization of a residual stream in one launch.

The Triton kernel here backs ``uniserve.nn.functional.sandwich_rms_norm``,
which defines the numbers with its portable composition, allocates the
outputs and raises on CUDA with the reason :func:`unsupported` reports. One
program owns one contiguous row: it normalizes and sums the updates,
normalizes the sum, adds it to the residual, applies the optional device
scale, stores the stream and then every requested normalization of that
stream. Each stored intermediate of the composition rounds to the row dtype
at the same point.

Every normalization of a row evaluates exactly as
``uniserve_kernels.norm.rms.rms_norm`` does on that row: the same
``x * x`` square sum reduced by ``tl.sum`` over the same power-of-two block
and warp count, the same compile-time width and epsilon, and
``x * rsqrt(mean + eps) * weight``. The FP32 reduction order is therefore
the one of the separate normalizations this launch replaces, and the
results are bit-identical to the composition of those kernels and the
tensor operations between them. The scale is read from device memory, so the
launch synchronizes nothing and CUDA graphs capture it.

An update may also be given as routes: ``[rows, K, width]`` rows and
``[rows, K]`` FP32 weights whose update value is the FP32 fused
multiply-add chain over ``k = 0 .. K - 1`` in order, from zero, rounded
once to the row dtype (``uniserve.nn.functional.Routes``). The launch
evaluates that value while reading the routes, so the combined rows are
never stored.

A normalization may instead be stored in a calibrated NVFP4 encoding: its
row, rounded to the row dtype as the unencoded output would store it, is
encoded by :func:`uniserve_kernels.quantization.nvfp4_encode` into packed
E2M1 values and linear E4M3 block scales, bit for bit as FlashInfer's NVFP4
encoder encodes the stored row, and the unencoded row is never written.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import torch

from uniserve_kernels.norm.rms import MAX_WIDTH, row_launch
from uniserve_kernels.quantization import nvfp4_encode
from uniserve_kernels.triton import tl, triton, unsupported_operands

#: Rows are 16-bit floating values.
ROW_DTYPES = (torch.bfloat16, torch.float16)
#: Updates summed before the normalization, and normalizations of the stream.
MAX_UPDATES = 2
MAX_NORMS = 4
#: Routes of an update given as routes; each is one unrolled row load.
MAX_ROUTES = 32
_VECTOR_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


if triton is not None:

    @triton.jit
    def _normalize(
        values,
        weight_ptr,
        columns,
        mask,
        WEIGHTED: tl.constexpr,  # noqa: N803
        WIDTH: tl.constexpr,  # noqa: N803
        EPS: tl.constexpr,  # noqa: N803
    ):
        """``values * rsqrt(mean(values^2) + EPS) * weight`` in FP32, as
        ``uniserve_kernels.norm.rms`` evaluates it; unweighted without the
        unit multiplication (exact either way).
        """  # noqa: D205
        var = tl.sum(values * values, axis=0) / WIDTH
        normalized = values * tl.rsqrt(var + EPS)
        if WEIGHTED:
            weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(
                tl.float32
            )
            normalized = normalized * weight
        return normalized

    @triton.jit
    def _update(
        rows_ptr,
        weights_ptr,
        row,
        columns,
        mask,
        ROUTES: tl.constexpr,  # noqa: N803
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """One update row in FP32: a stored row (``ROUTES`` 0) or the value
        of ``ROUTES`` weighted route rows, the FP32 fma chain in route order
        from zero rounded once to the row dtype.
        """  # noqa: D205
        if ROUTES == 0:
            return tl.load(
                rows_ptr + row * WIDTH + columns, mask=mask, other=0.0
            ).to(tl.float32)
        total = tl.zeros([BLOCK], dtype=tl.float32)
        for route in tl.static_range(ROUTES):
            values = tl.load(
                rows_ptr + (row * ROUTES + route) * WIDTH + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weights_ptr + row * ROUTES + route)
            total = tl.fma(values, weight, total)
        return total.to(rows_ptr.dtype.element_ty).to(tl.float32)

    @triton.jit
    def _sandwich_kernel(
        residual_ptr,
        first_ptr,
        second_ptr,
        first_routes_ptr,
        second_routes_ptr,
        first_weight_ptr,
        second_weight_ptr,
        weight_ptr,
        scale_ptr,
        stream_ptr,
        norm_weights,
        norm_factors,
        norm_outputs,
        norm_values,
        norm_block_scales,
        norm_tensor_scales,
        norm_encode_scales,
        WIDTH: tl.constexpr,  # noqa: N803
        EPS: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
        HAS_RESIDUAL: tl.constexpr,  # noqa: N803
        TWO: tl.constexpr,  # noqa: N803
        FIRST_ROUTES: tl.constexpr,  # noqa: N803
        SECOND_ROUTES: tl.constexpr,  # noqa: N803
        FIRST_NORMALIZED: tl.constexpr,  # noqa: N803
        SECOND_NORMALIZED: tl.constexpr,  # noqa: N803
        SCALED: tl.constexpr,  # noqa: N803
        NORM_WEIGHTED: tl.constexpr,  # noqa: N803
        NORM_FACTORED: tl.constexpr,  # noqa: N803
        NORM_SCALARS: tl.constexpr,  # noqa: N803
        NORM_ENCODED: tl.constexpr,  # noqa: N803
    ):
        """Sandwich-normalize one contiguous row.

        ``FIRST_ROUTES`` / ``SECOND_ROUTES`` count the routes of an update
        given as routes (rows at ``*_ptr``, FP32 weights at
        ``*_routes_ptr``), zero for a stored update.
        ``NORM_SCALARS[i]`` is the trailing number of normalization ``i``,
        or ``None``. An ``NORM_ENCODED[i]`` normalization stores its NVFP4
        encoding with calibrated scale ``norm_encode_scales[i]`` into
        ``norm_values[i]`` (``[rows, WIDTH / 2]`` packed E2M1),
        ``norm_block_scales[i]`` (``[rows, WIDTH / 16]`` E4M3) and, from the
        first row, ``norm_tensor_scales[i]``. Every ``.to(dtype)`` marks a
        rounding point of the composition.
        """
        dtype = stream_ptr.dtype.element_ty
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        mask = columns < WIDTH
        base = row * WIDTH

        update = _update(
            first_ptr,
            first_routes_ptr,
            row,
            columns,
            mask,
            FIRST_ROUTES,
            WIDTH,
            BLOCK,
        )
        if FIRST_NORMALIZED:
            update = _normalize(
                update, first_weight_ptr, columns, mask, True, WIDTH, EPS
            )
            update = update.to(dtype).to(tl.float32)
        if TWO:
            term = _update(
                second_ptr,
                second_routes_ptr,
                row,
                columns,
                mask,
                SECOND_ROUTES,
                WIDTH,
                BLOCK,
            )
            if SECOND_NORMALIZED:
                term = _normalize(
                    term, second_weight_ptr, columns, mask, True, WIDTH, EPS
                )
                term = term.to(dtype).to(tl.float32)
            update = (update + term).to(dtype).to(tl.float32)

        stream = _normalize(update, weight_ptr, columns, mask, True, WIDTH, EPS)
        if HAS_RESIDUAL:
            residual = tl.load(
                residual_ptr + base + columns, mask=mask, other=0.0
            ).to(tl.float32)
            stream = residual + stream.to(dtype).to(tl.float32)
        stream = stream.to(dtype).to(tl.float32)
        if SCALED:
            scale = tl.load(scale_ptr).to(tl.float32)
            stream = (stream * scale).to(dtype).to(tl.float32)
        tl.store(stream_ptr + base + columns, stream, mask=mask)

        for n in tl.static_range(len(NORM_WEIGHTED)):
            value = _normalize(
                stream,
                norm_weights[n],
                columns,
                mask,
                NORM_WEIGHTED[n],
                WIDTH,
                EPS,
            )
            value = value.to(dtype).to(tl.float32)
            if NORM_FACTORED[n]:
                factor = tl.load(
                    norm_factors[n] + columns, mask=mask, other=0.0
                ).to(tl.float32)
                value = (value * factor).to(dtype).to(tl.float32)
            if NORM_SCALARS[n] is not None:
                value = value * NORM_SCALARS[n]
            if NORM_ENCODED[n]:
                # Encode the row as the unencoded output would store it.
                packed, scale_bytes = nvfp4_encode(
                    value.to(dtype).to(tl.float32), norm_encode_scales[n], BLOCK
                )
                pairs = tl.arange(0, BLOCK // 2)
                tl.store(
                    norm_values[n] + row * (WIDTH // 2) + pairs,
                    packed,
                    mask=pairs < WIDTH // 2,
                )
                blocks = tl.arange(0, BLOCK // 16)
                tl.store(
                    norm_block_scales[n] + row * (WIDTH // 16) + blocks,
                    scale_bytes,
                    mask=blocks < WIDTH // 16,
                )
                if row == 0:
                    tl.store(norm_tensor_scales[n], norm_encode_scales[n])
            else:
                tl.store(norm_outputs[n] + base + columns, value, mask=mask)


def _unsupported_vector(vector: torch.Tensor, width: int, device) -> str | None:
    if vector.device != device:
        return "a weight or factor resides on another device"
    if vector.dtype not in _VECTOR_DTYPES:
        return f"weight or factor dtype {vector.dtype} is not floating"
    if vector.shape != (width,) or not vector.is_contiguous():
        return "a weight or factor is not one contiguous row-width vector"
    return None


def unsupported(
    residual: torch.Tensor,
    updates: tuple[
        tuple[
            torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            torch.Tensor | None,
        ],
        ...,
    ],
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    norms: tuple[tuple, ...],
    encodings: tuple = (),
) -> str | None:
    """Return why the kernel cannot take a sandwich call, or ``None``.

    ``residual`` and every update are contiguous CUDA rows of one BF16 or
    FP16 dtype and shape, at most :data:`MAX_WIDTH` wide. Weights and
    vector factors are contiguous FP32, BF16 or FP16 ``[width]`` vectors (a
    normalization weight may be ``None``: unweighted), ``scale`` holds one
    floating value, and each normalization's factors are at most one vector
    followed by at most one Python number. An update given as routes is a
    ``(rows, weights)`` pair: contiguous ``[rows, K, width]`` rows of the
    residual dtype, ``1 <= K <=`` :data:`MAX_ROUTES`, with contiguous
    ``[rows, K]`` FP32 weights. ``encodings`` holds, per normalization,
    ``None`` or a calibrated NVFP4 ``Quantizer`` whose encoding stores it;
    encoded rows are a whole number of K16 blocks.
    """
    routed = [
        (update, None) if isinstance(update, torch.Tensor) else update
        for update, _ in updates
    ]
    rows = (residual, *(values for values, _ in routed))
    reason = unsupported_operands(
        *rows,
        *(weights for _, weights in routed),
        weight,
        scale,
        *(w for _, w in updates),
    )
    if reason is not None:
        return reason
    if residual.dtype not in ROW_DTYPES:
        return f"row dtype {residual.dtype} is not bfloat16 or float16"
    width = int(residual.shape[-1]) if residual.ndim else 0
    if not 0 < width <= MAX_WIDTH:
        return f"row width {width} is outside 1..{MAX_WIDTH}"
    if not 0 < len(updates) <= MAX_UPDATES or len(norms) > MAX_NORMS:
        return (
            f"the kernel sums 1..{MAX_UPDATES} updates and stores at most "
            f"{MAX_NORMS} normalizations"
        )
    for row, weights in ((residual, None), *routed):
        # Route rows [rows, K, width] add the route axis to [rows, width].
        shape = (
            row.shape
            if weights is None
            else (row.shape[0], *row.shape[2:])
            if row.ndim == 3
            else None
        )
        if row.dtype != residual.dtype or shape != residual.shape:
            return "updates do not match the residual rows"
        if not row.is_contiguous():
            return "rows are not contiguous"
    for values, weights in routed:
        if weights is None:
            continue
        if not 1 <= values.shape[1] <= MAX_ROUTES:
            return f"routes number 1..{MAX_ROUTES} per row"
        if (
            weights.dtype != torch.float32
            or weights.shape != values.shape[:2]
            or not weights.is_contiguous()
        ):
            return "route weights are not contiguous FP32 [rows, K]"
    vectors = [weight, *(w for _, w in updates if w is not None)]
    for norm in norms:
        if norm[0] is not None:
            vectors.append(norm[0])
        factors = norm[1:]
        if factors and isinstance(factors[0], torch.Tensor):
            vectors.append(factors[0])
            factors = factors[1:]
        if len(factors) > 1 or any(
            not isinstance(factor, Real) for factor in factors
        ):
            return (
                "a normalization's factors must be at most one vector "
                "followed by at most one number"
            )
    for vector in vectors:
        reason = _unsupported_vector(vector, width, residual.device)
        if reason is not None:
            return reason
    if scale is not None and (
        scale.numel() != 1 or scale.dtype not in _VECTOR_DTYPES
    ):
        return "the scale is not one floating value"
    if len(encodings) > len(norms):
        return "more encodings than normalizations"
    for quantizer in encodings:
        if quantizer is None:
            continue
        if quantizer.format != "nvfp4" or quantizer.calibrated_scale is None:
            return "only calibrated NVFP4 encodings are stored"
        if width % 16:
            return f"row width {width} is not a whole number of K16 blocks"
    return None


def sandwich(
    residual: torch.Tensor,
    updates: tuple[
        tuple[
            torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            torch.Tensor | None,
        ],
        ...,
    ],
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    norms: tuple[tuple, ...],
    eps: float,
    stream: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    encodings: tuple = (),
) -> None:
    """Store the stream and its normalizations into caller outputs.

    ``updates`` entries are ``(update, branch_weight)`` with ``update`` a
    row tensor or a ``(rows, weights)`` routes pair (see
    :func:`unsupported`). ``stream`` and each unencoded entry of
    ``outputs`` (one per normalization) are contiguous rows like
    ``residual``, overlapping no operand. The output of a normalization
    ``encodings`` names is that quantizer's ``QuantizedTensor`` of the rows'
    shape with linear block scales; the launch fills its values, block
    scales and tensor scale.
    Callers first check :func:`unsupported`.
    """
    width = int(residual.shape[-1])
    rows = residual.numel() // width
    encodings = tuple(encodings) + (None,) * (len(norms) - len(encodings))
    # The storage of each encoded output (a QuantizedTensor), else None.
    fields: list[Mapping[str, torch.Tensor] | None] = [
        None if quantizer is None else output.buffers()  # type: ignore[attr-defined]
        for quantizer, output in zip(encodings, outputs, strict=True)
    ]
    if rows == 0:
        for quantizer, field in zip(encodings, fields, strict=True):
            if quantizer is not None and field is not None:
                field["tensor_scale"].fill_(quantizer.calibrated_scale)
        return
    weights, factors, scalars = [], [], []
    for norm in norms:
        rest = norm[1:]
        vector = rest[0] if rest and isinstance(rest[0], torch.Tensor) else None
        rest = rest[1:] if vector is not None else rest
        weights.append(norm[0])
        factors.append(vector)
        scalars.append(float(rest[0]) if rest else None)
    # The block and warp count of the rms row kernel give every
    # normalization its reduction layout, hence its FP32 order.
    block, warps = row_launch(width)
    # Each update is (rows, route weights or None, branch weight).
    terms = [
        (update, None, branch)
        if isinstance(update, torch.Tensor)
        else (*update, branch)
        for update, branch in updates
    ]
    first, first_routes, first_weight = terms[0]
    second, second_routes, second_weight = (
        terms[1] if len(terms) == 2 else (first, None, None)
    )
    # Absent pointers stand in as the stream; the kernel never reads them.
    _sandwich_kernel[(rows,)](
        residual,
        first,
        second,
        stream if first_routes is None else first_routes,
        stream if second_routes is None else second_routes,
        stream if first_weight is None else first_weight,
        stream if second_weight is None else second_weight,
        weight,
        stream if scale is None else scale,
        stream,
        tuple(stream if w is None else w for w in weights),
        tuple(stream if f is None else f for f in factors),
        tuple(
            stream if f is not None else output
            for f, output in zip(fields, outputs, strict=True)
        ),
        tuple(stream if f is None else f["values"] for f in fields),
        tuple(stream if f is None else f["block_scale"] for f in fields),
        tuple(stream if f is None else f["tensor_scale"] for f in fields),
        tuple(
            1.0 if quantizer is None else float(quantizer.calibrated_scale)
            for quantizer in encodings
        ),
        width,
        float(eps),
        block,
        True,
        len(updates) == 2,
        0 if first_routes is None else first.shape[1],
        0 if second_routes is None else second.shape[1],
        first_weight is not None,
        second_weight is not None,
        scale is not None,
        tuple(w is not None for w in weights),
        tuple(f is not None for f in factors),
        tuple(scalars),
        tuple(quantizer is not None for quantizer in encodings),
        num_warps=warps,
    )
