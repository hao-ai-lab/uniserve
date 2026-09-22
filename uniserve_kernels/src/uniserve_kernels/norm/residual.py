"""Channel-scaled residuals followed by RMS or layer normalization.

Projection bias, residual sums, statistics and affine transforms stay in FP32
until the store. RMS variants update the residual in place while normalizing
the unrounded sum; magnitude variants record one per-row absolute maximum of
the stored output.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

#: One program reduces a complete row, bounding the normalized width.
MAX_WIDTH = 32768


if triton is not None:

    @triton.jit
    def _weighted_rms_norm_kernel(
        hidden_ptr,
        weight_ptr,
        output_ptr,
        eps,
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Normalize one activation row and apply its learned scale."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)

        hidden = tl.load(
            hidden_ptr + row * WIDTH + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)

        # Reduction and scaling stay in fp32; the store casts to the output
        # dtype.
        inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / WIDTH + eps)
        tl.store(
            output_ptr + row * WIDTH + columns,
            hidden * inverse_rms * weight,
            mask=columns < WIDTH,
        )

    @triton.jit
    def _weighted_rms_norm_absmax_kernel(
        hidden_ptr,
        weight_ptr,
        output_ptr,
        partials_ptr,
        eps,
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Normalize one row and publish its output magnitude.

        Normalize one row while publishing its output magnitude for
        reduction.
        """
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)

        hidden = tl.load(
            hidden_ptr + row * WIDTH + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)

        inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / WIDTH + eps)
        output = (hidden * inverse_rms * weight).to(output_ptr.dtype.element_ty)
        tl.store(
            output_ptr + row * WIDTH + columns, output, mask=columns < WIDTH
        )

        # ``partials_ptr`` collects one magnitude per row; ``finish_absmax``
        # reduces them to the published scalar after the launch.
        tl.store(
            partials_ptr + row,
            tl.max(
                tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0),
                axis=0,
            ),
        )

    @triton.jit
    def _scaled_residual_rms_norm_kernel(
        hidden_ptr,
        update_ptr,
        update_bias_ptr,
        scale_ptr,
        weight_ptr,
        output_ptr,
        eps,
        HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Update a residual row in place and emit its RMS-normalized value."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        offsets = row * WIDTH + columns

        hidden = tl.load(
            hidden_ptr + offsets, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        update = tl.load(
            update_ptr + offsets, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(
                update_bias_ptr + columns, mask=columns < WIDTH, other=0.0
            ).to(tl.float32)
        scale = tl.load(
            scale_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)

        residual = hidden + update * scale
        inverse_rms = tl.rsqrt(
            tl.sum(residual * residual, axis=0) / WIDTH + eps
        )
        weight = tl.load(
            weight_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)

        # The residual stream is written back before its normalized projection
        # is rounded into the output dtype.
        tl.store(hidden_ptr + offsets, residual, mask=columns < WIDTH)
        tl.store(
            output_ptr + offsets,
            residual * inverse_rms * weight,
            mask=columns < WIDTH,
        )

    @triton.jit
    def _scaled_residual_rms_norm_absmax_kernel(
        hidden_ptr,
        update_ptr,
        update_bias_ptr,
        scale_ptr,
        weight_ptr,
        output_ptr,
        partials_ptr,
        eps,
        HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Fuse residual update, RMS normalization, and magnitude capture.

        Fuse residual update, RMS normalization, and rowwise magnitude
        capture.
        """
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        offsets = row * WIDTH + columns

        hidden = tl.load(
            hidden_ptr + offsets, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        update = tl.load(
            update_ptr + offsets, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(
                update_bias_ptr + columns, mask=columns < WIDTH, other=0.0
            ).to(tl.float32)
        scale = tl.load(
            scale_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)

        residual = hidden + update * scale
        inverse_rms = tl.rsqrt(
            tl.sum(residual * residual, axis=0) / WIDTH + eps
        )
        weight = tl.load(
            weight_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        output = (residual * inverse_rms * weight).to(
            output_ptr.dtype.element_ty
        )

        tl.store(hidden_ptr + offsets, residual, mask=columns < WIDTH)
        tl.store(output_ptr + offsets, output, mask=columns < WIDTH)
        tl.store(
            partials_ptr + row,
            tl.max(
                tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0),
                axis=0,
            ),
        )

    @triton.jit
    def _scaled_residual_kernel(
        hidden_ptr,
        update_ptr,
        update_bias_ptr,
        scale_ptr,
        elements,
        HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Apply a channel-scaled update to the residual stream."""
        # Flat element offsets address ``hidden``/``update``; the channel index
        # selects the per-channel scale and optional bias.
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        columns = offsets % WIDTH

        hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_UPDATE_BIAS:
            update += tl.load(
                update_bias_ptr + columns, mask=mask, other=0.0
            ).to(tl.float32)
        scale = tl.load(scale_ptr + columns, mask=mask, other=0.0).to(
            tl.float32
        )

        tl.store(hidden_ptr + offsets, hidden + update * scale, mask=mask)

    @triton.jit
    def _scaled_residual_layer_norm_kernel(
        hidden_ptr,
        update_ptr,
        update_bias_ptr,
        scale_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        eps,
        HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Fuse a scaled residual update with layer normalization."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        offsets = row * WIDTH + columns

        hidden = tl.load(
            hidden_ptr + offsets, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        update = tl.load(
            update_ptr + offsets, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(
                update_bias_ptr + columns, mask=columns < WIDTH, other=0.0
            ).to(tl.float32)
        scale = tl.load(
            scale_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)

        residual = hidden + update * scale
        mean = tl.sum(residual, axis=0) / WIDTH
        # Masked lanes center to zero so they contribute nothing to the
        # variance.
        centered = tl.where(columns < WIDTH, residual - mean, 0.0)
        inverse_std = tl.rsqrt(
            tl.sum(centered * centered, axis=0) / WIDTH + eps
        )
        weight = tl.load(
            weight_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        bias = tl.load(bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(
            tl.float32
        )

        tl.store(
            output_ptr + offsets,
            centered * inverse_std * weight + bias,
            mask=columns < WIDTH,
        )

    @triton.jit
    def _scaled_residual_layer_norm_absmax_kernel(
        hidden_ptr,
        update_ptr,
        update_bias_ptr,
        scale_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        partials_ptr,
        eps,
        HAS_UPDATE_BIAS: tl.constexpr,  # noqa: N803
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Compute fused residual layer norm and output magnitude.

        Compute fused residual layer normalization and rowwise output
        magnitude.
        """
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        offsets = row * WIDTH + columns

        hidden = tl.load(
            hidden_ptr + offsets, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        update = tl.load(
            update_ptr + offsets, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(
                update_bias_ptr + columns, mask=columns < WIDTH, other=0.0
            ).to(tl.float32)
        scale = tl.load(
            scale_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)

        residual = hidden + update * scale
        mean = tl.sum(residual, axis=0) / WIDTH
        # Masked lanes center to zero so they contribute nothing to the
        # variance.
        centered = tl.where(columns < WIDTH, residual - mean, 0.0)
        inverse_std = tl.rsqrt(
            tl.sum(centered * centered, axis=0) / WIDTH + eps
        )
        weight = tl.load(
            weight_ptr + columns, mask=columns < WIDTH, other=0.0
        ).to(tl.float32)
        bias = tl.load(bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(
            tl.float32
        )
        output = (centered * inverse_std * weight + bias).to(
            output_ptr.dtype.element_ty
        )

        tl.store(output_ptr + offsets, output, mask=columns < WIDTH)
        tl.store(
            partials_ptr + row,
            tl.max(
                tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0),
                axis=0,
            ),
        )


def can_run(value: torch.Tensor, *operands: torch.Tensor | None) -> bool:
    """Return whether contiguous CUDA rows and operands fit the kernels.

    Operands are channel vectors, updates or outputs on ``value``'s device.
    """
    return (
        launchable(value.device)
        and value.is_cuda
        and value.numel() > 0
        and 0 < int(value.shape[-1]) <= MAX_WIDTH
        and value.is_contiguous()
        and all(
            operand is None
            or (operand.device == value.device and operand.is_contiguous())
            for operand in operands
        )
    )


def _rows(value: torch.Tensor) -> tuple[int, int]:
    width = int(value.shape[-1])
    return value.numel() // width, width


def weighted_rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    partials: torch.Tensor | None = None,
) -> None:
    """Store weighted RMS rows; ``partials`` receives per-row absmax."""
    rows, width = _rows(hidden)
    kernel = (
        _weighted_rms_norm_kernel
        if partials is None
        else _weighted_rms_norm_absmax_kernel
    )
    kernel[(rows,)](
        hidden,
        weight,
        out,
        *(() if partials is None else (partials,)),
        eps,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        num_warps=8,
    )


def scaled_residual_rms_norm_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    update_bias: torch.Tensor | None,
    eps: float,
    out: torch.Tensor,
    partials: torch.Tensor | None = None,
) -> None:
    """Add the scaled update into ``hidden`` and store its RMS rows."""
    rows, width = _rows(hidden)
    kernel = (
        _scaled_residual_rms_norm_kernel
        if partials is None
        else _scaled_residual_rms_norm_absmax_kernel
    )
    kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        out,
        *(() if partials is None else (partials,)),
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        num_warps=8,
    )


def scaled_residual_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    update_bias: torch.Tensor | None,
) -> None:
    """Add the channel-scaled, optionally biased update into ``hidden``."""
    elements = hidden.numel()
    _scaled_residual_kernel[(triton.cdiv(elements, 1024),)](
        hidden,
        update,
        update_bias,
        scale,
        elements,
        HAS_UPDATE_BIAS=update_bias is not None,
        WIDTH=int(hidden.shape[-1]),
        BLOCK=1024,
        num_warps=4,
    )


def scaled_residual_layer_norm(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    update_bias: torch.Tensor | None,
    eps: float,
    out: torch.Tensor,
    partials: torch.Tensor | None = None,
) -> None:
    """Store layer-normalized residual rows without updating ``hidden``."""
    rows, width = _rows(hidden)
    kernel = (
        _scaled_residual_layer_norm_kernel
        if partials is None
        else _scaled_residual_layer_norm_absmax_kernel
    )
    kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        bias,
        out,
        *(() if partials is None else (partials,)),
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        num_warps=8,
    )
