"""Channel-scaled residuals and normalization with FP32 affine accumulation.

Projection bias rounds through the update dtype before scaling. Residual sums,
normalization statistics and learned affine transforms stay in FP32 until the
output dtype boundary. RMS variants update the residual storage in place while
normalizing the unrounded sum. Magnitude variants measure the rounded output.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ..backends.triton import triton_available
from .silu import finish_absmax

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _output_dtype(value: torch.Tensor) -> torch.dtype:
    return (
        torch.get_autocast_dtype("cuda")
        if value.is_cuda and torch.is_autocast_enabled("cuda")
        else value.dtype
    )


def _fused_rows(value: torch.Tensor, *vectors: torch.Tensor | None) -> bool:
    """Validate channel vectors and select contiguous bounded CUDA rows."""

    if value.ndim < 1 or value.numel() == 0 or not value.is_floating_point():
        raise ValueError("normalization requires nonempty floating activation rows")
    width = int(value.shape[-1])
    for vector in vectors:
        if vector is not None and (vector.numel() != width or vector.device != value.device):
            raise ValueError("normalization vectors must match the activation width and device")
    return (
        value.is_cuda
        and triton_available(value.device)
        and value.is_contiguous()
        and width <= 32768
        and all(vector is None or vector.is_contiguous() for vector in vectors)
    )


if triton is not None:

    @triton.jit
    def _weighted_rms_norm_kernel(
        hidden_ptr,
        weight_ptr,
        output_ptr,
        eps,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Normalize one activation row and apply its learned scale."""

        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        hidden = tl.load(hidden_ptr + row * WIDTH + columns, mask=columns < WIDTH, other=0.0).to(
            tl.float32
        )
        weight = tl.load(weight_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / WIDTH + eps)
        tl.store(
            output_ptr + row * WIDTH + columns, hidden * inverse_rms * weight, mask=columns < WIDTH
        )

    @triton.jit
    def _weighted_rms_norm_absmax_kernel(
        hidden_ptr,
        weight_ptr,
        output_ptr,
        partials_ptr,
        eps,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Normalize one row while publishing its output magnitude for reduction."""

        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        hidden = tl.load(hidden_ptr + row * WIDTH + columns, mask=columns < WIDTH, other=0.0).to(
            tl.float32
        )
        weight = tl.load(weight_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / WIDTH + eps)
        output = (hidden * inverse_rms * weight).to(output_ptr.dtype.element_ty)
        tl.store(output_ptr + row * WIDTH + columns, output, mask=columns < WIDTH)
        tl.store(
            partials_ptr + row,
            tl.max(tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0), axis=0),
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
        HAS_UPDATE_BIAS: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Update a residual row in place and emit its RMS-normalized value."""

        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        offsets = row * WIDTH + columns
        hidden = tl.load(hidden_ptr + offsets, mask=columns < WIDTH, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + offsets, mask=columns < WIDTH, other=0.0).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(update_bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(
                tl.float32
            )
            update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
        scale = tl.load(scale_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        residual = hidden + update * scale
        inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / WIDTH + eps)
        weight = tl.load(weight_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        tl.store(hidden_ptr + offsets, residual, mask=columns < WIDTH)
        tl.store(output_ptr + offsets, residual * inverse_rms * weight, mask=columns < WIDTH)

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
        HAS_UPDATE_BIAS: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fuse residual update, RMS normalization, and rowwise magnitude capture."""

        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        offsets = row * WIDTH + columns
        hidden = tl.load(hidden_ptr + offsets, mask=columns < WIDTH, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + offsets, mask=columns < WIDTH, other=0.0).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(update_bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(
                tl.float32
            )
            update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
        scale = tl.load(scale_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        residual = hidden + update * scale
        inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / WIDTH + eps)
        weight = tl.load(weight_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        output = (residual * inverse_rms * weight).to(output_ptr.dtype.element_ty)
        tl.store(hidden_ptr + offsets, residual, mask=columns < WIDTH)
        tl.store(output_ptr + offsets, output, mask=columns < WIDTH)
        tl.store(
            partials_ptr + row,
            tl.max(tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0), axis=0),
        )

    @triton.jit
    def _scaled_residual_kernel(
        hidden_ptr,
        update_ptr,
        update_bias_ptr,
        scale_ptr,
        elements,
        HAS_UPDATE_BIAS: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Apply a channel-scaled update to the residual stream."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        columns = offsets % WIDTH
        hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(update_bias_ptr + columns, mask=mask, other=0.0).to(tl.float32)
            update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
        scale = tl.load(scale_ptr + columns, mask=mask, other=0.0).to(tl.float32)
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
        HAS_UPDATE_BIAS: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fuse a scaled residual update with layer normalization."""

        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        offsets = row * WIDTH + columns
        hidden = tl.load(hidden_ptr + offsets, mask=columns < WIDTH, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + offsets, mask=columns < WIDTH, other=0.0).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(update_bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(
                tl.float32
            )
            update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
        scale = tl.load(scale_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        residual = hidden + update * scale
        mean = tl.sum(residual, axis=0) / WIDTH
        centered = tl.where(columns < WIDTH, residual - mean, 0.0)
        inverse_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / WIDTH + eps)
        weight = tl.load(weight_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        tl.store(output_ptr + offsets, centered * inverse_std * weight + bias, mask=columns < WIDTH)

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
        HAS_UPDATE_BIAS: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Compute fused residual layer normalization and rowwise output magnitude."""

        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        offsets = row * WIDTH + columns
        hidden = tl.load(hidden_ptr + offsets, mask=columns < WIDTH, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + offsets, mask=columns < WIDTH, other=0.0).to(tl.float32)
        if HAS_UPDATE_BIAS:
            update += tl.load(update_bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(
                tl.float32
            )
            update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
        scale = tl.load(scale_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        residual = hidden + update * scale
        mean = tl.sum(residual, axis=0) / WIDTH
        centered = tl.where(columns < WIDTH, residual - mean, 0.0)
        inverse_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / WIDTH + eps)
        weight = tl.load(weight_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + columns, mask=columns < WIDTH, other=0.0).to(tl.float32)
        output = (centered * inverse_std * weight + bias).to(output_ptr.dtype.element_ty)
        tl.store(output_ptr + offsets, output, mask=columns < WIDTH)
        tl.store(
            partials_ptr + row,
            tl.max(tl.where(columns < WIDTH, tl.abs(output.to(tl.float32)), 0.0), axis=0),
        )


def weighted_rms_norm(hidden: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Apply weighted RMS normalization with CPU and Triton execution paths."""

    if not _fused_rows(hidden, weight):
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().pow(2).mean(-1, keepdim=True) + eps
        )
        return (normalized * weight.float()).to(_output_dtype(hidden))

    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // int(hidden.shape[-1])
    _weighted_rms_norm_kernel[(rows,)](
        hidden,
        weight,
        output,
        eps,
        WIDTH=int(hidden.shape[-1]),
        BLOCK=triton.next_power_of_2(int(hidden.shape[-1])),
        num_warps=8,
    )
    return output


def weighted_rms_norm_absmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized activation rows and their global absolute maximum."""

    if not _fused_rows(hidden, weight):
        output = weighted_rms_norm(hidden, weight, eps=eps)
        return output, output.abs().amax()

    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // int(hidden.shape[-1])
    partials = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _weighted_rms_norm_absmax_kernel[(rows,)](
        hidden,
        weight,
        output,
        partials,
        eps,
        WIDTH=int(hidden.shape[-1]),
        BLOCK=triton.next_power_of_2(int(hidden.shape[-1])),
        num_warps=8,
    )
    return output, finish_absmax(partials, output.dtype)


def scaled_residual_rms_norm_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the residual stream in place and return its RMS-normalized view."""

    if hidden.shape != update.shape or hidden.device != update.device:
        raise ValueError("scaled residual operands must have the same shape and device")
    if not _fused_rows(hidden, scale, weight, update_bias) or not update.is_contiguous():
        if update_bias is not None:
            update = (update.float() + update_bias.float()).to(update.dtype)
        residual = hidden.float() + update.float() * scale.float()
        hidden.copy_(residual)
        normalized = residual * torch.rsqrt(residual.square().mean(-1, keepdim=True) + eps)
        return hidden, (normalized * weight.float()).to(_output_dtype(update))

    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else update.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // int(hidden.shape[-1])
    _scaled_residual_rms_norm_kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        output,
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        WIDTH=int(hidden.shape[-1]),
        BLOCK=triton.next_power_of_2(int(hidden.shape[-1])),
        num_warps=8,
    )
    return hidden, output


def scaled_residual_rms_norm_absmax_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update a residual in place and return normalized values with their magnitude."""

    if hidden.shape != update.shape or hidden.device != update.device:
        raise ValueError("scaled residual operands must have the same shape and device")
    if not _fused_rows(hidden, scale, weight, update_bias) or not update.is_contiguous():
        hidden, output = scaled_residual_rms_norm_(
            hidden,
            update,
            scale,
            weight,
            update_bias=update_bias,
            eps=eps,
        )
        return hidden, output, output.abs().amax()

    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else update.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // int(hidden.shape[-1])
    partials = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _scaled_residual_rms_norm_absmax_kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        output,
        partials,
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        WIDTH=int(hidden.shape[-1]),
        BLOCK=triton.next_power_of_2(int(hidden.shape[-1])),
        num_warps=8,
    )
    return hidden, output, finish_absmax(partials, output.dtype)


def scaled_residual_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    update_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a channel-scaled, optionally biased update to ``hidden`` in place."""

    if hidden.shape != update.shape or hidden.device != update.device:
        raise ValueError("scaled residual operands must have the same shape and device")
    if not _fused_rows(hidden, scale, update_bias) or not update.is_contiguous():
        if update_bias is not None:
            update = (update.float() + update_bias.float()).to(update.dtype)
        hidden.add_(update.float() * scale.float())
        return hidden

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
    return hidden


def scaled_residual_layer_norm(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> torch.Tensor:
    """Apply a scaled residual update followed by layer normalization."""

    if hidden.shape != update.shape or hidden.device != update.device:
        raise ValueError("scaled residual operands must have the same shape and device")
    if not _fused_rows(hidden, scale, weight, bias, update_bias) or not update.is_contiguous():
        if update_bias is not None:
            update = (update.float() + update_bias.float()).to(update.dtype)
        residual = hidden.float() + update.float() * scale.float()
        normalized = F.layer_norm(
            residual, (int(hidden.shape[-1]),), weight.float(), bias.float(), eps
        )
        return normalized.to(_output_dtype(update))

    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else update.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // int(hidden.shape[-1])
    _scaled_residual_layer_norm_kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        bias,
        output,
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        WIDTH=int(hidden.shape[-1]),
        BLOCK=triton.next_power_of_2(int(hidden.shape[-1])),
        num_warps=8,
    )
    return output


def scaled_residual_layer_norm_absmax(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fused residual layer normalization and its global absolute maximum."""

    if hidden.shape != update.shape or hidden.device != update.device:
        raise ValueError("scaled residual operands must have the same shape and device")
    if not _fused_rows(hidden, scale, weight, bias, update_bias) or not update.is_contiguous():
        output = scaled_residual_layer_norm(
            hidden,
            update,
            scale,
            weight,
            bias,
            update_bias=update_bias,
            eps=eps,
        )
        return output, output.abs().amax()

    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else update.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // int(hidden.shape[-1])
    partials = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _scaled_residual_layer_norm_absmax_kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        bias,
        output,
        partials,
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        WIDTH=int(hidden.shape[-1]),
        BLOCK=triton.next_power_of_2(int(hidden.shape[-1])),
        num_warps=8,
    )
    return output, finish_absmax(partials, output.dtype)
