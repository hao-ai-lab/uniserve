# SPDX-License-Identifier: Apache-2.0
"""Fixed-width fused operations for the MiniMax H3 video decoder."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

__all__ = [
    "qk_rmsnorm_partial_rope_",
    "scaled_residual_",
    "scaled_residual_layernorm",
    "scaled_residual_layernorm_absmax",
    "scaled_residual_rmsnorm_absmax_",
    "scaled_residual_rmsnorm_",
    "value_first_swiglu",
    "value_first_swiglu_absmax",
    "video_patch_output",
    "video_rmsnorm",
    "video_rmsnorm_absmax",
]

_WIDTH = 2048
_HEAD_DIM = 64
_ROTARY_DIM = 48
_WIDTH_TL = tl.constexpr(2048)
_HEAD_DIM_TL = tl.constexpr(64)
_ROTARY_DIM_TL = tl.constexpr(48)


@triton.jit
def _video_rmsnorm_kernel(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    hidden = tl.load(hidden_ptr + row * _WIDTH_TL + columns).to(tl.float32)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / _WIDTH_TL + eps)
    tl.store(output_ptr + row * _WIDTH_TL + columns, hidden * inverse_rms * weight)


@triton.jit
def _video_rmsnorm_absmax_kernel(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    partials_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    hidden = tl.load(hidden_ptr + row * _WIDTH_TL + columns).to(tl.float32)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / _WIDTH_TL + eps)
    output = (hidden * inverse_rms * weight).to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + row * _WIDTH_TL + columns, output)
    tl.store(partials_ptr + row, tl.max(tl.abs(output.to(tl.float32)), axis=0))


@triton.jit
def _scaled_residual_rmsnorm_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    weight_ptr,
    output_ptr,
    eps,
    HAS_UPDATE_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * _WIDTH_TL + columns
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    update = tl.load(update_ptr + offsets).to(tl.float32)
    if HAS_UPDATE_BIAS:
        update += tl.load(update_bias_ptr + columns).to(tl.float32)
        update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
    scale = tl.load(scale_ptr + columns).to(tl.float32)
    residual = hidden + update * scale
    inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / _WIDTH_TL + eps)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    tl.store(hidden_ptr + offsets, residual)
    tl.store(output_ptr + offsets, residual * inverse_rms * weight)


@triton.jit
def _scaled_residual_rmsnorm_absmax_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    weight_ptr,
    output_ptr,
    partials_ptr,
    eps,
    HAS_UPDATE_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * _WIDTH_TL + columns
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    update = tl.load(update_ptr + offsets).to(tl.float32)
    if HAS_UPDATE_BIAS:
        update += tl.load(update_bias_ptr + columns).to(tl.float32)
        update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
    scale = tl.load(scale_ptr + columns).to(tl.float32)
    residual = hidden + update * scale
    inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / _WIDTH_TL + eps)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    output = (residual * inverse_rms * weight).to(output_ptr.dtype.element_ty)
    tl.store(hidden_ptr + offsets, residual)
    tl.store(output_ptr + offsets, output)
    tl.store(partials_ptr + row, tl.max(tl.abs(output.to(tl.float32)), axis=0))


@triton.jit
def _scaled_residual_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    elements,
    HAS_UPDATE_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    columns = offsets % _WIDTH_TL
    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_UPDATE_BIAS:
        update += tl.load(update_bias_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
    scale = tl.load(scale_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(hidden_ptr + offsets, hidden + update * scale, mask=mask)


@triton.jit
def _scaled_residual_layernorm_kernel(
    hidden_ptr,
    update_ptr,
    update_bias_ptr,
    scale_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    HAS_UPDATE_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * _WIDTH_TL + columns
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    update = tl.load(update_ptr + offsets).to(tl.float32)
    if HAS_UPDATE_BIAS:
        update += tl.load(update_bias_ptr + columns).to(tl.float32)
        update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
    scale = tl.load(scale_ptr + columns).to(tl.float32)
    residual = hidden + update * scale
    mean = tl.sum(residual, axis=0) / _WIDTH_TL
    centered = residual - mean
    inverse_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / _WIDTH_TL + eps)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    bias = tl.load(bias_ptr + columns).to(tl.float32)
    tl.store(output_ptr + offsets, centered * inverse_std * weight + bias)


@triton.jit
def _scaled_residual_layernorm_absmax_kernel(
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
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * _WIDTH_TL + columns
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    update = tl.load(update_ptr + offsets).to(tl.float32)
    if HAS_UPDATE_BIAS:
        update += tl.load(update_bias_ptr + columns).to(tl.float32)
        update = update.to(update_ptr.dtype.element_ty).to(tl.float32)
    scale = tl.load(scale_ptr + columns).to(tl.float32)
    residual = hidden + update * scale
    mean = tl.sum(residual, axis=0) / _WIDTH_TL
    centered = residual - mean
    inverse_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / _WIDTH_TL + eps)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    bias = tl.load(bias_ptr + columns).to(tl.float32)
    output = (centered * inverse_std * weight + bias).to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + offsets, output)
    tl.store(partials_ptr + row, tl.max(tl.abs(output.to(tl.float32)), axis=0))


@triton.jit
def _qk_rmsnorm_partial_rope_kernel(
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
    ROW_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    head = tl.program_id(1)
    columns = tl.arange(0, _HEAD_DIM_TL)
    valid = row[:, None] < rows
    offsets = row[:, None] * row_stride + head * head_stride + columns[None, :]
    query = tl.load(query_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    key = tl.load(key_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias_offsets = head * _HEAD_DIM_TL + columns[None, :]
        query += tl.load(query_bias_ptr + bias_offsets).to(tl.float32)
        key += tl.load(key_bias_ptr + bias_offsets).to(tl.float32)
        query = query.to(query_ptr.dtype.element_ty).to(tl.float32)
        key = key.to(key_ptr.dtype.element_ty).to(tl.float32)
    if HAS_VALUE_BIAS:
        value = tl.load(value_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        value += tl.load(value_bias_ptr + head * _HEAD_DIM_TL + columns[None, :]).to(tl.float32)
        tl.store(value_ptr + offsets, value.to(value_ptr.dtype.element_ty), mask=valid)
    query_rstd = tl.rsqrt(tl.sum(query * query, axis=1) / _HEAD_DIM_TL + 1e-5)
    key_rstd = tl.rsqrt(tl.sum(key * key, axis=1) / _HEAD_DIM_TL + 1e-5)
    query = (query * query_rstd[:, None]).to(query_ptr.dtype.element_ty).to(tl.float32)
    key = (key * key_rstd[:, None]).to(key_ptr.dtype.element_ty).to(tl.float32)

    half_rotary: tl.constexpr = _ROTARY_DIM_TL // 2
    partner_columns = tl.where(
        columns < half_rotary,
        columns + half_rotary,
        columns - half_rotary,
    )
    partner_columns = tl.where(columns < _ROTARY_DIM_TL, partner_columns, columns)
    partner_offsets = row[:, None] * row_stride + head * head_stride + partner_columns[None, :]
    query_partner = tl.load(query_ptr + partner_offsets, mask=valid, other=0.0).to(tl.float32)
    key_partner = tl.load(key_ptr + partner_offsets, mask=valid, other=0.0).to(tl.float32)
    if HAS_BIAS:
        partner_bias_offsets = head * _HEAD_DIM_TL + partner_columns[None, :]
        query_partner += tl.load(query_bias_ptr + partner_bias_offsets).to(tl.float32)
        key_partner += tl.load(key_bias_ptr + partner_bias_offsets).to(tl.float32)
        query_partner = query_partner.to(query_ptr.dtype.element_ty).to(tl.float32)
        key_partner = key_partner.to(key_ptr.dtype.element_ty).to(tl.float32)
    query_partner = (
        (query_partner * query_rstd[:, None]).to(query_ptr.dtype.element_ty).to(tl.float32)
    )
    key_partner = (key_partner * key_rstd[:, None]).to(key_ptr.dtype.element_ty).to(tl.float32)

    rotary_mask = valid & (columns[None, :] < _ROTARY_DIM_TL)
    rotary_offsets = row[:, None] * rotary_row_stride + columns[None, :]
    cosine = tl.load(cosine_ptr + rotary_offsets, mask=rotary_mask, other=1.0).to(tl.float32)
    sine = tl.load(sine_ptr + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    sign = tl.where(columns[None, :] < half_rotary, -1.0, 1.0)
    query_rotated = query * cosine + sign * query_partner * sine
    key_rotated = key * cosine + sign * key_partner * sine
    tl.store(query_ptr + offsets, tl.where(rotary_mask, query_rotated, query), mask=valid)
    tl.store(key_ptr + offsets, tl.where(rotary_mask, key_rotated, key), mask=valid)


@triton.jit
def _value_first_swiglu_kernel(
    value_gate_ptr,
    bias_ptr,
    output_ptr,
    elements,
    width: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    row = offsets // width
    column = offsets - row * width
    value = tl.load(value_gate_ptr + row * (2 * width) + column, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(
        value_gate_ptr + row * (2 * width) + width + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(bias_ptr + column, mask=mask, other=0.0).to(tl.float32)
        gate += tl.load(bias_ptr + width + column, mask=mask, other=0.0).to(tl.float32)
        value = value.to(value_gate_ptr.dtype.element_ty).to(tl.float32)
        gate = gate.to(value_gate_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + offsets, value * gate / (1.0 + tl.exp(-gate)), mask=mask)


@triton.jit
def _value_first_swiglu_absmax_kernel(
    value_gate_ptr,
    bias_ptr,
    output_ptr,
    partials_ptr,
    elements: tl.constexpr,
    width: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    row = offsets // width
    column = offsets - row * width
    value = tl.load(
        value_gate_ptr + row * (2 * width) + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        value_gate_ptr + row * (2 * width) + width + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(bias_ptr + column, mask=mask, other=0.0).to(tl.float32)
        gate += tl.load(bias_ptr + width + column, mask=mask, other=0.0).to(tl.float32)
        value = value.to(value_gate_ptr.dtype.element_ty).to(tl.float32)
        gate = gate.to(value_gate_ptr.dtype.element_ty).to(tl.float32)
    output = (value * gate / (1.0 + tl.exp(-gate))).to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + offsets, output, mask=mask)
    partial = tl.max(tl.where(mask, tl.abs(output.to(tl.float32)), 0.0), axis=0)
    tl.store(partials_ptr + tl.program_id(0), partial)


@triton.jit
def _finish_absmax_kernel(
    partials_ptr,
    output_ptr,
    count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partials_ptr + offsets, mask=offsets < count, other=-float("inf"))
    tl.store(output_ptr, tl.max(values, axis=0))


@triton.jit
def _video_patch_output_kernel(
    source_ptr,
    bias_ptr,
    output_ptr,
    elements: tl.constexpr,
    sequence: tl.constexpr,
    frames: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    output_width: tl.constexpr = width * 16
    output_height: tl.constexpr = height * 16
    output_frames: tl.constexpr = frames * 4
    output_spatial: tl.constexpr = output_height * output_width
    output_volume: tl.constexpr = output_frames * output_spatial
    output_channels: tl.constexpr = 3 * output_volume
    batch = offsets // output_channels
    remainder = offsets - batch * output_channels
    channel = remainder // output_volume
    remainder -= channel * output_volume
    output_frame = remainder // output_spatial
    remainder -= output_frame * output_spatial
    output_row = remainder // output_width
    output_column = remainder - output_row * output_width
    frame = output_frame // 4
    temporal_patch = output_frame - frame * 4
    patch_row = output_row // 16
    inner_row = output_row - patch_row * 16
    patch_column = output_column // 16
    inner_column = output_column - patch_column * 16
    patch = (frame * height + patch_row) * width + patch_column
    patch_channel = (((channel * 4 + temporal_patch) * 16 + inner_row) * 16) + inner_column
    source_offsets = (batch * sequence + patch) * 3072 + patch_channel
    value = tl.load(source_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(bias_ptr + patch_channel, mask=mask, other=0.0).to(tl.float32)
        value = value.to(output_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + offsets, value, mask=mask)


def _finish_absmax(partials: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    maximum = torch.empty((), dtype=dtype, device=partials.device)
    finish_block = triton.next_power_of_2(int(partials.numel()))
    _finish_absmax_kernel[(1,)](
        partials,
        maximum,
        count=int(partials.numel()),
        BLOCK=finish_block,
        num_warps=8,
    )
    return maximum


def video_rmsnorm(hidden: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    if not hidden.is_cuda:
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().pow(2).mean(-1, keepdim=True) + eps
        )
        return (normalized * weight.float()).to(hidden.dtype)
    if hidden.shape[-1] != _WIDTH:
        raise ValueError("H3 video RMSNorm requires width 2048")
    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // _WIDTH
    _video_rmsnorm_kernel[(rows,)](
        hidden,
        weight,
        output,
        eps,
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return output


def video_rmsnorm_absmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hidden.is_cuda:
        output = video_rmsnorm(hidden, weight, eps=eps)
        return output, output.abs().amax()
    if hidden.shape[-1] != _WIDTH:
        raise ValueError("H3 video RMSNorm requires width 2048")
    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // _WIDTH
    partials = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _video_rmsnorm_absmax_kernel[(rows,)](
        hidden,
        weight,
        output,
        partials,
        eps,
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return output, _finish_absmax(partials, output.dtype)


def scaled_residual_rmsnorm_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hidden.is_cuda:
        if update_bias is not None:
            update = update + update_bias
        hidden.add_(update.float() * scale.float())
        return hidden, video_rmsnorm(hidden, weight, eps=eps)
    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else update.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // _WIDTH
    _scaled_residual_rmsnorm_kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        output,
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return hidden, output


def scaled_residual_rmsnorm_absmax_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hidden.is_cuda:
        hidden, output = scaled_residual_rmsnorm_(
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
    rows = hidden.numel() // _WIDTH
    partials = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _scaled_residual_rmsnorm_absmax_kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        output,
        partials,
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return hidden, output, _finish_absmax(partials, output.dtype)


def scaled_residual_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    update_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not hidden.is_cuda:
        if update_bias is not None:
            update = update + update_bias
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
        BLOCK=1024,
        num_warps=4,
    )
    return hidden


def scaled_residual_layernorm(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> torch.Tensor:
    if not hidden.is_cuda:
        if update_bias is not None:
            update = update + update_bias
        residual = hidden + update.float() * scale.float()
        return F.layer_norm(residual, (_WIDTH,), weight, bias, eps)
    output_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else update.dtype
    )
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // _WIDTH
    _scaled_residual_layernorm_kernel[(rows,)](
        hidden,
        update,
        update_bias,
        scale,
        weight,
        bias,
        output,
        eps,
        HAS_UPDATE_BIAS=update_bias is not None,
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return output


def scaled_residual_layernorm_absmax(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hidden.is_cuda:
        output = scaled_residual_layernorm(
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
    rows = hidden.numel() // _WIDTH
    partials = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _scaled_residual_layernorm_absmax_kernel[(rows,)](
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
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return output, _finish_absmax(partials, output.dtype)


def qk_rmsnorm_partial_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    query_bias: torch.Tensor | None = None,
    key_bias: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    value_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (query_bias is None) != (key_bias is None):
        raise ValueError("H3 video Q/K fusion requires both biases or neither")
    if (value is None) != (value_bias is None):
        raise ValueError("H3 video Q/K fusion requires both value and value bias or neither")
    if not query.is_cuda:
        if query_bias is not None and key_bias is not None:
            query = query + query_bias.view(query.shape[-2], query.shape[-1])
            key = key + key_bias.view(key.shape[-2], key.shape[-1])
        dtype = query.dtype
        query = query.float() * torch.rsqrt(query.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        key = key.float() * torch.rsqrt(key.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        query = query.to(dtype)
        key = key.to(dtype)
        first, second = query[..., :_ROTARY_DIM].chunk(2, dim=-1)
        query_rotary = torch.cat((-second, first), dim=-1)
        first, second = key[..., :_ROTARY_DIM].chunk(2, dim=-1)
        key_rotary = torch.cat((-second, first), dim=-1)
        query[..., :_ROTARY_DIM] = query[..., :_ROTARY_DIM] * cosine + query_rotary * sine
        key[..., :_ROTARY_DIM] = key[..., :_ROTARY_DIM] * cosine + key_rotary * sine
        if value is not None and value_bias is not None:
            value.add_(value_bias.view(value.shape[-2], value.shape[-1]))
        return query, key
    if query.shape != key.shape or query.shape[-1] != _HEAD_DIM:
        raise ValueError("H3 video Q/K fusion requires matching head dimension 64")
    if value is not None and value.shape != query.shape:
        raise ValueError("H3 video Q/K fusion requires matching value geometry")
    rows = query.numel() // (int(query.shape[-2]) * _HEAD_DIM)
    heads = int(query.shape[-2])
    row_block = 8
    _qk_rmsnorm_partial_rope_kernel[(triton.cdiv(rows, row_block), heads)](
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
        int(cosine.stride(-3)),
        HAS_BIAS=query_bias is not None,
        HAS_VALUE_BIAS=value_bias is not None,
        ROW_BLOCK=row_block,
        num_warps=4,
    )
    return query, key


def value_first_swiglu(
    value_gate: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    value, gate = value_gate.chunk(2, dim=-1)
    if not value_gate.is_cuda:
        if bias is not None:
            value_bias, gate_bias = bias.chunk(2)
            value = value + value_bias
            gate = gate + gate_bias
        return value * F.silu(gate)
    width = int(value.shape[-1])
    output = torch.empty_like(value)
    elements = output.numel()
    _value_first_swiglu_kernel[(triton.cdiv(elements, 1024),)](
        value_gate,
        bias,
        output,
        elements,
        width,
        HAS_BIAS=bias is not None,
        BLOCK=1024,
        num_warps=4,
    )
    return output


def value_first_swiglu_absmax(
    value_gate: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not value_gate.is_cuda:
        output = value_first_swiglu(value_gate, bias)
        return output, output.abs().amax()
    width = int(value_gate.shape[-1]) // 2
    elements = value_gate.numel() // 2
    block = 32768
    partial_count = triton.cdiv(elements, block)
    output = torch.empty(
        (*value_gate.shape[:-1], width), dtype=value_gate.dtype, device=value_gate.device
    )
    partials = torch.empty((partial_count,), dtype=torch.float32, device=value_gate.device)
    _value_first_swiglu_absmax_kernel[(partial_count,)](
        value_gate,
        bias,
        output,
        partials,
        elements=elements,
        width=width,
        HAS_BIAS=bias is not None,
        BLOCK=block,
        num_warps=8,
    )
    return output, _finish_absmax(partials, value_gate.dtype)


def video_patch_output(
    source: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    batch, sequence, channels = source.shape
    if channels != 3072:
        raise ValueError("H3 video patch output requires width 3072")
    patch_count = frames * height * width
    if sequence < patch_count:
        raise ValueError("H3 video patch output has fewer tokens than patches")
    if not source.is_cuda:
        if bias is not None:
            source = source + bias
        source = source[:, :patch_count].view(
            batch,
            frames,
            height,
            width,
            3,
            4,
            16,
            16,
        )
        return (
            source.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .contiguous()
            .reshape(batch, 3, frames * 4, height * 16, width * 16)
        )
    output = torch.empty(
        (batch, 3, frames * 4, height * 16, width * 16),
        dtype=source.dtype,
        device=source.device,
    )
    elements = output.numel()
    _video_patch_output_kernel[(triton.cdiv(elements, 1024),)](
        source,
        bias,
        output,
        elements=elements,
        sequence=sequence,
        frames=frames,
        height=height,
        width=width,
        HAS_BIAS=bias is not None,
        BLOCK=1024,
        num_warps=4,
    )
    return output
