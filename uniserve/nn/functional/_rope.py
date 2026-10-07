"""Rotary position rotations and fused Q/K normalization with rotation.

Factors are compact: one ``[..., rotated / 2]`` table per rotary axis over the
tensor's token axes, broadcast over heads. Arithmetic accumulates in FP32 and
rounds once to the input dtype; ``qk_norm_rope`` also evaluates the
``Rounding.STEPWISE`` recipe of eager PyTorch. CUDA calls run UniServe's
kernels and raise ``ValueError`` when no kernel accepts their operands;
other devices evaluate the same formulas with tensor operations.
"""

from __future__ import annotations

from typing import Literal

import torch
from uniserve_kernels.triton import require_kernel

from ._tensors import Rounding, check_output, result


def _factors_match(x, cos, sin) -> bool:
    """Check compact factors against the token axes of ``[..., heads, dim]``."""
    return (
        cos.shape == sin.shape
        and cos.ndim == x.ndim - 1
        and cos.shape[:-1] == x.shape[:-2]
        and cos.device == x.device
        and sin.device == x.device
    )


def _rotate(x, cos, sin, rotation):
    """Rotate the leading ``2 * cos.shape[-1]`` coordinates of ``x``.

    The factors are cast to ``x.dtype`` and every product and sum evaluates
    in that dtype: an FP32 ``x`` accumulates the rotation in FP32, while an
    activation-dtype ``x`` rounds after each operation as eager PyTorch does.
    Factors broadcast over the omitted head axis: ``[..., 1, width / 2]``.
    Coordinates beyond the rotated width are returned unchanged.
    """
    width = cos.shape[-1] * 2
    if width == 0:
        return x

    cosine = cos.to(x.dtype).unsqueeze(-2)
    sine = sin.to(x.dtype).unsqueeze(-2)
    prefix = x[..., :width]
    # Split pairs the two half-widths; interleaved pairs adjacent coordinates.
    first, second = (
        prefix.chunk(2, dim=-1)
        if rotation == "split"
        else (prefix[..., ::2], prefix[..., 1::2])
    )
    left, right = first * cosine - second * sine, second * cosine + first * sine
    rotated = (
        torch.cat((left, right), dim=-1)
        if rotation == "split"
        else torch.stack((left, right), dim=-1).flatten(-2)
    )
    if width == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., width:]), dim=-1)


def apply_rotary(
    x, cos, sin, *, rotation: Literal["interleaved", "split"], out=None
):
    """Rotate the leading coordinates of token/head vectors with compact
    factors.

    ``x`` has shape ``[..., heads, head_dim]``. ``cos`` and ``sin`` have shape
    ``[..., rotary_dim / 2]`` over the same token axes; the omitted head axis
    broadcasts over every head. ``rotation="split"`` pairs coordinate ``i``
    with ``i + rotary_dim / 2``; ``"interleaved"`` pairs adjacent coordinates.
    Coordinates from ``rotary_dim`` onward are preserved. Products and sums
    use FP32 and the result is rounded once to ``x.dtype``.

    ``out`` must match ``x`` in shape, dtype and device and may alias it.
    On CUDA, ``x`` and ``out`` may be strided views whose token axes merge
    into one strided axis with unit-strided features.
    """  # noqa: D205
    if (
        rotation not in {"interleaved", "split"}
        or x.ndim < 2
        or not _factors_match(x, cos, sin)
        or cos.shape[-1] * 2 > x.shape[-1]
    ):
        raise ValueError(
            "rotary factors must match tokens and a prefix of the head width"
        )
    if out is not None:
        check_output(x, out)

    if x.is_cuda:
        from uniserve_kernels import rope

        target = (
            torch.empty_like(x, memory_format=torch.contiguous_format)
            if out is None
            else out
        )
        require_kernel(
            "apply_rotary",
            rope.unsupported_rotary(x, cos, sin, target),
            x=x,
            cos=cos,
            sin=sin,
            out=target,
        )
        rope.rotary(x, cos, sin, target, interleaved=rotation == "interleaved")
        return target

    return result(_rotate(x.float(), cos, sin, rotation).to(x.dtype), out)


def _domain_axes(widths, axis_dims):
    """Count the complete rotary axes each normalization domain spans.

    Returns ``None`` unless the domains, in order, cover every axis and each
    domain ends on an axis boundary.
    """
    counts, axis = [], 0
    for width in widths:
        start, covered = axis, 0
        while axis < len(axis_dims) and covered < width:
            covered += axis_dims[axis]
            axis += 1
        if covered != width or axis == start:
            return None
        counts.append(axis - start)
    return tuple(counts) if axis == len(axis_dims) else None


def qk_norm_rope(
    q,
    k,
    q_weights,
    k_weights,
    cos,
    sin,
    *,
    eps: float,
    axis_dims,
    out=None,
    rounding: Rounding = Rounding.ONCE,
):
    """Normalize Q/K over explicit RMS domains, then rotate each rotary axis.

    ``q`` and ``k`` have shapes ``[..., heads, head_dim]`` and
    ``[..., kv_heads, head_dim]`` over the same token axes. ``axis_dims``
    partitions ``head_dim`` into rotary axes. ``q_weights`` and ``k_weights``
    hold one RMS weight per normalization domain in head order. A domain's
    width is its weight length and spans consecutive complete axes, so every
    axis in one domain shares that domain's FP32 variance.

    ``cos`` and ``sin`` hold one compact factor table ``[..., rotated / 2]``
    per axis. Each axis rotates its leading ``rotated`` coordinates split-half
    and only normalizes the remainder; zero-width factors leave an axis
    unrotated. With ``Rounding.ONCE`` normalization, scaling and rotation
    accumulate in FP32 and round once to the input dtype. With
    ``Rounding.STEPWISE`` each weighted normalization domain rounds once, the
    factors round to the input dtype, and each rotation product and sum
    rounds in turn. ``out`` receives the query and key results and may alias
    ``q`` and ``k``.
    """
    widths = tuple(
        weight.shape[0] if weight.ndim == 1 else -1 for weight in q_weights
    )
    counts = (
        _domain_axes(widths, axis_dims)
        if isinstance(axis_dims, tuple)
        else None
    )
    if (
        counts is None
        or q.ndim < 2
        or k.ndim != q.ndim
        or k.shape[:-2] != q.shape[:-2]
        or k.shape[-1] != q.shape[-1]
        or k.device != q.device
        or any(type(width) is not int or width < 1 for width in axis_dims)
        or sum(axis_dims) != q.shape[-1]
        or len(cos) != len(axis_dims)
        or len(sin) != len(axis_dims)
        or any(
            not _factors_match(q, cosine, sine) or cosine.shape[-1] * 2 > width
            for cosine, sine, width in zip(cos, sin, axis_dims, strict=True)
        )
        or len(k_weights) != len(q_weights)
        or any(
            key.shape != query.shape
            or query.device != q.device
            or key.device != q.device
            for query, key in zip(q_weights, k_weights, strict=True)
        )
    ):
        raise ValueError(
            "Q/K normalization domains must cover complete rotary axes that "
            "partition the head width"
        )

    if out is None:
        query = torch.empty_like(q, memory_format=torch.contiguous_format)
        key = torch.empty_like(k, memory_format=torch.contiguous_format)
    else:
        query, key = out
        check_output(q, query)
        check_output(k, key)

    if q.is_cuda:
        from uniserve_kernels import rope

        require_kernel(
            "qk_norm_rope",
            rope.unsupported_qk_norm_rope(
                q, k, q_weights, k_weights, cos, sin, query, key
            ),
            q=q,
            k=k,
            q_out=query,
            k_out=key,
        )
        rope.qk_norm_rope(
            q,
            k,
            q_weights,
            k_weights,
            cos,
            sin,
            eps,
            query,
            key,
            axis_dims=axis_dims,
            stepwise=rounding is Rounding.STEPWISE,
        )
        return query, key

    # General composition reads the complete source before storing. A single
    # rounding keeps every domain and axis in FP32 until the store; stepwise
    # rounding rounds each normalized domain and rotates in the input dtype.
    for source, weights, target in ((q, q_weights, query), (k, k_weights, key)):
        normalized = torch.cat(
            tuple(
                part
                * torch.rsqrt(part.square().mean(-1, keepdim=True) + eps)
                * weight.float()
                for part, weight in zip(
                    source.float().split(widths, dim=-1), weights, strict=True
                )
            ),
            dim=-1,
        )
        if rounding is Rounding.STEPWISE:
            normalized = normalized.to(source.dtype)
        target.copy_(
            torch.cat(
                tuple(
                    _rotate(part, cosine, sine, "split")
                    for part, cosine, sine in zip(
                        normalized.split(axis_dims, dim=-1),
                        cos,
                        sin,
                        strict=True,
                    )
                ),
                dim=-1,
            )
        )
    return query, key


def qk_bias_rms_norm_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_bias: torch.Tensor | None = None,
    key_bias: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    value_bias: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias and RMS-normalize Q/K heads, then rotate them, in place.

    ``query`` and ``key`` have shape ``[..., heads, head_dim]``; biases cover
    the complete ``heads * head_dim`` projection width. Heads normalize
    without a learned weight, and compact factors ``[..., rotated / 2]`` rotate
    the leading coordinates split-half. When supplied, ``value`` receives its
    bias in the same pass. Arithmetic accumulates in FP32 before each store.
    """
    from uniserve_kernels import rope

    if (query_bias is None) != (key_bias is None):
        raise ValueError("Q/K normalization requires both biases or neither")
    if (value is None) != (value_bias is None):
        raise ValueError(
            "Q/K normalization requires both value and value bias or neither"
        )
    if (
        query.shape != key.shape
        or query.ndim < 3
        or query.numel() == 0
        or query.dtype != key.dtype
        or not query.is_floating_point()
        or not _factors_match(query, cos, sin)
        or not 0 < cos.shape[-1] * 2 <= query.shape[-1]
        or key.device != query.device
    ):
        raise ValueError(
            "Q/K normalization requires equal floating heads and compact "
            "factors for a prefix of each head"
        )
    heads, head_dim = int(query.shape[-2]), int(query.shape[-1])
    for bias in (query_bias, key_bias, value_bias):
        if bias is not None and (
            bias.numel() != heads * head_dim or bias.device != query.device
        ):
            raise ValueError(
                "projection bias must match the complete head width and device"
            )
    if value is not None and (
        value.shape != query.shape or value.device != query.device
    ):
        raise ValueError("value projection must match Q/K shape and device")

    if query.is_cuda:
        require_kernel(
            "qk_bias_rms_norm_rope_",
            rope.unsupported_qk_bias_rms_norm_rope(
                query, key, value, cos, sin, query_bias, key_bias, value_bias
            ),
            query=query,
            key=key,
            value=value,
            cos=cos,
            sin=sin,
        )
        rope.qk_bias_rms_norm_rope_(
            query, key, cos, sin, query_bias, key_bias, value, value_bias, eps
        )
        return query, key

    for target, bias in ((query, query_bias), (key, key_bias)):
        normalized = target.float()
        if bias is not None:
            normalized = normalized + bias.reshape(heads, head_dim).float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + eps
        )
        target.copy_(_rotate(normalized, cos, sin, "split"))
    if value is not None and value_bias is not None:
        value.copy_(
            (value.float() + value_bias.reshape(heads, head_dim).float()).to(
                value.dtype
            )
        )
    return query, key
