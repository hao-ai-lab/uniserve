"""RMS and layer normalization with residual and modulation epilogues.

Statistics, residual sums and affine transforms accumulate in FP32 and round
once at the output dtype boundary. The modulation epilogues also evaluate the
``Rounding.STEPWISE`` recipe, which rounds after each operation of the eager
expression. Eligible CUDA calls use UniServe's kernels; every other call
evaluates the same formula with tensor operations.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ._tensors import Rounding, check_output, result


def _rms(value: torch.Tensor, eps: float) -> torch.Tensor:
    """Return FP32 ``value * rsqrt(mean(value^2) + eps)`` over the last axis."""
    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)


def _check_vectors(value: torch.Tensor, *vectors: torch.Tensor | None) -> None:
    if value.ndim < 1 or value.numel() == 0 or not value.is_floating_point():
        raise ValueError(
            "normalization requires nonempty floating activation rows"
        )
    width = value.shape[-1]
    for vector in vectors:
        if vector is not None and (
            vector.numel() != width or vector.device != value.device
        ):
            raise ValueError(
                "normalization vectors must match the activation width and "
                "device"
            )


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize the last axis as ``x * rsqrt(mean(x^2) + eps) * weight``.

    Variance and scaling accumulate in FP32; the result has ``x``'s dtype.
    ``out`` receives the result and may alias ``x``.
    """
    from uniserve_kernels.norm import rms

    if weight.shape != x.shape[-1:] or weight.device != x.device:
        raise ValueError("RMS weight must match the final width and device")
    target = torch.empty_like(x) if out is None else out
    check_output(x, target)
    if rms.can_run(x, weight, target):
        rms.rms_norm(x, weight, eps, target)
        return target
    return result((_rms(x, eps) * weight.float()).to(x.dtype), out)


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the RMS normalization of ``x + residual`` and the sum itself.

    Both results derive from the unrounded FP32 sum and round once to ``x``'s
    dtype. Inputs are preserved unless ``out`` aliases them.
    """
    from uniserve_kernels.norm import rms

    if (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("residual must match the activation representation")
    if weight.shape != x.shape[-1:] or weight.device != x.device:
        raise ValueError("RMS weight must match the final width and device")
    targets = (torch.empty_like(x), torch.empty_like(x)) if out is None else out
    for target in targets:
        check_output(x, target)
    if rms.can_run(x, weight, residual, *targets):
        rms.add_rms_norm(x, residual, weight, eps, *targets)
        return targets

    summed = x.float() + residual.float()
    normalized = (_rms(summed, eps) * weight.float()).to(x.dtype)
    if out is None:
        return normalized, summed.to(x.dtype)
    return result(normalized, out[0]), result(summed.to(x.dtype), out[1])


def _autocast_dtype(value: torch.Tensor) -> torch.dtype:
    """Return the dtype CUDA autocast would produce for ``value``."""
    return (
        torch.get_autocast_dtype("cuda")
        if value.is_cuda and torch.is_autocast_enabled("cuda")
        else value.dtype
    )


def _absmax(output: torch.Tensor, partials: torch.Tensor) -> torch.Tensor:
    """Finish one FP32 absolute maximum from per-row kernel partials."""
    from uniserve_kernels import reduction

    maximum = torch.empty((), dtype=torch.float32, device=output.device)
    reduction.absmax(partials, maximum)
    return maximum


def _partials(value: torch.Tensor) -> torch.Tensor:
    rows = value.numel() // value.shape[-1]
    return torch.empty((rows,), dtype=torch.float32, device=value.device)


def weighted_rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, *, eps: float
) -> torch.Tensor:
    """Return ``rms(hidden) * weight`` in the CUDA autocast output dtype."""
    from uniserve_kernels.norm import residual

    _check_vectors(hidden, weight)
    output = torch.empty_like(hidden, dtype=_autocast_dtype(hidden))
    if residual.can_run(hidden, weight, output):
        residual.weighted_rms_norm(hidden, weight, eps, output)
        return output
    return (_rms(hidden, eps) * weight.float()).to(output.dtype)


def weighted_rms_norm_absmax(
    hidden: torch.Tensor, weight: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted RMS rows and the FP32 absmax of the rounded rows."""
    from uniserve_kernels.norm import residual

    _check_vectors(hidden, weight)
    output = torch.empty_like(hidden, dtype=_autocast_dtype(hidden))
    if residual.can_run(hidden, weight, output):
        partials = _partials(hidden)
        residual.weighted_rms_norm(hidden, weight, eps, output, partials)
        return output, _absmax(output, partials)
    output = weighted_rms_norm(hidden, weight, eps=eps)
    return output, output.float().abs().amax()


def _check_update(hidden: torch.Tensor, update: torch.Tensor) -> None:
    if hidden.shape != update.shape or hidden.device != update.device:
        raise ValueError(
            "scaled residual operands must have the same shape and device"
        )


def _scaled_sum(hidden, update, scale, update_bias) -> torch.Tensor:
    """Return FP32 ``hidden + (update + update_bias) * scale``."""
    biased = update.float()
    if update_bias is not None:
        biased = biased + update_bias.float()
    return hidden.float() + biased * scale.float()


def scaled_residual_rms_norm_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the residual stream in place and return ``(hidden, normalized)``.

    ``hidden`` becomes the FP32 sum ``hidden + (update + update_bias) * scale``
    rounded to its dtype; ``normalized`` is the weighted RMS of the unrounded
    sum in the CUDA autocast dtype of ``update``.
    """
    from uniserve_kernels.norm import residual

    _check_update(hidden, update)
    _check_vectors(hidden, scale, weight, update_bias)
    output = torch.empty_like(hidden, dtype=_autocast_dtype(update))
    if residual.can_run(hidden, update, scale, weight, update_bias, output):
        residual.scaled_residual_rms_norm_(
            hidden, update, scale, weight, update_bias, eps, output
        )
        return hidden, output
    summed = _scaled_sum(hidden, update, scale, update_bias)
    hidden.copy_(summed)
    return hidden, (_rms(summed, eps) * weight.float()).to(output.dtype)


def scaled_residual_rms_norm_absmax_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    update_bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return :func:`scaled_residual_rms_norm_` and the normalized absmax.

    The FP32 absolute maximum spans the rounded normalized output.
    """
    from uniserve_kernels.norm import residual

    _check_update(hidden, update)
    _check_vectors(hidden, scale, weight, update_bias)
    output = torch.empty_like(hidden, dtype=_autocast_dtype(update))
    if residual.can_run(hidden, update, scale, weight, update_bias, output):
        partials = _partials(hidden)
        residual.scaled_residual_rms_norm_(
            hidden, update, scale, weight, update_bias, eps, output, partials
        )
        return hidden, output, _absmax(output, partials)
    hidden, output = scaled_residual_rms_norm_(
        hidden, update, scale, weight, update_bias, eps=eps
    )
    return hidden, output, output.float().abs().amax()


def scaled_residual_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    update_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add ``(update + update_bias) * scale`` to ``hidden`` in place."""
    from uniserve_kernels.norm import residual

    _check_update(hidden, update)
    _check_vectors(hidden, scale, update_bias)
    if residual.can_run(hidden, update, scale, update_bias):
        residual.scaled_residual_(hidden, update, scale, update_bias)
        return hidden
    biased = update.float()
    if update_bias is not None:
        biased = biased + update_bias.float()
    return hidden.add_(biased * scale.float())


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
    """Return the layer normalization of the scaled residual sum.

    Unlike the RMS variants, ``hidden`` is not updated in place.
    """
    from uniserve_kernels.norm import residual

    _check_update(hidden, update)
    _check_vectors(hidden, scale, weight, bias, update_bias)
    output = torch.empty_like(hidden, dtype=_autocast_dtype(update))
    if residual.can_run(
        hidden, update, scale, weight, bias, update_bias, output
    ):
        residual.scaled_residual_layer_norm(
            hidden, update, scale, weight, bias, update_bias, eps, output
        )
        return output
    summed = _scaled_sum(hidden, update, scale, update_bias)
    normalized = F.layer_norm(
        summed, (hidden.shape[-1],), weight.float(), bias.float(), eps
    )
    return normalized.to(output.dtype)


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
    """Return :func:`scaled_residual_layer_norm` and its rounded absmax."""
    from uniserve_kernels.norm import residual

    _check_update(hidden, update)
    _check_vectors(hidden, scale, weight, bias, update_bias)
    output = torch.empty_like(hidden, dtype=_autocast_dtype(update))
    if residual.can_run(
        hidden, update, scale, weight, bias, update_bias, output
    ):
        partials = _partials(hidden)
        residual.scaled_residual_layer_norm(
            hidden,
            update,
            scale,
            weight,
            bias,
            update_bias,
            eps,
            output,
            partials,
        )
        return output, _absmax(output, partials)
    output = scaled_residual_layer_norm(
        hidden, update, scale, weight, bias, update_bias, eps=eps
    )
    return output, output.float().abs().amax()


def _modulate(
    value, weight, shift, scale, row_indices, eps, rounding, dtype
) -> torch.Tensor:
    """Return ``rms(value) * weight * (1 + scale[i]) + shift[i]`` in FP32.

    Stepwise rounding returns FP32 values that the activation ``dtype``
    holds exactly: the weighted normalization, ``1 + scale``, the product and
    the shifted sum each round to it. ``value`` may be an FP32 sum.
    """
    scale = scale.index_select(0, row_indices).float()
    shift = shift.index_select(0, row_indices).float()
    if rounding is Rounding.ONCE:
        return _rms(value, eps) * weight.float() * (1.0 + scale) + shift

    normalized = (_rms(value, eps) * weight.float()).to(dtype).float()
    factor = (1.0 + scale).to(dtype).float()
    product = (normalized * factor).to(dtype).float()
    return (product + shift).to(dtype).float()


def _gated_sum(hidden, update, gate, row_indices, rounding) -> torch.Tensor:
    """Return ``hidden + gate[i] * update`` in FP32.

    Stepwise rounding rounds the gated update and then the sum to
    ``hidden.dtype``.
    """
    gated = gate.index_select(0, row_indices).float() * update.float()
    if rounding is Rounding.ONCE:
        return hidden.float() + gated
    gated = gated.to(hidden.dtype).float()
    return (hidden.float() + gated).to(hidden.dtype).float()


def modulated_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
    retain: torch.Tensor | None = None,
    rounding: Rounding = Rounding.ONCE,
) -> torch.Tensor:
    """Return ``rms(value) * weight * (1 + scale[i]) + shift[i]`` per row.

    ``row_indices`` selects each row's modulation parameters. ``retain``
    receives a copy of ``value`` in the same pass, so a caller keeping the
    residual while ``value`` lives in borrowed scratch reads it once.
    ``rounding`` selects where the expression rounds to ``value.dtype``.
    """
    from uniserve_kernels.norm import modulation

    if retain is not None and (
        retain.shape != value.shape
        or retain.dtype != value.dtype
        or not retain.is_contiguous()
    ):
        raise ValueError("retained rows must match the normalized value")
    if value.shape[-1] <= modulation.MAX_WIDTH and modulation.can_run(
        value, weight, shift, scale, row_indices
    ):
        output = torch.empty_like(value)
        modulation.modulated_rms_norm(
            value,
            weight,
            shift,
            scale,
            row_indices,
            eps,
            output,
            retain=retain,
            stepwise=rounding is Rounding.STEPWISE,
        )
        return output

    if retain is not None:
        retain.copy_(value)
    return _modulate(
        value, weight, shift, scale, row_indices, eps, rounding, value.dtype
    ).to(value.dtype)


def gated_residual(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    rounding: Rounding = Rounding.ONCE,
) -> torch.Tensor:
    """Return ``hidden + gate[i] * update``; the kernel consumes ``update``.

    ``rounding`` selects whether the gated update rounds to ``hidden.dtype``
    before the sum.
    """
    from uniserve_kernels.norm import modulation

    if update.is_contiguous() and modulation.can_run(
        hidden, update, gate, row_indices
    ):
        modulation.gated_residual(
            hidden,
            update,
            gate,
            row_indices,
            stepwise=rounding is Rounding.STEPWISE,
        )
        return update
    return _gated_sum(hidden, update, gate, row_indices, rounding).to(
        hidden.dtype
    )


def gated_residual_rms_norm(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
    rounding: Rounding = Rounding.ONCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a gated residual and its modulated normalization.

    With ``Rounding.ONCE`` the normalization reads the unrounded FP32
    residual sum; with ``Rounding.STEPWISE`` it reads the rounded residual
    and every intermediate rounds. ``update`` is consumed and may back the
    returned activation-dtype residual.
    """
    from uniserve_kernels.norm import modulation

    if (
        hidden.shape[-1] <= modulation.MAX_WIDTH
        and update.is_contiguous()
        and modulation.can_run(
            hidden, update, gate, weight, shift, scale, row_indices
        )
    ):
        normalized = torch.empty_like(hidden)
        modulation.modulated_rms_norm(
            hidden,
            weight,
            shift,
            scale,
            row_indices,
            eps,
            normalized,
            update=update,
            gate=gate,
            stepwise=rounding is Rounding.STEPWISE,
        )
        return update, normalized

    summed = _gated_sum(hidden, update, gate, row_indices, rounding)
    normalized = _modulate(
        summed, weight, shift, scale, row_indices, eps, rounding, hidden.dtype
    )
    return summed.to(hidden.dtype), normalized.to(hidden.dtype)


def gated_residual_rms_norm_fp8(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
    rounding: Rounding = Rounding.ONCE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the residual and its row-scaled E4M3 modulated normalization.

    ``update`` is consumed and may back the returned residual. The
    ``[rows, 1]`` FP32 scales dequantize the returned values. ``rounding``
    applies as in ``gated_residual_rms_norm``; a stepwise normalization
    rounds to the activation dtype before it is encoded.
    """
    from uniserve_kernels.norm import modulation

    if (
        hidden.shape[-1] <= modulation.MAX_WIDTH
        and update.is_contiguous()
        and modulation.can_run(
            hidden, update, gate, weight, shift, scale, row_indices
        )
    ):
        rows = hidden.numel() // hidden.shape[-1]
        values = torch.empty_like(hidden, dtype=torch.float8_e4m3fn)
        scales = torch.empty(
            (rows, 1), dtype=torch.float32, device=hidden.device
        )
        modulation.modulated_rms_norm(
            hidden,
            weight,
            shift,
            scale,
            row_indices,
            eps,
            values,
            update=update,
            gate=gate,
            output_scale=scales,
            stepwise=rounding is Rounding.STEPWISE,
        )
        return update, values, scales

    from uniserve.quantization import Quantizer

    summed = _gated_sum(hidden, update, gate, row_indices, rounding)
    normalized = _modulate(
        summed, weight, shift, scale, row_indices, eps, rounding, hidden.dtype
    )
    encoded = Quantizer("fp8", axis=0).quantize(
        normalized.reshape(-1, normalized.shape[-1])
    )
    buffers = encoded.buffers()
    return (
        summed.to(hidden.dtype),
        buffers["values"].reshape(normalized.shape),
        buffers["scale"],
    )
