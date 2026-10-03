"""Gated activations over packed or separate value and gate channels.

Bias, activation and multiplication accumulate in FP32 before one rounding;
``silu_and_mul`` also evaluates the ``Rounding.STEPWISE`` recipe, which
rounds the activated gate before the product. Eligible CUDA calls use
UniServe's kernels; every other call evaluates the same formula with tensor
operations.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

from uniserve.quantization import QuantizedTensor

from ._tensors import Rounding, result


def _row_fp8(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode FP32 rows as E4M3 values with one ``[rows, 1]`` scale each."""
    from uniserve.quantization import Quantizer

    encoded = Quantizer("fp8", axis=0).quantize(
        values.reshape(-1, values.shape[-1])
    )
    buffers = encoded.buffers()
    return buffers["values"].reshape(values.shape), buffers["scale"]


def _gated_silu(gate, value, rounding) -> torch.Tensor:
    """Return FP32 ``silu(gate) * value`` under ``rounding``.

    Stepwise rounding rounds the activated gate and the product to the
    input dtype, so the FP32 result holds an activation-dtype value.
    """
    activated = F.silu(gate.float())
    if rounding is Rounding.ONCE:
        return activated * value.float()
    activated = activated.to(gate.dtype).float()
    return (activated * value.float()).to(gate.dtype).float()


def silu_and_mul(
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    rounding: Rounding = Rounding.ONCE,
) -> torch.Tensor:
    """Apply SiLU to the first channel half and multiply by the second.

    ``x`` packs ``[gate, value]`` along its final axis. A row-scaled FP8
    ``QuantizedTensor`` output receives the encoded activation directly.
    ``rounding`` selects whether the activated gate rounds to ``x.dtype``
    before the product.
    """
    from uniserve_kernels import activation

    if x.ndim < 1 or x.shape[-1] % 2 or x.shape[-1] == 0:
        raise ValueError("gating requires equal channel halves")
    shape = (*x.shape[:-1], x.shape[-1] // 2)
    stepwise = rounding is Rounding.STEPWISE

    if isinstance(out, QuantizedTensor):
        from uniserve.quantization import Quantizer

        if out.quantizer != Quantizer("fp8", axis=0):
            raise ValueError("fused SiLU output requires row-scaled FP8")
        if x.ndim != 2:
            gate, value = x.chunk(2, dim=-1)
            activated = _gated_silu(gate, value, rounding)
            # Axis zero of a higher-rank tensor retains that axis alone; it
            # must not silently become one separate scale per flattened row.
            encoded = out.quantizer.quantize(activated)
            return result(encoded.to(dtype=x.dtype), out)

        if activation.can_run(x):
            values = torch.empty(
                shape, dtype=torch.float8_e4m3fn, device=x.device
            )
            scales = torch.empty(
                (x.shape[0], 1), dtype=torch.float32, device=x.device
            )
            activation.silu_and_mul_fp8(x, values, scales, stepwise=stepwise)
        else:
            gate, value = x.chunk(2, dim=-1)
            values, scales = _row_fp8(_gated_silu(gate, value, rounding))
        encoded = out.quantizer.from_tensors(
            {"values": values, "scale": scales},
            shape=tuple(values.shape),
            dtype=x.dtype,
        )
        return result(encoded, out)

    if activation.can_run(x):
        target = (
            torch.empty(shape, dtype=x.dtype, device=x.device)
            if out is None or not out.is_contiguous()
            else out
        )
        activation.silu_and_mul(x, target, stepwise=stepwise)
        return target if out is None else result(target, out)
    gate, value = x.chunk(2, dim=-1)
    return result(_gated_silu(gate, value, rounding).to(x.dtype), out)


def gelu_and_mul(
    x: torch.Tensor,
    *,
    approximate: Literal["none", "tanh"],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the selected GELU formula to the first channel half, then
    multiply by the second.
    """  # noqa: D205
    if x.ndim < 1 or x.shape[-1] % 2 or approximate not in {"none", "tanh"}:
        raise ValueError(
            "GELU gating requires equal channel halves and a supported "
            "approximation"
        )
    gate, value = x.chunk(2, dim=-1)
    return result(F.gelu(gate, approximate=approximate) * value, out)


def _value_first_width(
    value_gate: torch.Tensor, bias: torch.Tensor | None
) -> int:
    if (
        value_gate.ndim < 1
        or value_gate.shape[-1] < 2
        or value_gate.shape[-1] % 2
    ):
        raise ValueError("SwiGLU requires two equal, nonempty packed halves")
    if bias is not None and (
        bias.shape != value_gate.shape[-1:] or bias.device != value_gate.device
    ):
        raise ValueError("SwiGLU bias must match the packed width and device")
    return int(value_gate.shape[-1]) // 2


def _value_first(
    value_gate: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Return FP32 ``value * silu(gate)`` for packed ``[value, gate]`` rows."""
    if bias is not None:
        value_gate = value_gate.float() + bias.float()
    value, gate = value_gate.chunk(2, dim=-1)
    return value.float() * F.silu(gate.float())


def _absmax(output: torch.Tensor, partials: torch.Tensor) -> torch.Tensor:
    from uniserve_kernels import reduction

    maximum = torch.empty((), dtype=torch.float32, device=output.device)
    reduction.absmax(partials, maximum)
    return maximum


def value_first_swiglu(
    value_gate: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply SiLU gating to packed ``[value, gate]`` projection rows."""
    from uniserve_kernels import activation

    width = _value_first_width(value_gate, bias)
    if not activation.can_run(value_gate):
        return _value_first(value_gate, bias).to(value_gate.dtype)
    output = torch.empty(
        (*value_gate.shape[:-1], width),
        dtype=value_gate.dtype,
        device=value_gate.device,
    )
    activation.value_first_swiglu(value_gate, bias, output)
    return output


def value_first_swiglu_absmax(
    value_gate: torch.Tensor, bias: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return value-first SwiGLU and the FP32 absmax of its rounded result."""
    from uniserve_kernels import activation

    width = _value_first_width(value_gate, bias)
    if not activation.can_run(value_gate):
        output = _value_first(value_gate, bias).to(value_gate.dtype)
        return output, output.float().abs().amax()
    output = torch.empty(
        (*value_gate.shape[:-1], width),
        dtype=value_gate.dtype,
        device=value_gate.device,
    )
    partials = torch.empty(
        (-(-output.numel() // activation.ABSMAX_BLOCK),),
        dtype=torch.float32,
        device=value_gate.device,
    )
    activation.value_first_swiglu(value_gate, bias, output, partials)
    return output, _absmax(output, partials)


def value_first_swiglu_fp8(
    value_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize value-first SwiGLU with one E4M3 scale per row."""
    from uniserve_kernels import activation

    width = _value_first_width(value_gate, None)
    if width > activation.MAX_FP8_WIDTH or not activation.can_run(value_gate):
        return _row_fp8(_value_first(value_gate, None))
    rows = value_gate.numel() // (2 * width)
    output = torch.empty(
        (*value_gate.shape[:-1], width),
        dtype=torch.float8_e4m3fn,
        device=value_gate.device,
    )
    scales = torch.empty(
        (rows, 1), dtype=torch.float32, device=value_gate.device
    )
    activation.value_first_swiglu_fp8(value_gate, output, scales)
    return output, scales


def _swiglu_width(value, gate, value_bias, gate_bias) -> int:
    if (
        value.ndim < 1
        or value.shape != gate.shape
        or value.dtype != gate.dtype
        or value.device != gate.device
        or value.shape[-1] < 1
    ):
        raise ValueError("SwiGLU value and gate tensors must match")
    width = int(value.shape[-1])
    for name, bias in (("value", value_bias), ("gate", gate_bias)):
        if bias is not None and (
            bias.shape != (width,)
            or bias.device != value.device
            or bias.dtype != value.dtype
        ):
            raise ValueError(
                f"SwiGLU {name} bias must match the channel width and tensor"
            )
    return width


def _strided_rows(value: torch.Tensor, gate: torch.Tensor) -> bool:
    """Report whether separate views flatten to unit-strided channel rows."""
    if value.stride(-1) != 1 or gate.stride(-1) != 1:
        return False
    try:
        value.view(-1, value.shape[-1])
        gate.view(-1, gate.shape[-1])
    except RuntimeError:
        return False
    return True


def _swiglu(value, gate, value_bias, gate_bias, *, absmax: bool):
    from uniserve_kernels.triton import launchable

    from uniserve_kernels import activation

    _swiglu_width(value, gate, value_bias, gate_bias)
    if not (value.is_cuda and launchable(value.device)) or not _strided_rows(
        value, gate
    ):
        value_fp32, gate_fp32 = value.float(), gate.float()
        if value_bias is not None:
            value_fp32 = value_fp32 + value_bias.float()
        if gate_bias is not None:
            gate_fp32 = gate_fp32 + gate_bias.float()
        output = (value_fp32 * F.silu(gate_fp32)).to(value.dtype)
        return (output, output.float().abs().amax()) if absmax else output

    output = torch.empty_like(value, memory_format=torch.contiguous_format)
    partials = (
        torch.empty(
            (-(-output.numel() // activation.ABSMAX_BLOCK),),
            dtype=torch.float32,
            device=value.device,
        )
        if absmax
        else None
    )
    activation.swiglu(value, gate, value_bias, gate_bias, output, partials)
    if partials is None:
        return output
    return output, _absmax(output, partials)


def swiglu(
    value: torch.Tensor,
    gate: torch.Tensor,
    *,
    value_bias: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(value + value_bias) * silu(gate + gate_bias)``.

    Separate views let merged projections keep their shared backing rather
    than copying both branches into one packed layout.
    """
    return _swiglu(value, gate, value_bias, gate_bias, absmax=False)


def swiglu_absmax(
    value: torch.Tensor,
    gate: torch.Tensor,
    *,
    value_bias: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return :func:`swiglu` and the FP32 absmax of its rounded output."""
    return _swiglu(value, gate, value_bias, gate_bias, absmax=True)
