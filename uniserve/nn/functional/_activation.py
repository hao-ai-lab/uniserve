"""Gated activations over packed or separate value and gate channels.

Bias, activation and multiplication accumulate in FP32 before one rounding.
CUDA calls run UniServe's kernels and raise ``ValueError`` when no kernel
accepts their operands; other devices evaluate the same formula with tensor
operations.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F
from uniserve_kernels.triton import require_kernel

from uniserve.quantization import QuantizedTensor

from ._tensors import result


def _row_fp8(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode FP32 rows as E4M3 values with one ``[rows, 1]`` scale each."""
    from uniserve.quantization import Quantizer

    encoded = Quantizer("fp8", axis=0).quantize(
        values.reshape(-1, values.shape[-1])
    )
    buffers = encoded.buffers()
    return buffers["values"].reshape(values.shape), buffers["scale"]


def _gate_value(x: torch.Tensor, activation: str) -> torch.Tensor:
    """Return FP32 ``activation(gate) * value`` for packed ``[gate, value]``."""
    gate, value = x.float().chunk(2, dim=-1)
    if activation == "silu":
        activated = F.silu(gate)
    else:
        activated = F.gelu(
            gate, approximate="tanh" if activation == "gelu_tanh" else "none"
        )
    return activated * value


def _act_and_mul(
    function: str,
    x: torch.Tensor,
    activation: str,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate packed gating into a plain or row-scaled FP8 output.

    ``activation`` names a formula of ``uniserve_kernels.activation``. A
    row-scaled FP8 ``QuantizedTensor`` output receives the E4M3 encoding of
    the unrounded FP32 products with one scale per row.
    """
    from uniserve.quantization import Quantizer
    from uniserve_kernels import activation as kernels

    if x.ndim < 1 or x.shape[-1] % 2 or x.shape[-1] == 0:
        raise ValueError("gating requires equal channel halves")
    shape = (*x.shape[:-1], x.shape[-1] // 2)

    if isinstance(out, QuantizedTensor):
        if out.quantizer != Quantizer("fp8", axis=0):
            raise ValueError("fused gating output requires row-scaled FP8")
        if x.is_cuda:
            require_kernel(
                function,
                "a row-scaled FP8 output keeps one scale per leading index "
                "of a rank-2 input; other ranks have no kernel"
                if x.ndim != 2
                else kernels.unsupported(x, fp8_width=shape[-1]),
                x=x,
            )
            values = torch.empty(
                shape, dtype=torch.float8_e4m3fn, device=x.device
            )
            scales = torch.empty(
                (x.shape[0], 1), dtype=torch.float32, device=x.device
            )
            kernels.act_and_mul_fp8(x, values, scales, activation=activation)
        elif x.ndim != 2:
            # Axis zero of a higher-rank tensor retains that axis alone; it
            # must not silently become one separate scale per flattened row.
            encoded = out.quantizer.quantize(_gate_value(x, activation))
            return result(encoded.to(dtype=x.dtype), out)
        else:
            values, scales = _row_fp8(_gate_value(x, activation))
        encoded = out.quantizer.from_tensors(
            {"values": values, "scale": scales},
            shape=tuple(values.shape),
            dtype=x.dtype,
        )
        return result(encoded, out)

    if out is not None and (
        out.shape != shape or out.dtype != x.dtype or out.device != x.device
    ):
        raise ValueError(
            "output must match the numerical result's shape, dtype and device"
        )
    if x.is_cuda:
        # The kernel stores contiguous rows; another caller layout receives
        # a copy of them.
        target = (
            out
            if out is not None and out.is_contiguous()
            else torch.empty(shape, dtype=x.dtype, device=x.device)
        )
        require_kernel(function, kernels.unsupported(x, target), x=x, out=out)
        kernels.act_and_mul(x, target, activation=activation)
        return target if out is None else result(target, out)
    return result(_gate_value(x, activation).to(x.dtype), out)


def silu_and_mul(
    x: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply SiLU to the first channel half and multiply by the second.

    ``x`` packs ``[gate, value]`` along its final axis. A row-scaled FP8
    ``QuantizedTensor`` output receives the encoded activation directly.
    """
    return _act_and_mul("silu_and_mul", x, "silu", out)


def gelu_and_mul(
    x: torch.Tensor,
    *,
    approximate: Literal["none", "tanh"],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the selected GELU formula to the first channel half, then
    multiply by the second.

    ``approximate="none"`` is the exact erf form; ``"tanh"`` is PyTorch's
    tanh approximation (``gelu_pytorch_tanh``). Outputs follow
    :func:`silu_and_mul`, including a row-scaled FP8 ``QuantizedTensor``.
    """  # noqa: D205
    if approximate not in {"none", "tanh"}:
        raise ValueError("GELU gating requires a supported approximation")
    return _act_and_mul(
        "gelu_and_mul",
        x,
        "gelu_tanh" if approximate == "tanh" else "gelu",
        out,
    )


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
    if not value_gate.is_cuda:
        return _value_first(value_gate, bias).to(value_gate.dtype)
    require_kernel(
        "value_first_swiglu",
        activation.unsupported(value_gate, bias),
        value_gate=value_gate,
        bias=bias,
    )
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
    if not value_gate.is_cuda:
        output = _value_first(value_gate, bias).to(value_gate.dtype)
        return output, output.float().abs().amax()
    require_kernel(
        "value_first_swiglu_absmax",
        activation.unsupported(value_gate, bias),
        value_gate=value_gate,
        bias=bias,
    )
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
    if not value_gate.is_cuda:
        return _row_fp8(_value_first(value_gate, None))
    require_kernel(
        "value_first_swiglu_fp8",
        activation.unsupported(value_gate, fp8_width=width),
        value_gate=value_gate,
    )
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


def _swiglu(value, gate, value_bias, gate_bias, *, absmax: bool):
    from uniserve_kernels import activation

    _swiglu_width(value, gate, value_bias, gate_bias)
    if not value.is_cuda:
        value_fp32, gate_fp32 = value.float(), gate.float()
        if value_bias is not None:
            value_fp32 = value_fp32 + value_bias.float()
        if gate_bias is not None:
            gate_fp32 = gate_fp32 + gate_bias.float()
        output = (value_fp32 * F.silu(gate_fp32)).to(value.dtype)
        return (output, output.float().abs().amax()) if absmax else output

    # Separate views keep their own row strides; each must flatten to rows.
    require_kernel(
        "swiglu_absmax" if absmax else "swiglu",
        activation.unsupported(value, value_bias, gate_bias)
        or activation.unsupported(gate),
        value=value,
        gate=gate,
        value_bias=value_bias,
        gate_bias=gate_bias,
    )
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


def softcap(
    x: torch.Tensor,
    cap: float,
    *,
    dtype: torch.dtype = torch.float32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bound values to ``(-cap, cap)`` as ``tanh(x / cap) * cap``.

    Evaluates ``torch.tanh(x.float() / cap) * cap`` in FP32, rounded once to
    ``dtype`` (or ``out``'s dtype). On CUDA one launch reads ``x`` and
    writes the result, bit-identical to that tensor expression; ``out``, a
    contiguous tensor of ``x``'s shape, receives it directly.
    """
    from uniserve_kernels import activation

    if out is not None:
        dtype = out.dtype
    if x.is_cuda:
        target = (
            torch.empty(x.shape, dtype=dtype, device=x.device)
            if out is None
            else out
        )
        require_kernel(
            "softcap",
            activation.unsupported_softcap(x, target),
            x=x,
            out=target,
        )
        activation.softcap(x, cap, target)
        return target

    capped = (torch.tanh(x.float() / cap) * cap).to(dtype)
    return capped if out is None else out.copy_(capped)
