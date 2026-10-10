"""Per-channel bias addition with an optional residual, channel-first."""

from __future__ import annotations

import torch
from uniserve_kernels.triton import require_kernel


def bias_add(
    values: torch.Tensor,
    bias: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a per-channel bias, then an optional residual.

    ``values`` are ``[batch, channels, *spatial]`` and ``bias`` holds one
    value per channel. Returns ``values + bias``, or with a ``residual`` of
    the values' shape ``residual + (values + bias)``; ``residual_bias``,
    another per-channel vector, first adds to the residual: ``(residual +
    residual_bias) + (values + bias)``. Every sum rounds once in the
    operands' dtype in this grouping, as separate tensor additions round
    it, so the result equals those additions bit for bit. The result is a
    new contiguous (channel-first) tensor.

    A cuDNN convolution's bias added here instead of by the convolution,
    as in ``conv(x, weight) + bias``, gives the biased convolution bit for
    bit: PyTorch adds that bias in a separate pass after the convolution. A
    convolution of channels-last input computes channels-last values; this
    addition stores them channel-first in the same pass, and adds a
    residual block's shortcut with them.

    On CUDA one kernel adds float32 operands, reading each once and
    writing the result; ``values`` and ``residual`` may use any strides
    under which their spatial axes merge into one strided axis, such as
    contiguous or channels-last storage. Other devices evaluate the tensor
    additions.

    Raises:
        ValueError: when the shapes or dtypes disagree, or on CUDA when the
            kernel cannot take the operands.
    """
    if values.ndim < 2:
        raise ValueError("bias addition takes [batch, channels, *spatial]")
    channels = values.shape[1]
    if bias.shape != (channels,):
        raise ValueError("the bias holds one value per channel")
    if residual is None and residual_bias is not None:
        raise ValueError("a residual bias requires a residual")
    if residual is not None and residual.shape != values.shape:
        raise ValueError("the residual must match the values' shape")
    if residual_bias is not None and residual_bias.shape != (channels,):
        raise ValueError("the residual bias holds one value per channel")
    operands = (values, bias, residual, residual_bias)
    if any(
        tensor is not None and tensor.dtype != values.dtype
        for tensor in operands
    ):
        raise ValueError("bias addition operands share one dtype")

    if values.is_cuda:
        from uniserve_kernels import bias as kernels

        require_kernel(
            "bias_add",
            kernels.unsupported(values, bias, residual, residual_bias),
            values=values,
            bias=bias,
            residual=residual,
            residual_bias=residual_bias,
        )
        out = torch.empty(
            values.shape, dtype=values.dtype, device=values.device
        )
        kernels.add(
            values,
            bias,
            out,
            residual=residual,
            residual_bias=residual_bias,
        )
        return out

    # Biases broadcast along every axis but the channels.
    shape = (channels,) + (1,) * (values.ndim - 2)
    result = values + bias.view(shape)
    if residual is not None:
        if residual_bias is not None:
            residual = residual + residual_bias.view(shape)
        result = residual + result
    return result.contiguous()
