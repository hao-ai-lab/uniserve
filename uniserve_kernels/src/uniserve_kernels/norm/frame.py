"""Per-frame group normalization, SiLU and causal padding in one pass.

A causal video convolution after a pre-activation reads
``pad(silu(group_norm(frame)))`` for every frame of ``[batch, channels,
frames, height, width]`` values. :func:`normalize_pad` computes the moments of
every frame's channel groups straight from the values and writes the padded,
normalized and activated input in one further pass, into channel-first or
channels-last storage. The result equals PyTorch's composition (a
frame-folding copy, ``F.group_norm``, ``F.silu`` and the causal padding) bit
for bit: the native module :mod:`uniserve_kernels.norm._frame` reproduces
PyTorch's CUDA moments reduction, normalization branches and SiLU expression
(see ``csrc/frame.cu``).

The kernels back ``uniserve.nn.functional.frame_norm_pad``, which validates
the padding, allocates the output and raises on CUDA whenever
:func:`unsupported` reports a reason; the launcher does not revalidate.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import records_autograd


def _extension():
    # Imported on first use: a CPU build of the package has no native module.
    from uniserve_kernels.norm import _frame

    return _frame


def unsupported(
    values: torch.Tensor,
    out: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> str | None:
    """Return why the kernels cannot normalize ``values`` into ``out``.

    The values are CUDA float32 whose frames store each channel's
    ``height * width`` pixels contiguously, such as channel-first or
    frame-major storage; the output is contiguous or channels-last float32
    storage of the values' batch and channels; ``weight`` and ``bias``, when
    present, are contiguous float32 per-channel vectors. Returns ``None``
    when every condition holds.
    """
    operands = (values, out, weight, bias)
    present = tuple(tensor for tensor in operands if tensor is not None)
    if any(tensor.device != values.device for tensor in present):
        return "operands reside on different devices"
    if values.device.type != "cuda":
        return f"operands reside on {values.device}, not a CUDA device"
    if records_autograd(*present):
        return (
            "autograd is recording operands that require grad, and the "
            "kernel defines no backward; call under torch.inference_mode()"
        )
    if any(tensor.dtype != torch.float32 for tensor in present):
        return "the kernels normalize float32 values"
    if values.ndim != 5 or out.ndim != 5:
        return "values and output are not [batch, channels, frames, h, w]"
    if values.stride(4) != 1 or values.stride(3) != values.shape[4]:
        return "a frame's pixels of one channel are not contiguous"
    if not (
        out.is_contiguous()
        or out.is_contiguous(memory_format=torch.channels_last_3d)
    ):
        return "the output is neither contiguous nor channels-last"
    if out.shape[:2] != values.shape[:2]:
        return "the output's batch and channels differ from the input's"
    for parameter in (weight, bias):
        if parameter is not None and (
            parameter.shape != (values.shape[1],)
            or not parameter.is_contiguous()
        ):
            return "normalization parameters are not per-channel vectors"
    return None


def normalize_pad(
    values: torch.Tensor,
    out: torch.Tensor,
    *,
    groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    padding: tuple[int, int, int],
    reflect: bool,
) -> None:
    """Write ``values`` normalized, activated and padded into ``out``.

    ``padding`` is ``(top, left, front)``: the rows above, columns left of
    and zero frames before the source within ``out``; the remaining rows and
    columns of ``out`` pad the bottom and right, reflected or replicated.
    """
    extension = _extension()
    mean, rstd = extension.frame_moments(values, groups, eps)
    top, left, front = padding
    extension.frame_norm_pad(
        values, mean, rstd, weight, bias, out, groups, top, left, front, reflect
    )
