"""Softmax top-k expert routing in one launch (CUDA).

The kernel in ``csrc/topk_softmax.cu`` backs
``uniserve.nn.functional.topk_softmax``, which defines the numbers with its
tensor composition, allocates the outputs and raises on CUDA with the reason
:func:`unsupported` reports. One warp routes one token and reproduces the
composition's FP32 arithmetic and order: PyTorch's warp softmax over the
scores, the k largest probabilities in descending order with equal ones in
the order ``torch.topk`` gives them (its gather order sorted by its bitonic
network), their renormalization by PyTorch's butterfly row sum clamped below
by the FP32 epsilon, and an optional per-expert scale. Outputs are int32
expert ids and FP32 weights, both ``[tokens, k]``.

The extension builds with :func:`uniserve_kernels.jit.load` on first use;
:func:`unsupported` never compiles. Launches synchronize nothing, so CUDA
graphs capture them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

#: Largest top-k: the picks and their renormalizing butterfly stay in one
#: warp's registers.
MAX_TOP_K = 8
#: Largest expert count one warp holds (eight scores per lane).
MAX_EXPERTS = 256
_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def unsupported(
    scores: torch.Tensor, k: int, scale: torch.Tensor | None
) -> str | None:
    """Return why the kernel cannot route ``scores``, or ``None``.

    ``scores`` are FP32, BF16 or FP16 ``[tokens, experts]`` rows with unit
    expert stride and at most :data:`MAX_EXPERTS` experts, ``k`` is at most
    :data:`MAX_TOP_K`, and ``scale`` is a floating ``[experts]`` vector of
    one of those dtypes on the same device.
    """
    if scores.device.type != "cuda":
        return f"scores reside on {scores.device}, not a CUDA device"
    if torch.is_grad_enabled() and scores.requires_grad:
        return (
            "autograd is recording scores that require grad, and the kernel "
            "defines no backward; call under torch.inference_mode()"
        )
    if scores.dtype not in _DTYPES:
        return f"score dtype {scores.dtype} is not float32, bfloat16 or float16"
    if scores.ndim != 2 or (scores.shape[1] > 1 and scores.stride(1) != 1):
        return "scores are not [tokens, experts] rows with unit expert stride"
    if scores.shape[1] > MAX_EXPERTS:
        return f"{scores.shape[1]} experts exceed {MAX_EXPERTS}"
    if k > MAX_TOP_K:
        return f"top-k {k} exceeds {MAX_TOP_K}"
    if scale is not None and (
        scale.device != scores.device
        or scale.dtype not in _DTYPES
        or scale.shape != (scores.shape[1],)
        or not scale.is_contiguous()
    ):
        return "the expert scale is not one contiguous [experts] vector"
    return None


@lru_cache(maxsize=1)
def _extension():
    from uniserve_kernels import jit

    return jit.load(
        "uniserve_topk_softmax",
        [Path(__file__).parent / "csrc" / "topk_softmax.cu"],
        cuda_flags=("-O3", "-std=c++20"),
    )


def load() -> None:
    """Compile or load the cached extension, for example before capture."""
    _extension()


def topk_softmax(
    scores: torch.Tensor,
    k: int,
    renormalize: bool,
    scale: torch.Tensor | None,
    ids: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Route every row of ``scores`` into contiguous ``ids`` and ``weights``.

    ``ids`` is int32 and ``weights`` FP32, both ``[tokens, k]``. Callers
    first check :func:`unsupported`.
    """
    _extension().topk_softmax(scores, k, renormalize, scale, ids, weights)
