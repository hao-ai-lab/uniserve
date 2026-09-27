"""Portable routed expert evaluation over stacked expert weights."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from uniserve.quantization import QuantizedTensor


def _dense(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (
        weight.dequantize(dtype=dtype)
        if isinstance(weight, QuantizedTensor)
        else weight.to(dtype)
    )


def _encoded(value: torch.Tensor, quantizer) -> torch.Tensor:
    """Round activations through their static encoding, as kernels read them.

    Activations already in that encoding decode as they are stored.
    """
    if isinstance(value, QuantizedTensor):
        return value.dequantize()
    return value if quantizer is None else quantizer.round_trip(value)


def topk_softmax(
    scores: torch.Tensor,
    k: int,
    *,
    renormalize: bool,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route tokens to their ``k`` most probable experts.

    ``scores`` is ``[tokens, experts]``. Returns int32 ``ids`` and FP32
    ``weights``, both ``[tokens, k]`` in descending probability: the full
    FP32 softmax of the scores, its top-k, divided by their sum (clamped
    below by the FP32 epsilon) when ``renormalize`` is set, then multiplied
    by ``scale[id]`` (an ``[experts]`` vector) when given.
    """
    from uniserve_kernels.triton import require_kernel

    if scores.ndim != 2 or not 0 < k <= scores.shape[-1]:
        raise ValueError("routing requires [tokens, experts] scores and k")
    if scale is not None and scale.shape != scores.shape[-1:]:
        raise ValueError("the expert scale must hold one value per expert")

    if scores.is_cuda:
        from uniserve_kernels import routing

        require_kernel(
            "topk_softmax",
            routing.unsupported(scores, k, scale),
            scores=scores,
            scale=scale,
        )
        ids = torch.empty(
            (scores.shape[0], k), dtype=torch.int32, device=scores.device
        )
        weights = torch.empty(
            (scores.shape[0], k), dtype=torch.float32, device=scores.device
        )
        routing.topk_softmax(scores, k, renormalize, scale, ids, weights)
        return ids, weights

    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    weights, ids = torch.topk(probabilities, k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
    if scale is not None:
        weights = weights * scale.index_select(0, ids.reshape(-1)).reshape(
            ids.shape
        )
    return ids.to(torch.int32), weights


def fused_moe(
    hidden: torch.Tensor,
    up_gate: torch.Tensor,
    down: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    input_quantizers=(None, None),
) -> torch.Tensor:
    """Evaluate routed experts one selected expert at a time.

    ``hidden`` is ``[T, H]``, dense or already in ``input_quantizers[0]``'s
    encoding; ``up_gate`` is ``[E, 2I, H]`` (up rows then gate rows) and
    ``down`` is ``[E, H, I]``, dense or encoded. ``topk_ids``
    and ``topk_weights`` are ``[T, K]``. ``input_quantizers`` are the static
    activation encodings read by the two projections, or None. Each expert's
    projections run in the hidden dtype; the weighted combination
    accumulates in float32 and rounds once to the hidden dtype. The loop
    reads the expert counts on the host, so this path is eager-only: it is
    the portable reference and CPU implementation, not a GPU serving path.
    """
    tokens, width = hidden.shape
    intermediate = down.shape[-1]
    dtype = hidden.dtype
    up_gate_values, down_values = _dense(up_gate, dtype), _dense(down, dtype)
    expert_input = _encoded(hidden, input_quantizers[0])

    output = torch.zeros(
        (tokens, width), dtype=torch.float32, device=hidden.device
    )
    rows = torch.arange(tokens, device=hidden.device).repeat_interleave(
        topk_ids.shape[1]
    )
    experts = topk_ids.reshape(-1).to(torch.int64)
    weights = topk_weights.reshape(-1)
    for expert in torch.unique(experts).tolist():
        selected = (experts == expert).nonzero().squeeze(1)
        token_rows = rows[selected]
        projected = F.linear(expert_input[token_rows], up_gate_values[expert])
        up, gate = projected.split(intermediate, dim=-1)
        activated = (
            F.silu(gate)
            if activation == "silu"
            else F.gelu(gate, approximate="tanh")
        ) * up
        result = F.linear(
            _encoded(activated, input_quantizers[1]), down_values[expert]
        )
        output.index_add_(
            0, token_rows, result.float() * weights[selected, None]
        )
    return output.to(dtype)
