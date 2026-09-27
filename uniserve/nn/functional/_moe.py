"""Portable routed expert evaluation over stacked expert weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from uniserve.quantization import QuantizedTensor


@dataclass(frozen=True)
class Routes:
    """A weighted sum of rows: each token's routed expert outputs.

    ``rows`` is ``[T, K, H]`` in a floating dtype and ``weights`` ``[T, K]``
    FP32. The value of token ``t`` is defined as the FP32 fused
    multiply-add chain ``s = fma(rows[t, k], weights[t, k], s)`` over the
    routes ``k = 0 .. K - 1`` in order, starting from ``s = 0``, rounded
    once to the rows' dtype (the combination of TensorRT-LLM's
    ``moeUnpermuteKernel``). Routed experts return their per-route outputs
    in this form so that the consumer of the sum evaluates it while it
    reads it (``sandwich_rms_norm`` does), instead of the sum being stored
    and read back.

    ``Routes(y[:, None], ones[T, 1])`` is an ordinary instance with one
    route of unit weight, and its value is ``y`` exactly, since
    ``fma(y, 1, 0) = y``: experts whose kernels combine routes internally
    return their combined rows that way.
    """

    rows: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self):
        rows, weights = self.rows, self.weights
        if (
            rows.ndim != 3
            or weights.shape != rows.shape[:2]
            or weights.dtype != torch.float32
            or not rows.dtype.is_floating_point
            or weights.device != rows.device
        ):
            raise ValueError(
                "routes are [T, K, H] floating rows with [T, K] FP32 weights "
                "on one device"
            )

    @property
    def shape(self) -> torch.Size:
        """The ``[T, H]`` shape of the value."""
        return torch.Size((self.rows.shape[0], self.rows.shape[2]))

    @property
    def dtype(self) -> torch.dtype:
        return self.rows.dtype

    @property
    def device(self) -> torch.device:
        return self.rows.device

    def combine(self) -> torch.Tensor:
        """Evaluate the value with tensor operations.

        Each step is the FP32 fused multiply-add evaluated through its exact
        FP64 product and sum; kernels that consume routes compute it in
        FP32 directly.
        """
        total = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        for route in range(self.rows.shape[1]):
            total = (
                total.double()
                + self.rows[:, route].double()
                * self.weights[:, route, None].double()
            ).float()
        return total.to(self.dtype)


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
    combine: bool = True,
) -> torch.Tensor | Routes:
    """Evaluate routed experts one selected expert at a time.

    ``hidden`` is ``[T, H]``, dense or already in ``input_quantizers[0]``'s
    encoding; ``up_gate`` is ``[E, 2I, H]`` (up rows then gate rows) and
    ``down`` is ``[E, H, I]``, dense or encoded. ``topk_ids``
    and ``topk_weights`` are ``[T, K]``. ``input_quantizers`` are the static
    activation encodings read by the two projections, or None. Each expert's
    projections run in the hidden dtype and every route's output rounds to
    it; with ``combine`` the result is the value of those :class:`Routes`
    (their weighted sum, accumulated in float32 and rounded once to the
    hidden dtype), without it the routes themselves. The loop
    reads the expert counts on the host, so this path is eager-only: it is
    the portable reference and CPU implementation, not a GPU serving path.
    """
    tokens, width = hidden.shape
    intermediate = down.shape[-1]
    dtype = hidden.dtype
    up_gate_values, down_values = _dense(up_gate, dtype), _dense(down, dtype)
    expert_input = _encoded(hidden, input_quantizers[0])

    # Route rows in token-major route order: row t * K + k is route k of t.
    routes = torch.zeros(
        (tokens * topk_ids.shape[1], width), dtype=dtype, device=hidden.device
    )
    rows = torch.arange(tokens, device=hidden.device).repeat_interleave(
        topk_ids.shape[1]
    )
    experts = topk_ids.reshape(-1).to(torch.int64)
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
        routes[selected] = F.linear(
            _encoded(activated, input_quantizers[1]), down_values[expert]
        ).to(dtype)
    result = Routes(routes.reshape(tokens, -1, width), topk_weights)
    return result.combine() if combine else result
