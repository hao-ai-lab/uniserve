"""Routed mixture-of-experts over stacked expert weights."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from uniserve.distributed import Communicator
from uniserve.nn import _binding, functional
from uniserve.quantization import Quantizer

__all__ = ["ExpertLinear", "FusedMoE", "TopK"]

Activation = Literal["silu", "gelu_tanh"]


class TopK(nn.Module):
    """Select the highest-probability experts of each token.

    Returns ``(topk_ids int32 [T, K], topk_weights fp32 [T, K])``: a full
    softmax over the router scores followed by top-k, with the selected
    weights renormalized to sum to one when ``renormalize`` is set.
    """

    def __init__(self, k: int, *, renormalize: bool = True) -> None:
        super().__init__()
        if type(k) is not int or k <= 0:
            raise ValueError("expert top-k must be a positive integer")
        self.k = k
        self.renormalize = renormalize

    def forward(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(probabilities, self.k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).eps
            )
        return ids.to(torch.int32), weights


class ExpertLinear(nn.Module):
    """One projection stacked over experts: ``weight [E, out, in]``.

    The module owns its weight's representation and the static activation
    encoding applied to its inputs (``input_quantizer``), exactly as
    ``Linear`` does for one matrix. Tensor-parallel binding narrows the
    output or input channels of every expert alike.
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if min(num_experts, in_features, out_features) < 1:
            raise ValueError("expert projections require positive extents")
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                num_experts,
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.input_quantizer: Quantizer | None = None


class FusedMoE(nn.Module):
    """Routed expert FFN over stacked expert weights.

    ``up_gate.weight`` is ``[E, 2I, H]`` with each expert's up rows followed
    by its gate rows; ``down.weight`` is ``[E, H, I]``. These are logical
    orders: an encoded weight may store each expert's rows in the physical
    ``RowOrder`` its prepared kernel reads, which the runtime provider
    places once and every consumer decodes. ``forward(hidden [T, H],
    topk_ids [T, K] int32, topk_weights [T, K] fp32) -> [T, H]`` computes

        sum_k topk_weights[t, k] * down_e(act(gate_e(x_t)) * up_e(x_t)),
        e = topk_ids[t, k],

    where ``act`` is SiLU or tanh-approximated GELU. Routing belongs to the
    model; this module consumes its result. Tensor-parallel binding splits
    ``I`` across the tensor-parallel group and sums the partial outputs once
    after combining the experts. An active ``ExecutionContext`` supplies a
    prepared expert kernel. A standalone call prepares a native kernel for
    itself on a GPU and evaluates the portable reference on the CPU; a GPU
    representation no native kernel covers is an error, not a slower path.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        top_k: int,
        activation: Activation,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"unsupported expert activation {activation!r}")
        if type(top_k) is not int or not 0 < top_k <= num_experts:
            raise ValueError(
                "expert top-k must select between one and all experts"
            )
        self.num_experts = num_experts
        # Routes per token; it sizes kernel workspace, so calls must match it.
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation: Activation = activation
        self.up_gate = ExpertLinear(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            device=device,
            dtype=dtype,
        )
        self.down = ExpertLinear(
            num_experts,
            intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        # Tensor-parallel binding records this rank's interval of I, in the
        # global intermediate coordinate, and the group whose partial sums
        # complete each output.
        self.intermediate_slice = slice(0, intermediate_size)
        self.group = Communicator()
        self.communication_groups = (self.group,)

    def forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hidden.ndim != 2
            or topk_ids.shape != topk_weights.shape
            or topk_ids.ndim != 2
            or topk_ids.shape != (hidden.shape[0], self.top_k)
            or topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
        ):
            raise ValueError(
                "expert routing requires [tokens, hidden] states with int32 "
                "ids and fp32 weights of shape [tokens, top_k]"
            )
        operator = _binding.moe.get().get(id(self))
        if operator is not None:
            result = operator(hidden, topk_ids, topk_weights)
        elif hidden.device.type == "cpu":
            result = functional.fused_moe(
                hidden,
                self.up_gate.weight,
                self.down.weight,
                topk_ids,
                topk_weights,
                activation=self.activation,
                input_quantizers=(
                    self.up_gate.input_quantizer,
                    self.down.input_quantizer,
                ),
            )
        else:
            from uniserve.runtime.backends.moe import evaluate

            result = evaluate(self, hidden, topk_ids, topk_weights)
        if self.group.size > 1:
            self.group.all_reduce(result)
        return result
