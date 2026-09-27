"""Routed-expert operators and workspace for one ``FusedMoE`` call site."""

from __future__ import annotations

import torch

from uniserve.model.inputs import TextSize
from uniserve.nn.functional import Routes
from uniserve.quantization import QuantizedTensor, RowOrder

from ..backends import moe as moe_backend
from ..backends import record_kernel_choice
from . import capturing


class MoEBinding:
    """Specialize one ``FusedMoE`` call site for a token capacity.

    The operator is prepared for the largest token count seen, never during
    CUDA capture; a smaller call reuses it.

    An expert-parallel call site (``module.expert_group`` spans several
    ranks) borrows the worker's ``ExpertExchange``: each call sends its
    tokens to the ranks holding their experts, runs this rank's experts over
    the tokens it receives, ``P * C`` rows for a group of ``P`` ranks and the
    open step's capacity ``C``, and combines the partial sums back. Its
    local operator is prepared once for ``P`` times the exchange's largest
    capacity. Hidden states of NVFP4 experts travel in the experts' input
    encoding, dense hidden states otherwise.
    """

    def __init__(self, module, backend, size, device, allocate, exchange=None):
        self.module, self.backend, self.device = module, backend, device
        self.allocate = allocate
        self.size = size
        self.operator = None
        self.provider = None
        self.exchange = exchange if module.expert_group.size > 1 else None
        if module.expert_group.size > 1 and exchange is None:
            raise ValueError(
                "an expert-parallel call site needs the worker's expert "
                "exchange"
            )
        # Unit route weights of an uncombined expert-parallel call, filled
        # once for the exchange's largest local token count.
        self._unit = (
            None
            if self.exchange is None
            else torch.ones(
                (self.exchange.max_tokens, 1),
                dtype=torch.float32,
                device=device,
            )
        )

    def prepare(self, size: TextSize):
        previous = self.operator
        if self.exchange is not None:
            # Received rows, not local tokens, reach the local experts.
            size = TextSize(
                self.module.expert_group.size * self.exchange.max_tokens, 1
            )
        if previous is not None and previous.size.num_tokens >= size.num_tokens:
            return previous
        if capturing(self.device):
            raise RuntimeError(
                "expert capacity must be prepared before capture"
            )
        if self.size is not None:
            size = TextSize(
                max(size.num_tokens, self.size.num_tokens),
                max(size.batch_size, self.size.batch_size),
            )
        provider = moe_backend.resolve(
            self.backend, module=self.module, device=self.device
        )
        requirements = provider.workspace_buffers(module=self.module, size=size)
        operator = provider.prepare(
            module=self.module,
            size=size,
            workspace=self.allocate(requirements, self.device),
        )
        if previous is not None:
            previous.close()
        self.operator = operator
        self.provider = provider.name
        self._invalid_expert = provider.invalid_expert(self.module)
        record_kernel_choice()
        return operator

    def kernels(self):
        """Describe the grouped-expert kernel serving this call site.

        The record names the prepared provider, the expert count, and each
        projection's weight representation and resident row order (the
        physical order the provider placed; ``linear`` is logical order).
        An expert-parallel call site also names its expert group size, its
        all-to-all exchange and the representation its hidden states travel
        in. Empty until the call site is prepared.
        """
        if self.operator is None:
            return []
        projections = {}
        for name in ("up_gate", "down"):
            weight = getattr(self.module, name).weight
            quantized = isinstance(weight, QuantizedTensor)
            projections[name] = {
                "weight": weight.quantizer.format
                if quantized
                else str(weight.dtype).removeprefix("torch."),
                "row_order": weight.row_order.value
                if quantized
                else RowOrder.LINEAR.value,
            }
        parallel = (
            {}
            if self.exchange is None
            else {
                "expert_parallel": self.module.expert_group.size,
                "exchange": "flashinfer_mnnvl_alltoall",
                # The representation hidden states travel in.
                "payload": "nvfp4"
                if isinstance(self.operator, moe_backend.NVFP4Operator)
                else "dense",
            }
        )
        return [
            {
                "op": "moe",
                "provider": self.provider,
                "experts": self.module.num_experts,
                "activation": self.module.activation,
                **projections,
                **parallel,
            }
        ]

    def __call__(self, hidden, topk_ids, topk_weights, *, combine=True):
        exchange = self.exchange
        if exchange is None:
            operator = self.prepare(TextSize(hidden.shape[0], 1))
            return operator(hidden, topk_ids, topk_weights, combine=combine)

        operator = self.prepare(TextSize(hidden.shape[0], 1))
        if isinstance(operator, moe_backend.NVFP4Operator):
            # NVFP4 experts read their input encoding, so the hidden states
            # travel in it: rows already stored in it as they are, BF16
            # rows (a join's empty rows included) encoded here to the bytes
            # the operator would encode after the exchange. The
            # representation is thus a property of the layer, the same on
            # every rank of the step, including ranks that only join it.
            hidden = operator.encode(hidden)
        received = exchange.dispatch(
            id(self.module),
            hidden,
            topk_ids,
            topk_weights,
            invalid_expert=self._invalid_expert,
        )
        # The local experts combine their routes into this rank's partial
        # sums, and the exchange sums the ranks' partials (FP32, one BF16
        # rounding); uncombined, those combined rows are the one-route
        # Routes of unit weight.
        output = exchange.combine(operator(*received), hidden.shape[0])
        if combine:
            return output
        assert self._unit is not None
        return Routes(output.unsqueeze(1), self._unit[: output.shape[0]])

    def join(self, hidden_size: int, dtype: torch.dtype) -> None:
        """Exchange at this layer with no tokens of this rank's own.

        The other ranks' tokens routed to this rank's experts still arrive,
        run through them, and return; this rank contributes and receives
        nothing for itself. A rank joins the expert layers a step's forward
        did not reach, so every rank exchanges at every layer of every step.
        """
        top_k = self.module.top_k
        self(
            torch.empty((0, hidden_size), dtype=dtype, device=self.device),
            torch.empty((0, top_k), dtype=torch.int32, device=self.device),
            torch.empty((0, top_k), dtype=torch.float32, device=self.device),
        )

    def close(self):
        if self.operator is not None:
            self.operator.close()
        self.operator = self.provider = None
        self._unit = None
