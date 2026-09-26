"""Routed-expert operators and workspace for one ``FusedMoE`` call site."""

from __future__ import annotations

from uniserve.model.inputs import TextSize
from uniserve.quantization import QuantizedTensor, RowOrder

from ..backends import moe as moe_backend
from . import capturing


class MoEBinding:
    """Specialize one ``FusedMoE`` call site for a token capacity.

    The operator is prepared for the largest token count seen, never during
    CUDA capture; a smaller call reuses it.
    """

    def __init__(self, module, backend, size, device, allocate):
        self.module, self.backend, self.device = module, backend, device
        self.allocate = allocate
        self.size = size
        self.operator = None
        self.provider = None

    def prepare(self, size: TextSize):
        previous = self.operator
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
        return operator

    def kernels(self):
        """Describe the grouped-expert kernel serving this call site.

        The record names the prepared provider, the expert count, and each
        projection's weight representation and resident row order (the
        physical order the provider placed; ``linear`` is logical order).
        Empty until the call site is prepared.
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
        return [
            {
                "op": "moe",
                "provider": self.provider,
                "experts": self.module.num_experts,
                "activation": self.module.activation,
                **projections,
            }
        ]

    def __call__(self, hidden, topk_ids, topk_weights):
        operator = self.prepare(TextSize(hidden.shape[0], 1))
        return operator(hidden, topk_ids, topk_weights)

    def close(self):
        if self.operator is not None:
            self.operator.close()
        self.operator = self.provider = None
