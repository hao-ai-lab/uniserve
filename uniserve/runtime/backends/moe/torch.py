"""Portable routed-expert evaluation for CPUs and numerical reference."""

from uniserve.nn import functional

from . import Backend as _Backend
from . import Operator as _Operator


class _Torch(_Operator):
    def __call__(self, hidden, topk_ids, topk_weights):
        self._validate(hidden, topk_ids, topk_weights)
        module = self.module
        return functional.fused_moe(
            hidden,
            module.up_gate.weight,
            module.down.weight,
            topk_ids,
            topk_weights,
            activation=module.activation,
            input_quantizers=(
                module.up_gate.input_quantizer,
                module.down.input_quantizer,
            ),
        )


class Backend(_Backend):
    """The eager per-expert reference; it reads expert counts on the host."""

    name = "torch"
    operator_class = _Torch
