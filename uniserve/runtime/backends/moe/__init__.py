"""Prepared routed-expert operators borrowing stacked expert weights.

A provider evaluates one ``FusedMoE`` call site: routed tokens of up to the
prepared ``TextSize`` capacity through that module's resident expert
weights. Operators borrow the module's parameters and context-owned
workspace; host planning happens only in ``prepare`` so calls can be
captured in CUDA graphs.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

import torch

from uniserve.model.inputs import TextSize
from uniserve.tensors import BufferConfig

# Native providers in automatic selection order.
_NATIVE = ("cutlass",)


class Operator:
    """One ``FusedMoE`` call site's prepared kernel and borrowed resources."""

    def __init__(self, *, module, size: TextSize, workspace):
        self.module = module
        self.size = size
        self.workspace = workspace
        self._closed = False

    def _validate(self, hidden, topk_ids, topk_weights):
        if self._closed:
            raise RuntimeError("expert operator is closed")
        if hidden.shape[0] > self.size.num_tokens:
            raise ValueError("routed tokens exceed the prepared token capacity")

    def __call__(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True
        self.module = None
        self.workspace = {}


class Backend:
    """Factory for routed-expert operators and their workspace declarations."""

    name: str
    operator_class: type[Operator]

    def supports(self, module) -> bool:
        """Report whether this provider evaluates ``module``'s weights."""
        return True

    def workspace_buffers(
        self, *, module, size: TextSize
    ) -> Mapping[str, BufferConfig]:
        return {}

    def prepare(self, *, module, size: TextSize, workspace) -> Operator:
        return self.operator_class(
            module=module, size=size, workspace=workspace
        )


def resolve(backend: str | Backend, *, module, device: torch.device) -> Backend:
    """Resolve a provider name for ``module`` on ``device``.

    ``auto`` selects a native grouped-expert kernel on a GPU and the
    portable implementation on the CPU. A GPU representation that no native
    kernel covers raises here, at preparation: the portable loop is a
    reference, not a GPU serving path.
    """
    if isinstance(backend, Backend):
        return backend
    if backend == "auto":
        if device.type != "cuda":
            backend = "torch"
        else:
            for name in _NATIVE:
                provider = import_module(f"{__name__}.{name}").Backend()
                if provider.supports(module):
                    return provider
            raise ValueError(
                "no native expert kernel covers this FusedMoE representation "
                f"on {device}"
            )
    if backend not in {"torch", "cutlass"}:
        raise ValueError(f"unknown expert backend {backend!r}")
    provider = import_module(f"{__name__}.{backend}").Backend()
    if not provider.supports(module):
        raise ValueError(
            f"expert backend {backend!r} does not support this representation"
        )
    return provider


def evaluate(module, hidden, topk_ids, topk_weights):
    """Run one standalone call through a provider prepared just for it.

    The call owns its workspace for its duration; execution contexts bind a
    reusable operator instead.
    """
    from uniserve.runtime.tensor_buffers import TensorBuffers

    provider = resolve("auto", module=module, device=hidden.device)
    size = TextSize(hidden.shape[0], 1)
    requirements = provider.workspace_buffers(module=module, size=size)
    with TensorBuffers.allocate(requirements, device=hidden.device) as buffers:
        operator = provider.prepare(
            module=module, size=size, workspace=buffers.view(requirements)
        )
        try:
            return operator(hidden, topk_ids, topk_weights)
        finally:
            operator.close()
