"""VSA operators and projected input backing for one layer call site."""

from __future__ import annotations

from uniserve.tensors import BufferConfig

from ..backends import record_kernel_choice
from . import capturing


class VsaBinding:
    """Own one VSA call site's plans and packed input backing.

    Own one VSA call site's plans, mutable query maps, and packed input
    backing.
    """

    def __init__(self, backend, transient, scratch, shared_buffers, exchange):
        # ``transient(role, requirements, device)`` lends the context's
        # shared per-call work areas (``ExecutionContext.scratch``);
        # ``scratch`` allocates the operators' own workspace.
        self.backend, self.transient, self.scratch = backend, transient, scratch
        self._shared_buffers, self.exchange = shared_buffers, exchange
        self.operators, self._buffers = {}, {}
        # Provider name of each prepared operator key.
        self.providers = {}

    def prepare(self, pattern, q):
        key = (q.device, q.dtype, q.shape[1], q.shape[2], pattern)
        if key not in self.operators:
            if capturing(q.device):
                raise RuntimeError(
                    "prepare VSA numerical shapes before capture"
                )
            from ..backends.attention import vsa

            provider = vsa.resolve(self.backend, device=q.device)
            options = {
                "num_heads": q.shape[1],
                "head_dim": q.shape[2],
                "dtype": q.dtype,
            }
            requirements = provider.workspace_buffers(pattern, **options)
            self.operators[key] = provider.prepare(
                pattern,
                **options,
                workspace=self.scratch(requirements, q.device),
                transient=self.transient,
            )
            self.providers[key] = provider.name
            record_kernel_choice()

        return self.operators[key]

    def kernels(self):
        """Describe the block-sparse kernel of each prepared query shape.

        One record per prepared (device, dtype, heads, head dimension,
        pattern): the provider name, local heads, head dimension and dtype.
        """
        return [
            {
                "op": "vsa",
                "provider": name,
                "heads": heads,
                "head_dim": head_dim,
                "dtype": str(dtype).removeprefix("torch."),
            }
            for (_, dtype, heads, head_dim, _), name in self.providers.items()
        ]

    def buffers(self, requirements, device):
        key = (device, tuple(requirements.items()))
        if key not in self._buffers:
            if capturing(device):
                raise RuntimeError(
                    "prepare VSA projected input backing before capture"
                )
            self._buffers[key] = self._shared_buffers(
                {
                    name: BufferConfig(shape, dtype)
                    for name, (shape, dtype) in requirements.items()
                },
                device,
            )

        return self._buffers[key]

    def close(self):
        for operator in self.operators.values():
            operator.close()
        self.operators.clear()
        self.providers.clear()
        self._buffers.clear()
