"""Shared VSA numerical plans for a layer call site."""

from __future__ import annotations

from ..backends import record_kernel_choice
from . import capturing


class VsaBinding:
    """Borrow the context's VSA plans for one call site.

    Every operator is bound only to numerical dimensions and a block-count
    pattern. Serialized layers with the same signature can share its plans;
    the live block maps and projected rows remain inputs of each invocation.
    """

    def __init__(self, backend, transient, scratch, exchange, operators):
        # ``transient(role, requirements, device)`` lends the context's
        # shared per-call work areas (``ExecutionContext.scratch``);
        # ``scratch`` allocates the operators' own workspace.
        self.backend, self.transient, self.scratch = backend, transient, scratch
        self.exchange = exchange
        self.operators = operators
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

            provider = vsa.resolve(
                self.backend, device=q.device, tile=pattern.tile
            )
            options = {
                "num_heads": q.shape[1],
                "head_dim": q.shape[2],
                "dtype": q.dtype,
            }
            requirements = provider.workspace_buffers(pattern, **options)
            operator = provider.prepare(
                pattern,
                **options,
                workspace=self.scratch(requirements, q.device),
                transient=self.transient,
            )
            self.operators[key] = (provider.name, operator)

        provider_name, operator = self.operators[key]
        if key not in self.providers:
            self.providers[key] = provider_name
            record_kernel_choice()

        return operator

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

    def close(self):
        # The execution context retires shared operators once every call site
        # and graph has finished borrowing them.
        self.providers.clear()
