"""Shared VSA numerical plans and projected backing for a layer call site."""

from __future__ import annotations

from uniserve.tensors import BufferConfig

from . import capturing


class VsaBinding:
    """Borrow the context's VSA plans and retain call-site input backing.

    Every operator is bound only to numerical dimensions and a block-count
    pattern. Serialized layers with the same signature can share its plans;
    the live block maps and projected rows remain inputs of each invocation.
    """

    def __init__(
        self, backend, transient, scratch, shared_buffers, exchange, operators
    ):
        # ``transient(role, requirements, device)`` lends the context's
        # shared per-call work areas (``ExecutionContext.scratch``);
        # ``scratch`` allocates the operators' own workspace.
        self.backend, self.transient, self.scratch = backend, transient, scratch
        self._shared_buffers, self.exchange = shared_buffers, exchange
        self.operators, self._buffers = operators, {}

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
            self.operators[key] = provider.prepare(
                pattern,
                **options,
                workspace=self.scratch(requirements, q.device),
                transient=self.transient,
            )

        return self.operators[key]

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
        # The execution context retires shared operators once every call site
        # and graph has finished borrowing them.
        self._buffers.clear()
