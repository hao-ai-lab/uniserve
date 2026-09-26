"""Standalone local attention over packed or dense numerical inputs."""

from __future__ import annotations

import torch


def attention(
    q, k, v, batch, *, scale: float, window: int | None = None, out=None
):
    """Apply local attention without implicit cache allocation or collectives.

    Paged read-only calls pass physical K/V tensors and write_indices=None.
    Persistent cache writes use Attention bound through ExecutionContext.
    ``window`` bounds the visible history as documented on ``Attention``.
    """
    from uniserve.model.inputs import TextSize
    from uniserve.nn.attention.inputs import DenseInput
    from uniserve.runtime.backends.attention import resolve
    from uniserve.runtime.backends.attention._sequences import host_lengths
    from uniserve.runtime.tensor_buffers import TensorBuffers

    if q.ndim not in {3, 4}:
        raise ValueError("attention requires packed THD or dense BHTD tensors")

    size = (
        TextSize(
            q.shape[0] if q.ndim == 3 else q.shape[0] * q.shape[2],
            1 if q.ndim == 3 else q.shape[0],
        )
        if isinstance(batch, DenseInput)
        else TextSize(q.shape[0], batch.queries.batch_size)
    )
    kv_axis = 1 if k.ndim == 3 or isinstance(batch, DenseInput) else 2
    options = {
        "num_heads": q.shape[1],
        "num_kv_heads": k.shape[kv_axis],
        "head_dim": q.shape[-1],
        "dtype": q.dtype,
        "size": size,
        "cache": None,
        "window": window,
    }

    provider = resolve("auto", device=q.device)
    requirements = provider.workspace_buffers(**options)
    with TensorBuffers.allocate(requirements, device=q.device) as buffers:
        operator = provider.prepare(
            **options, workspace=buffers.view(requirements)
        )
        try:
            if operator.requires_host_lengths(batch):
                batch = host_lengths(batch)
            operator.bind(batch)
            destination = torch.empty_like(q) if out is None else out
            return operator(q, k, v, batch, scale=scale, out=destination)
        finally:
            operator.close()
