"""Factories and independently bound operators for Video Sparse Attention."""

from __future__ import annotations

from importlib import import_module

import torch

from uniserve.nn.attention.vsa.inputs import BlockInput, Pattern


class Operator:
    """One call site's plans and borrowed scratch; block IDs remain live inputs."""

    def __init__(self, pattern, *, num_heads, head_dim, dtype, workspace):
        if min(num_heads, head_dim) < 1 or len(pattern.row_counts) not in (1, num_heads):
            raise ValueError("VSA operator requires compatible positive head dimensions")
        self.pattern, self.num_heads, self.head_dim, self.dtype = (
            pattern,
            num_heads,
            head_dim,
            dtype,
        )
        self.workspace, self._closed = workspace, False

    def bind(self, batch: BlockInput) -> None:
        if self._closed:
            raise RuntimeError("VSA operator is closed")
        if batch.pattern != self.pattern:
            raise ValueError("VSA block cardinalities differ from the prepared pattern")

    def _validate(self, q, k, v, batch, out):
        self.bind(batch)
        if (
            q.shape != (len(self.pattern.row_counts[0]) * 64, self.num_heads, self.head_dim)
            or k.shape != v.shape
            or k.ndim != 3
            or k.shape[1:] != q.shape[1:]
            or k.shape[0] != batch.valid_sizes.numel() * 64
            or out.shape != q.shape
            or out.dtype != q.dtype
            or any(value.dtype != self.dtype for value in (q, k, v))
            or any(
                value.device != q.device
                for value in (k, v, out, batch.block_indices, batch.block_counts, batch.valid_sizes)
            )
            or batch.block_indices.shape[:2] != (self.num_heads, q.shape[0] // 64)
        ):
            raise ValueError("VSA tensors disagree with the prepared numerical dimensions")

    def __call__(self, q, k, v, batch, *, scale, out):
        raise NotImplementedError

    def close(self):
        self._closed = True
        self.workspace = {}


class Backend:
    operator_class: type[Operator]

    def workspace_buffers(
        self, pattern: Pattern, *, num_heads: int, head_dim: int, dtype: torch.dtype
    ):
        return {}

    def prepare(
        self, pattern: Pattern, *, num_heads: int, head_dim: int, dtype: torch.dtype, workspace
    ):
        return self.operator_class(
            pattern, num_heads=num_heads, head_dim=head_dim, dtype=dtype, workspace=workspace
        )


def resolve(backend, *, device):
    if isinstance(backend, Backend):
        return backend
    if backend == "auto":
        for name in ("sm100", "flashinfer", "triton"):
            candidate = import_module(f"{__name__}.{name}")
            if candidate.available(device):
                return candidate.Backend()
        raise RuntimeError(f"no installed VSA backend supports {device}")
    if backend not in {"sm100", "cute", "flashinfer", "triton"}:
        raise ValueError(f"unknown VSA backend {backend!r}")
    module = import_module(f"{__name__}.{backend}")
    if not module.available(device):
        raise RuntimeError(f"VSA backend {backend!r} is unavailable on {device}")
    return module.Backend()
