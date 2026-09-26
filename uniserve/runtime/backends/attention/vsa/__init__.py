"""Factories and independently bound operators for Video Sparse Attention."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib import import_module

import torch

from uniserve.nn.attention.vsa.inputs import BlockInput, Pattern

from ._rows import _Rows


class Operator:
    """One call site's plans, borrowed scratch and per-scale row producers.

    Block IDs remain live inputs. A concrete operator supplies ``kernel``, the
    block-64 numerical call ``kernel(q, k, v, out, indices, counts,
    valid_sizes, scale=...)`` that evaluates complete calls, and may supply a
    different ``row_kernel`` for row production.
    """

    kernel: Callable[..., object]

    def __init__(
        self,
        pattern,
        *,
        num_heads,
        head_dim,
        dtype,
        workspace,
        transient=None,
    ):
        if min(num_heads, head_dim) < 1 or len(pattern.row_counts) not in (
            1,
            num_heads,
        ):
            raise ValueError(
                "VSA operator requires compatible positive head dimensions"
            )

        self.pattern, self.num_heads, self.head_dim, self.dtype = (
            pattern,
            num_heads,
            head_dim,
            dtype,
        )
        self.workspace, self._closed = workspace, False
        # ``transient(role, requirements, device)`` lends per-call work areas
        # that every operator of the owning context shares; without it the
        # operator allocates its own.
        self.transient = transient
        # Row producers own the query maps they plan, one per softmax scale.
        self._rows: dict[float, _Rows] = {}

    @property
    def row_kernel(self) -> Callable[..., object]:
        """Return the numerical call used for row production."""
        return self.kernel

    def bind(self, batch: BlockInput) -> None:
        """Check a live batch against the prepared pattern before use."""
        if self._closed:
            raise RuntimeError("VSA operator is closed")
        if batch.pattern != self.pattern:
            raise ValueError(
                "VSA block cardinalities differ from the prepared pattern"
            )

    def _validate(self, q, k, v, batch, out):
        """Bind the batch and check every tensor.

        Bind the batch and check every tensor against the prepared
        dimensions.
        """
        self.bind(batch)
        if (
            q.shape
            != (
                len(self.pattern.row_counts[0]) * 64,
                self.num_heads,
                self.head_dim,
            )
            or k.shape != v.shape
            or k.ndim != 3
            or k.shape[1:] != q.shape[1:]
            or k.shape[0] != batch.valid_sizes.numel() * 64
            or out.shape != q.shape
            or out.dtype != q.dtype
            or any(value.dtype != self.dtype for value in (q, k, v))
            or any(
                value.device != q.device
                for value in (
                    k,
                    v,
                    out,
                    batch.block_indices,
                    batch.block_counts,
                    batch.valid_sizes,
                )
            )
            or batch.block_indices.shape[:2]
            != (self.num_heads, q.shape[0] // 64)
        ):
            raise ValueError(
                "VSA tensors disagree with the prepared numerical dimensions"
            )

    def __call__(self, q, k, v, batch, *, scale, out):
        """Evaluate one attention call into ``out``."""
        self._validate(q, k, v, batch, out)
        self.kernel(
            q,
            k,
            v,
            out,
            batch.block_indices,
            batch.block_counts,
            batch.valid_sizes,
            scale=scale,
        )
        return out

    def rows(
        self,
        q,
        k,
        v,
        batch,
        *,
        gate,
        compressed,
        out,
        owners,
        chunk_tokens,
        packed,
        scale,
    ):
        """Return a producer of owner row intervals over one fine attention.

        The first produced interval launches ``row_kernel`` over the complete
        packed query domain; every interval then composes its own rows into
        its transport destinations. ``out`` stays borrowed until the last
        interval has been composed.
        """
        self.bind(batch)
        producer = self._rows.get(scale)
        if producer is None:
            producer = self._rows[scale] = _Rows(
                partial(self.row_kernel, scale=scale), self.transient
            )
        return producer.prepare(
            q,
            k,
            v,
            mask_block_indices=batch.block_indices,
            mask_block_count=batch.block_counts,
            valid_sizes=batch.valid_sizes,
            gate=gate,
            compressed=compressed,
            attention_output=out,
            owners=owners,
            chunk_rows=chunk_tokens,
            packed=packed,
        )

    def close(self):
        """Release plans and borrowed workspace; the operator is not reused."""
        self._closed = True
        self._rows.clear()
        self.workspace = {}


class Backend:
    """Construct operators and their workspace for one concrete VSA provider."""

    operator_class: type[Operator]

    def workspace_buffers(
        self,
        pattern: Pattern,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        """Describe extra device buffers.

        Describe extra device buffers the backend needs beyond caller
        tensors.
        """
        return {}

    def prepare(
        self,
        pattern: Pattern,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        workspace,
        transient=None,
    ):
        """Build an operator bound to one pattern.

        Build an operator bound to one pattern, dtype, and borrowed
        workspace. ``transient`` lends the owning context's shared per-call
        work areas (see ``ExecutionContext.scratch``).
        """
        return self.operator_class(
            pattern,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            workspace=workspace,
            transient=transient,
        )


def resolve(backend, *, device):
    """Return the Backend for a name or instance.

    ``"auto"`` selects the native SM100 kernels (``sm100``) and raises on any
    other device, including a CUDA device without them: FlashInfer's
    block-sparse wrapper runs its FlashAttention-2 templates and the Triton
    kernel is a portable reference, so neither stands in for native
    coverage. Both remain selectable by name.
    """
    if isinstance(backend, Backend):
        return backend
    if backend == "auto":
        native = import_module(f"{__name__}.sm100")
        if native.available(device):
            return native.Backend()
        raise RuntimeError(
            f"no native VSA kernel supports {device}: the SM100 block-sparse "
            "kernels require a compute capability 10.x CUDA device with the "
            "CUTLASS DSL, cuDNN frontend and CUDA driver bindings"
        )

    if backend not in {"sm100", "cute", "flashinfer", "triton"}:
        raise ValueError(f"unknown VSA backend {backend!r}")
    module = import_module(f"{__name__}.{backend}")
    if not module.available(device):
        raise RuntimeError(
            f"VSA backend {backend!r} is unavailable on {device}"
        )
    return module.Backend()
