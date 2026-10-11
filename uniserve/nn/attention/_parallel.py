"""Attention communication composed around typed local compute calls."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from uniserve.distributed.mesh import DeviceMesh

from .config import AttentionParallelConfig


@dataclass(frozen=True)
class AttentionBuffers:
    """Fixed-capacity K/V transport storage, separate from sparse compute.

    Gather storage holds a compact replicated key domain: every owner's active
    rows, in topology order, in this rank's own storage.
    """

    key: torch.Tensor
    value: torch.Tensor
    local_key: torch.Tensor
    local_value: torch.Tensor


class ParallelAttention(torch.nn.Module):
    """Exchange attention tensors independently of the numerical backend.

    Ulysses partitions heads over the complete sequence. A context binding
    gathers every owner's K/V into each rank, which is what an attention
    kernel reads fastest: the copy engines move the rows once, where reading
    a peer's storage inside the kernel pays for every tile it touches. Compute
    backends retain their mask, selection and softmax semantics. The runtime
    owns communication storage.
    """

    def __init__(
        self,
        *,
        mesh: DeviceMesh,
        parallel: AttentionParallelConfig = AttentionParallelConfig(),
    ) -> None:
        torch.nn.Module.__init__(self)
        # The mathematical partition this exchange realizes.
        self.mesh, self.parallel = mesh, parallel

        heads = () if parallel.heads is None else (parallel.heads.axis,)
        context = parallel.context
        context_axes: tuple[str, ...] = (
            () if context is None else (context.gather_axis,)
        )
        # Validate before ordering by topology: misspelled axes must not
        # silently disappear, and flattened token order follows the declared
        # mesh.
        mesh.size((*heads, *context_axes))
        context_axes = tuple(axis for axis in mesh.axes if axis in context_axes)

        self.ulysses_group = mesh.get_group(heads)
        self.context_group = mesh.get_group(context_axes)
        self.communication_groups = (self.ulysses_group, self.context_group)

    @property
    def context_buffers(self) -> AttentionBuffers | None:
        """Borrow this invocation's context tensors from the caller's scope."""
        if self.context_group.size == 1:
            return None
        buffers = _CONTEXT.get().get(self)
        if buffers is None:
            raise RuntimeError(
                "context attention requires bound numerical buffers"
            )
        return buffers

    def distribute_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Publish this owner's K/V and return the whole context's rows.

        Every owner contributes its own rows and receives all of them, so the
        returned views cover the active rows of the complete context in
        topology order and live in this rank's own storage for the attention
        call to read.
        """
        context = self.context_group
        if context.size == 1:
            return key, value

        workspace = self.context_buffers
        assert workspace is not None
        if (
            key.ndim != 3
            or key.shape != value.shape
            or key.shape[1:] != workspace.local_key.shape[1:]
            or not 0 < key.shape[0] <= workspace.local_key.shape[0]
            or key.dtype != workspace.key.dtype
            or value.dtype != workspace.value.dtype
            or key.device != workspace.key.device
            or value.device != workspace.value.device
        ):
            raise ValueError("context K/V exceeds its declared tensor storage")

        rows = key.shape[0] * context.size
        context_key, context_value = (
            workspace.key[:rows],
            workspace.value[:rows],
        )
        context.all_gather_into(context_key, key.contiguous())
        context.all_gather_into(context_value, value.contiguous())
        return context_key, context_value


# Call-scoped buffer bindings: layers borrow caller-owned storage through this
# context variable instead of holding references between invocations.
_CONTEXT: ContextVar[Mapping[ParallelAttention, AttentionBuffers]] = ContextVar(
    "parallel_attention_context", default={}
)


@contextmanager
def context_scope(
    bindings: Mapping[ParallelAttention, AttentionBuffers],
) -> Iterator[None]:
    """Borrow context buffers for one numerical execution domain.

    The caller retains backing until all kernels, graphs, and readers finish.
    Layers retain no bindings; nested scopes restore the enclosing buffers,
    including when computation raises. Shared weights can use independent
    allocations in separate runtime owners.
    """
    token = _CONTEXT.set(bindings)
    try:
        yield
    finally:
        _CONTEXT.reset(token)
