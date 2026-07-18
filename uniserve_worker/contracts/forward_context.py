"""Per-forward ambient context published to shared layers.

:class:`ForwardContext` is the contextvar-scoped companion to
:class:`~.forward_batch.ForwardBatch`. The batch is the explicit argument into
model and graph-runner entry points; this context is what attention backends,
shared layers, and timing helpers read via :func:`get_forward_context` /
:func:`use_forward_context` for the duration of one forward.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Protocol

from .attention_plan import (
    AttentionPlanBase,
    GraphBinding,
    PagedDecodePlan,
    PagedVarlenPlan,
)
from .forward_stats import ForwardStats

__all__ = [
    "KVPool",
    "AttentionCache",
    "GraphBinding",
    "AttentionPlanBase",
    "PagedDecodePlan",
    "PagedVarlenPlan",
    "ForwardContext",
    "component_timer_start",
    "record_component_elapsed",
    "get_forward_context",
    "use_forward_context",
]

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import torch

    from ..backends.attention.base import AttentionBackend


class KVPool(Protocol):
    """System-owned KV pool published on :class:`ForwardContext`.

    Structural surface consumers read: per-layer device cache pair, per-request
    cache view, and page geometry. ``PagedKVPool`` is the concrete implementer
    and satisfies this without an explicit subclass edge.
    """

    @property
    def block_size(self) -> int: ...

    def layer_cache(self, layer: int) -> "tuple[torch.Tensor, torch.Tensor]": ...

    def view(self, block_ids: "Iterable[int]", base_len: int) -> "AttentionCache": ...


class AttentionCache(Protocol):
    """Paged request-cache surface read by the attention path and graph runner.

    Members consumed off the attention plan's ``residency_cache``: per-row persistent
    lengths, page-table / cache-seqlens tensors, single- and ragged append entry
    points, and the backing pool. Both ``PagedRequestCache`` and
    ``BatchedPagedRequestCache`` satisfy it structurally.
    """

    @property
    def pool(self) -> KVPool: ...

    @property
    def base_len(self) -> int: ...

    @property
    def base_lens(self) -> "Sequence[int]": ...

    def block_table(self, *, device: "torch.device | str | None" = ...) -> "torch.Tensor": ...

    def cache_seqlens(self, *, device: "torch.device | str | None" = ...) -> "torch.Tensor": ...

    def append(self, layer: int, k: "torch.Tensor", v: "torch.Tensor") -> None: ...

    def append_varlen(
        self,
        layer: int,
        k: "torch.Tensor",
        v: "torch.Tensor",
        query_lens: "Sequence[int]",
        *,
        block_table: "torch.Tensor | None" = ...,
        cache_seqlens: "torch.Tensor | None" = ...,
        cu_seqlens_q: "torch.Tensor | None" = ...,
    ) -> None: ...


@dataclass(frozen=True)
class ForwardContext:
    """Ambient per-forward state for attention and shared layers.

    Published by the runtime for the duration of one forward. Carries:

    - the selected attention backend (and the routing preference)
    - the system-built :class:`~.attention_plan.AttentionPlanBase` and the
      system-owned :class:`KVPool` (the model builds neither; it resolves
      residency from here)
    - an optional :class:`~.attention_plan.GraphBinding` identity token for
      CUDA-graph attention wrappers (plan tensors stay on ``attention_plan``;
      backends that need exclusive wrappers route by ``id(graph_binding)``)
    - whether a missing physical graph may be captured during this forward
    - optional :class:`~.forward_stats.ForwardStats` for per-component timing

    Frozen, and so is the plan it references: replace the whole context between
    forwards. Graph runners publish a fresh plan per replay while the device
    buffers that plan names are refreshed in place.
    """

    attention_backend: "AttentionBackend | None" = None
    attention_preference: str | None = None
    attention_plan: AttentionPlanBase | None = None
    graph_binding: GraphBinding | None = None
    kv_pool: "KVPool | None" = None
    stats: ForwardStats | None = None
    allow_capture: bool = True

    def component_timer_start(self) -> int:
        """Start a per-component timer against this context's ``stats``."""

        return component_timer_start(self.stats)

    def record_component_elapsed(self, component: str, start_ns: int) -> None:
        """Accumulate elapsed ns for ``component`` into this context's ``stats``."""

        record_component_elapsed(self.stats, component, start_ns)


def component_timer_start(stats: ForwardStats | None) -> int:
    """Return a perf-counter start stamp, or 0 when timing is disabled.

    Shared by the text driver and model forward paths. Timing is taken only
    when a :class:`ForwardStats` is being collected for this forward.
    """

    return time.perf_counter_ns() if stats is not None else 0


def record_component_elapsed(stats: ForwardStats | None, component: str, start_ns: int) -> None:
    """Accumulate elapsed ns since ``start_ns`` into ``stats.component_ns``.

    Companion to :func:`component_timer_start`; the ``None``-guard lives here so
    the two consuming modules cannot drift.
    """

    if stats is None:
        return
    stats.add_component_elapsed(component, start_ns)


_CURRENT: ContextVar[ForwardContext | None] = ContextVar("uniserve_forward_context", default=None)


def get_forward_context() -> ForwardContext:
    """Return the ambient :class:`ForwardContext`, or an empty default."""

    ctx = _CURRENT.get()
    if ctx is None:
        return ForwardContext()
    return ctx


@contextmanager
def use_forward_context(ctx: ForwardContext) -> Iterator[ForwardContext]:
    """Publish ``ctx`` as the ambient forward context for a ``with`` block."""

    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)
