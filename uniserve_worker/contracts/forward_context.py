"""Per-forward context published to shared layers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Protocol, Self

from .forward_stats import ForwardStats

__all__ = [
    'KVPool',
    'AttentionCache',
    'TextAttentionMetadata',
    'TextAttentionMetadataBuilder',
    'ForwardContext',
    'component_timer_start',
    'record_component_elapsed',
    'get_forward_context',
    'use_forward_context',
]

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import torch

    from ..backends.attention.base import AttentionBackend
    from .forward_mode import ForwardMode


class KVPool(Protocol):
    """System-owned KV pool published on :class:`ForwardContext`.

    The structural surface consumers read off the pool: a per-layer device cache
    pair, a per-request cache view, and the page geometry. ``PagedKVPool`` is the
    concrete implementer; it satisfies this without an explicit subclass edge.
    """

    @property
    def block_size(self) -> int: ...

    def layer_cache(self, layer: int) -> "tuple[torch.Tensor, torch.Tensor]": ...

    def view(self, block_ids: "Iterable[int]", base_len: int) -> "AttentionCache": ...


class AttentionCache(Protocol):
    """Paged request-cache surface read by the attention path and graph runner.

    Captures exactly the members consumed off ``TextAttentionMetadata.cache``:
    per-row persistent lengths, the page-table / cache-seqlens tensors, the
    single- and ragged append entry points, and the backing pool. Both the
    single-request (``PagedRequestCache``) and batched (``BatchedPagedRequestCache``)
    caches satisfy it structurally.
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


# Not frozen: the CUDA-graph runner keeps one captured metadata object resident
# and rewrites its per-replay CPU summary fields (cache_seqlens_cpu/query_lens_cpu/
# kv_seqlens_cpu) in place each replay so captured closures keep pointing at the
# same object. Modelling that as a frozen dataclass forced object.__setattr__
# escapes; a plain mutable companion is the honest contract.
@dataclass
class TextAttentionMetadata:
    """Per-step text attention metadata shared by all decoder layers.

    The tensors here are dynamic per batch, but they are invariant across
    decoder layers for one forward.  Building them once is also the contract a
    future CUDA graph runner can split into out-of-graph and in-graph state.
    """

    # ``cache`` is the paged request cache object (BatchedPagedRequestCache or a
    # single-request paged cache); the rest are per-step device/host tensors.
    cache: "AttentionCache"
    block_table: "torch.Tensor | None"
    cache_seqlens: "torch.Tensor | None"
    cache_seqlens_cpu: tuple[int, ...] = ()
    query_lens: "torch.Tensor | None" = None
    query_lens_cpu: tuple[int, ...] = ()
    kv_seqlens: "torch.Tensor | None" = None
    kv_seqlens_cpu: tuple[int, ...] = ()
    decode_page_ids: "torch.Tensor | None" = None
    decode_page_offsets: "torch.Tensor | None" = None
    cu_seqlens_q: "torch.Tensor | None" = None
    cu_seqlens_k: "torch.Tensor | None" = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    max_context_len: int = 0
    mode: "ForwardMode | None" = None

    @classmethod
    def for_decode_graph(
        cls,
        *,
        cache: "AttentionCache",
        batch_size: int,
        block_table: "torch.Tensor",
        cache_seqlens: "torch.Tensor",
        kv_seqlens: "torch.Tensor",
        query_lens: "torch.Tensor",
        decode_page_ids: "torch.Tensor",
        decode_page_offsets: "torch.Tensor",
        max_context_len: int = 0,
    ) -> "TextAttentionMetadata":
        """Build a fully-initialized decode-graph metadata in one step.

        The captured one-token decode graph holds a single resident metadata
        object whose device tensors are the runner's static input buffers. This
        constructs that object directly (no ``None`` placeholders / later
        ``replace``) with the fixed decode initial values: one query token per
        row and unit kv length per row.
        """

        from .forward_mode import ForwardMode

        batch_size = int(batch_size)
        query_lens.fill_(1)
        return cls(
            cache=cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cache_seqlens_cpu=tuple(0 for _ in range(batch_size)),
            kv_seqlens=kv_seqlens,
            query_lens=query_lens,
            query_lens_cpu=tuple(1 for _ in range(batch_size)),
            kv_seqlens_cpu=tuple(1 for _ in range(batch_size)),
            decode_page_ids=decode_page_ids,
            decode_page_offsets=decode_page_offsets,
            max_context_len=int(max_context_len),
            mode=ForwardMode.DECODE,
        )


class TextAttentionMetadataBuilder:
    """Fluent builder for :class:`TextAttentionMetadata`."""

    _REQUIRED = ("cache", "block_table", "cache_seqlens")
    _SETTER_FIELDS = (
        "cache",
        "block_table",
        "cache_seqlens",
        "cache_seqlens_cpu",
        "query_lens",
        "query_lens_cpu",
        "kv_seqlens",
        "kv_seqlens_cpu",
        "decode_page_ids",
        "decode_page_offsets",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "max_context_len",
        "mode",
    )

    def __init__(self) -> None:
        self._fields: dict[str, object] = {}

    def set(self, name: str, value: object) -> "Self":
        if name not in self._SETTER_FIELDS:
            raise ValueError(f"unknown TextAttentionMetadata field {name!r}")
        self._fields[name] = value
        return self

    def build(self) -> "TextAttentionMetadata":
        missing = [name for name in self._REQUIRED if name not in self._fields]
        if missing:
            raise ValueError(
                "TextAttentionMetadataBuilder is missing required field(s): "
                + ", ".join(missing)
            )
        return TextAttentionMetadata(**self._fields)  # type: ignore[arg-type]


def _metadata_setter(name: str):
    def setter(
        self: TextAttentionMetadataBuilder,
        value: object,
    ) -> TextAttentionMetadataBuilder:
        return self.set(name, value)

    setter.__name__ = name
    setter.__qualname__ = f"TextAttentionMetadataBuilder.{name}"
    return setter


def _install_text_attention_metadata_builder_setters() -> None:
    for field_name in TextAttentionMetadataBuilder._SETTER_FIELDS:
        setattr(TextAttentionMetadataBuilder, field_name, _metadata_setter(field_name))


_install_text_attention_metadata_builder_setters()



@dataclass(frozen=True)
class ForwardContext:
    attention_backend: "AttentionBackend | None" = None
    attention_backend_name: str | None = None
    # System-built per-forward attention plan + the system-owned KV pool, both
    # published by the runtime (the ``ForwardBatchBuilder`` builds the metadata;
    # the ``ResidencyManager`` owns the pool). The model resolves its residency
    # from here — it builds neither (SGLang's ``get_token_to_kv_pool()`` move).
    attention_metadata: TextAttentionMetadata | None = None
    kv_pool: "KVPool | None" = None
    stats: ForwardStats | None = None

    def component_timer_start(self) -> int:
        """Start a per-component timer against this context's ``stats``."""

        return component_timer_start(self.stats)

    def record_component_elapsed(self, component: str, start_ns: int) -> None:
        """Accumulate elapsed ns for ``component`` into this context's ``stats``."""

        record_component_elapsed(self.stats, component, start_ns)


def component_timer_start(stats: ForwardStats | None) -> int:
    """Return a perf-counter start stamp, or 0 when timing is disabled.

    The single source of truth for per-component timing shared by the text
    driver and the model forward paths. Timing is only taken when a
    :class:`ForwardStats` is being collected for this forward.
    """

    return time.perf_counter_ns() if stats is not None else 0


def record_component_elapsed(stats: ForwardStats | None, component: str, start_ns: int) -> None:
    """Accumulate elapsed ns since ``start_ns`` into ``stats.component_ns``.

    The companion to :func:`component_timer_start`; the ``None``-guard lives here
    so the two consuming modules cannot drift.
    """

    if stats is None:
        return
    stats.add_component_elapsed(component, start_ns)


_CURRENT: ContextVar[ForwardContext | None] = ContextVar("uniserve_forward_context", default=None)


def get_forward_context() -> ForwardContext:
    ctx = _CURRENT.get()
    if ctx is None:
        return ForwardContext()
    return ctx


@contextmanager
def use_forward_context(ctx: ForwardContext) -> Iterator[ForwardContext]:
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)
