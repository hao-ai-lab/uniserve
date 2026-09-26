"""Prepared attention operators borrowing numerical inputs and cache state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module

import torch as torch_lib

from uniserve.cache import mha
from uniserve.model.inputs import TextSize
from uniserve.nn.attention.inputs import (
    AttentionInput,
    DenseInput,
    PagedInput,
    SegmentedInput,
    VisibleInput,
)
from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig


@dataclass(frozen=True, slots=True)
class CachePages:
    """How a layer's paged prefix cache stores its pages.

    ``page_tokens`` is the tokens of one page, ``dtype`` the element type
    the pages store and ``quantized`` whether they hold per-block FP8
    values with scales.
    """

    page_tokens: int
    dtype: torch_lib.dtype
    quantized: bool = False

    @classmethod
    def of(cls, state: mha.State) -> CachePages:
        """Describe the pages of a bound cache state."""
        key = state.key
        return cls(
            state.block_size, key.dtype, isinstance(key, QuantizedTensor)
        )


class Operator:
    """One layer invocation's backend state and borrowed numerical resources.

    ``window`` is the layer's history bound in tokens (``None`` reads the
    whole history); its visibility rule is documented on
    :class:`uniserve.nn.attention.Attention`.

    ``reads_retired_tables`` states whether the provider consumes block
    tables whose rows start after logical page zero
    (``BlockTable.start_page``). A provider that does not rejects such a
    table instead of reading its columns from the wrong logical pages.
    """

    reads_retired_tables = False

    def __init__(
        self,
        *,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype,
        size,
        cache,
        workspace,
        window=None,
    ):
        if (
            min(num_heads, num_kv_heads, head_dim) < 1
            or num_heads % num_kv_heads
        ):
            raise ValueError(
                "attention requires compatible positive query and KV heads"
            )
        if window is not None and (type(window) is not int or window < 0):
            raise ValueError(
                "attention windows must be nonnegative token counts"
            )
        self.window = window
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.size = size
        self.cache = cache
        self.workspace = workspace
        self._closed = False

        if cache is not None and cache.key.shape[2:] != (
            num_kv_heads,
            head_dim,
        ):
            raise ValueError(
                "prefix state does not match the prepared KV head dimensions"
            )

    def bind(self, batch: AttentionInput) -> None:
        """Prepare changed numerical metadata.

        Prepare changed numerical metadata without relying on object
        identity.
        """
        self._check_batch(batch)

        table = getattr(batch, "block_table", None)
        if self.requires_host_lengths(batch) and (
            any(
                lengths is not None and lengths.host is None
                for lengths in (
                    getattr(batch, name, None)
                    for name in ("queries", "keys", "prefixes")
                )
            )
            or (
                table is not None
                and table.start_page is not None
                and table.start_page_host is None
            )
        ):
            raise ValueError(
                "this attention preparation requires exact host sequence "
                "lengths"
            )

    def selections(self) -> Mapping[str, str]:
        """Return the provider that served each input class this operator met.

        Keys name an input class by its path and mask semantics, for example
        ``"paged attention: causal rows"``; values are provider names. Only a
        dispatching operator, which chooses a provider per input, reports
        choices; a provider's own operator serves every input itself and
        returns an empty mapping.
        """
        return {}

    def requires_host_lengths(self, batch: AttentionInput) -> bool:
        """Whether preparation needs exact CPU sequence lengths.

        Infrastructure supplies missing mirrors before bind/capture. Providers
        whose kernels consume device offsets directly override this query.
        """
        return not isinstance(batch, DenseInput)

    def _check_batch(self, batch):
        if self._closed:
            raise RuntimeError("attention operator is closed")

        if self.window is not None and isinstance(batch, VisibleInput):
            # Visible inputs carry key endpoints but no query positions, so a
            # history bound relative to each query is not defined for them.
            raise ValueError(
                "windowed attention requires paged, segmented, variable-length "
                "or dense inputs"
            )

        if not isinstance(batch, DenseInput) and (
            (
                batch.queries.num_tokens is not None
                and batch.queries.num_tokens > self.size.num_tokens
            )
            or batch.queries.batch_size > self.size.batch_size
        ):
            raise ValueError(
                "attention input exceeds prepared token or sequence capacity"
            )

        if (
            isinstance(batch, (PagedInput, SegmentedInput))
            and self.cache is not None
        ):
            if batch.block_table.block_size != self.cache.block_size:
                raise ValueError(
                    "attention block table and cache block sizes differ"
                )

        table = getattr(batch, "block_table", None)
        if (
            table is not None
            and table.start_page is not None
            and not self.reads_retired_tables
        ):
            raise ValueError(
                "this attention provider reads block tables from logical "
                "page zero and cannot consume retired window pages"
            )

    def update_cache(
        self,
        k: torch_lib.Tensor,
        v: torch_lib.Tensor,
        *,
        indices: torch_lib.Tensor,
    ) -> None:
        """Write current K/V rows into the bound prefix cache."""
        if self._closed:
            raise RuntimeError("attention operator is closed")
        if self.cache is None:
            raise RuntimeError(
                "attention cache update requires bound prefix state"
            )

        self.cache.update(k, v, indices=indices)

    def _validate(self, q, k, v, batch, out):
        # Host-driven provider preparation belongs to bind(), before capture.
        # The numerical call only validates capacity and borrowed tensor views.
        self._check_batch(batch)
        head_axis = 1

        if (
            q.ndim not in {3, 4}
            or q.shape[head_axis] != self.num_heads
            or q.shape[-1] != self.head_dim
        ):
            raise ValueError(
                "query tensor does not match prepared attention dimensions"
            )

        if (
            q.dtype != self.dtype
            or out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
        ):
            raise ValueError(
                "attention output and query representation must match"
            )

        if q.ndim == 3 and q.shape[0] > self.size.num_tokens:
            raise ValueError("attention queries exceed prepared token capacity")

        if isinstance(batch, DenseInput):
            tokens = q.shape[0] if q.ndim == 3 else q.shape[0] * q.shape[2]
            batches = 1 if q.ndim == 3 else q.shape[0]
            if tokens > self.size.num_tokens or batches > self.size.batch_size:
                raise ValueError(
                    "dense attention exceeds the prepared token or batch "
                    "capacity"
                )

        if k.shape != v.shape or k.device != q.device or v.device != q.device:
            raise ValueError(
                "key and value tensors must have matching dimensions and "
                "devices"
            )

        # Packed K/V are [tokens, heads, dim]; paged cache rows are
        # [blocks, tokens, heads, dim] with the head axis shifted.
        kv_axis = 1 if k.ndim == 3 or isinstance(batch, DenseInput) else 2
        if (
            k.ndim not in {3, 4}
            or k.shape[kv_axis] != self.num_kv_heads
            or k.shape[-1] != self.head_dim
            or k.dtype != self.dtype
            or v.dtype != self.dtype
        ):
            raise ValueError(
                "key and value tensors do not match prepared head "
                "dimensions or dtype"
            )

    def __call__(
        self,
        q: torch_lib.Tensor,
        k: torch_lib.Tensor,
        v: torch_lib.Tensor,
        batch: AttentionInput,
        *,
        scale: float,
        out: torch_lib.Tensor,
    ) -> torch_lib.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        """Release borrowed state.

        Release borrowed state; the operator must not be invoked afterward.
        """
        self._closed = True
        self.cache = None
        self.workspace = {}


class Backend:
    """Factory for independent layer operators.

    Factory for independent layer operators and their workspace
    declarations.
    """

    name: str

    operator_class: type[Operator]

    def reads_pages(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch_lib.dtype,
        window: int | None,
        pages: CachePages,
    ) -> bool:
        """Report whether the backend serves a cache layer on ``pages``.

        A layer with a paged prefix cache receives causal and non-causal
        paged calls and segmented reads of its prefix. A backend serving
        one provider reports True and validates its own page constraints
        when a layer is prepared.
        """
        return True

    def workspace_buffers(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch_lib.dtype,
        size: TextSize,
        cache: mha.State | None,
        window: int | None = None,
    ) -> Mapping[str, BufferConfig]:
        return {}

    def prepare(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch_lib.dtype,
        size: TextSize,
        cache: mha.State | None,
        workspace: Mapping[str, torch_lib.Tensor],
        window: int | None = None,
    ) -> Operator:
        return self.operator_class(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            size=size,
            cache=cache,
            workspace=workspace,
            window=window,
        )


def resolve(
    backend: str | Backend, *, device: torch_lib.device, flashinfer=None
) -> Backend:
    """Resolve a provider, preserving an optional configured FlashInfer factory.

    ``"auto"`` selects native kernels per call (see ``_auto``) and uses the
    FlashInfer factory's workspace grant for the TensorRT-LLM kernels that
    ship with FlashInfer. Any other name selects that provider for every
    call, including the FlashInfer, FlashAttention-2 and portable torch
    providers that automatic selection never chooses on CUDA.
    """
    if isinstance(backend, Backend):
        return backend

    if backend == "auto":
        from ._auto import Backend as _Auto

        return _Auto(device, flashinfer=flashinfer)

    if backend not in {
        "torch",
        "flash_attn",
        "flash_attn_4",
        "flashinfer",
        "trtllm",
        "sgl_kernel",
        "prefix_block",
    }:
        raise ValueError(f"unknown attention backend {backend!r}")

    if backend == "flashinfer" and flashinfer is not None:
        return flashinfer

    factory = import_module(f"{__name__}.{backend}").Backend
    if backend == "trtllm" and flashinfer is not None:
        return factory(workspace_size=flashinfer.config.workspace_size)
    return factory()
