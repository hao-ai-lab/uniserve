"""SM100 block attention over a borrowed paged prefix.

The provider binds ``uniserve_kernels.attention.prefix_block`` to a layer's
prefix cache. It serves the reads whose query rows see a window of the
sequence's paged prefix and their current block, with the visibility
:class:`uniserve.nn.attention.Attention` defines:

- a segmented input whose queries see every current key (a canvas read of a
  read-only prefix): each query reads prefix tokens
  ``[max(P - window, 0), P)``, or the whole prefix without a window;
- a non-causal paged input (a block that attends to itself in both
  directions, such as an image block): query ``i`` at absolute position
  ``P + i`` reads prefix tokens from ``max(P + i - window, 0)`` on, or the
  whole prefix without a window;
- on a full-attention layer (head dimension 512 without a history window),
  a causal paged input (a prompt chunk or committed rows continuing their
  history): query ``i`` reads the whole prefix and the chunk's keys
  ``[0, i]``.

``P`` is the row's absolute prefix length. Current K/V come from the packed
projections at the query rows and the prefix from the cache through each
row's block table, whose first column is its start page when the table
carries one. A paged input's write addresses are committed before the read,
so later calls find the block in the cache. A paged batch mixing causal and
non-causal rows is evaluated as its contiguous runs of equal causality after
its complete write commits once. Lengths, offsets, tables and start pages
are device values: a launch records into a CUDA graph and replays with
changed values in the same buffers.

Causal launches claim work tiles from a ticket counter in the operator's
``schedule`` workspace buffer, which the layers serving causal chunks
declare. The counter must be zero when a causal launch starts and the launch
leaves it nonzero, so the preparation launch that commits a causal read's
cache write (``_paged_inputs.prepare``) also zeroes it, immediately before
the attention launch on the same stream, eagerly and in a captured graph
alike. The buffer is therefore per invocation domain, never shared across
streams: it is not the shared ``scratch`` grant, so an execution context
allocates it privately for each call site (automatic selection namespaces it
as ``prefix_block.schedule``), and a context runs its preparation, eager
calls and graph replays on its one stream; separate execution lanes use
separate contexts.
"""

from __future__ import annotations

import torch

from uniserve.nn.attention.inputs import PagedInput, SegmentedInput
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import CachePages
from . import Operator as _Operator
from ._sequences import causal_runs


def unsupported(
    *, num_heads, num_kv_heads, head_dim, dtype, pages: CachePages | None
) -> str | None:
    """Return why the kernel cannot serve a layer, or None.

    ``pages`` describes the layer's bound prefix cache: the kernel reads
    prefixes only from paged caches storing the query dtype in 16-, 32- or
    64-token pages.
    """
    from uniserve_kernels.attention import prefix_block

    if pages is None:
        return "requires a bound paged prefix cache"
    if pages.quantized:
        return "does not read per-block FP8 prefix caches"
    if pages.dtype != dtype:
        return "requires the cache to store the query dtype"
    return prefix_block.unsupported_configuration(
        query_heads=num_heads,
        kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_tokens=pages.page_tokens,
        dtype=dtype,
    )


def _serves_causal(*, num_heads, num_kv_heads, head_dim, dtype, cache, window):
    """Whether the kernel reads a layer's causal chunks.

    They are the full-attention layers: head dimension 512 without a
    history window.
    """
    from uniserve_kernels.attention import prefix_block

    if cache is None:
        return False
    pages = CachePages.of(cache)
    return (
        unsupported(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            pages=pages,
        )
        is None
        and prefix_block.unsupported_configuration(
            query_heads=num_heads,
            kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_tokens=pages.page_tokens,
            dtype=dtype,
            causal=True,
            window=window,
        )
        is None
    )


class _PrefixBlock(_Operator):
    # Each row's prefix is addressed from its table's first column, which
    # holds absolute logical page ``start_page``; no row reads a retired
    # page because the window starts at or after its start page.
    reads_retired_tables = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from uniserve_kernels.attention import prefix_block

        problem = unsupported(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            pages=None if self.cache is None else CachePages.of(self.cache),
        )
        if problem is not None:
            raise ValueError(f"prefix-block attention {problem}")
        self._attention = prefix_block.prefix_block_attention
        # The causal launches' workspace; see the module docstring.
        self._schedule = self.workspace.get("schedule")

    def requires_host_lengths(self, batch):
        # The kernels read every length, offset and table origin on device;
        # only the runs of a mixed-causality batch are sliced on the host.
        return isinstance(batch, PagedInput) and len(set(batch.causal)) > 1

    def bind(self, batch):
        super().bind(batch)
        self._require_reads(batch)

    def __call__(self, q, k, v, batch, *, scale, out):
        self._require_reads(batch)
        self._validate(q, k, v, batch, out)
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(
                "prefix-block attention requires packed query and current "
                "K/V rows"
            )

        if not isinstance(batch, PagedInput) or len(set(batch.causal)) <= 1:
            self._read(q, k, v, batch, scale, out)
            return out

        # A later run may read rows an earlier run of the same sequence
        # writes, so the batch's complete write commits before any run; the
        # runs carry no write addresses.
        if batch.write_indices is not None:
            self.update_cache(k, v, indices=batch.write_indices)
        for rows, _, run in causal_runs(batch):
            if rows.start == rows.stop:
                continue
            # Current K/V share the query rows of their run.
            self._read(q[rows], k[rows], v[rows], run, scale, out[rows])
        return out

    def _read(self, q, k, v, batch, scale, out):
        """Evaluate a segmented read or paged rows of one causality."""
        from ._paged_inputs import prepare

        causal = isinstance(batch, PagedInput) and any(batch.causal)
        if causal:
            # One launch commits the rows' write, fused where the storage
            # allows, and zeroes the ticket counter of the causal launch.
            prepare(self.cache, k, v, batch, semaphore=self._schedule[:1])
        elif batch.write_indices is not None:
            self.update_cache(k, v, indices=batch.write_indices)
        if not q.shape[0] or not batch.queries.batch_size:
            return

        table = batch.block_table
        indices = table.indices
        if indices.dtype != torch.int32 or indices.stride(1) != 1:
            indices = indices.to(dtype=torch.int32).contiguous()
        if not indices.shape[1]:
            # Rows without any prefix page read no prefix token; the kernel
            # still takes one column, which the sentinel unit fills.
            indices = indices.new_zeros((indices.shape[0], 1))

        # The launch grid is sized by a host bound on every block length. A
        # captured launch replays with changed lengths: a segmented read's
        # visibility extent bounds every block it replays with, since its
        # rows must stay within it, and the packed query capacity bounds a
        # paged block's. An eager call uses its exact longest block.
        longest = batch.queries.maximum
        if torch.cuda.is_current_stream_capturing():
            longest = (
                batch.visible_current_end.shape[1]
                if isinstance(batch, SegmentedInput)
                else q.shape[0]
            )
        elif longest is None:
            longest = q.shape[0]

        start_page = table.start_page
        self._attention(
            q,
            k,
            v,
            self.cache.key,
            self.cache.value,
            indices,
            batch.queries.offsets.contiguous(),
            batch.prefixes.values.contiguous(),
            max_query_len=longest,
            window=self.window,
            # A paged block's history window follows each query's absolute
            # position; a segmented read keeps one prefix interval.
            query_window=isinstance(batch, PagedInput),
            causal=causal,
            workspace=self._schedule if causal else None,
            start_page=None if start_page is None else start_page.contiguous(),
            scale=scale,
            out=out,
        )

    def _require_reads(self, batch):
        """Reject inputs outside the kernels' block visibility."""
        if isinstance(batch, SegmentedInput) and batch.fully_visible_current:
            return
        if isinstance(batch, PagedInput) and (
            not any(batch.causal) or self._schedule is not None
        ):
            return
        raise ValueError(
            "prefix-block attention reads non-causal paged blocks, segmented "
            "prefixes whose queries see every current key, and causal paged "
            "chunks of head dimension 512 without a history window"
        )


class Backend(_Backend):
    name = "prefix_block"

    operator_class = _PrefixBlock

    def workspace_buffers(
        self,
        *,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype,
        size,
        cache,
        window=None,
    ):
        from uniserve_kernels.attention import prefix_block

        if not _serves_causal(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            cache=cache,
            window=window,
        ):
            return {}
        return {
            "schedule": BufferConfig(
                (prefix_block.workspace_words(size.batch_size),), torch.int32
            )
        }
