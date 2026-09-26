"""SM100 non-causal block attention over a borrowed paged prefix.

The provider binds ``uniserve_kernels.attention.prefix_block`` to a layer's
prefix cache. It serves the two non-causal reads whose query rows see their
whole current block and a window of the sequence's paged prefix, with the
visibility :class:`uniserve.nn.attention.Attention` defines:

- a segmented input whose queries see every current key (a canvas read of a
  read-only prefix): each query reads prefix tokens
  ``[max(P - window, 0), P)``, or the whole prefix without a window;
- a non-causal paged input (a block that attends to itself in both
  directions, such as an image block): query ``i`` at absolute position
  ``P + i`` reads prefix tokens from ``max(P + i - window, 0)`` on, or the
  whole prefix without a window.

``P`` is the row's absolute prefix length. Current K/V come from the packed
projections at the query rows and the prefix from the cache through each
row's block table, whose first column is its start page when the table
carries one. A paged input's write addresses are committed before the read,
so later calls find the block in the cache. Lengths, offsets, tables and
start pages are device values: a launch records into a CUDA graph and
replays with changed values in the same buffers.
"""

from __future__ import annotations

import torch

from uniserve.nn.attention.inputs import PagedInput, SegmentedInput

from . import Backend as _Backend
from . import CachePages
from . import Operator as _Operator


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

    def requires_host_lengths(self, batch):
        # The kernel reads every length, offset and table origin on device.
        return False

    def bind(self, batch):
        super().bind(batch)
        _require_block_reads(batch)

    def __call__(self, q, k, v, batch, *, scale, out):
        _require_block_reads(batch)
        self._validate(q, k, v, batch, out)
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(
                "prefix-block attention requires packed query and current "
                "K/V rows"
            )

        if batch.write_indices is not None:
            self.update_cache(k, v, indices=batch.write_indices)
        if not q.shape[0] or not batch.queries.batch_size:
            return out

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
            start_page=None if start_page is None else start_page.contiguous(),
            scale=scale,
            out=out,
        )
        return out


def _require_block_reads(batch):
    """Reject inputs outside the kernel's non-causal block visibility."""
    if isinstance(batch, PagedInput) and not any(batch.causal):
        return
    if isinstance(batch, SegmentedInput) and batch.fully_visible_current:
        return
    raise ValueError(
        "prefix-block attention reads non-causal paged blocks and segmented "
        "prefixes whose queries see every current key"
    )


class Backend(_Backend):
    name = "prefix_block"

    operator_class = _PrefixBlock
