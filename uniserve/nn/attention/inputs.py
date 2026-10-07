"""Borrowed sequence lengths and numerical attention index representations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate, chain
from types import MappingProxyType

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SequenceLengths:
    """Borrowed device lengths with an optional exact host mirror.

    Callers keep host and device values consistent while an invocation uses
    them. Construction validates representation without synchronizing a device.
    """

    values: torch.Tensor
    offsets: torch.Tensor
    host: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.host is not None and (
            not isinstance(self.host, tuple)
            or any(
                type(length) is not int or length < 0 for length in self.host
            )
        ):
            raise ValueError("sequence lengths must be nonnegative integers")
        if (
            self.values.ndim != 1
            or self.offsets.shape != (self.batch_size + 1,)
            or self.values.dtype != torch.int32
            or self.offsets.dtype != torch.int32
            or self.values.device != self.offsets.device
        ):
            raise ValueError(
                "sequence values and offsets require matching int32 device "
                "columns"
            )
        if self.host is not None and len(self.host) != self.batch_size:
            raise ValueError(
                "host lengths must match the device sequence count"
            )
        if (
            self.num_tokens is not None
            and self.num_tokens > torch.iinfo(torch.int32).max
        ):
            raise ValueError("sequence offsets exceed int32 indexing")

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def num_tokens(self) -> int | None:
        return None if self.host is None else sum(self.host)

    @property
    def maximum(self) -> int | None:
        return None if self.host is None else max(self.host, default=0)

    @classmethod
    def from_lengths(
        cls, lengths: tuple[int, ...], *, device: torch.device | str
    ) -> SequenceLengths:
        if (
            not isinstance(lengths, tuple)
            or any(type(length) is not int or length < 0 for length in lengths)
            or sum(lengths) > torch.iinfo(torch.int32).max
        ):
            raise ValueError(
                "sequence lengths must fit nonnegative int32 offsets"
            )
        return cls(
            values=torch.tensor(lengths, dtype=torch.int32, device=device),
            offsets=torch.tensor(
                tuple(accumulate(lengths, initial=0)),
                dtype=torch.int32,
                device=device,
            ),
            host=lengths,
        )


@dataclass(frozen=True, slots=True)
class BlockTable:
    """Map [sequence, page column] to physical block IDs.

    Column ``j`` of row ``b`` holds absolute logical page
    ``start_page[b] + j``, so the absolute token position ``p`` of the row
    lies in column ``p // block_size - start_page[b]`` at page offset
    ``p % block_size``. Sequence lengths stay absolute: the valid columns of a
    row end at the page holding its last key. Pages before a row's start page
    are retired, a representation only history-windowed caches produce; the
    row's reader must need none of their keys. Unused trailing entries have
    no numerical meaning and are never part of an attention read.

    ``start_page`` is an int32 ``[B]`` device column, or ``None`` when every
    row starts at page zero, as every full-history table does.
    ``start_page_host`` optionally mirrors the column exactly on the host;
    the caller keeps both consistent while an invocation uses them, and
    construction never reads the device column.
    """

    indices: torch.Tensor
    block_size: int
    start_page: torch.Tensor | None = None
    start_page_host: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            type(self.block_size) is not int
            or self.block_size < 1
            or self.indices.ndim != 2
            or self.indices.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError(
                "block tables require a positive block size and an integer "
                "matrix"
            )
        if self.start_page is None:
            if self.start_page_host is not None:
                raise ValueError(
                    "a host start-page mirror requires its device column"
                )
            return
        if (
            self.start_page.shape != (self.indices.shape[0],)
            or self.start_page.dtype != torch.int32
            or self.start_page.device != self.indices.device
        ):
            raise ValueError(
                "table start pages require one int32 page per row on the "
                "table's device"
            )
        if self.start_page_host is not None and (
            not isinstance(self.start_page_host, tuple)
            or len(self.start_page_host) != self.indices.shape[0]
            or any(
                type(page) is not int or page < 0
                for page in self.start_page_host
            )
        ):
            raise ValueError(
                "host start pages must mirror one nonnegative page per row"
            )


def _sequences(queries: SequenceLengths, keys: SequenceLengths) -> None:
    if (
        queries.batch_size != keys.batch_size
        or queries.values.device != keys.values.device
    ):
        raise ValueError(
            "query and key sequences must share batch size and device"
        )


def _causal(values: tuple[bool, ...], count: int) -> None:
    if (
        not isinstance(values, tuple)
        or len(values) != count
        or any(type(value) is not bool for value in values)
    ):
        raise ValueError("causality must specify one boolean per sequence")


def _paged(
    queries: SequenceLengths,
    prefixes: SequenceLengths,
    block_table: BlockTable,
    write_indices: torch.Tensor | None,
) -> None:
    _sequences(queries, prefixes)
    if (
        block_table.indices.shape[0] != queries.batch_size
        or block_table.indices.device != queries.values.device
    ):
        raise ValueError("block tables must match query batch size and device")
    if write_indices is not None and (
        write_indices.ndim != 1
        or (
            queries.num_tokens is not None
            and write_indices.shape != (queries.num_tokens,)
        )
        or write_indices.dtype != torch.int64
        or write_indices.device != queries.values.device
    ):
        raise ValueError(
            "write indices must be an int64 address per query token on its "
            "device"
        )


@dataclass(frozen=True, slots=True)
class DenseInput:
    causal: bool
    mask: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class VarlenInput:
    queries: SequenceLengths
    keys: SequenceLengths
    causal: tuple[bool, ...]

    def __post_init__(self) -> None:
        _sequences(self.queries, self.keys)
        _causal(self.causal, self.queries.batch_size)


@dataclass(frozen=True, slots=True)
class PagedInput:
    """Packed cache-appending rows with uniform or per-row causality.

    ``causal_values`` optionally borrows an int32 device mirror of ``causal``
    (0 or 1). Callers keep the two consistent. Native mixed-block readers
    consume the device column so graph replay can change row visibility
    without changing the launch or synchronizing the device.
    The auto, prefix-block, FlashAttention-4, FlashInfer and portable providers
    implement this representation; uniform-only native kernels reject it.
    """

    queries: SequenceLengths
    prefixes: SequenceLengths
    block_table: BlockTable
    write_indices: torch.Tensor | None
    causal: tuple[bool, ...]
    causal_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _paged(
            self.queries, self.prefixes, self.block_table, self.write_indices
        )
        _causal(self.causal, self.queries.batch_size)
        flags = self.causal_values
        if flags is not None and (
            flags.shape != self.queries.values.shape
            or flags.dtype != torch.int32
            or flags.device != self.queries.values.device
            or flags.stride(0) != 1
        ):
            raise ValueError(
                "causal values require one contiguous int32 per row"
            )

    @classmethod
    def from_blocks(
        cls,
        *,
        blocks: tuple[tuple[int, ...], ...],
        query_lengths: tuple[int, ...],
        prefix_lengths: tuple[int, ...],
        block_size: int,
        causal: bool | tuple[bool, ...],
        device: torch.device | str,
        start_pages: tuple[int, ...] | None = None,
    ) -> PagedInput:
        """Build a block table and addresses for appending each query to its
        prefix.

        Each row of ``blocks`` holds the physical blocks of its logical pages
        from its entry of ``start_pages`` on (page zero when ``None``). The
        blocks must cover every appended position, which must lie at or
        after the row's start page.
        """  # noqa: D205
        table, addresses = paged_append(
            blocks,
            query_lengths=query_lengths,
            prefix_lengths=prefix_lengths,
            block_size=block_size,
            start_pages=start_pages,
        )
        queries = SequenceLengths.from_lengths(query_lengths, device=device)
        prefixes = SequenceLengths.from_lengths(prefix_lengths, device=device)
        flags = (causal,) * len(blocks) if type(causal) is bool else causal
        _causal(flags, len(blocks))
        return cls(
            queries,
            prefixes,
            BlockTable(
                torch.from_numpy(table).to(device),
                block_size,
                None
                if start_pages is None
                else torch.tensor(
                    start_pages, dtype=torch.int32, device=device
                ),
                start_pages,
            ),
            torch.from_numpy(addresses).to(device),
            flags,
        )


def paged_append(
    blocks: tuple[tuple[int, ...], ...],
    *,
    query_lengths: tuple[int, ...],
    prefix_lengths: tuple[int, ...],
    block_size: int,
    start_pages: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lay out appending each query to its prefix on the host.

    Each row of ``blocks`` holds the physical blocks of its logical pages
    from its entry of ``start_pages`` on (page zero when ``None``). The
    blocks must cover every appended position, which must lie at or after
    the row's start page.

    Returns:
        The zero-padded int32 block table ``[rows, width]`` and the int64
        physical token address of every appended position, row after row.

    Raises:
        ValueError: If the rows, lengths and start pages disagree in number,
            a length or start page is negative, a block id is not a
            nonnegative int32 Python int, or a row does not cover its prefix
            and query.
    """
    if (
        not isinstance(blocks, tuple)
        or type(block_size) is not int
        or block_size < 1
        or len(blocks) != len(query_lengths)
        or len(blocks) != len(prefix_lengths)
    ):
        raise ValueError(
            "block lists and sequence lengths must have matching batch sizes"
        )
    if any(
        type(length) is not int or length < 0
        for length in (*query_lengths, *prefix_lengths)
    ):
        raise ValueError("sequence lengths must be nonnegative integers")
    starts = (0,) * len(blocks) if start_pages is None else start_pages
    if (
        not isinstance(starts, tuple)
        or len(starts) != len(blocks)
        or any(type(page) is not int or page < 0 for page in starts)
    ):
        raise ValueError(
            "start pages must give one nonnegative page per block row"
        )

    if any(not isinstance(row, tuple) for row in blocks) or not set(
        map(type, chain.from_iterable(blocks))
    ) <= {int}:
        raise ValueError(
            "physical block IDs must be nonnegative int32 integers"
        )
    for row, query, prefix, start in zip(
        blocks, query_lengths, prefix_lengths, starts, strict=True
    ):
        if (start + len(row)) * block_size < prefix + query or (
            query and prefix < start * block_size
        ):
            raise ValueError("block table does not cover the prefix and query")

    # Short rows are zero-padded to the shared table width; padded entries
    # are never read because lengths bound the valid span.
    count = len(blocks)
    width = max(map(len, blocks), default=0)
    table = np.zeros((count, width), dtype=np.int64)
    try:
        for index, row in enumerate(blocks):
            table[index, : len(row)] = row
    except OverflowError:
        raise ValueError(
            "physical block IDs must be nonnegative int32 integers"
        ) from None
    if table.size and (table.min() < 0 or table.max() > np.iinfo(np.int32).max):
        raise ValueError(
            "physical block IDs must be nonnegative int32 integers"
        )

    # One physical token address per appended query position, in the row's
    # columns counted from its start page, gathered for every position at
    # once: token t of row r is position prefix[r] + t - first[r], where
    # first[r] is the row's first token.
    lengths = np.asarray(query_lengths, dtype=np.int64)
    owners = np.repeat(np.arange(count), lengths)
    first = np.cumsum(lengths) - lengths
    positions = (
        np.asarray(prefix_lengths, dtype=np.int64)[owners]
        + np.arange(owners.size)
        - first[owners]
    )
    columns = (
        positions // block_size - np.asarray(starts, dtype=np.int64)[owners]
    )
    addresses = table[owners, columns] * block_size + positions % block_size
    return table.astype(np.int32), addresses


@dataclass(frozen=True, slots=True)
class VisibleInput:
    """Attend each query row to a prefix of its sequence's keys.

    ``visible_end`` holds exclusive key endpoints on the query device, as
    int32 or int64: either ``[batch, query rows]``, one endpoint per
    sequence-local query row, or ``[batch, 1]``, one endpoint that every
    query row of the sequence shares. An endpoint beyond the sequence's keys
    sees all of them. The shared form bounds each sequence's key length, so
    kernels attend the visible prefix without evaluating a per-row mask.
    ``fully_visible`` declares every key visible and leaves the endpoints
    unread.
    """

    queries: SequenceLengths
    keys: SequenceLengths
    visible_end: torch.Tensor
    block_table: BlockTable | None
    prefix_bounds: bool
    fully_visible: bool

    def __post_init__(self) -> None:
        _sequences(self.queries, self.keys)
        _visibility(self.visible_end, self.queries, shared=True)
        if self.block_table is not None and (
            self.block_table.indices.shape[0] != self.queries.batch_size
            or self.block_table.indices.device != self.queries.values.device
        ):
            raise ValueError(
                "visible block table must match query batch and device"
            )


@dataclass(frozen=True, slots=True)
class SegmentedInput:
    queries: SequenceLengths
    prefixes: SequenceLengths
    block_table: BlockTable
    write_indices: torch.Tensor | None
    visible_current_end: torch.Tensor
    fully_visible_current: bool

    def __post_init__(self) -> None:
        _paged(
            self.queries, self.prefixes, self.block_table, self.write_indices
        )
        _visibility(self.visible_current_end, self.queries)


def _visibility(
    value: torch.Tensor, queries: SequenceLengths, *, shared: bool = False
) -> None:
    # ``shared`` admits one endpoint column that every query row reads.
    if (
        value.ndim != 2
        or value.shape[0] != queries.batch_size
        or (
            queries.maximum is not None
            and value.shape[1] < queries.maximum
            and not (shared and value.shape[1] == 1)
        )
        or value.dtype not in {torch.int32, torch.int64}
        or value.device != queries.values.device
    ):
        raise ValueError(
            "visibility must provide an integer endpoint per query token"
        )


# Union of every index representation the numerical attention layers accept.
AttentionInput = (
    DenseInput | VarlenInput | PagedInput | VisibleInput | SegmentedInput
)


@dataclass(frozen=True, slots=True)
class AttentionBatch:
    """One call's numerical attention inputs, one entry per cache table.

    A layer bound to a prefix cache reads the entry of the table that holds
    its cache; a layer without a cache table reads the batch's only entry.
    Table IDs come from the prefix cache's layout and are not model
    configuration. Every packed entry shares ``queries``: the same query
    order, length and offset tensors. Entries may differ in their prefixes,
    block tables, write addresses and key visibility. A dense input forms a
    singleton batch without a packed query domain. The batch borrows its
    tensors and owns no cache, request or execution state.
    """

    entries: Mapping[int, AttentionInput]
    queries: SequenceLengths | None

    def __post_init__(self) -> None:
        entries = dict(self.entries)
        if not entries or any(
            type(table) is not int or table < 0 for table in entries
        ):
            raise ValueError(
                "attention batches map nonnegative table IDs to inputs"
            )
        # Every input except a dense one carries a packed query domain.
        packed = tuple(
            entry
            for entry in entries.values()
            if not isinstance(entry, DenseInput)
        )
        if len(packed) != len(entries):
            if len(entries) != 1 or self.queries is not None:
                raise ValueError(
                    "dense attention forms a singleton batch without packed "
                    "queries"
                )
        elif self.queries is None or any(
            entry.queries.values is not self.queries.values
            or entry.queries.offsets is not self.queries.offsets
            for entry in packed
        ):
            raise ValueError(
                "attention tables must share one packed query domain"
            )
        object.__setattr__(self, "entries", MappingProxyType(entries))

    @classmethod
    def single(cls, value: AttentionInput) -> AttentionBatch:
        """Wrap one input as the only entry, for layers without cache tables."""
        return cls(
            {0: value},
            None if isinstance(value, DenseInput) else value.queries,
        )

    @property
    def batch_size(self) -> int | None:
        """Packed sequence count, or None for a dense singleton."""
        return None if self.queries is None else self.queries.batch_size

    def entry(self, table: int | None) -> AttentionInput:
        """Select one table's input; ``None`` selects the only entry.

        Raises:
            ValueError: ``table`` is absent, or ``None`` names a batch with
                several entries.
        """
        if table is None:
            if len(self.entries) != 1:
                raise ValueError(
                    "a layer without a cache table requires a single-entry "
                    "attention batch"
                )
            return next(iter(self.entries.values()))
        try:
            return self.entries[table]
        except KeyError:
            raise ValueError(
                f"attention batch has no entry for cache table {table}"
            ) from None
