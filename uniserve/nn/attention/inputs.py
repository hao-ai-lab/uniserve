"""Borrowed sequence lengths and numerical attention index representations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

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
            or any(type(length) is not int or length < 0 for length in self.host)
        ):
            raise ValueError("sequence lengths must be nonnegative integers")
        if (
            self.values.ndim != 1
            or self.offsets.shape != (self.batch_size + 1,)
            or self.values.dtype != torch.int32
            or self.offsets.dtype != torch.int32
            or self.values.device != self.offsets.device
        ):
            raise ValueError("sequence values and offsets require matching int32 device columns")
        if self.host is not None and len(self.host) != self.batch_size:
            raise ValueError("host lengths must match the device sequence count")
        if self.num_tokens is not None and self.num_tokens > torch.iinfo(torch.int32).max:
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
            raise ValueError("sequence lengths must fit nonnegative int32 offsets")
        return cls(
            values=torch.tensor(lengths, dtype=torch.int32, device=device),
            offsets=torch.tensor(
                tuple(accumulate(lengths, initial=0)), dtype=torch.int32, device=device
            ),
            host=lengths,
        )


@dataclass(frozen=True, slots=True)
class BlockTable:
    """Map [sequence, logical block] to physical block IDs.

    Sequence lengths determine the valid portion of each row. Unused trailing
    entries have no numerical meaning and are never part of an attention read.
    """

    indices: torch.Tensor
    block_size: int

    def __post_init__(self) -> None:
        if (
            type(self.block_size) is not int
            or self.block_size < 1
            or self.indices.ndim != 2
            or self.indices.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError("block tables require a positive block size and an integer matrix")


def _sequences(queries: SequenceLengths, keys: SequenceLengths) -> None:
    if queries.batch_size != keys.batch_size or queries.values.device != keys.values.device:
        raise ValueError("query and key sequences must share batch size and device")


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
        or (queries.num_tokens is not None and write_indices.shape != (queries.num_tokens,))
        or write_indices.dtype != torch.int64
        or write_indices.device != queries.values.device
    ):
        raise ValueError("write indices must be an int64 address per query token on its device")


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
    queries: SequenceLengths
    prefixes: SequenceLengths
    block_table: BlockTable
    write_indices: torch.Tensor | None
    causal: tuple[bool, ...]

    def __post_init__(self) -> None:
        _paged(self.queries, self.prefixes, self.block_table, self.write_indices)
        _causal(self.causal, self.queries.batch_size)

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
    ) -> PagedInput:
        """Build a block table and addresses for appending each query to its prefix."""

        if (
            not isinstance(blocks, tuple)
            or type(block_size) is not int
            or block_size < 1
            or len(blocks) != len(query_lengths)
            or len(blocks) != len(prefix_lengths)
        ):
            raise ValueError("block lists and sequence lengths must have matching batch sizes")
        queries = SequenceLengths.from_lengths(query_lengths, device=device)
        prefixes = SequenceLengths.from_lengths(prefix_lengths, device=device)
        flags = (causal,) * len(blocks) if type(causal) is bool else causal
        _causal(flags, len(blocks))
        width = max(map(len, blocks), default=0)
        rows = []
        addresses = []
        for row, query, prefix in zip(blocks, query_lengths, prefix_lengths, strict=True):
            if not isinstance(row, tuple) or any(
                type(block) is not int or block < 0 or block > torch.iinfo(torch.int32).max
                for block in row
            ):
                raise ValueError("physical block IDs must be nonnegative int32 integers")
            if len(row) * block_size < prefix + query:
                raise ValueError("block table does not cover the prefix and query")
            rows.append((*row, *((0,) * (width - len(row)))))
            addresses.extend(
                row[position // block_size] * block_size + position % block_size
                for position in range(prefix, prefix + query)
            )
        table = torch.tensor(rows, dtype=torch.int32, device=device).reshape(len(blocks), width)
        return cls(
            queries,
            prefixes,
            BlockTable(table, block_size),
            torch.tensor(addresses, dtype=torch.int64, device=device),
            flags,
        )


@dataclass(frozen=True, slots=True)
class VisibleInput:
    queries: SequenceLengths
    keys: SequenceLengths
    visible_end: torch.Tensor
    block_table: BlockTable | None
    prefix_bounds: bool
    fully_visible: bool

    def __post_init__(self) -> None:
        _sequences(self.queries, self.keys)
        _visibility(self.visible_end, self.queries)
        if self.block_table is not None and (
            self.block_table.indices.shape[0] != self.queries.batch_size
            or self.block_table.indices.device != self.queries.values.device
        ):
            raise ValueError("visible block table must match query batch and device")


@dataclass(frozen=True, slots=True)
class SegmentedInput:
    queries: SequenceLengths
    prefixes: SequenceLengths
    block_table: BlockTable
    write_indices: torch.Tensor | None
    visible_current_end: torch.Tensor
    fully_visible_current: bool

    def __post_init__(self) -> None:
        _paged(self.queries, self.prefixes, self.block_table, self.write_indices)
        _visibility(self.visible_current_end, self.queries)


def _visibility(value: torch.Tensor, queries: SequenceLengths) -> None:
    if (
        value.ndim != 2
        or value.shape[0] != queries.batch_size
        or (queries.maximum is not None and value.shape[1] < queries.maximum)
        or value.dtype not in {torch.int32, torch.int64}
        or value.device != queries.values.device
    ):
        raise ValueError("visibility must provide an integer endpoint per query token")


AttentionInput = DenseInput | VarlenInput | PagedInput | VisibleInput | SegmentedInput
