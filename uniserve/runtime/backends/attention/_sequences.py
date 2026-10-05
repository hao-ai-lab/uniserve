"""Numerical sequence slicing for kernels with one causality flag per launch."""

from dataclasses import dataclass, replace
from itertools import groupby

import torch

from uniserve.nn.attention.inputs import BlockTable, PagedInput, SequenceLengths


def host_lengths(batch, *, prepared=None, derive=True):
    """Supply exact mirrors at preparation, retaining borrowed device columns.

    Sequence lengths and a block table's start pages are mirrored. During
    capture, prepared mirrors must describe these same fixed addresses.
    Reading device values is forbidden there; replay metadata is prepared by
    the caller before each invocation that requires host-driven planning.

    Without ``prepared``, a missing mirror is read from its column. A CUDA
    column is read only with ``derive``, a synchronizing device-to-host copy
    that direct library callers may rely on; serving contexts pass False,
    because their input buffers supply every mirror, and a missing CUDA
    mirror then raises ``ValueError`` instead of copying. Columns on the CPU
    are mirrored either way, since reading them transfers nothing.
    """
    changes = {}
    table = getattr(batch, "block_table", None)
    if (
        table is not None
        and table.start_page is not None
        and table.start_page_host is None
    ):
        if prepared is not None:
            previous = prepared.block_table
            if (
                previous.start_page is None
                or previous.start_page_host is None
                or previous.start_page.data_ptr() != table.start_page.data_ptr()
                or previous.start_page.shape != table.start_page.shape
            ):
                raise RuntimeError(
                    "prepare these table start pages before CUDA capture"
                )
            host = previous.start_page_host
        else:
            host = _read(table.start_page, "table start pages", derive)
        changes["block_table"] = replace(table, start_page_host=host)
    for name, what in (
        ("queries", "query lengths"),
        ("keys", "key lengths"),
        ("prefixes", "prefix lengths"),
    ):
        lengths = getattr(batch, name, None)
        if lengths is None or lengths.host is not None:
            continue
        if prepared is not None:
            previous = getattr(prepared, name)
            if (
                previous.host is None
                or previous.values.shape != lengths.values.shape
                or previous.values.data_ptr() != lengths.values.data_ptr()
                or previous.offsets.data_ptr() != lengths.offsets.data_ptr()
            ):
                raise RuntimeError(
                    "prepare these sequence lengths before CUDA capture"
                )
            host = previous.host
        else:
            host = _read(lengths.values, what, derive)
        changes[name] = replace(lengths, host=host)
    return replace(batch, **changes) if changes else batch


def _read(values, what, derive):
    """Mirror one metadata column on the host, as ``host_lengths`` permits."""
    if values.is_cuda:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"prepare host {what} before CUDA capture")
        if not derive:
            raise ValueError(
                f"attention planning requires host {what}, which this "
                "context never reads from the device; stage them with the "
                "call's attention input"
            )
    return tuple(values.cpu().tolist())


def batch_host_lengths(batch, *, derive=True):
    """Mirror every entry of an ``AttentionBatch`` with one read per column.

    Entries share their query lengths, so the shared domain is read once and
    each entry keeps borrowing the same device columns. Reading device values
    during CUDA capture is forbidden, and without ``derive`` a missing CUDA
    mirror raises, as for ``host_lengths``.
    """
    from uniserve.nn.attention.inputs import AttentionBatch

    if batch.queries is None:
        return batch
    queries = (
        batch.queries
        if batch.queries.host is not None
        else host_lengths(_Queries(batch.queries), derive=derive).queries
    )
    entries = {
        table: host_lengths(replace(entry, queries=queries), derive=derive)
        for table, entry in batch.entries.items()
    }
    return AttentionBatch(entries, queries)


@dataclass(frozen=True, slots=True)
class _Queries:
    """A query-only view letting ``host_lengths`` mirror a shared domain."""

    queries: SequenceLengths


def _lengths(lengths, start, stop):
    # Rebase cumulative offsets to zero at the run's first sequence.
    return SequenceLengths(
        host=lengths.host[start:stop],
        values=lengths.values[start:stop],
        offsets=lengths.offsets[start : stop + 1] - sum(lengths.host[:start]),
    )


def causal_runs(batch):
    """Yield contiguous runs without changing sequence order or visible values.

    Homogeneous batches use their original borrowed views. Mixed causality is
    a mathematical property of sequences, independent of serving workloads.
    """
    if len(set(batch.causal)) <= 1:
        yield slice(0, batch.queries.num_tokens), slice(None), batch
        return

    query_start = key_start = 0
    for _, rows in groupby(
        range(len(batch.causal)), key=batch.causal.__getitem__
    ):
        members = tuple(rows)
        start, stop = members[0], members[-1] + 1
        queries = _lengths(batch.queries, start, stop)
        query_slice = slice(query_start, query_start + queries.num_tokens)
        changes = {"queries": queries, "causal": batch.causal[start:stop]}

        if isinstance(batch, PagedInput):
            # Paged keys stay in the cache; only the query domain is sliced.
            table = batch.block_table
            changes.update(
                prefixes=_lengths(batch.prefixes, start, stop),
                block_table=BlockTable(
                    table.indices[start:stop],
                    table.block_size,
                    None
                    if table.start_page is None
                    else table.start_page[start:stop],
                    None
                    if table.start_page_host is None
                    else table.start_page_host[start:stop],
                ),
                write_indices=None,
            )
            key_slice = slice(None)
        else:
            keys = _lengths(batch.keys, start, stop)
            changes["keys"] = keys
            key_slice = slice(key_start, key_start + keys.num_tokens)
            key_start += keys.num_tokens

        yield query_slice, key_slice, replace(batch, **changes)
        query_start += queries.num_tokens
