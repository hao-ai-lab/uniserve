"""Numerical sequence slicing for kernels with one causality flag per launch."""

from dataclasses import replace
from itertools import groupby

import torch

from uniserve.nn.attention.inputs import BlockTable, PagedInput, SequenceLengths


def host_lengths(batch, *, prepared=None):
    """Supply exact mirrors at preparation, retaining borrowed device columns.

    During capture, prepared mirrors must describe these same fixed addresses.
    Reading device values is forbidden there; replay metadata is prepared by
    the caller before each invocation that requires host-driven planning.
    """
    changes = {}
    for name in ("queries", "keys", "prefixes"):
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
            if (
                lengths.values.is_cuda
                and torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError(
                    "prepare host sequence lengths before CUDA capture"
                )
            host = tuple(lengths.values.cpu().tolist())
        changes[name] = replace(lengths, host=host)
    return replace(batch, **changes) if changes else batch


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
            changes.update(
                prefixes=_lengths(batch.prefixes, start, stop),
                block_table=BlockTable(
                    batch.block_table.indices[start:stop],
                    batch.block_table.block_size,
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
