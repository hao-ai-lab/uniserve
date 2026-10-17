"""Chunked row gathers and exchanges over borrowed numerical communicators."""

from __future__ import annotations

from collections.abc import Iterator

import torch

# Bytes of one staged row interval: large enough that each peer publication
# amortizes its launch, small enough that consumers start on the first
# interval while later transfers are in flight.
_PAYLOAD_BYTES = 64 * 1024 * 1024
# Bytes one member sends all its peers in one exchanged row interval, which
# sets the interval's transfer time. An exchange returns rows whose consumers
# (an output projection and a feed-forward update) take longer per row than
# the transfer, so smaller intervals than a gather's start them sooner: the
# first interval's transfer is the part of the exchange its consumers cannot
# hide.
_EXCHANGE_BYTES = 48 * 1024 * 1024
# Intervals hold whole units of this many rows, a multiple of every sparse
# attention tile, so consumers see tile-aligned intervals.
_ROW_UNIT = 128


def payload_rows(width: int, itemsize: int) -> int:
    """Return the rows of one gathered interval of ``width``-element rows.

    The bound is 64 MiB of one member's rows, rounded down to whole
    128-row units and never below one unit.
    """
    return max(
        _ROW_UNIT, _PAYLOAD_BYTES // (width * itemsize) // _ROW_UNIT * _ROW_UNIT
    )


def exchange_rows(row_bytes: int, peers: int) -> int:
    """Return the rows of one exchanged interval of ``row_bytes``-byte rows.

    ``row_bytes`` is what one member sends one peer per row and ``peers``
    the members it sends to. The bound is 48 MiB sent per member, rounded
    down to whole 128-row units and never below one unit.
    """
    return max(
        _ROW_UNIT,
        _EXCHANGE_BYTES // (row_bytes * max(peers, 1)) // _ROW_UNIT * _ROW_UNIT,
    )


def gather_chunks(
    group, input: torch.Tensor, workspace: torch.Tensor
) -> Iterator[tuple[slice, torch.Tensor]]:
    """Expose complete local rows, then ready remote intervals in logical order.

    All source staging precedes asynchronous gathers. Consumers may enqueue
    numerical work between yields while subsequent transfers make progress.
    Workspace is caller-owned and remains live until iterator exhaustion.
    """
    if input.ndim != 2 or min(input.shape) < 1:
        raise ValueError("row gathering requires a nonempty matrix")
    rows, width = input.shape
    if group.size == 1:
        yield slice(0, rows), input
        return

    byte_count = input.numel() * input.element_size() * group.size
    if not workspace.is_contiguous() or workspace.device != input.device:
        raise ValueError(
            "row gathering requires contiguous scratch on the input device"
        )
    if workspace.numel() * workspace.element_size() < byte_count:
        raise ValueError("row gathering scratch cannot hold all input rows")

    # Scratch layout per interval: [group.size, count, width] with one member
    # slot per rank in backend order.
    storage = (
        workspace.view(torch.uint8).view(-1)[:byte_count].view(input.dtype)
    )
    interval_rows = payload_rows(width, input.element_size())
    local_rank = group.backend_order.index(group.rank)

    intervals = []
    for start in range(0, rows, interval_rows):
        count = min(interval_rows, rows - start)
        sources = storage.narrow(
            0, start * group.size * width, count * group.size * width
        )
        sources = sources.view(group.size, count, width)
        sources[local_rank].copy_(input[start : start + count])
        intervals.append((start, count, sources))

    pending = []
    consumed = 0
    try:
        for _, _, sources in intervals:
            pending.append(
                group.start_all_gather(
                    sources.flatten(0, 1), sources[local_rank]
                )
            )

        begin = group.rank * rows
        yield slice(begin, begin + rows), input
        for (start, count, sources), work in zip(
            intervals, pending, strict=True
        ):
            work.wait()
            consumed += 1
            for backend_rank, logical_rank in enumerate(group.backend_order):
                if backend_rank != local_rank:
                    begin = logical_rank * rows + start
                    yield slice(begin, begin + count), sources[backend_rank]
    finally:
        # Cancellation retires published transfers before the caller can lend
        # this scratch to another projection on its computation stream.
        for work in pending[consumed:]:
            work.wait()
