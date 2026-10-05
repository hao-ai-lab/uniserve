"""Chunked tensor exchange over borrowed numerical communicators."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from math import prod

import torch

ChunkProducer = Callable[[slice, tuple[torch.Tensor, ...]], None]


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

    # Scratch layout per segment: [group.size, count, width] with one member
    # slot per rank in backend order. Segments stay near 64 MiB of staged
    # source rows, rounded down to whole 128-row units.
    storage = (
        workspace.view(torch.uint8).view(-1)[:byte_count].view(input.dtype)
    )
    segment_rows = max(
        1, ((64 * 1024 * 1024) // (width * input.element_size()) // 128) * 128
    )
    local_rank = group.backend_order.index(group.rank)

    segments = []
    for start in range(0, rows, segment_rows):
        count = min(segment_rows, rows - start)
        sources = storage.narrow(
            0, start * group.size * width, count * group.size * width
        )
        sources = sources.view(group.size, count, width)
        sources[local_rank].copy_(input[start : start + count])
        segments.append((start, count, sources))

    pending = []
    consumed = 0
    try:
        for _, _, sources in segments:
            pending.append(
                group.start_all_gather(
                    sources.flatten(0, 1), sources[local_rank]
                )
            )

        begin = group.rank * rows
        yield slice(begin, begin + rows), input
        for (start, count, sources), work in zip(
            segments, pending, strict=True
        ):
            work.wait()
            consumed += 1
            for backend_rank, logical_rank in enumerate(group.backend_order):
                if backend_rank != local_rank:
                    begin = logical_rank * rows + start
                    yield slice(begin, begin + count), sources[backend_rank]
    finally:
        # Cancellation drains pending transfers before the caller can lend
        # this scratch to another projection on its computation stream.
        for work in pending[consumed:]:
            work.wait()


def produce_exchange(
    group,
    source: torch.Tensor,
    destination: torch.Tensor,
    producer: Callable[[tuple[torch.Tensor, ...]], None],
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Exchange equal peer payloads and return their deferred completion.

    Physical buffers have a leading member axis. Producer and consumer
    views use logical member order. Contiguous registered buffers permit
    NCCL's zero-CTA AlltoAll; tensor layout and numerical work belong to
    the caller. Both buffers must remain live until completion is consumed.
    """
    if (
        source.ndim < 2
        or source.shape[0] != group.size
        or source.shape != destination.shape
        or source.dtype != destination.dtype
        or source.device != destination.device
        or not source.is_contiguous()
        or not destination.is_contiguous()
        or source.data_ptr() == destination.data_ptr()
    ):
        raise ValueError(
            "produced exchange requires distinct matching peer buffers"
        )

    order = group.backend_order
    producer(tuple(source[order.index(rank)] for rank in range(group.size)))
    if group.size == 1:
        destination.copy_(source)
        work = None
    else:
        work = group.start_all_to_all(destination, source)

    def complete() -> tuple[torch.Tensor, ...]:
        if work is not None:
            work.wait()
        return tuple(
            destination[order.index(rank)] for rank in range(group.size)
        )

    return complete


def _produce_chunks(
    group,
    shape: tuple[int, ...],
    workspace: torch.Tensor,
    output: torch.Tensor,
    chunk_rows: int,
    producer: ChunkProducer,
) -> Iterator[tuple[slice, tuple[torch.Tensor, ...]]]:
    """Exchange row intervals as a stream-ordered producer writes them.

    Shape axes are logical destination, row, and payload. The producer
    writes each supplied logical destination view on the current stream.
    It must finish enqueuing its writes before returning. Transport owns
    ordering and scratch layout, and starts each exchange immediately.
    All production is enqueued before yielding to consumers, allowing its
    temporary inputs to be released before consumer allocations begin.
    Both distinct scratch buffers remain live until iterator exhaustion.
    """
    if (
        len(shape) < 3
        or shape[0] != group.size
        or min(shape) < 1
        or chunk_rows < 1
    ):
        raise ValueError(
            "row production requires a member axis and positive row chunks"
        )
    if (
        not workspace.is_contiguous()
        or not output.is_contiguous()
        or workspace.numel() != prod(shape)
        or output.numel() != prod(shape)
        or workspace.device != output.device
        or workspace.dtype != output.dtype
        or workspace.data_ptr() == output.data_ptr()
    ):
        raise ValueError(
            "row production requires distinct matching contiguous buffers"
        )

    rows = shape[1]
    row_elements = prod(shape[2:])
    source_flat, target_flat = workspace.view(-1), output.view(-1)
    segments = []
    consumed = 0
    try:
        for start in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - start)
            offset = start * group.size * row_elements
            elements = count * group.size * row_elements
            segment_shape = (group.size, count, *shape[2:])
            source = source_flat.narrow(0, offset, elements).view(segment_shape)
            target = target_flat.narrow(0, offset, elements).view(segment_shape)
            interval = slice(start, start + count)
            complete = produce_exchange(
                group, source, target, partial(producer, interval)
            )
            segments.append((interval, complete))

        del producer
        for interval, complete in segments:
            values = complete()
            consumed += 1
            yield interval, values
    finally:
        for _, complete in segments[consumed:]:
            complete()
