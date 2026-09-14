"""Logical packed rows and their sequence-parallel storage ownership."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.attention.metadata import RouteSpan
from uniserve.distributed.mesh import Communicator


@dataclass(frozen=True)
class SequencePartition:
    """Own a contiguous logical interval with equal-capacity transport storage.

    Padding exists only in collective payloads. Model math, expert routing and
    paged-cache writes see the original logical rows, including empty shards.
    """

    rows: int
    group: Communicator

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError("sequence execution requires positive global rows")

    @property
    def capacity(self) -> int:
        return (self.rows + self.group.world_size - 1) // self.group.world_size

    @property
    def start(self) -> int:
        return min(self.rows, self.capacity * self.group.rank_in_group)

    @property
    def count(self) -> int:
        return min(self.capacity, self.rows - self.start)

    def local(self, value: torch.Tensor, *, axis: int = 0) -> torch.Tensor:
        """Return the local logical rows of an unpartitioned tensor."""

        if value.shape[axis] != self.rows:
            raise ValueError("sequence input does not match its global row extent")
        return value.narrow(axis, self.start, self.count).contiguous()

    def pad(self, value: torch.Tensor) -> torch.Tensor:
        """Create a transport payload without introducing logical model rows."""

        if value.shape[0] != self.count:
            raise ValueError("sequence payload does not match its owned row extent")
        if self.count == self.capacity:
            return value.contiguous()
        padded = value.new_zeros((self.capacity, *value.shape[1:]))
        padded[: self.count].copy_(value)
        return padded

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        """Materialize the global logical rows in communicator member order."""

        if self.group.world_size == 1:
            return value
        local = self.pad(value)
        gathered = local.new_empty((self.capacity * self.group.world_size, *local.shape[1:]))
        self.group.all_gather_into_tensor(gathered, local)
        return gathered[: self.rows]

    def routes(self, spans: tuple[RouteSpan, ...]) -> tuple[RouteSpan, ...]:
        """Slice and rebase scheduler route intervals without creating zero-token spans."""

        end = self.start + self.count
        return tuple(
            RouteSpan(
                span.route,
                max(self.start, span.token_start) - self.start,
                min(end, span.token_end) - max(self.start, span.token_start),
            )
            for span in spans
            if span.token_start < end and span.token_end > self.start
        )
