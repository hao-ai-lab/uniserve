"""Backend collectives behind the logical model communicators.

Numerical code calls :class:`uniserve.distributed.Communicator`, which orders
logical members. This module resolves each call to the stream-bound
communicator selected by the active execution scope or, without one, to the
initialized process group, and owns asynchronous completion.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

import torch
import torch.distributed as dist

from uniserve.profiling import profile_range


class StreamCollectives(Protocol):
    """Ordered collectives for one computation binding.

    Ordered collectives and deferred transfers for one computation binding.
    """

    def all_reduce(self, value: torch.Tensor, op: str = "sum") -> None: ...
    def all_gather(self, output: torch.Tensor, value: torch.Tensor) -> None: ...
    def start_all_gather(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> Any: ...
    def start_all_to_all(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> Any: ...
    def all_to_all(
        self,
        output: torch.Tensor,
        value: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> None: ...
    def gather(
        self, outputs: list[torch.Tensor] | None, value: torch.Tensor, root: int
    ) -> None: ...
    def broadcast(self, value: torch.Tensor, root: int) -> None: ...
    def reduce_scatter(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> None: ...

    def send(self, value: torch.Tensor, peer: int) -> None: ...
    def recv(self, value: torch.Tensor, peer: int) -> None: ...
    def send_recv(
        self, output: torch.Tensor, value: torch.Tensor, dst: int, src: int
    ) -> None: ...


_BOUND_COLLECTIVES: ContextVar[Mapping[str, StreamCollectives] | None] = (
    ContextVar("stream_collectives", default=None)
)


@contextmanager
def stream_collective_scope(
    bindings: Mapping[str, StreamCollectives] | None,
) -> Iterator[None]:
    """Select communication resources for one numerical invocation.

    ``None`` selects initialized process-group communication on an ordinary
    device stream. A mapping selects explicit stream bindings and must cover
    every used communicator, including when that mapping is empty.
    """
    token = _BOUND_COLLECTIVES.set(bindings)
    try:
        yield
    finally:
        _BOUND_COLLECTIVES.reset(token)


def stream_collectives(group_name: str) -> StreamCollectives | None:
    """Resolve a stream provider.

    Resolve a stream provider, rejecting incomplete explicit execution
    bindings.

    An unscoped numerical caller may use the initialized process group. Once a
    caller selects an explicit stream scope, every used group must belong to it;
    falling back would enqueue communication on a different execution stream.
    """
    bindings = _BOUND_COLLECTIVES.get()
    if bindings is None:
        return None
    try:
        return bindings[group_name]
    except KeyError:
        raise RuntimeError(
            f"stream scope has no binding for communicator {group_name!r}"
        ) from None


class Transfer:
    """A started collective whose consumer joins its completion.

    Outputs are readable on the consumer's stream after :meth:`wait`, which
    enqueues the dependency without blocking the host for CUDA transfers.
    """

    def __init__(self, work: Any, device: torch.device) -> None:
        self._work, self._device = work, device

    def wait(self) -> None:
        """Join the transfer; later calls do nothing."""
        work, self._work = self._work, None
        if work is None:
            return
        if self._device.type == "cuda":
            work.block_current_stream()
        else:
            work.wait()


def start_all_gather(
    output: torch.Tensor, input: torch.Tensor, group
) -> Transfer:
    """Publish a gather into backend-ordered ``output`` and return it.

    The range covers the launch rather than the transfer, which is what a
    profile correlates a device kernel back to.
    """
    with profile_range(
        f"uniserve.collective kind=all_gather_start "
        f"group={group.group_name} rank={dist.get_rank()}"
    ):
        bound = stream_collectives(group.group_name)
        if bound is not None:
            work = bound.start_all_gather(output, input)
        else:
            work = dist.all_gather_into_tensor(
                output, input, group=group, async_op=True
            )
    return Transfer(work, input.device)


def start_all_to_all(
    output: torch.Tensor, input: torch.Tensor, group
) -> Transfer:
    """Publish an equal-split exchange in backend member order."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        work = bound.start_all_to_all(output, input)
    else:
        work = dist.all_to_all_single(output, input, group=group, async_op=True)
    return Transfer(work, input.device)


def all_gather(output: torch.Tensor, input: torch.Tensor, group) -> None:
    """Gather equal contributions into ``output`` in backend rank order.

    ``output`` is contiguous storage of any shape holding one ``input``-sized
    contribution per member.
    """
    with profile_range(
        f"uniserve.collective kind=all_gather group={group.group_name} "
        f"rank={dist.get_rank()}"
    ):
        # In-place AllGather registers one stable allocation for both source
        # and destination. Symmetric workspaces can then use NCCL copy engines.
        # Both operands are passed flat: Gloo accepts a concatenation of the
        # contributions along the leading axis but not a stack of them.
        flat = output.view(-1)
        local = flat.narrow(0, group.rank() * input.numel(), input.numel())
        local.view_as(input).copy_(input)
        bound = stream_collectives(group.group_name)
        if bound is not None:
            # Immediate consumers need no fork onto the transport stream.
            bound.all_gather(flat, local)
        else:
            start_all_gather(flat, local, group).wait()


def all_to_all(
    output: torch.Tensor,
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group,
) -> None:
    """Exchange leading-axis partitions in backend rank order."""
    with profile_range(
        f"uniserve.collective kind=all_to_all group={group.group_name} "
        f"rank={dist.get_rank()}"
    ):
        bound = stream_collectives(group.group_name)
        if bound is not None:
            bound.all_to_all(output, input, output_splits, input_splits)
            return
        # Empty split lists select the native equal-count collective. Explicit
        # lists select variable-count send/recv, including when counts match.
        equal_counts = (
            input.numel() == output.numel()
            and len(set(output_splits + input_splits)) == 1
        )
        work = dist.all_to_all_single(
            output,
            input,
            output_split_sizes=None if equal_counts else output_splits,
            input_split_sizes=None if equal_counts else input_splits,
            group=group,
            async_op=True,
        )
        Transfer(work, input.device).wait()


def send_recv(
    value: torch.Tensor,
    output: torch.Tensor,
    dst: int,
    src: int,
    group,
) -> None:
    """Send ``value`` bytes to ``dst`` and receive ``output`` from ``src``."""
    with profile_range(
        f"uniserve.collective kind=send_recv group={group.group_name} "
        f"rank={dist.get_rank()}"
    ):
        bound = stream_collectives(group.group_name)
        if bound is not None:
            bound.send_recv(output, value, dst, src)
            return
        calls = [
            dist.P2POp(
                dist.isend, value.reshape(-1).view(torch.uint8), dst, group
            ),
            dist.P2POp(
                dist.irecv, output.reshape(-1).view(torch.uint8), src, group
            ),
        ]
        for work in dist.batch_isend_irecv(calls):
            Transfer(work, value.device).wait()


_REDUCTIONS = {
    "sum": dist.ReduceOp.SUM,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}


def all_reduce(value: torch.Tensor, op: str, group) -> None:
    """Reduce ``value`` in place over ``group``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.all_reduce(value, op)
    else:
        dist.all_reduce(value, op=_REDUCTIONS[op], group=group)


def gather(
    gather_list: list[torch.Tensor] | None,
    value: torch.Tensor,
    dst: int,
    group,
) -> None:
    """Gather ``value`` into ``gather_list`` on global rank ``dst``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.gather(gather_list, value, dst)
        return
    work = dist.gather(
        value, gather_list=gather_list, dst=dst, group=group, async_op=True
    )
    Transfer(work, value.device).wait()


def broadcast(value: torch.Tensor, src: int, group) -> None:
    """Broadcast ``value`` in place from global rank ``src``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.broadcast(value, src)
    else:
        dist.broadcast(value, src=src, group=group)


def reduce_scatter(output: torch.Tensor, value: torch.Tensor, group) -> None:
    """Sum backend-ordered partitions of ``value`` into ``output``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.reduce_scatter(output, value)
    else:
        dist.reduce_scatter_tensor(output, value, group=group)


def send(payload: torch.Tensor, dst: int, group) -> None:
    """Send contiguous bytes to global rank ``dst``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.send(payload, dst)
    else:
        dist.send(payload, dst=dst, group=group)


def recv(payload: torch.Tensor, src: int, group) -> None:
    """Receive contiguous bytes from global rank ``src``."""
    bound = stream_collectives(group.group_name)
    if bound is not None:
        bound.recv(payload, src)
    else:
        dist.recv(payload, src=src, group=group)
