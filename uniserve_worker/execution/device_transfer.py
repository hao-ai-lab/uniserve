"""In-process tensor placement for execution across modality devices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..nn.mesh import GroupCoordinator


@dataclass(frozen=True)
class DeviceTransfer:
    """Copy stage tensors to an explicitly assigned in-process device."""

    devices: tuple[torch.device, ...] = ()

    def dispatch(self, value: torch.Tensor, coordinate: int) -> torch.Tensor:
        if not self.devices:
            return value
        if not 0 <= coordinate < len(self.devices):
            raise ValueError(f"device coordinate {coordinate} is outside placement")
        return value.to(self.devices[coordinate], non_blocking=True)

    def combine(self, value: torch.Tensor, target: torch.device) -> torch.Tensor:
        return value.to(target, non_blocking=True) if value.device != target else value


@dataclass(frozen=True)
class ComponentTensorTransfer:
    """Transfer component products over one physical worker command lane.

    Producers and consumers use their declared logical order. Tensor contents
    and any reduction/assembly semantics remain with the component consumer.
    Callers own the send and receive buffers through completion of the operation.
    """

    group: "GroupCoordinator"
    producers: tuple[int, ...]
    consumers: tuple[int, ...]

    def __post_init__(self) -> None:
        for members in (self.producers, self.consumers):
            if not members or len(set(members)) != len(members):
                raise ValueError("component transfer requires nonempty unique membership")
            if any(rank not in self.group.ranks for rank in members):
                raise ValueError("component transfer members must belong to its process group")

    def broadcast(self, value: torch.Tensor | None, output: torch.Tensor | None) -> None:
        """Send one canonical product only to the component's input owners."""

        if len(self.producers) != 1:
            raise ValueError("canonical product broadcast requires exactly one producer")
        source = self.producers[0]
        rank = self.group.rank
        if rank == source:
            if value is None:
                raise ValueError("component producer must provide its tensor product")
            for destination in self.consumers:
                if destination == source:
                    if output is None:
                        raise ValueError("local component consumer requires output storage")
                    output.copy_(value)
                else:
                    self.group.send(value, dst=self.group.ranks.index(destination))
        elif rank in self.consumers:
            if output is None:
                raise ValueError("component consumer requires output storage")
            self.group.recv(output, src=self.group.ranks.index(source))

    def exchange(self, send: torch.Tensor, receive: torch.Tensor) -> None:
        """Exchange one equal-shaped product per producer/consumer pair.

        Send storage has [consumer, ...] shape on producers and [0, ...]
        elsewhere. Receive storage has [producer, ...] shape on consumers and
        [0, ...] elsewhere. Every process enters this cooperative operation.
        """

        from math import prod

        rank = self.group.rank
        send_count = len(self.consumers) if rank in self.producers else 0
        receive_count = len(self.producers) if rank in self.consumers else 0
        if send.ndim < 2 or tuple(send.shape[1:]) != tuple(receive.shape[1:]):
            raise ValueError("component products must agree on their item shape")
        if send.shape[0] != send_count or receive.shape[0] != receive_count:
            raise ValueError("component transfer storage disagrees with membership")
        if send.dtype != receive.dtype or send.device != receive.device:
            raise ValueError("component transfer storage must share dtype and device")
        if self.producers == self.group.ranks and len(self.consumers) == 1:
            self.group.gather_into_tensor(
                receive if receive_count else None,
                send[0],
                dst=self.group.ranks.index(self.consumers[0]),
            )
            return
        item_size = prod(send.shape[1:])
        output_splits = [
            item_size if receive_count and member in self.producers else 0
            for member in self.group.ranks
        ]
        input_splits = [
            item_size if send_count and member in self.consumers else 0
            for member in self.group.ranks
        ]
        consumer_order = tuple(
            self.consumers.index(member) for member in self.group.ranks if member in self.consumers
        )
        producer_order = tuple(
            self.producers.index(member) for member in self.group.ranks if member in self.producers
        )
        outgoing = send
        if send_count and consumer_order != tuple(range(send_count)):
            outgoing = torch.stack([send[index] for index in consumer_order])
        incoming = receive
        if receive_count and producer_order != tuple(range(receive_count)):
            incoming = torch.empty_like(receive)
        self.group.all_to_all_single_into(
            incoming.reshape(-1), outgoing.reshape(-1), output_splits, input_splits
        )
        if incoming is not receive:
            for source, destination in enumerate(producer_order):
                receive[destination].copy_(incoming[source])
