"""Layer ownership and tensor transfer for iterative pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.distributed.mesh import Communicator


@dataclass(frozen=True)
class LayerPipeline:
    """Bind disjoint layers and recurrence to an ordered pipeline group.

    Peers share the same tensor and sequence coordinates. The existing worker
    command lane supplies request identity and step order; every stage enters
    each forward and feedback operation for that request. Callers retain tensor
    storage through completion on the current execution stream.
    """

    group: Communicator
    layer_count: int

    def __post_init__(self) -> None:
        if self.layer_count < self.group.world_size:
            raise ValueError("every pipeline stage requires at least one model layer")

    @property
    def layers(self) -> range:
        rank, size = self.group.rank_in_group, self.group.world_size
        return range(self.layer_count * rank // size, self.layer_count * (rank + 1) // size)

    @property
    def first(self) -> bool:
        return self.group.rank_in_group == 0

    @property
    def last(self) -> bool:
        return self.group.rank_in_group == self.group.world_size - 1

    def receive_activation(self, *states: torch.Tensor) -> None:
        """Receive numerical states separately, preserving their declared dtypes."""

        if not self.first:
            for state in states:
                self.group.recv(state, src=self.group.rank_in_group - 1)

    def send_activation(self, *states: torch.Tensor) -> None:
        """Publish numerical states in the receiving stage's declared order."""

        if not self.last:
            for state in states:
                self.group.send(state, dst=self.group.rank_in_group + 1)

    def nonresident_layer_names(self, prefix: str, parameters: tuple[str, ...]) -> frozenset[str]:
        """Enumerate valid off-stage parameter names from an architecture's layer schema."""

        return frozenset(
            f"{prefix}.{layer}.{parameter}"
            for layer in range(self.layer_count)
            if layer not in self.layers
            for parameter in parameters
        )

    def feedback(self, products: tuple[torch.Tensor, ...]) -> None:
        """Return final-stage solver products to the first stage in tuple order.

        The two endpoints provide identical tensor shapes, dtypes and order.
        Intermediate stages need no copy: they consume the next forward's
        activations. A singleton pipeline already owns its updated products.
        """

        if self.group.world_size == 1:
            return
        for product in products:
            if self.last:
                self.group.send(product, dst=0)
            elif self.first:
                self.group.recv(product, src=self.group.world_size - 1)
