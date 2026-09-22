"""Shared transformer traversal and mathematical activation communication."""

from __future__ import annotations

import torch
from torch import nn

from uniserve import cache
from uniserve.cache import mha
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.distributed.tokens import TokenShard
from uniserve.nn.attention import Attention, AttentionInput, VarlenInput
from uniserve.nn.functional import add_rms_norm
from uniserve.nn.norm import RMSNorm
from uniserve.nn.routing import RoutedTensor, RouteSpan


class TransformerDecoder(nn.Module):
    """Traverse resident layers, carrying a separate residual stream.

    Layers return (hidden, residual). Pipeline peers exchange those two values
    with their original dtypes; only the final stage normalizes and gathers the
    token shards. Placement binding selects resident modules before loading.
    """

    def __init__(
        self,
        embedding,
        layers: nn.ModuleDict,
        norm: nn.Module,
        *,
        default_route: str | None = None,
    ):
        super().__init__()
        if not layers:
            raise ValueError(
                "a transformer decoder requires at least one layer"
            )
        self.embedding, self.layers, self.norm = embedding, layers, norm
        self.mesh = DeviceMesh(ranks=(0,), shape=(1,), axes=("tp",), rank=0)
        self._pipeline = Communicator()
        self._tokens = Communicator()
        self.hidden_size = embedding.embedding_dim
        self.vocab_size = embedding.num_embeddings
        # Logical cache identities survive pipeline pruning. Resource owners
        # use this order to describe global K/V transfers without retaining
        # parameters or modules belonging to another pipeline stage.
        self.cache_names = tuple(
            child.cache_name
            for layer in layers.values()
            for child in layer.modules()
            if isinstance(child, Attention) and child.cache_name is not None
        )
        if default_route is not None and (
            not isinstance(norm, nn.ModuleDict) or default_route not in norm
        ):
            raise ValueError(
                "a default expert route must name one of the decoder's "
                "output norms"
            )
        self._default_route = default_route

    @property
    def cache_config(self) -> cache.Config:
        result = {}
        for layer in self.layers.values():
            # Parameters expose their logical dtype even for encoded weights.
            dtype = next(layer.parameters()).dtype
            for child in layer.modules():
                if (
                    isinstance(child, Attention)
                    and child.cache_name is not None
                ):
                    result[child.cache_name] = mha.Config(
                        child.num_kv_heads,
                        child.head_dim,
                        child.head_indices,
                        dtype,
                    )
        return cache.Config(result)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._pipeline.rank != 0:
            raise ValueError(
                "token embeddings belong to the first pipeline stage"
            )
        return self.embedding(input_ids)

    def forward(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        attention: AttentionInput,
        *,
        routes: tuple[RouteSpan, ...] = (),
    ) -> torch.Tensor:
        count = positions.shape[-1]
        if not routes and self._default_route is not None:
            routes = (RouteSpan(self._default_route, 0, count),)
        partition = TokenShard(count, self._tokens)
        positions = partition.local(positions, dim=positions.ndim - 1)
        if self._pipeline.rank == 0:
            if embeddings is None or embeddings.shape != (
                count,
                self.hidden_size,
            ):
                raise ValueError(
                    "the first stage requires complete packed token embeddings"
                )
            hidden = partition.local(embeddings)
            residual = None
        else:
            reference = next(self.layers.parameters())
            hidden = reference.new_empty((partition.count, self.hidden_size))
            residual = torch.empty_like(hidden)
            for value in (hidden, residual):
                self._pipeline.recv(src=self._pipeline.rank - 1, out=value)

        # Clip route spans to this rank's token shard, in shard-local offsets.
        local_routes = ()
        if routes:
            local_routes = tuple(
                RouteSpan(
                    span.route,
                    max(span.start, partition.token_slice.start)
                    - partition.token_slice.start,
                    min(span.stop, partition.token_slice.stop)
                    - max(span.start, partition.token_slice.start),
                )
                for span in routes
                if span.stop > partition.token_slice.start
                and span.start < partition.token_slice.stop
            )
            keys = frozenset(span.route for span in routes)
            hidden = RoutedTensor.from_packed(hidden, local_routes, routes=keys)
            if residual is not None:
                residual = RoutedTensor.from_packed(
                    residual, local_routes, routes=keys
                )

        for layer in self.layers.values():
            if routes:
                hidden, residual = layer(
                    hidden, residual, positions, attention, routes=local_routes
                )
            else:
                hidden, residual = layer(hidden, residual, positions, attention)

        last = self._pipeline.rank == self._pipeline.size - 1
        if not last:
            for value in (hidden, residual):
                packed = (
                    value.packed(local_routes)
                    if isinstance(value, RoutedTensor)
                    else value
                )
                self._pipeline.send(packed, dst=self._pipeline.rank + 1)
            return (
                hidden.packed(local_routes)
                if isinstance(hidden, RoutedTensor)
                else hidden
            )

        # Fold the residual into the output norm only on the final stage.
        if isinstance(hidden, RoutedTensor):
            result = hidden.add(residual).apply(self.norm).packed(local_routes)
        elif isinstance(self.norm, RMSNorm):
            result, _ = add_rms_norm(
                hidden, residual, self.norm.weight, self.norm.eps
            )
        else:
            result = self.norm(hidden + residual)
        return partition.gather(result)


class TransformerEncoder(nn.Module):
    """Evaluate ordinary encoder layers over the same packed sequence domain."""

    def __init__(self, layers: nn.ModuleList, norm: nn.Module):
        super().__init__()
        self.layers, self.norm = layers, norm

    def forward(
        self, features: torch.Tensor, attention: VarlenInput
    ) -> torch.Tensor:
        if (
            attention.queries.num_tokens is not None
            and features.shape[0] != attention.queries.num_tokens
        ):
            raise ValueError(
                "encoder features must cover the declared query sequences"
            )
        for layer in self.layers:
            features = layer(features, attention)
        return self.norm(features)
